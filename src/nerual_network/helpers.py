#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File name: helpers.py
Author: Filip Cerny
Created: 10.04.2025
Version: 1.0
Description: 
"""
import logging
from dataclasses import dataclass
from typing import TypeAlias, Dict

import numpy as np
import torch
from torch import nn
from trimesh import Trimesh

from data_processing.class_mapping import SurfacePointsFrameList
from nerual_network.class_model import NNDataset
from nerual_network.loss_functions import run_through_encoder, run_through_decoder_at_time
from utils.constants import TrainConfig, NN_DEVICE_STR, ModelType
from utils.nn_config_utils import init_training_config, init_model

RGBColorArray : TypeAlias = np.ndarray
ProcessedPointsListSplitByTimeValue : TypeAlias = dict[int, np.ndarray]


class ClusterIndex(int):
    pass


@dataclass
class VisualizationData:
    points: np.ndarray
    colors: np.ndarray
    """values normalized to 0-1"""

    def __post_init__(self):
        if len(self.points) != len(self.colors):
            raise ValueError("points and rgb_colors must have the same length")


class LoadedModelDic(Dict[ClusterIndex, nn.Module]):
    pass

@dataclass
class CentersMetricsInfo:
    data: SurfacePointsFrameList
    loaded_models: LoadedModelDic
    num_points: int


@dataclass
class NNOutputForVisualization:
    rgb_colors: RGBColorArray
    processed_points: ProcessedPointsListSplitByTimeValue

#
# def get_closest_centers_indices(closest_centers_indicies_all_frames : np.ndarray, inputs : torch.tensor, input_time_index : int) -> np.ndarray:
#     inputs_points_index_column = NNDataset.get_point_indices_column(inputs)
#     inputs_points_index_column = [int(element) for element in inputs_points_index_column]
#     closest_centers_indices = closest_centers_indicies_all_frames[input_time_index][inputs_points_index_column]
#
#     return closest_centers_indices


def _run_model_with_one_encoder_time_to_all_decoder_times_prepare_for_visualization(
        surface_data_list: SurfacePointsFrameList, time_index: int, loaded_models : LoadedModelDic) -> NNOutputForVisualization:

    def __prepare_data_for_visualization(original_points_all: list[torch.tensor],
                                         processed_points_all: list[list[torch.tensor]]) -> NNOutputForVisualization:
        """

        :param original_points_all:
        :param processed_points_all:
        :return:
        rgb_colors: numpy array of shape (num_points, 3) - RGB colors for each point
        processed_points_split_by_time_value: dict: key: time_index
                                                    value: numpy arrays, each of shape (num_points, 3) - processed points
        """

        # region original_points_all
        # convert original_points_all to only array of points
        original_points_combined = torch.cat(original_points_all, dim=0)

        # region SANITY CHECK
        # check if time is the same for all elements
        time_indices = NNDataset.get_time_indices_column(original_points_combined)
        if not (time_indices == time_indices[0]).all(dim=1).all():
            raise ValueError("All time indices must be the same for original points.")
        time_values = NNDataset.get_time_values_column(original_points_combined)
        if not (time_values == time_values[0]).all(dim=1).all():
            raise ValueError("All time values must be the same for original points.")

        # check if point indexes are not duplicit
        point_indices = NNDataset.get_point_indices_column(original_points_combined)
        if not torch.unique(point_indices).numel() == point_indices.numel():
            raise ValueError("Point indices must be unique for original points.")
        # endregion

        # sort by point index
        #indices = torch.lexsort([NNDataset.get_point_indices_column(original_points_combined)])
        indices = torch.argsort(NNDataset.get_point_indices_column(original_points_combined).squeeze(), dim=0)
        original_points_combined_sorted = original_points_combined[indices]

        # select only points
        original_points_combined_only_points = NNDataset.get_points_columns(original_points_combined_sorted)

        # convert it to numpy
        original_points_combined_only_points_numpy = original_points_combined_only_points.cpu().detach().numpy()

        rgb_colors = __convert_points_to_colors(original_points_combined_only_points_numpy)
        # endregion

        # region processed_points_all
        # convert processed_points to only one big tensor
        processed_points_combined = torch.cat([torch.cat(points, dim=0) for points in processed_points_all], dim=0)

        # sort by time index
        indices = torch.argsort(NNDataset.get_time_indices_column(processed_points_combined).squeeze(),dim=0)
        processed_points_combined_sorted = processed_points_combined[indices]

        # create dic of processed points split by time index
        processed_points_split_by_time_index : dict[int, np.ndarray] = dict()
        # iterate over time indices
        unique_time_indices = torch.unique(NNDataset.get_time_indices_column(processed_points_combined_sorted).squeeze())
        if unique_time_indices.numel() <= 1:
            raise ValueError("There must be more than one time index in processed points.")

        unique_time_indices_list = unique_time_indices.tolist()
        for current_time_index in unique_time_indices_list:
            # filter by time index
            time_indices = NNDataset.get_time_indices_column(processed_points_combined_sorted).squeeze()
            processed_points_filtered = processed_points_combined_sorted[time_indices == current_time_index]

            # sort by point index
            indices = torch.argsort(NNDataset.get_point_indices_column(processed_points_filtered).squeeze(), dim=0)
            processed_points_filtered_sorted = processed_points_filtered[indices]

            # region SANITY CHECK
            # check if point indexes are not duplicit
            point_indices = NNDataset.get_point_indices_column(processed_points_filtered_sorted)
            if not torch.unique(point_indices).numel() == point_indices.numel():
                raise ValueError("Point indices must be unique for processed points.")

            # check if original_point indices column is the same as processed_points indices column
            original_point_indices = NNDataset.get_point_indices_column(original_points_combined_sorted)
            processed_point_indices = NNDataset.get_point_indices_column(processed_points_filtered_sorted)
            if not torch.equal(original_point_indices, processed_point_indices):
                raise ValueError("Original point indices must be the same as processed points indices.")

            # check if time_value column is same for all elements
            time_values = NNDataset.get_time_values_column(processed_points_filtered_sorted)
            if not (time_values == time_values[0]).all(dim=1).all():
                raise ValueError("All time values must be the same for processed points.")
            # check if time index is the same for all elements
            check_time_indices = NNDataset.get_time_indices_column(processed_points_filtered_sorted)
            if not (check_time_indices == check_time_indices[0]).all(dim=1).all():
                raise ValueError("All time indices must be the same for processed points.")
            # endregion

            # select only points
            processed_points_filtered_sorted_only_points = NNDataset.get_points_columns(processed_points_filtered_sorted)

            # convert it to numpy
            processed_points_filtered_sorted_only_points_numpy = processed_points_filtered_sorted_only_points.detach().cpu().numpy()

            # add to list
            processed_points_split_by_time_index[int(current_time_index)] = processed_points_filtered_sorted_only_points_numpy

        # endregion

        return NNOutputForVisualization(rgb_colors=rgb_colors,
                                        processed_points=processed_points_split_by_time_index)

    original_points_all_tensor, processed_points_all_tensor = _run_model_decoder_all_times_with_selected_encoder_time(
        surface_data_list=surface_data_list, time_index=time_index, loaded_models=loaded_models)

    return __prepare_data_for_visualization(original_points_all_tensor, processed_points_all_tensor)


def _run_model_decoder_all_times_with_selected_encoder_time(surface_data_list: SurfacePointsFrameList, time_index: int,
                                                            loaded_models : LoadedModelDic) -> tuple[
    list[torch.tensor], list[list[torch.tensor]]]:
    """

    :param loaded_models:
    :param surface_data_list:
    :param time_index:
    :return:
    original_points_all:  list in a shape (num_clusters, num_points) element - data in format of NNDataset_points
    processed_points_all: list in a shape (num_clusters, num_times, num_points) element - data in format of NNDataset_points
    """

    device = torch.device(NN_DEVICE_STR)
    time_list = surface_data_list.get_time_list()

    original_points_all = []
    """ list in a shape (num_clusters, num_points) element - data in format of NNDataset_points"""
    processed_points_all = []
    """ list in a shape (num_clusters, num_times, num_points) element - data in format of NNDataset_points"""

    unique_clusters = loaded_models.keys()

    # select original points where time is 0
    original_points_frame = surface_data_list.get_element_by_time_index(time_index)

    original_points_frame_dataset = NNDataset(SurfacePointsFrameList([original_points_frame]))
    original_points_frame_tensor = torch.tensor(original_points_frame_dataset.data, dtype=torch.float32).to(device)

    # iterate over clusters
    for cluster_index in unique_clusters:
        # Load the trained model for the current cluster
        model = loaded_models[ClusterIndex(cluster_index)]

        # Prepare a DataLoader for original points
        input_tensor = NNDataset.filter_by_cluster_label(input_tensor=original_points_frame_tensor, cluster_index=cluster_index)

        input_tensor_point_indices = NNDataset.get_point_indices_column(input_tensor)
        input_tensor_point_labels = NNDataset.get_point_cluster_label_column(input_tensor)

        original_points_all.append(input_tensor)  # You can store the numpy array directly

        encoded_features = run_through_encoder(inputs=input_tensor, encoder=model.encoder)
        # encoded_features = []
        #
        # # Step 1: Encode the original data
        # with torch.no_grad():  # No need to calculate gradients during evaluation
        #     for inputs in original_points_loader:
        #         inputs = inputs[0].float().to(device)
        #         encoder_input_data = NNDataset.get_encoder_input(inputs)
        #         encoded_features_element = model.encoder(encoder_input_data)
        #         encoded_features.append(encoded_features_element)
        #
        # # Process the original points through the model
        # encoded_features = torch.cat(encoded_features).to(device)

        processed_points_one_cluster = []
        # Step 2: Decode in all times
        for time in time_list:
            decoded_output_tensor = run_through_decoder_at_time(encoded_output=encoded_features,
                                                                decoder=model.decoder,
                                                                time=time)
            # .cpu().detach().numpy())

            time_index_tensor = torch.full((decoded_output_tensor.shape[0], 1), time.index, device=device)
            time_value_tensor = torch.full((decoded_output_tensor.shape[0], 1), time.value, device=device)

            # add columns (x,y,z) from decoded_output_tensor and metadata columns from input_tensor_metadata
            decoded_output_tensor_with_metadata = NNDataset.create_tensor(point_columns_tensor=decoded_output_tensor,
                                                                           time_value_column_tensor=time_value_tensor,
                                                                            time_index_column_tensor=time_index_tensor,
                                                                            point_indices_column_tensor=input_tensor_point_indices,
                                                                          point_cluster_label_column_tensor=input_tensor_point_labels)

            # Append the modified decoded output
            processed_points_one_cluster.append(decoded_output_tensor_with_metadata)

        # Convert processed points to a single numpy array
        processed_points_all.append(processed_points_one_cluster)

    return original_points_all, processed_points_all


def __convert_points_to_colors(points) -> np.ndarray:
    # Funkce pro normalizaci pro jednotlivé osy
    def normalize_slices(data):
        normalized = np.zeros_like(data)

        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)

        for index, value in enumerate(data):
            normalized_value = (value - min_val) / (max_val - min_val)

            normalized[index] = normalized_value

        return normalized

    # Normalizace podle osy X, Y, Z
    normalized_x = normalize_slices(points[:, 0])
    normalized_y = normalize_slices(points[:, 1])
    normalized_z = normalize_slices(points[:, 2])

    # Spojení barev
    rgb_colors = np.stack([normalized_x, normalized_y, normalized_z], axis=1)

    return rgb_colors


def _load_trained_model(model_weights_filepath: str, train_config: TrainConfig):
    """
    Loads the trained neural network model weights from a specified file path.

    Args:
        model_weights_filepath (str): The path to the file containing the model weights.
        input_size (int): The number of input features for the model.
        hidden_size (int): The number of hidden units in the model.
        output_size (int): The number of output features for the model.

    Returns:
        model (nn.Module): The loaded neural network model.
    """

    # Load the checkpoint
    model, optimizer, loss_function = init_training_config(train_config)

    checkpoint = torch.load(model_weights_filepath)  # Load the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])  # Load the model state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load the optimizer state
    epoch = checkpoint['epoch']  # Get the epoch number
    val_loss = checkpoint['val_loss']  # Get the validation loss

    # Set the model to evaluation mode
    model.eval()

    return model


def load_trained_nn_from_files(train_config: TrainConfig) -> LoadedModelDic:

    logging.info(f"START: loading nn from model weights files {train_config.file_path_config.model_weights_folderpath_template}")
    loaded_models = LoadedModelDic()
    num_clusters = train_config.num_clusters
    model_type = train_config.nn_config.model_type
    model_weights_template = train_config.file_path_config.model_weights_folderpath_template
    for i in range(num_clusters):
        cluster_index = ClusterIndex(i + 1)
        filepath = model_weights_template.format(cluster=cluster_index)

        loaded_model = load_trained_nn_from_file(filepath, model_type)
        loaded_models[cluster_index] = loaded_model

    logging.info(
        f"END: loading nn from model weights files")
    return loaded_models


MeshStruct : TypeAlias = Trimesh
Folderpath : TypeAlias = str


@dataclass
class MeshData:
    time_index: int


@dataclass
class ProcessedMeshData:
    processed_visualization_data: NNOutputForVisualization
    origin_mesh: MeshStruct


def load_trained_nn_from_file(model_weights_filepath: str, model_type: ModelType) -> nn.Module:
    # Load the checkpoint
    model = init_model(model_type)
    checkpoint = torch.load(model_weights_filepath)  # Load the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])  # Load the model state

    # Set the model to evaluation mode
    model.eval()

    # set device
    device = torch.device(NN_DEVICE_STR)
    model.to(device)

    return model
