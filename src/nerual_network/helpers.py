#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File name: helpers.py
Author: Filip Cerny
Created: 10.04.2025
Version: 1.0
Description: 
"""
from typing import Any

import numpy as np
import torch
from numpy import ndarray, dtype
from torch.utils.data import DataLoader

from data_processing.class_mapping import SurfacePointsFrameList
from nerual_network.class_model import NNDataset
from nerual_network.loss_functions import _add_time_column
from utils.constants import TrainConfig, NN_DEVICE_STR
from utils.nn_config_utils import init_training_config


def get_closest_centers_indices(closest_centers_indicies_all_frames : np.ndarray, inputs : torch.tensor, input_time_index : int) -> np.ndarray:
    inputs_points_index_column = NNDataset.get_point_indices_column(inputs)
    inputs_points_index_column = [int(element) for element in inputs_points_index_column]
    closest_centers_indices = closest_centers_indicies_all_frames[input_time_index][inputs_points_index_column]

    return closest_centers_indices


def _run_model_with_one_encoder_time_to_all_decoder_times_prepare_for_visualization(
        surface_data_list: SurfacePointsFrameList, time_index: int, train_config: TrainConfig) -> tuple[
    np.ndarray, list[np.ndarray]]:
    def __prepare_data_for_visualization(original_points_all: list[np.ndarray],
                                         processed_points_all: list[np.ndarray]) -> tuple[ndarray, list[np.ndarray]]:
        # convert original_points_all to only array of points
        original_points_list_of_points = np.vstack(original_points_all)
        # get only x, y, z from points
        original_points_list_of_points = original_points_list_of_points[:, :3]
        rgb_colors = __convert_points_to_colors(original_points_list_of_points)

        # Extract unique time values assuming the last column contains time values
        # convert processed_points_all to only array of points
        processed_points_list_of_points = []
        for processed_points_one_cluster in processed_points_all:
            processed_points_list_of_points.extend(processed_points_one_cluster)
        processed_points_list_of_points = np.vstack(processed_points_list_of_points)

        processed_points_split_by_time_value = NNDataset.split_by_time_value(processed_points_list_of_points)
        # get only x,y,z from points
        processed_points_split_by_time_value = [points[:, :3] for points in processed_points_split_by_time_value]

        return rgb_colors, processed_points_split_by_time_value

    original_points_all, processed_points_all = _run_model_decoder_all_times_with_selected_encoder_time(
        surface_data_list=surface_data_list, time_index=time_index, train_config=train_config)
    return __prepare_data_for_visualization(original_points_all, processed_points_all)


def _run_model_decoder_all_times_with_selected_encoder_time(surface_data_list: SurfacePointsFrameList,
                                                            time_index: int, train_config: TrainConfig) -> tuple[
    list[ndarray[Any, dtype[Any]]], list[list[ndarray[Any, dtype[Any]]]]]:
    model_weights_template = train_config.file_path_config.model_weights_folderpath_template
    batch_size = train_config.nn_config.batch_size

    original_points_all = []
    processed_points_all = []

    unique_clusters = surface_data_list.get_unique_clusters_indexes()

    # select original points where time is 0
    original_points_frame = surface_data_list.get_element_by_time_index(time_index)

    # iterate over clusters
    for i, cluster in enumerate(unique_clusters):
        if i >= train_config.num_clusters:
            break
        # Load the original surface points for the current cluster
        surface_data_cluster_timeframe = original_points_frame.filter_by_label(cluster)

        # Create a SurfaceDataset instance with the filtered surface data
        original_points_dataset_one_cluster = NNDataset(SurfacePointsFrameList([surface_data_cluster_timeframe]))

        # Prepare a DataLoader for original points
        original_points_loader = DataLoader(original_points_dataset_one_cluster, batch_size=batch_size, shuffle=False)

        # Load the trained model for the current cluster
        model_weights_filepath = model_weights_template.format(cluster=cluster)
        model = _load_trained_model(model_weights_filepath, train_config)
        device = torch.device(NN_DEVICE_STR)
        model.to(device)

        encoded_features = []

        # Step 1: Encode the original data
        with torch.no_grad():  # No need to calculate gradients during evaluation
            for inputs in original_points_loader:
                inputs = inputs[0].float().to(device)
                encoder_input_data = NNDataset.get_encoder_input(inputs)
                encoded_features_element = model.encoder(encoder_input_data)
                encoded_features.append(encoded_features_element)

        # Process the original points through the model
        encoded_features = torch.cat(encoded_features).to(device)

        processed_points_one_cluster = []
        # Step 2: Decode in all times
        for i, surface_points_frame in enumerate(surface_data_list.public_list):
            if surface_points_frame.time.index != i:
                raise Exception("Not same time")

            time_value = surface_points_frame.time.value
            encoded_with_time = _add_time_column(encoded_features, time_value)

            # Pass through the decoder
            decoded_output = model.decoder(encoded_with_time)
            # Add the time_value as a column to decoded_output

            decoded_output = decoded_output.cpu().detach().numpy()
            time_column = np.full((decoded_output.shape[0], 1), time_value)
            decoded_output_with_time = np.hstack((decoded_output, time_column))

            # Append the modified decoded output
            processed_points_one_cluster.append(decoded_output_with_time)

        # Convert processed points to a single numpy array
        processed_points_all.append(processed_points_one_cluster)
        # Accumulate all original and processed points
        original_points_all.append(
            np.array(original_points_dataset_one_cluster.data))  # You can store the numpy array directly

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
