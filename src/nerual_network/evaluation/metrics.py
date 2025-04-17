#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File name: metrics.py
Author: Filip Cerny
Created: 17.04.2025
Version: 1.0
Description: 
"""
import json
import logging
import os

import numpy as np
import torch
import trimesh
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

from data_processing.class_clustering import ClusteredCenterPointsAllFrames
from data_processing.class_mapping import SurfacePointsFrameList, TimeFrame
from nerual_network.class_model import NNDataset
from nerual_network.helpers import LoadedModelDic, ClusterIndex, CentersMetricsInfo, \
    _run_model_decoder_all_times_with_selected_encoder_time
from nerual_network.loss_functions import run_through_nn_at_decoder_time
from utils.constants import NN_DEVICE_STR, TrainConfig
from utils.helpers import get_meshes_list


def _compute_variance(evaluation_results_list):
    variance_list = []
    for evaluation_result in evaluation_results_list._list:
        variance_list_list = []
        for decoder_element in evaluation_result.decoder_pair_list._list:
            unique_ids = decoder_element.pair_processed_center.get_unique_ids()
            for id in unique_ids:
                pair_list = decoder_element.pair_processed_center.get_decoder_element_by_id(id)
                distances = [pair.distance for pair in pair_list]
                # compute statistical dispersion
                variance = np.var(distances)
                variance_list_list.append({"variance": variance, "id": id})
        variance_list.append({"time": evaluation_result.encoder_time, "variance_list": variance_list_list})

    return variance_list


def compute_centers_metrics2(data: SurfacePointsFrameList, loaded_models: LoadedModelDic, num_points: int) -> list[
    torch.tensor]:
    logging.info(f"START: computing centers metrics for points: {num_points}")
    device = torch.device(NN_DEVICE_STR)
    clusteres_indexes_unique = data.get_unique_clusters_indexes()

    time_list = data.get_time_list()

    closest_centers_matrix = np.array(data.create_all_frames_all_points_closest_centers_indices(), dtype=int)

    num_points_cluster = num_points // len(clusteres_indexes_unique)

    variances_list = []

    for cluster_index in clusteres_indexes_unique:
        data_filtered_cluster = data.filter_by_label(cluster_index)
        model = loaded_models[ClusterIndex(cluster_index)]

        dataset = NNDataset(data_filtered_cluster)
        input_tensor = torch.tensor(dataset.data, dtype=torch.float32).to(device)

        for time in time_list:
            time_index = time.index

            # filter by time index
            filtered_input_tensor = NNDataset.filter_by_time_index(input_tensor, time_index)
            # select num_points_cluster randomly from input_tensor
            filtered_input_tensor = NNDataset.select_random_values(filtered_input_tensor, num_points_cluster)

            # CLOSEST CENTERS INDICES
            closest_centers_indices_tensor = get_closes_centers_indices_by_points_and_time_index(closest_centers_matrix,
                                                                                                 filtered_input_tensor,
                                                                                                 time, 1).to(device)

            # INPUT POINTS
            inputs_points = NNDataset.get_points_columns(filtered_input_tensor)

            # CENTERS POINTS at input_time
            centers_point_inputs_time = get_centers_points_by_time_and_closestcentersindicies(data,
                                                                                              closest_centers_indices_tensor,
                                                                                              time).to(device)

            # DISTANCE at input time - input_points and center_points
            distances_input_time = torch.norm(inputs_points - centers_point_inputs_time, dim=1)

            all_distance_differences = []  # To store differences between input and decoded distances
            for decoder_time in time_list:
                # DECODED POINTS
                with torch.no_grad():
                    processed_points = run_through_nn_at_decoder_time(inputs=filtered_input_tensor, model=model, decoder_time=decoder_time)

                # CENTERS POINTS at decoded_time
                centers_point_decoded_time = get_centers_points_by_time_and_closestcentersindicies(data,
                                                                                                   closest_centers_indices_tensor,
                                                                                                   decoder_time).to(device)

                # DISTANCE at decoded time - processed_points and center_points
                distances_decoded_time = torch.norm(processed_points - centers_point_decoded_time, dim=1)

                # Compute the difference between the input and decoded distances for this time step
                distance_difference = distances_input_time - distances_decoded_time

                # Store the distance differences for variance calculation
                all_distance_differences.append(distance_difference)

            # Now, compute variance of the differences across time for each point
            all_distance_differences_tensor = torch.stack(all_distance_differences,
                                                          dim=1)  # Shape: (num_points, num_time_steps)
            variance_per_point_difference = torch.var(all_distance_differences_tensor,
                                                      dim=1)  # Variance across time (for each point)

            # Save the variance for each point (converted to numpy for saving)
            variances_list.append(variance_per_point_difference)

    logging.info(f"END: computing centers metrics for points: {num_points}")
    return variances_list


def compute_save_centers_metrics(centers_metrics_info: CentersMetricsInfo, filepath: str) -> torch.Tensor:
    variance_list = compute_centers_metrics2(centers_metrics_info.data, centers_metrics_info.loaded_models,
                                             centers_metrics_info.num_points)



    variances_list_numpy = [variance.detach().cpu().numpy() for variance in variance_list]
    # Save the variance list to a file
    np.save(filepath, variances_list_numpy)

    logging.info(f"Variance list saved to {filepath}")

    # create from list of tensors one tensor where all values of tensors are in one dimension
    variance_tensor = torch.cat(variance_list, dim=0)

    return variance_tensor


def get_centers_points_by_time_and_closestcentersindicies(data: SurfacePointsFrameList,
                                                          closest_centers_indices_tensor: torch.Tensor, time: TimeFrame) -> torch.Tensor:
    all_centers_info_input_time = data.get_element_by_time_index(time.index).normalized_centers_info
    all_input_centers_points = all_centers_info_input_time.points
    all_input_centers_points = torch.tensor(all_input_centers_points).to(closest_centers_indices_tensor.device)
    centers_point_inputs_time = all_input_centers_points[closest_centers_indices_tensor]
    return centers_point_inputs_time


def get_closes_centers_indices_by_points_and_time_index(closest_centers_matrix: np.ndarray, input_tensor: torch.Tensor,
                                                        time: TimeFrame, num_closest_centers : int) -> torch.Tensor:
    inputs_points_index_column = NNDataset.get_point_indices_column(input_tensor)
    inputs_points_index_column = [int(element) for element in inputs_points_index_column]
    closest_centers_filtered = closest_centers_matrix[:, :, num_closest_centers - 1]  # selects only one closest center
    closest_centers_indices = closest_centers_filtered[time.index][inputs_points_index_column]
    closest_centers_indices_tensor = torch.tensor(closest_centers_indices)
    return closest_centers_indices_tensor


def _compute_mesh_shape_metrics(surface_data_list: SurfacePointsFrameList, train_config: TrainConfig,
                                clustered_data: ClusteredCenterPointsAllFrames, nn_lr):
    """
    Computes metrics which:
    1. loads mesh in one specified time
    2. runs these mesh points through the encoder in specified time
    3. runs output through the decoder in all times
    4. compares the output of an time with a mesh in the same time
    :param surface_data_list:
    :param train_config:
    :return:
    """

    def compute_laplacian_eigenvalues(mesh, k=10):
        """
        Compute the first k eigenvalues of the Laplace-Beltrami operator for a mesh.

        Args:
            mesh (trimesh.Trimesh): Trimesh mesh object.
            k (int): Number of smallest eigenvalues to compute.

        Returns:
            np.ndarray: Sorted eigenvalues.
        """
        # Get the adjacency matrix of the mesh
        adjacency_matrix = mesh.vertex_adjacency_matrix

        # Compute the Laplacian matrix
        laplacian = csgraph.laplacian(adjacency_matrix, normed=True)

        # Compute the smallest k eigenvalues of the Laplacian matrix
        eigenvalues, _ = eigsh(laplacian, k=k, which='SM')
        return np.sort(eigenvalues)

    def compute_similarity(original_mesh, processed_mesh):
        # Compute eigenvalues
        eigenvalues1 = compute_laplacian_eigenvalues(original_mesh)
        eigenvalues2 = compute_laplacian_eigenvalues(processed_mesh)
        # Compare eigenvalues (e.g., L2 distance)
        return np.linalg.norm(eigenvalues1 - eigenvalues2)

    def label_points_by_clustered_data(surface_data_list: SurfacePointsFrameList,
                                       clustered_data: ClusteredCenterPointsAllFrames):
        raise NotImplementedError("Not implemented yet")
        #
        # final_surface_data_list = SurfacePointsFrameList([])
        #
        # for i, surface_data_frame in enumerate(surface_data_list.list):
        #     mesh_points = surface_data_frame.points_list
        #     centers_points_frame = clustered_data.points_allframes[i]
        #     centers_labels_frame = clustered_data.labels_frame
        #     # check indexes with filepath names
        #
        #     surface_labels = categorize_points_with_labels(centers_labels_frame, centers_points_frame, mesh_points)
        #     # append both values to list with names in the list
        #     final_surface_data_list.append(SurfacePointsFrame(mesh_points, surface_labels, None))
        # return final_surface_data_list

    mesh_folder_path = train_config.file_path_config.raw_data_folderpath
    loaded_meshes_list = get_loaded_meshes_list(mesh_folder_path)

    all_vertices = []
    for mesh in loaded_meshes_list:
        # Access vertices as a NumPy array
        vertices = mesh.vertices
        all_vertices.append(vertices)

    mesh_points_allframes: SurfacePointsFrameList = _convert_to_surfacepointsframelist(all_vertices)
    mesh_points_allframes = label_points_by_clustered_data(mesh_points_allframes, clustered_data)

    mesh_points_allframes.assign_time_to_all_elements()
    mesh_points_allframes.normalize_all_elements()

    time = 0
    # todo is not working - change
    original_points_all, processed_points_all, cluster_labels = _run_model_decoder_all_times_with_selected_encoder_time(
        surface_data_list=mesh_points_allframes, time_index=time, loaded_models=LoadedModelDic)

    if len(loaded_meshes_list) != len(surface_data_list.public_list) and len(mesh_points_allframes.public_list) != len(
            surface_data_list.public_list):
        raise Exception("Not same number of loaded meshes and surface data")

    similarity_list = []
    # compare the output of the decoder with the mesh in the same time
    for i, surface_data_frame in enumerate(surface_data_list.public_list):
        if surface_data_frame.time.index != i:
            raise Exception("Not same time")
        time = surface_data_frame.time.value

        loaded_mesh_timeframe = loaded_meshes_list[i]

        mesh_points_timeframe = mesh_points_allframes.public_list[i]
        if mesh_points_timeframe.time.index != i:
            raise Exception("Not same time")
        mesh_points_timeframe_points = mesh_points_timeframe.normalized_points_list

        processed_points_timeframe = processed_points_all[processed_points_all[:, 3] == time]

        # convert to meshes
        original_mesh = trimesh.Trimesh(vertices=mesh_points_timeframe_points, faces=loaded_mesh_timeframe.faces)
        processed_mesh = trimesh.Trimesh(vertices=processed_points_timeframe, faces=loaded_mesh_timeframe.faces)

        similarity = compute_similarity(original_mesh, processed_mesh)
        similarity_list.append({"time": time, "similarity": similarity})

    return similarity_list


def _convert_to_surfacepointsframelist(all_vertices):
    raise NotImplementedError("Not implemented yet")
    # # transform mesh vertices to  and normalize them and add time
    # mesh_points_list = SurfacePointsFrameList([])
    # for vertices in all_vertices:
    #     mesh_points = SurfacePointsFrame(vertices)
    #     mesh_points_list.append(mesh_points)
    # return mesh_points_list


def get_loaded_meshes_list(meshes_folder_path: str):
    meshes_filepaths_list = get_meshes_list(meshes_folder_path)
    loaded_meshes_list = []
    for mesh_filepath in meshes_filepaths_list:
        mesh = trimesh.load(mesh_filepath)
        loaded_meshes_list.append(mesh)
    return loaded_meshes_list


def save_metrics_centers_pipeline(evaluation_folderpath, loaded_models, surface_data_list, train_config):
    eval_surface_points_num = 100
    variance_tensor = compute_save_centers_metrics(
        CentersMetricsInfo(surface_data_list, loaded_models, eval_surface_points_num),
        train_config.file_path_config.center_metric_eval_filepath)
    standard_deviation_tensor = torch.sqrt(variance_tensor)
    mean_variance = variance_tensor.mean()
    # Prepare dictionary
    metrics_dict = {
        "standard_deviation": standard_deviation_tensor.item(),
        "mean_variance": mean_variance.item()
    }
    filepath = os.path.join(evaluation_folderpath, "centers_metrics_values.json")
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    logging.info(f"centrics metrics saved to file {filepath}")
