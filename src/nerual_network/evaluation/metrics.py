#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File name: metrics.py
Author: Filip Cerny
Created: 17.04.2025
Version: 1.0
Description: 
"""
import csv
import json
import logging
import os
import pickle
import re
import subprocess
from itertools import groupby

import numpy as np
import torch

from data_processing.class_clustering import ClusteredCenterPointsAllFrames
from data_processing.class_mapping import SurfacePointsFrameList, TimeFrame
from nerual_network.class_model import NNDataset
from nerual_network.evaluation.meshes import process_mesh_through_model, MeshDataVisualizer
from nerual_network.helpers import MeshFilepathsDic, ClusterIndex, CentersMetricsInfo, \
    FilePath, CenterMetricsElement, MetroMetrics, TimeIndex, LoadedModelDic, MeshData
from nerual_network.loss_functions import run_through_nn_at_decoder_time
from utils.constants import NN_DEVICE_STR, TrainConfig
from utils.helpers import load_pickle_file


#
# def _compute_variance(evaluation_results_list):
#     variance_list = []
#     for evaluation_result in evaluation_results_list._list:
#         variance_list_list = []
#         for decoder_element in evaluation_result.decoder_pair_list._list:
#             unique_ids = decoder_element.pair_processed_center.get_unique_ids()
#             for id in unique_ids:
#                 pair_list = decoder_element.pair_processed_center.get_decoder_element_by_id(id)
#                 distances = [pair.distance for pair in pair_list]
#                 # compute statistical dispersion
#                 variance = np.var(distances)
#                 variance_list_list.append({"variance": variance, "id": id})
#         variance_list.append({"time": evaluation_result.encoder_time, "variance_list": variance_list_list})
#
#     return variance_list

# region CENTERS METRICS


def _get_centers_points_by_time_and_closestcentersindicies(data: SurfacePointsFrameList,
                                                           closest_centers_indices_tensor: torch.Tensor, time: TimeFrame) -> torch.Tensor:
    all_centers_info_input_time = data.get_element_by_time_index(time.index).normalized_centers_info
    all_input_centers_points = all_centers_info_input_time.points
    all_input_centers_points = torch.tensor(all_input_centers_points).to(closest_centers_indices_tensor.device)
    centers_point_inputs_time = all_input_centers_points[closest_centers_indices_tensor]
    return centers_point_inputs_time


def _get_closes_centers_indices_by_points_and_time_index(closest_centers_matrix: np.ndarray, input_tensor: torch.Tensor,
                                                         time: TimeFrame, num_closest_centers : int) -> torch.Tensor:
    inputs_points_index_column = NNDataset.get_point_indices_column(input_tensor)
    inputs_points_index_column = [int(element) for element in inputs_points_index_column]
    closest_centers_filtered = closest_centers_matrix[:, :, num_closest_centers - 1]  # selects only one closest center
    closest_centers_indices = closest_centers_filtered[time.index][inputs_points_index_column]
    closest_centers_indices_tensor = torch.tensor(closest_centers_indices)
    return closest_centers_indices_tensor


def compute_centers_metrics2(data: SurfacePointsFrameList, loaded_models: LoadedModelDic, num_points: int) -> dict[int, list[
    CenterMetricsElement]]:
    """
    returns list of tensors with distance differences (original point - closest_center) - FOR_ALL_TIME(processed_point - closest_center)
    :param data:
    :param loaded_models:
    :param num_points:
    :return:
    """
    logging.info(f"START: computing centers metrics for points: {num_points}")
    device = torch.device(NN_DEVICE_STR)
    clusteres_indexes_unique = data.get_unique_clusters_indexes()

    time_list = data.get_time_list()

    closest_centers_matrix = np.array(data.create_all_frames_all_points_closest_centers_indices(), dtype=int)

    num_points_cluster = num_points // len(clusteres_indexes_unique)

    centers_metrics_data_list = []

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
            closest_centers_indices_tensor = _get_closes_centers_indices_by_points_and_time_index(closest_centers_matrix,
                                                                                                  filtered_input_tensor,
                                                                                                  time, 1).to(device)

            # INPUT POINTS
            inputs_points = NNDataset.get_points_columns(filtered_input_tensor)

            # CENTERS POINTS at input_time
            centers_point_inputs_time = _get_centers_points_by_time_and_closestcentersindicies(data,
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
                centers_point_decoded_time = _get_centers_points_by_time_and_closestcentersindicies(data,
                                                                                                    closest_centers_indices_tensor,
                                                                                                    decoder_time).to(device)

                # DISTANCE at decoded time - processed_points and center_points
                distances_decoded_time = torch.norm(processed_points - centers_point_decoded_time, dim=1)

                # Compute the difference between the input and decoded distances for this time step
                distance_difference = distances_input_time - distances_decoded_time

                # absolute of difference
                distance_difference = torch.abs(distance_difference)

                # Store the distance differences for variance calculation
                all_distance_differences.append(distance_difference)

            # Now, compute variance of the differences across time for each point
            all_distance_differences_tensor = torch.stack(all_distance_differences,
                                                          dim=1)  # Shape: (num_points, num_time_steps)

            mean_per_point_difference = torch.mean(all_distance_differences_tensor,
                                                   dim=1)  # Variance across time (for each point)
            max_per_point_difference = torch.max(all_distance_differences_tensor,
                                                 dim=1).values  # Max distance difference across time (for each point)
            min_per_point_difference = torch.min(all_distance_differences_tensor,
                                                 dim=1).values  # Min distance difference across time (for each point)

            centers_metrics_data_list.append(CenterMetricsElement(mean_per_point_difference=mean_per_point_difference,
                                                                  max_per_point_difference=max_per_point_difference,
                                                                  min_per_point_difference=min_per_point_difference,
                                                                  time_index=time_index,
                                                                  cluster_index=cluster_index))


    #join the list elements by time_index
    sorted_centers_metrics_data_list = sorted(centers_metrics_data_list, key=lambda x: x.time_index)
    grouped_centers_metrics_list_by_time_index = {key: list(group) for key, group in groupby(sorted_centers_metrics_data_list, key=lambda x: x.time_index)}


    logging.info(f"END: computing centers metrics for points: {num_points}")
    return grouped_centers_metrics_list_by_time_index


def compute_save_centers_metrics(centers_metrics_info: CentersMetricsInfo, folderpath: str):
    centers_metrics_data_list = compute_centers_metrics2(centers_metrics_info.data, centers_metrics_info.loaded_models,
                                                         centers_metrics_info.num_points)

    filepath = os.path.join(folderpath, f"centers_metrics_values_{centers_metrics_info.num_points}.pkl")
    # save as an pickle file
    with open(filepath, 'wb') as f:
        pickle.dump(centers_metrics_data_list, f)

    logging.info(f"Centers metrics data list saved to {filepath}")

    # save as json file all values
    def convert_to_dict(element):
        return {
            'mean_per_point_difference': element.mean_per_point_difference.tolist(),
            'max_per_point_difference': element.max_per_point_difference.tolist(),
            'min_per_point_difference': element.min_per_point_difference.tolist(),
            'time_index': element.time_index,
            'cluster_index': element.cluster_index
        }

    elements_list = []
    for time_index, centers_metrics_element in centers_metrics_data_list.items():
        for element in centers_metrics_element:
            elements_list.append(convert_to_dict(element))

    # save to json file
    file_name = f"centers_metrics_all_values_{centers_metrics_info.num_points}.json"
    file_path = os.path.join(folderpath, file_name)
    with open(file_path, 'w') as f:
        json.dump(elements_list, f)


    # cat all tensors in one tensor big tensor with all points num_points * num_time_steps
    min_per_point_tensor = None
    max_per_point_tensor = None
    mean_per_point_tensor = None
    for time_index, centers_metrics_element in centers_metrics_data_list.items():
        logging.info(f"Time index: {time_index}")
        for element in centers_metrics_element:
            if min_per_point_tensor is None:
                min_per_point_tensor = element.min_per_point_difference
            else:
                min_per_point_tensor = torch.cat((min_per_point_tensor, element.min_per_point_difference))

            if max_per_point_tensor is None:
                max_per_point_tensor = element.max_per_point_difference
            else:
                max_per_point_tensor = torch.cat((max_per_point_tensor, element.max_per_point_difference))

            if mean_per_point_tensor is None:
                mean_per_point_tensor = element.mean_per_point_difference
            else:
                mean_per_point_tensor = torch.cat((mean_per_point_tensor, element.mean_per_point_difference))


    min_value = torch.min(min_per_point_tensor).item()
    max_value = torch.max(max_per_point_tensor).item()
    mean_value = torch.mean(mean_per_point_tensor).item()

    # save to json file
    dict = {
        "min": min_value,
        "max": max_value,
        "mean": mean_value
    }
    file_name = f"centers_metrics_results_{centers_metrics_info.num_points}.json"
    file_path = os.path.join(folderpath, file_name)

    with open(file_path, 'w') as f:
        json.dump(dict, f)
# endregion

# region CENTERS MASH SHAPE
# def _compute_mesh_shape_metrics(surface_data_list: SurfacePointsFrameList, train_config: TrainConfig,
#                                 clustered_data: ClusteredCenterPointsAllFrames, nn_lr):
#     """
#     Computes metrics which:
#     1. loads mesh in one specified time
#     2. runs these mesh points through the encoder in specified time
#     3. runs output through the decoder in all times
#     4. compares the output of an time with a mesh in the same time
#     :param surface_data_list:
#     :param train_config:
#     :return:
#     """
#
#     def compute_laplacian_eigenvalues(mesh, k=10):
#         """
#         Compute the first k eigenvalues of the Laplace-Beltrami operator for a mesh.
#
#         Args:
#             mesh (trimesh.Trimesh): Trimesh mesh object.
#             k (int): Number of smallest eigenvalues to compute.
#
#         Returns:
#             np.ndarray: Sorted eigenvalues.
#         """
#         # Get the adjacency matrix of the mesh
#         adjacency_matrix = mesh.vertex_adjacency_matrix
#
#         # Compute the Laplacian matrix
#         laplacian = csgraph.laplacian(adjacency_matrix, normed=True)
#
#         # Compute the smallest k eigenvalues of the Laplacian matrix
#         eigenvalues, _ = eigsh(laplacian, k=k, which='SM')
#         return np.sort(eigenvalues)
#
#     def compute_similarity(original_mesh, processed_mesh):
#         # Compute eigenvalues
#         eigenvalues1 = compute_laplacian_eigenvalues(original_mesh)
#         eigenvalues2 = compute_laplacian_eigenvalues(processed_mesh)
#         # Compare eigenvalues (e.g., L2 distance)
#         return np.linalg.norm(eigenvalues1 - eigenvalues2)
#
#     def label_points_by_clustered_data(surface_data_list: SurfacePointsFrameList,
#                                        clustered_data: ClusteredCenterPointsAllFrames):
#         raise NotImplementedError("Not implemented yet")
#         #
#         # final_surface_data_list = SurfacePointsFrameList([])
#         #
#         # for i, surface_data_frame in enumerate(surface_data_list.list):
#         #     mesh_points = surface_data_frame.points_list
#         #     centers_points_frame = clustered_data.points_allframes[i]
#         #     centers_labels_frame = clustered_data.labels_frame
#         #     # check indexes with filepath names
#         #
#         #     surface_labels = categorize_points_with_labels(centers_labels_frame, centers_points_frame, mesh_points)
#         #     # append both values to list with names in the list
#         #     final_surface_data_list.append(SurfacePointsFrame(mesh_points, surface_labels, None))
#         # return final_surface_data_list
#
#     mesh_folder_path = train_config.file_path_config.raw_data_folderpath
#     loaded_meshes_list = _get_loaded_meshes_list(mesh_folder_path)
#
#     all_vertices = []
#     for mesh in loaded_meshes_list:
#         # Access vertices as a NumPy array
#         vertices = mesh.vertices
#         all_vertices.append(vertices)
#
#     mesh_points_allframes: SurfacePointsFrameList = _convert_to_surfacepointsframelist(all_vertices)
#     mesh_points_allframes = label_points_by_clustered_data(mesh_points_allframes, clustered_data)
#
#     mesh_points_allframes.assign_time_to_all_elements()
#     mesh_points_allframes.normalize_all_elements()
#
#     time = 0
#     # todo is not working - change
#     original_points_all, processed_points_all, cluster_labels = _run_model_decoder_all_times_with_selected_encoder_time(
#         surface_data_list=mesh_points_allframes, time_index=time, loaded_models=LoadedModelDic)
#
#     if len(loaded_meshes_list) != len(surface_data_list.public_list) and len(mesh_points_allframes.public_list) != len(
#             surface_data_list.public_list):
#         raise Exception("Not same number of loaded meshes and surface data")
#
#     similarity_list = []
#     # compare the output of the decoder with the mesh in the same time
#     for i, surface_data_frame in enumerate(surface_data_list.public_list):
#         if surface_data_frame.time.index != i:
#             raise Exception("Not same time")
#         time = surface_data_frame.time.value
#
#         loaded_mesh_timeframe = loaded_meshes_list[i]
#
#         mesh_points_timeframe = mesh_points_allframes.public_list[i]
#         if mesh_points_timeframe.time.index != i:
#             raise Exception("Not same time")
#         mesh_points_timeframe_points = mesh_points_timeframe.normalized_points_list
#
#         processed_points_timeframe = processed_points_all[processed_points_all[:, 3] == time]
#
#         # convert to meshes
#         original_mesh = trimesh.Trimesh(vertices=mesh_points_timeframe_points, faces=loaded_mesh_timeframe.faces)
#         processed_mesh = trimesh.Trimesh(vertices=processed_points_timeframe, faces=loaded_mesh_timeframe.faces)
#
#         similarity = compute_similarity(original_mesh, processed_mesh)
#         similarity_list.append({"time": time, "similarity": similarity})
#
#     return similarity_list
#
#
# def _convert_to_surfacepointsframelist(all_vertices):
#     raise NotImplementedError("Not implemented yet")
#     # # transform mesh vertices to  and normalize them and add time
#     # mesh_points_list = SurfacePointsFrameList([])
#     # for vertices in all_vertices:
#     #     mesh_points = SurfacePointsFrame(vertices)
#     #     mesh_points_list.append(mesh_points)
#     # return mesh_points_list
#
#
# def _get_loaded_meshes_list(meshes_folder_path: str):
#     meshes_filepaths_list = get_meshes_list(meshes_folder_path)
#     loaded_meshes_list = []
#     for mesh_filepath in meshes_filepaths_list:
#         mesh = trimesh.load(mesh_filepath)
#         loaded_meshes_list.append(mesh)
#     return loaded_meshes_list

def _run_metro(mesh1_path, mesh2_path, metro_path="metro.exe"):
    logging.info(f"Running metro.exe with {mesh1_path} and {mesh2_path}")
    # Construct the command to run metro.exe
    cmd = [metro_path, mesh1_path, mesh2_path]

    # Run metro.exe and capture the output
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None

    return result.stdout

def _parse_metro_output(output_text) -> MetroMetrics:
    # Define regex patterns to extract max, mean, and RMS
    forward_distance_pattern = r"Forward distance \(M1 -> M2\):.*?distances:\s*max\s*:\s*([\d\.]+).*?mean\s*:\s*([\d\.]+).*?RMS\s*:\s*([\d\.]+)"
    backward_distance_pattern = r"Backward distance \(M2 -> M1\):.*?distances:\s*max\s*:\s*([\d\.]+).*?mean\s*:\s*([\d\.]+).*?RMS\s*:\s*([\d\.]+)"

    # Search for the forward distance (M1 -> M2)
    forward_match = re.search(forward_distance_pattern, output_text, re.DOTALL)
    if forward_match:
        forward_max = float(forward_match.group(1))
        forward_mean = float(forward_match.group(2))
        forward_rms = float(forward_match.group(3))
    else:
        forward_max = forward_mean = forward_rms = None

    # Search for the backward distance (M2 -> M1)
    backward_match = re.search(backward_distance_pattern, output_text, re.DOTALL)
    if backward_match:
        backward_max = float(backward_match.group(1))
        backward_mean = float(backward_match.group(2))
        backward_rms = float(backward_match.group(3))
    else:
        backward_max = backward_mean = backward_rms = None

    return MetroMetrics(
        max=forward_max,
        mean=forward_mean,
        rms=forward_rms,
    )

def _compute_one_mesh_metric(origin_mesh_filepath : FilePath, processed_mesh_filepath : FilePath, metro_exe_filepath : str) -> MetroMetrics:
    """
    Computes metrics which:
    1. loads mesh in one specified time
    2. runs these mesh points through the encoder in specified time
    3. runs output through the decoder in all times
    4. compares the output of an time with a mesh in the same time
    :param metro_exe_filepath:
    :param origin_mesh_filepath:
    :param processed_mesh_filepath:
    :param metro_path:
    :return:
    """
    # Run metro.exe and capture the output
    output_text = _run_metro(origin_mesh_filepath, processed_mesh_filepath, metro_exe_filepath)

    # Parse the output text to extract max, mean, and RMS values
    metrics = _parse_metro_output(output_text)

    return metrics

def compute_mesh_metrics(original_mesh_filepaths : MeshFilepathsDic, processed_mesh_filepaths : MeshFilepathsDic, metro_exe_filepath : str) -> dict[TimeIndex, MetroMetrics]:
    """
    Computes metrics which:
    1. loads mesh in one specified time
    2. runs these mesh points through the encoder in specified time
    3. runs output through the decoder in all times
    4. compares the output of an time with a mesh in the same time
    :param original_mesh_filepaths:
    :param processed_mesh_filepaths:
    :param metro_exe_filepath:
    :return:
    """
    logging.info(f"START: computing mesh metrics")
    # region SANITY CHECK

    # check if original_mesh_filepaths and processed_mesh_filepaths have the same keys
    if len(original_mesh_filepaths) != len(processed_mesh_filepaths):
        raise Exception("Not same number of original and processed meshes")
    if original_mesh_filepaths.keys() != processed_mesh_filepaths.keys():
        raise Exception("Not same keys in original and processed meshes")

    # endregion


    metrics_dict = dict()
    for time_index, original_mesh_filepath in original_mesh_filepaths.items():
        processed_mesh_filepath = processed_mesh_filepaths[time_index]
        metrics = _compute_one_mesh_metric(original_mesh_filepath, processed_mesh_filepath, metro_exe_filepath)
        metrics_dict[time_index] = metrics

    logging.info(f"END: computing mesh metrics")

    return metrics_dict

def compute_save_mesh_shape_metrics(folderpath : str, surface_data_list : SurfacePointsFrameList, train_config : TrainConfig, loaded_models : LoadedModelDic, mesh_time_index : int):
    # region STEP Read clustered data, surface data
    clustered_data: ClusteredCenterPointsAllFrames = load_pickle_file(
        train_config.file_path_config.session_clustered_data_filepath)
    if clustered_data is None:
        logging.error("Clustered data could not be loaded. Exiting.")
        return None

    processed_data = process_mesh_through_model(origin_mesh_data=MeshData(time_index=mesh_time_index),
                                                loaded_models=loaded_models,
                                                clustered_data=clustered_data,
                                                surface_data_list=surface_data_list
                                                )
    visualizer = MeshDataVisualizer(processed_data)

    os.makedirs(folderpath, exist_ok=True)

    mesh_files_folderpath = os.path.join(folderpath, "meshes")
    os.makedirs(mesh_files_folderpath, exist_ok=True)

    # Save origin mesh
    origin_meshes_list = surface_data_list.get_original_meshes_list()
    origin_meshes_filepaths = MeshFilepathsDic()

    for mesh_element in origin_meshes_list:
        time_index = mesh_element[0]
        mesh = mesh_element[1]

        mesh_file_path = os.path.join(mesh_files_folderpath, f"original_mesh_{time_index}.obj")
        mesh.export(mesh_file_path)
        origin_meshes_filepaths[TimeIndex(time_index)] = FilePath(mesh_file_path)

    processed_points_filepaths = visualizer.save_as_obj_file(mesh_files_folderpath)

    metrics_dict = compute_mesh_metrics(original_mesh_filepaths=origin_meshes_filepaths,
                                        processed_mesh_filepaths=processed_points_filepaths,
                                        metro_exe_filepath=train_config.file_path_config.metrics_mesh_shape_metro_filepath)

    # save as pickle
    metrics_filepath = os.path.join(folderpath, "mesh_shape_metrics.pkl")
    with open(metrics_filepath, "wb") as file:
        pickle.dump(metrics_dict, file)

    # save as csv
    metrics_csv_filepath = os.path.join(folderpath, "mesh_shape_metrics.csv")
    csv_data = []
    csv_data.append(["time_index", "max", "mean", "rms"])
    for time_index, element in metrics_dict.items():
        csv_data.append([time_index, element.max, element.mean, element.rms])
    with open(metrics_csv_filepath, "w") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)



    # compute max and mean and save to json
    max_list = []
    mean_list = []
    for time_index, element in metrics_dict.items():
        max_list.append(element.max)
        mean_list.append(element.mean)

    max_value = max(max_list)
    mean_value = sum(mean_list) / len(mean_list)

    metrics_json_filepath = os.path.join(folderpath, "mesh_shape_metrics.json")
    dict_metrics = {
        "max": max_value,
        "mean": mean_value
    }
    with open(metrics_json_filepath, "w") as file:
        json.dump(dict_metrics, file)

# endregion