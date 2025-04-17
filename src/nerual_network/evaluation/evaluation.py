import copy
import json
import logging
import os
from datetime import datetime

import numpy as np
import torch
import trimesh
from matplotlib import pyplot as plt
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from torch.utils.data import DataLoader

from data_processing.class_clustering import ClusteredCenterPointsAllFrames
from data_processing.class_mapping import SurfacePointsFrameList, TimeFrame, SurfacePointsFrame, MeshList
from data_processing.mapping import categorize_points_with_labels
from nerual_network.helpers import _run_model_with_one_encoder_time_to_all_decoder_times_prepare_for_visualization, \
    _run_model_decoder_all_times_with_selected_encoder_time, NNOutputForVisualization, \
    ClusterIndex, LoadedModelDic, VisualizationData, CentersMetricsInfo, load_trained_nn_from_files, \
    ProcessedPointsListSplitByTimeValue, Folderpath, MeshData, ProcessedMeshData, visualization_set_view_ax
from nerual_network.loss_functions import run_through_nn_at_decoder_time, run_through_nn_at_same_time
from src.nerual_network.class_model import NNDataset
from utils.constants import NN_DEVICE_STR, TrainConfig
from utils.helpers import load_pickle_file, get_meshes_list


# Restrict access to underscore-prefixed functions
def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"{name} is a private function and cannot be imported.")
    raise AttributeError(f"Module has no attribute {name}")


# region PRIVATE FUNCTIONS
def _save_pointcloud_to_file(original_points_all, processed_points_all, original_filepath, processed_filepath):
    np.savetxt(original_filepath, original_points_all, delimiter=",")
    np.savetxt(processed_filepath, processed_points_all, delimiter=",")


def _create_pointclouds_from_time_to_all_times(surface_data_list: SurfacePointsFrameList, images_save_folderpath: str,
                                               time_index: int, loaded_models : LoadedModelDic):
    def __save_pointcloud_to_file(visualization_data : NNOutputForVisualization,
                                  images_save_folderpath: str):
        rgb_colors = visualization_data.rgb_colors
        processed_points_split_by_time_value = visualization_data.processed_points

        # make dir if not made
        os.makedirs(images_save_folderpath, exist_ok=True)

        # Save the RGB values to another file
        rgb_colors_filepath = os.path.join(images_save_folderpath, 'rgb_colors.txt')
        np.savetxt(rgb_colors_filepath, rgb_colors, delimiter=",")

        logging.info(f"Saved RGB colors to {rgb_colors_filepath}")
        # todo finish - add normalization

        for time_index, processed_points_one_time_value in processed_points_split_by_time_value.items():
            processed_points_filepath = os.path.join(images_save_folderpath, f'processed_points_{time_index}.xyz')
            np.savetxt(processed_points_filepath, processed_points_one_time_value, delimiter=",")
            logging.info(f"Saved processed points to {processed_points_filepath}")

    visualization_data = _run_model_with_one_encoder_time_to_all_decoder_times_prepare_for_visualization(
        surface_data_list=surface_data_list, time_index=time_index, loaded_models=loaded_models)
    __save_pointcloud_to_file(visualization_data, images_save_folderpath)


def _visualize_all_clusters_for_each_time(surface_data_list: SurfacePointsFrameList, image_save_folder):
    """
    Visualizes the original and processed points in 3D in one image for each time slice.
    :param original_points_all:
    :param processed_points_all:
    :param image_save_folder:
    :return:
    """
    # Ensure the image save folder exists
    os.makedirs(image_save_folder, exist_ok=True)

    # Loop through each unique time value
    for i, surface_data_frame in enumerate(surface_data_list.public_list):
        if surface_data_frame.time.index != i:
            raise Exception("Not same time")

        cluster_labels = surface_data_frame.labels_list
        # transfrom surface_data_slice to array with points

        points_slice = np.array(surface_data_frame.normalized_points_list)
        # transform cluster_labels to array
        cluster_labels = np.array(cluster_labels)
        _visualize_clusters(points_slice, cluster_labels, image_save_folder, f'time_{i}_clusters_time.png')


def _visualize_combined_surface_points_for_each_time(original_points_all, processed_points_all,
                                                     image_save_folder):
    """
    Visualizes the original and processed points in 3D in one image for each time slice.
    :param original_points_all:
    :param processed_points_all:
    :param image_save_folder:
    :return:
    """
    logging.info("START: Visulazing for each time")
    # Ensure the image save folder exists
    os.makedirs(image_save_folder, exist_ok=True)

    # Extract unique time values assuming the last column contains time values
    unique_times = np.unique(original_points_all[:, 3])

    # Loop through each unique time value
    for i, time in enumerate(unique_times):
        original_points_slice = original_points_all[original_points_all[:, 3] == time]
        processed_points_slice = processed_points_all[processed_points_all[:, 3] == time]

        _visualize_combined_surface_points_for_one_time(image_save_folder, original_points_slice,
                                                        processed_points_slice,
                                                        f'time_{i}_combined_surface_points_time.png', time)

        # visulize clusters
    logging.info("END: Visulazing for each time")


def _visualize_points_with_time(original_points_all, processed_points_all, image_save_folderpath):
    """
    Visualizes the original and processed points in 3D in one image with time as color.
    :param original_points_all:
    :param processed_points_all:
    :param image_save_folder:
    :return:
    """
    os.makedirs(image_save_folderpath, exist_ok=True)

    # Create a new figure for the 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extracting x, y, z coordinates and time from original points
    original_x = original_points_all[:, 0]
    original_y = original_points_all[:, 1]
    original_z = original_points_all[:, 2]
    original_time = original_points_all[:, 3]  # Assuming the time column is the 4th column

    # Extracting x, y, z coordinates from processed points
    processed_x = processed_points_all[:, 0]
    processed_y = processed_points_all[:, 1]
    processed_z = processed_points_all[:, 2]
    processed_time = processed_points_all[:, 3]  # Assuming the time column is the 4th column

    # Scatter plot for original points (using time for color)
    scatter_original = ax.scatter(original_x, original_y, original_z,
                                  c=original_time, cmap='viridis', label='Original Points', alpha=0.5, s=50)

    # Scatter plot for processed points (using time for color)
    scatter_processed = ax.scatter(processed_x, processed_y, processed_z,
                                   c=processed_time, cmap='plasma', label='Processed Points', alpha=0.5, marker='^',
                                   s=50)

    # Setting labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('3D Visualization of Original and Processed Points with Time as Color')

    visualization_set_view_ax(ax)
    # Adding a color bar
    cbar_original = plt.colorbar(scatter_original, ax=ax, pad=0.1)
    cbar_original.set_label('Time (Original Points)')

    cbar_processed = plt.colorbar(scatter_processed, ax=ax, pad=0.1)
    cbar_processed.set_label('Time (Processed Points)')

    ax.legend()

    # Save the plot
    image_path = os.path.join(image_save_folderpath, 'combined_surface_points_time_colored.png')
    plt.savefig(image_path)
    plt.close(fig)

    print(f"Saved combined surface points image at {image_path}")


def _visualize_original_and_processed_points(original_points_all, processed_points_all, image_save_folder):
    """
    Visualizes the original and processed points in 3D in two separate images with time as color.
    :param original_points_all:
    :param processed_points_all:
    :param image_save_folder:
    :return:
    """
    # Ensure the save folder exists
    os.makedirs(image_save_folder, exist_ok=True)

    # Extracting x, y, z coordinates and time from original points
    original_x = original_points_all[:, 0]
    original_y = original_points_all[:, 1]
    original_z = original_points_all[:, 2]
    original_time = original_points_all[:, 3]  # Assuming the time column is the 4th column

    # Create a new figure for the original points 3D plot
    fig_original = plt.figure(figsize=(12, 8))
    ax_original = fig_original.add_subplot(111, projection='3d')

    # Scatter plot for original points (using time for color)
    scatter_original = ax_original.scatter(original_x, original_y, original_z,
                                           c=original_time, cmap='viridis', label='Original Points', alpha=1, s=70)

    # Setting labels
    ax_original.set_xlabel('X Label')
    ax_original.set_ylabel('Y Label')
    ax_original.set_zlabel('Z Label')
    ax_original.set_title('3D Visualization of Original Points with Time as Color')
    visualization_set_view_ax(ax_original)

    # Adding a color bar for original points
    cbar_original = plt.colorbar(scatter_original, ax=ax_original, pad=0.1)
    cbar_original.set_label('Time (Original Points)')

    # Save the original points plot
    original_image_path = os.path.join(image_save_folder, 'original_surface_points_time_colored.png')
    plt.savefig(original_image_path)
    plt.close(fig_original)

    print(f"Saved original surface points image at {original_image_path}")

    # Extracting x, y, z coordinates and time from processed points
    processed_x = processed_points_all[:, 0]
    processed_y = processed_points_all[:, 1]
    processed_z = processed_points_all[:, 2]
    processed_time = processed_points_all[:, 3]  # Assuming the time column is the 4th column

    # Create a new figure for the processed points 3D plot
    fig_processed = plt.figure(figsize=(12, 8))
    ax_processed = fig_processed.add_subplot(111, projection='3d')

    # Scatter plot for processed points (using time for color)
    scatter_processed = ax_processed.scatter(processed_x, processed_y, processed_z,
                                             c=processed_time, cmap='plasma', label='Processed Points', alpha=1, s=70)

    # Setting labels
    ax_processed.set_xlabel('X Label')
    ax_processed.set_ylabel('Y Label')
    ax_processed.set_zlabel('Z Label')
    ax_processed.set_title('3D Visualization of Processed Points with Time as Color')

    visualization_set_view_ax(ax_processed)

    # Adding a color bar for processed points
    cbar_processed = plt.colorbar(scatter_processed, ax=ax_processed, pad=0.1)
    cbar_processed.set_label('Time (Processed Points)')

    # Save the processed points plot
    processed_image_path = os.path.join(image_save_folder, 'processed_surface_points_time_colored.png')
    plt.savefig(processed_image_path)
    plt.close(fig_processed)

    print(f"Saved processed surface points image at {processed_image_path}")


def _prepare_export_data(surface_data_list : SurfacePointsFrameList, loaded_models : LoadedModelDic):
    logging.info("START: Preparing data for visualization")
    device = torch.device(NN_DEVICE_STR)

    original_points_all = []
    processed_points_all = []
    unique_clusters = loaded_models.keys()

    all_data_dataset = NNDataset(surface_data_list)
    all_data_tensor = torch.tensor(all_data_dataset.data, dtype=torch.float32).to(device)

    for cluster_index in unique_clusters:

        input_tensor = NNDataset.filter_by_cluster_label(all_data_tensor, cluster_index)

        model = loaded_models[cluster_index]

        with torch.no_grad():
            output_tensor = run_through_nn_at_same_time(input_tensor, model)

        # Accumulate all original and processed points
        original_points_all.append(input_tensor.detach().cpu().numpy())  # You can store the numpy array directly
        processed_points_all.append(output_tensor.detach().cpu().numpy())

    # Convert lists to numpy arrays for plotting
    original_points_all = np.vstack(original_points_all)
    processed_points_all = np.vstack(processed_points_all)

    # Add all columns from original_points_all after the third column to processed_points_all
    processed_points_all = np.hstack((processed_points_all, original_points_all[:, 3:]))

    logging.info("END: Preparing data for visualization")

    return original_points_all, processed_points_all


def _visualize_clusters(points, labels, image_save_folder, image_name):
    """
    Visualizes 3D points with cluster labels in distinct colors.

    Parameters:
    - points: np.ndarray of shape (N, 3), where N is the number of points.
    - labels: np.ndarray of shape (N,), cluster labels for each point.
    - image_save_folder: str, folder path to save the image.
    - image_name: str, name of the image file.
    """
    # Check input shapes
    if points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")
    if points.shape[0] != labels.shape[0]:
        raise ValueError(f"Number of points ({points.shape[0]}) must match number of labels ({labels.shape[0]})")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Generate distinct colors for each cluster
    unique_labels = np.unique(labels)
    colormap = plt.get_cmap('jet')
    colors = colormap(np.linspace(0, 1, len(unique_labels)))

    # Map labels to colors
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    point_colors = np.array([label_to_color[label] for label in labels])

    # Create scatter plot
    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=point_colors, s=50
    )

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    visualization_set_view_ax(ax) # Adjust these values as needed

    # Set plot title
    plt.title("3D Clusters with Mesh")

    # Save the image
    image_path = os.path.join(image_save_folder, image_name)
    plt.savefig(image_path)
    plt.close(fig)


def save_points_with_colors(visualization_data: VisualizationData, filepath: str, axis_label: str):
    # visualize point cloud for original point and make the color of the point based on the processed_points
    # where the x,y,z is r,g,b values

    # Create a new figure for each time slice

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Convert points and colors to numpy arrays for efficient plotting
    points = np.array(visualization_data.points)
    colors = np.array(visualization_data.colors)

    # Scatter all points at once
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=50)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title(axis_label)
    ax.legend()

    visualization_set_view_ax(ax)
    # Save the plot for the current time slice

    plt.savefig(filepath)
    plt.close(fig)
    print(f"Saved image at {filepath}")

def visualize_uv_points_in_3d(surface_data_list: SurfacePointsFrameList, images_save_folderpath: str, time_index: int,
                              loaded_models : LoadedModelDic, modulo : int):
    def visualize_for_eachtime(visualization_data : NNOutputForVisualization):
        rgb_colors = visualization_data.rgb_colors
        processed_points_split_by_time_value = visualization_data.processed_points

        # create folder if not created
        os.makedirs(images_save_folderpath, exist_ok=True)

        # Loop through each unique time value
        for time_index, processed_points_slice in processed_points_split_by_time_value.items():
            if time_index % modulo != 0:
                continue

            filepath = os.path.join(images_save_folderpath, f'time_{time_index}_uv_color_representation.png')

            axis_label = f'3D Visualization of Original and Processed Points from Time {time_index}'

            save_points_with_colors(VisualizationData(processed_points_slice, rgb_colors), filepath, axis_label)



    visualization_data = _run_model_with_one_encoder_time_to_all_decoder_times_prepare_for_visualization(
        surface_data_list=surface_data_list, time_index=time_index, loaded_models=loaded_models)

    visualize_for_eachtime(visualization_data)


def _visualize_combined_surface_points_for_one_time(image_save_folder, original_points_slice, processed_points_slice,
                                                    image_name, time):
    # Create a new figure for each time slice
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Filter original points for this time slice

    # Plot original points for this time slice
    # ax.scatter(original_points_slice[:, 0], original_points_slice[:, 1], original_points_slice[:, 2],
    #            color='blue', label='Original Points', alpha=0.5)
    # todo change

    # Plot processed points for this time slice
    ax.scatter(processed_points_slice[:, 0], processed_points_slice[:, 1], processed_points_slice[:, 2],
               color='red', label='Processed Points', alpha=0.5)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title(f'3D Visualization of Original and Processed Points from Time {time}')
    ax.legend()
    visualization_set_view_ax(ax)
    # Save the plot for the current time slice
    image_path = os.path.join(image_save_folder, image_name)
    plt.savefig(image_path)
    plt.close(fig)
    logging.info(f"Saved combined surface points image at {image_path}")


# endregion


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


# def _compute_centers_metrics(surface_data_list, train_config, num_points, nn_lr):
#     """
#     Computes metrics which:
#     1. for every sequenco of points
#     1.a selects num_points from the sequence
#     2. finds closest center point loaded from files
#     3. puts original points through the encoder
#     4. puts ouput throug the decoder in every time
#     5. computes the distances between the ceneter point again
#     6. outputs statistical dispersion for every point
#     7. do this for all sequences of points
#     :param surface_data_list:
#     :param train_config:
#     :param num_points:
#     :return:
#     """
#
#     def load_centers_data_from_files(folder_path, time_steps):
#         raise NotImplementedError("Not implemented yet")
#
#         center_points, num_points_in_file = load_centers_files(folder_path, time_steps)
#         # make it a Surface data list where each SurfacePoint is one num_points_in_file slice of center_points
#
#         # itarate over center_points and create SurfacePoint with num_points_in_file points
#         center_points_list_current = SurfacePointsFrameList([])
#         for i in range(0, center_points.shape[0]):
#             center_points_list_current.append(SurfacePointsFrame(center_points[i]))
#
#         center_points_list_current.assign_time_to_all_elements()
#         center_points_list_current.normalize_all_elements()
#         return center_points_list_current
#
#     def compute_distance(first_point, second_point):
#         return np.linalg.norm(first_point - second_point)
#
#     def find_closest_centers(center_points_list: SurfacePointsFrame, points_list: SurfacePointsFrame):
#         pair_original_center = []
#         index = 0
#         for index_point, point in enumerate(points_list.normalized_points_list):
#             closest_center_point = None
#             min_distance = float('inf')
#             for center_point in center_points_list.normalized_points_list:
#                 distance = compute_distance(point, center_point)
#                 if distance < min_distance:
#                     closest_center_point = center_point
#                     min_distance = distance
#
#             if points_list.time != center_points_list.time:
#                 raise Exception("Not same time")
#             time = points_list.time
#
#             label = points_list.labels_list[index_point]
#
#             pair_original_center.append(
#                 PairPointCenterPoint(point, closest_center_point, min_distance, index + 1, time, label))
#         return PairPointCenterPointList(pair_original_center)
#
#     def run_through_model(center_points_list: SurfacePointsFrameList, surface_data_list: SurfacePointsFrameList,
#                           model_weights_template: str, batch_size: int, num_points: int, nn_lr):
#         raise NotImplementedError("Not implemented yet")
#
#         # todo check if should be normalized
#         if len(center_points_list.list) != len(surface_data_list.list):
#             raise Exception("Not same number of center points and surface data")
#         pair_list_len = len(surface_data_list.list)
#
#         unique_clusters = surface_data_list.get_unique_clusters_indexes()
#         unique_times = surface_data_list.get_unique_times()
#
#         evaluation_result_list = EvaluationResultList([])
#
#         for i in range(0, pair_list_len):
#
#             # select original points where time is 0
#             surface_data_timeframe = surface_data_list.list[i]
#             if surface_data_timeframe.time.index != i:
#                 raise Exception("Not same time")
#
#             encoder_time = surface_data_timeframe.time.value
#
#             center_points_timeframe = center_points_list.list[i]
#             if center_points_timeframe.time.index != i:
#                 raise Exception("Not same time")
#
#             surface_data_timeframe = surface_data_list.select_random_points(num_points)
#
#             pair_original_center = find_closest_centers(center_points_timeframe, surface_data_timeframe)
#
#             for cluster in unique_clusters:
#
#                 # Load the original surface points for the current cluster
#                 pair_original_center_cluster = pair_original_center.filter_by_point_clusterlabel(cluster)
#                 surface_data_cluster = pair_original_center_cluster.get_points_list()
#                 surface_data_cluster = _convert_to_surfacepointsframelist(surface_data_cluster)
#
#                 # Create a SurfaceDataset instance with the filtered surface data
#                 original_points_dataset = NNDataset(surface_data_cluster)
#                 # Prepare a DataLoader for original points
#                 original_points_loader = DataLoader(original_points_dataset, batch_size=batch_size, shuffle=False)
#
#                 # Load the trained model for the current cluster
#                 model_weights_filepath = model_weights_template.format(cluster=cluster)
#                 model = _load_trained_model(model_weights_filepath, train_config)
#                 device = torch.device(NN_DEVICE_STR)
#
#                 model.to(device)
#
#                 # Step 1: Encode the original data
#                 with torch.no_grad():  # No need to calculate gradients during evaluation
#                     encoded_features = model.encoder(original_points_loader)
#
#                 decoder_pair_list = DecoderPairList([])
#
#                 # iterate to decoder over all times
#                 for decoder_time in unique_times:
#                     # Create a tensor of the same shape as the time feature in the input
#                     time_tensor = torch.full((encoded_features.size(0), 1), decoder_time, dtype=torch.float32)
#                     # Concatenate the encoded features with the time tensor
#                     encoded_with_time = torch.cat((encoded_features, time_tensor), dim=1)
#                     # Pass through the decoder
#                     decoded_output = model.decoder(encoded_with_time)
#
#                     decoder_processed_points = decoded_output
#                     decoder_processed_points_timeframe = SurfacePointsFrame([], None, decoder_time)
#                     # convert to
#                     for point in decoder_processed_points:
#                         decoder_processed_points_timeframe.points_list.append(point)
#
#                     decoder_center_points_timeframe = center_points_timeframe.get_element_by_time_index(decoder_time)
#
#                     decoder_pair_processed_center = find_closest_centers(decoder_center_points_timeframe,
#                                                                          decoder_processed_points_timeframe)
#
#                     decoder_pair_list.append(DecoderElement(decoder_pair_processed_center, decoder_time))
#
#                 evaluation_result_list.append(
#                     EvaluationResult(pair_original_center_cluster, encoder_time, decoder_pair_list))
#
#         return evaluation_result_list
#
#         # load center points
#
#     center_points_list = load_centers_data_from_files(train_config.file_path_config.raw_data_folderpath,
#                                                       train_config.max_time_steps)
#     # just loads the center points from files (already loaded in surface_data_list
#
#     # run through the model
#     evaluation_results_list = run_through_model(center_points_list,
#                                                 surface_data_list,
#                                                 train_config.file_path_config.model_weights_folderpath_template,
#                                                 train_config.nn_config.batch_size, num_points, nn_lr)
#
#     variance_list = _compute_variance(evaluation_results_list)
#
#     return variance_list, evaluation_results_list


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


# def get_centers_points_by_time_and_closestcentersindicies(data, closest_centers_indices_tensor, time, device):
def get_centers_points_by_time_and_closestcentersindicies(data: SurfacePointsFrameList,
                                                          closest_centers_indices_tensor: torch.Tensor, time: TimeFrame) -> torch.Tensor:
    all_centers_info_input_time = data.get_element_by_time_index(time.index).normalized_centers_info
    all_input_centers_points = all_centers_info_input_time.points
    all_input_centers_points = torch.tensor(all_input_centers_points).to(closest_centers_indices_tensor.device)
    centers_point_inputs_time = all_input_centers_points[closest_centers_indices_tensor]
    return centers_point_inputs_time


# def get_centers_matrix_by_cluster_frame(closest_centers_matrix, input_tensor, device, time_index):
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



# create an alias type for an model dictionary where key is cluster index


def save_visualize_centers(data: SurfacePointsFrameList, folderpath: str):
    """
    Will visualize points with color based on the index of center
    :param data:
    :param folderpath:
    :return:
    """

    centers_max_index = len(data.public_list[0].normalized_centers_info.points)
    # create rgb colors based on the index of center
    colors = np.zeros((centers_max_index, 3))
    for i in range(centers_max_index):
        colors[i] = [i / centers_max_index, 0, 1 - i / centers_max_index]


    for frame in data.public_list:
        centers_points = frame.normalized_centers_info.points
        file_name = f"centers_img_{frame.time.index}"
        filepath = os.path.join(folderpath, file_name)

        save_points_with_colors(VisualizationData(centers_points, colors), filepath, f"Centers at time {frame.time.index}")

    logging.info(f"Saved centers visualization to {folderpath}")


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


def save_centers_pipeline(evaluation_folderpath, surface_data_list):
    centers_image_foldername = "centers_img"
    centers_image_folderpath = os.path.join(evaluation_folderpath, centers_image_foldername)
    os.makedirs(centers_image_folderpath, exist_ok=True)
    save_visualize_centers(surface_data_list, centers_image_folderpath)


def save_mesh_thrugh_model_pipeline(evaluation_folderpath, loaded_models, train_config):
    processed_data = process_mesh_through_model(MeshData(time_index=0), train_config, loaded_models)
    visualizer = DataVisualizer(processed_data)
    mesh_files_folderpath = os.path.join(evaluation_folderpath, "mesh_files")
    os.makedirs(mesh_files_folderpath, exist_ok=True)
    visualizer.save_as_meshes_to_file(mesh_files_folderpath)


def evaluate(train_config: TrainConfig):
    surface_data_list = load_pickle_file(train_config.file_path_config.surface_data_filepath)

    if surface_data_list is None or surface_data_list.public_list is None:
        logging.error("Surface data list could not be loaded. Exiting.")
        return

    evaluation_folderpath = train_config.file_path_config.evaluation_folderpath

    loaded_models = load_trained_nn_from_files(train_config)


    # region Save Evaluation files
    # save_mesh_thrugh_model_pipeline(evaluation_folderpath, loaded_models, train_config)
    #
    save_centers_pipeline(evaluation_folderpath, surface_data_list)
    #
    # # region Bundle
    original_points_all, processed_points_all = _prepare_export_data(
         surface_data_list=surface_data_list, loaded_models=loaded_models)

    _visualize_combined_surface_points_for_each_time(original_points_all, processed_points_all,
                                                     os.path.join(evaluation_folderpath,
                                                                  "time_combined_only_processed"))
    _visualize_all_clusters_for_each_time(surface_data_list, os.path.join(evaluation_folderpath, "time_clusters"))

    _visualize_points_with_time(original_points_all, processed_points_all, evaluation_folderpath)
    # # Save the combined image
    _visualize_original_and_processed_points(original_points_all, processed_points_all, evaluation_folderpath)

    point_cloud_original_filepath = train_config.file_path_config.point_cloud_original_filepath
    point_cloud_processed_filepath = train_config.file_path_config.point_cloud_processed_filepath
    # _save_pointcloud_to_file(original_points_all, processed_points_all, point_cloud_original_filepath,
    #                          point_cloud_processed_filepath)
    # endregion

    visualize_uv_points_in_3d(surface_data_list=surface_data_list,
                              images_save_folderpath=os.path.join(evaluation_folderpath, "time_uv_points_0"),
                              time_index=0, loaded_models=loaded_models, modulo=5)

    visualize_uv_points_in_3d(surface_data_list=surface_data_list,
                              images_save_folderpath=os.path.join(evaluation_folderpath, "time_uv_points_59"),
                              time_index=59, loaded_models=loaded_models, modulo=5)

    # _create_pointclouds_from_time_to_all_times(surface_data_list=surface_data_list,
    #                                            images_save_folderpath=os.path.join(evaluation_folderpath,
    #                                                                                "point_clouds_all_times_time_0"),
    #                                            time_index=0, loaded_models=loaded_models)

    # endregion

    # region Save Metrics
    #save_metrics_centers_pipeline(evaluation_folderpath, loaded_models, surface_data_list, train_config)
    # mesh_shape_metrics = _compute_mesh_shape_metrics(surface_data_list, train_config, clustered_data)
    # # save mesh_shape_metrics to file
    # with open(train_config.file_path_config.mesh_shape_metrics_filepath, "w") as file:
    #     file.write(str(mesh_shape_metrics))
    # endregion


class DataVisualizer:
    def __init__(self, processed_data: ProcessedMeshData):
        self.processed_mesh_data : ProcessedMeshData = processed_data

    @staticmethod
    def create_dir(output_folderpath: Folderpath) -> Folderpath:
        # create output folder if not exists
        # add current time to folder name
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folderpath = os.path.join(output_folderpath, f"mesh_{current_time_str}")
        os.makedirs(output_folderpath, exist_ok=True)
        return output_folderpath

    def save_as_pointcloud_to_file(self, save_folderpath : str):
        rgb_colors = self.processed_mesh_data.processed_visualization_data.rgb_colors
        processed_points_split_by_time_value = self.processed_mesh_data.processed_visualization_data.processed_points

        # make dir if not made
        save_folderpath = self.create_dir(save_folderpath)

        # Save the RGB values to another file
        rgb_colors_filepath = os.path.join(save_folderpath, 'rgb_colors.txt')
        np.savetxt(rgb_colors_filepath, rgb_colors, delimiter=",")

        logging.info(f"Saved RGB colors to {rgb_colors_filepath}")
        # todo finish - add denormalization

        for i, processed_points_one_time_value in processed_points_split_by_time_value.items():
            processed_points_filepath = os.path.join(save_folderpath, f'processed_points_{i}.xyz')
            np.savetxt(processed_points_filepath, processed_points_one_time_value, delimiter=",")
            logging.info(f"Saved processed points to {processed_points_filepath}")

    def save_as_meshes_to_file(self, save_folderpath: str):
        processed_points_split_by_time_value = self.processed_mesh_data.processed_visualization_data.processed_points
        origin_mesh = self.processed_mesh_data.origin_mesh

        # make dir if not made
        save_folderpath = self.create_dir(save_folderpath)

        for time_index, processed_points_one_time_value in processed_points_split_by_time_value.items():
            processed_points_filepath = os.path.join(save_folderpath, f'processed_points_{time_index}.obj')
            origin_mesh_filepath = os.path.join(save_folderpath, f'origin_mesh_{time_index}.obj')

            # check if origin_mesh vertices is the same number of points as processed points
            if len(origin_mesh.vertices) != len(processed_points_one_time_value):
                raise ValueError(f"Number of vertices in origin mesh is not the same as number of processed points. Origin mesh vertices: {len(origin_mesh.vertices)}, processed points: {len(processed_points_one_time_value)}")

            # test_origin_mesh_verticies = np.random.rand(len(origin_mesh.vertices), 3)
            #
            # test_origin_mesh = trimesh.Trimesh(vertices=test_origin_mesh_verticies, faces=origin_mesh.faces)
            # test_origin_mesh.export(origin_mesh_filepath)

            mesh = trimesh.Trimesh(vertices=processed_points_one_time_value, faces=origin_mesh.faces)
            mesh.export(processed_points_filepath)
            logging.info(f"Saved processed points to {processed_points_filepath}")


def __save_pointcloud_to_file(processed_data : NNOutputForVisualization, images_save_folderpath : str):
    rgb_colors = processed_data.rgb_colors
    processed_points_split_by_time_value = processed_data.processed_points

    # make dir if not made
    os.makedirs(images_save_folderpath, exist_ok=True)

    # Save the RGB values to another file
    rgb_colors_filepath = os.path.join(images_save_folderpath, 'rgb_colors.txt')
    np.savetxt(rgb_colors_filepath, rgb_colors, delimiter=",")

    logging.info(f"Saved RGB colors to {rgb_colors_filepath}")
    # todo finish - add denormalization

    for time_index, processed_points_one_time_value in processed_points_split_by_time_value.items():
        processed_points_filepath = os.path.join(images_save_folderpath, f'processed_points_{time_index}.xyz')
        np.savetxt(processed_points_filepath, processed_points_one_time_value, delimiter=",")
        logging.info(f"Saved processed points to {processed_points_filepath}")

def create_mesh_surfacedatalist(clustered_data : ClusteredCenterPointsAllFrames, surface_data_list : SurfacePointsFrameList) -> SurfacePointsFrameList:
    original_loaded_meshes = surface_data_list.get_original_meshes_list()

    mesh_surface_points_frame_list = SurfacePointsFrameList([])

    for original_mesh in original_loaded_meshes:
        time_index = original_mesh[0]
        mesh = original_mesh[1]

        mesh_vertices = np.array(mesh.vertices)

        ## Categorize points
        centers_labels_frame = clustered_data.labels_frame
        centers_points_frame = clustered_data.points_allframes[time_index]
        labels = categorize_points_with_labels(centers_labels_frame, centers_points_frame, mesh_vertices)

        ## Create Surface data
        mesh_surface_points_frame = SurfacePointsFrame.create_instance(surface_points=mesh_vertices,
                                                                       surface_labels=labels, mesh=mesh,
                                                                       centers_points=centers_points_frame)

        ## region get time value
        surface_data_frame = surface_data_list.get_element_by_time_index(time_index)
        if surface_data_frame is None:
            logging.error(f"Surface data frame for time index {time_index} could not be found. Exiting.")
            raise ValueError(f"Surface data frame for time index {time_index} could not be found. Exiting.")

        time_value = surface_data_frame.time.value
        ## endregion

        mesh_surface_points_frame.time = TimeFrame(index=time_index, value=time_value)
        # endregion

        mesh_surface_points_frame_list.append(mesh_surface_points_frame)

    return mesh_surface_points_frame_list

def process_mesh_through_model(origin_mesh_data: MeshData, train_config: TrainConfig,
                               loaded_models : LoadedModelDic) -> ProcessedMeshData | None:
    """
    Function to process mesh through model
    :param loaded_models:
    :param origin_mesh_data:
    :param train_config:
    :return:
    """



    # region PREPARE MESH FOR MODEL
    # region STEP Read clustered data, surface data
    clustered_data: ClusteredCenterPointsAllFrames = load_pickle_file(
        train_config.file_path_config.clustered_data_filepath)
    if clustered_data is None:
        logging.error("Clustered data could not be loaded. Exiting.")
        return None

    surface_data_list: SurfacePointsFrameList = load_pickle_file(train_config.file_path_config.surface_data_filepath)
    if surface_data_list is None or surface_data_list.public_list is None:
        logging.error("Surface data list could not be loaded. Exiting.")
        return None
    # endregion

    # region STEP Create Surface data from this

    # create surface data list where input vertices are meshes vertices and they are clustered by labels

    mesh_surface_points_frame_list: SurfacePointsFrameList = create_mesh_surfacedatalist(clustered_data=
        clustered_data, surface_data_list=surface_data_list)

    # endregion

    # region PROCESS MESH THROUGH MODEL

    # deep copy of mesh_surface_points_frame_list
    normalized_mesh_surface_points_frame_list: SurfacePointsFrameList = copy.deepcopy(mesh_surface_points_frame_list)
    normalized_mesh_surface_points_frame_list.normalize_labeled_points_by_values(surface_data_list.normalize_values)
    input_model_data: SurfacePointsFrameList = normalized_mesh_surface_points_frame_list

    visualization_data = _run_model_with_one_encoder_time_to_all_decoder_times_prepare_for_visualization(
        surface_data_list=input_model_data, time_index=origin_mesh_data.time_index, loaded_models=loaded_models)
    processed_points_split_by_time_value = visualization_data.processed_points


    # _create_pointclouds_from_time_to_all_times(surface_data_list=input_model_data,
    #                                            images_save_folderpath=os.path.join(output_folderpath,
    #                                                                                "point_clouds_all_times"),
    #                                            time_index=origin_mesh_data.time_index, train_config=train_config)

    denormalized_points_split_by_time_value : ProcessedPointsListSplitByTimeValue = dict()
    for time_index, processed_points_one_time_value in processed_points_split_by_time_value.items():
        denormalized_points = SurfacePointsFrameList.denormalize_points(surface_data_list.normalize_values, processed_points_one_time_value)
        denormalized_points_split_by_time_value[time_index] = denormalized_points

    origin_mesh = input_model_data.get_element_by_time_index(origin_mesh_data.time_index).original_mesh
    return ProcessedMeshData(
        NNOutputForVisualization(rgb_colors=visualization_data.rgb_colors, processed_points=denormalized_points_split_by_time_value), origin_mesh)
