#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File name: visualization.py
Author: Filip Cerny
Created: 17.04.2025
Version: 1.0
Description: 
"""
import logging
import os
from datetime import time, datetime

import numpy as np
import torch
from matplotlib import pyplot as plt

from data_processing.class_mapping import SurfacePointsFrameList
from nerual_network.class_model import NNDataset
from nerual_network.helpers import MeshFilepathsDic, NNOutputForVisualization, \
    _run_model_with_one_encoder_time_to_all_decoder_times_prepare_for_visualization, VisualizationData, \
    create_timestemp_dir, LoadedModelDic
from nerual_network.loss_functions import run_through_nn_at_same_time
from utils.constants import NN_DEVICE_STR


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
        images_save_folderpath = create_timestemp_dir(images_save_folderpath)

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

    image_save_folder = create_timestemp_dir(image_save_folder)

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

    image_save_folder = create_timestemp_dir(image_save_folder)

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

    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the plot
    image_path = os.path.join(image_save_folderpath, f'combined_surface_points_time_colored_{current_time_str}.png')
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
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

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
    original_image_path = os.path.join(image_save_folder, f'original_surface_points_time_colored_{current_time_str}.png')
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
    processed_image_path = os.path.join(image_save_folder, f'processed_surface_points_time_colored_{current_time_str}.png')
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

        images_save_folderpath = create_timestemp_dir(images_save_folderpath)

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


def visualization_set_view_ax(ax):
    min_value = -1
    max_value = 1
    ax.set_xlim(min_value, max_value)
    ax.set_ylim(min_value, max_value)
    ax.set_zlim(min_value, max_value)

    # Set the initial view angle
    ax.view_init(elev=-90, azim=90)  # Adjust these values as needed
