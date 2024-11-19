import logging
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from data_processing.clustering import visualize_clusters
from data_processing.mapping import SurfaceDataList, convert_to_surface_data_list
from nerual_network.training import get_device
from src.utils.constants import nn_optimizer, nn_model, TrainConfig
from src.nerual_network.model import NNDataset
from utils.helpers import load_pickle_file


# Restrict access to underscore-prefixed functions
def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"{name} is a private function and cannot be imported.")
    raise AttributeError(f"Module has no attribute {name}")


def _save_pointcloud_to_file(original_points_all, processed_points_all, original_filepath, processed_filepath):
    np.savetxt(original_filepath, original_points_all, delimiter=",")
    np.savetxt(processed_filepath, processed_points_all, delimiter=",")


def evaluate(train_config: TrainConfig):
    surface_data_list = load_pickle_file(train_config.file_path_config.surface_data_filepath)

    if surface_data_list is None or surface_data_list.list is None:
        logging.error("Surface data list could not be loaded. Exiting.")
        return
    # surface_data_list = convert_to_surface_data_list(surface_data_list)

    images_save_folderpath = train_config.file_path_config.images_save_folderpath
    model_weights_template = train_config.file_path_config.model_weights_folderpath
    point_cloud_original_filepath = train_config.file_path_config.point_cloud_original_filepath
    point_cloud_processed_filepath = train_config.file_path_config.point_cloud_processed_filepath
    batch_size = train_config.nn_config.batch_size

    original_points_all, processed_points_all, cluster_labels = _prepare_export_data(surface_data_list,
                                                                                     model_weights_template,
                                                                                     batch_size)

    # Save the combined image
    _save_pointcloud_to_file(original_points_all, processed_points_all, point_cloud_original_filepath,
                             point_cloud_processed_filepath)

    _visualize_for_each_time(original_points_all, processed_points_all,
                             images_save_folderpath, cluster_labels)
    _visualize_points_with_time(original_points_all, processed_points_all, images_save_folderpath)
    _visualize_original_and_processed_points(original_points_all, processed_points_all, images_save_folderpath)


def _visualize_for_each_time(original_points_all, processed_points_all,
                             image_save_folder, cluster_labels):
    """
    Visualizes the original and processed points in 3D in one image for each time slice.
    :param original_points_all:
    :param processed_points_all:
    :param image_save_folder:
    :return:
    """
    # Ensure the image save folder exists
    os.makedirs(image_save_folder, exist_ok=True)

    # Extract unique time values assuming the last column contains time values
    unique_times = np.unique(original_points_all[:, 3])

    # Loop through each unique time value
    for i, time in enumerate(unique_times):
        original_points_slice = original_points_all[original_points_all[:, 3] == time]
        processed_points_slice = processed_points_all[processed_points_all[:, 3] == time]

        visualize_combined_surface_points_for_each_time(image_save_folder, original_points_slice,
                                                        processed_points_slice, f'time_{i}_combined_surface_points_time.png')

        # visulize clusters
        visualize_clusters(original_points_slice, cluster_labels, image_save_folder,
                           f'time_{i}_clustered_surface_points_time.png')


def visualize_combined_surface_points_for_each_time(image_save_folder, original_points_slice, processed_points_slice,
                                                    image_name):
    # Create a new figure for each time slice
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Filter original points for this time slice

    # Plot original points for this time slice
    ax.scatter(original_points_slice[:, 0], original_points_slice[:, 1], original_points_slice[:, 2],
               color='blue', label='Original Points', alpha=0.5)

    # Plot processed points for this time slice
    ax.scatter(processed_points_slice[:, 0], processed_points_slice[:, 1], processed_points_slice[:, 2],
               color='red', label='Processed Points', alpha=0.5)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title(f'3D Visualization of Original and Processed Points from Time {time}')
    ax.legend()
    # Save the plot for the current time slice
    image_path = os.path.join(image_save_folder, image_name)
    plt.savefig(image_path)
    plt.close(fig)
    print(f"Saved combined surface points image at {image_path}")


def _visualize_points_with_time(original_points_all, processed_points_all, image_save_folder):
    """
    Visualizes the original and processed points in 3D in one image with time as color.
    :param original_points_all:
    :param processed_points_all:
    :param image_save_folder:
    :return:
    """
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

    # Adding a color bar
    cbar_original = plt.colorbar(scatter_original, ax=ax, pad=0.1)
    cbar_original.set_label('Time (Original Points)')

    cbar_processed = plt.colorbar(scatter_processed, ax=ax, pad=0.1)
    cbar_processed.set_label('Time (Processed Points)')

    ax.legend()

    # Save the plot
    image_path = os.path.join(image_save_folder, 'combined_surface_points_time_colored.png')
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

    # Adding a color bar for processed points
    cbar_processed = plt.colorbar(scatter_processed, ax=ax_processed, pad=0.1)
    cbar_processed.set_label('Time (Processed Points)')

    # Save the processed points plot
    processed_image_path = os.path.join(image_save_folder, 'processed_surface_points_time_colored.png')
    plt.savefig(processed_image_path)
    plt.close(fig_processed)

    print(f"Saved processed surface points image at {processed_image_path}")


def _prepare_export_data(surface_data_list, model_weights_template, batch_size):
    original_points_all = []
    processed_points_all = []
    cluster_labels = []
    unique_clusters = surface_data_list.get_unique_clusters()
    for cluster in unique_clusters:
        # Load the original surface points for the current cluster
        surface_data_cluster = surface_data_list.filter_by_label(cluster)

        # Create a SurfaceDataset instance with the filtered surface data
        original_points_dataset = NNDataset(surface_data_cluster.list)

        # Load the trained model for the current cluster
        model_weights_filepath = model_weights_template.format(cluster=cluster)
        model = _load_trained_model(model_weights_filepath)
        device = get_device()

        model.to(device)

        # Prepare a DataLoader for original points
        original_points_loader = DataLoader(original_points_dataset, batch_size=batch_size, shuffle=True)

        # Process the original points through the model
        processed_points = []
        with torch.no_grad():
            for batch in original_points_loader:
                inputs = batch[0]  # Get only the points with time
                inputs = inputs.float().to(device)

                outputs = model(inputs)  # Forward pass through the model
                processed_points.append(outputs)

        # Convert processed points to a single numpy array
        processed_points = torch.cat(processed_points).cpu().numpy()

        # Accumulate all original and processed points
        original_points_all.append(original_points_dataset.data)  # You can store the numpy array directly
        processed_points_all.append(processed_points)
        # cluster labels store
        cluster_labels.extend([cluster] * len(original_points_dataset.data))
    # Convert lists to numpy arrays for plotting
    original_points_all = np.vstack(original_points_all) if original_points_all else np.empty((0, 4))
    processed_points_all = np.vstack(processed_points_all) if processed_points_all else np.empty((0, 4))

    processed_points_all = np.hstack((processed_points_all, original_points_all[:, 3].reshape(-1, 1)))

    return original_points_all, processed_points_all, cluster_labels


def _load_trained_model(model_weights_filepath):
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
    model = nn_model
    optimizer = nn_optimizer

    checkpoint = torch.load(model_weights_filepath)  # Load the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])  # Load the model state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load the optimizer state
    epoch = checkpoint['epoch']  # Get the epoch number
    val_loss = checkpoint['val_loss']  # Get the validation loss

    # Set the model to evaluation mode
    model.eval()

    return model
