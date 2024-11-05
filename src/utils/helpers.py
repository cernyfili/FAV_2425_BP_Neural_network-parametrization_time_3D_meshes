import json
import logging
import os
import pickle
import re

import numpy as np
from matplotlib import pyplot as plt

from utils.constants import IMAGE_SAVE_FOLDERPATH

# Restrict access to underscore-prefixed functions
def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"{name} is a private function and cannot be imported.")
    raise AttributeError(f"Module has no attribute {name}")

def get_filepaths_from_json(folder_path, json_file_path):
    """
    Load the file paths from the JSON file and add the folder path as a prefix.

    Parameters:
    - folder_path: str, the folder path to be prefixed
    - json_file_path: str, path to the JSON file

    Returns:
    - meshes_filepaths_list: list of str, list of file paths with the folder path prefixed
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    meshes_filepaths_list = []
    for pair in data['pairs']:
        mesh_file_path = os.path.join(folder_path, pair['mesh_filename'])
        meshes_filepaths_list.append(mesh_file_path)

    return meshes_filepaths_list

def get_file_pairs_from_numbers(folder_path):
    """
    Get pairs of files from the specified folder based on matching numbers in specific formats.

    Parameters:
    - folder_path: str, the folder path containing the files.

    Returns:
    - file_pairs_list: list of tuples, each containing a pair of file paths.
    """
    file_pairs_list = []
    files = os.listdir(folder_path)

    # Create dictionaries to hold the relevant files
    obj_files = {}
    res_files = {}

    # Categorize files into obj and res based on their naming patterns
    for filename in files:
        if filename.endswith('.obj'):
            match = re.search(r'((\d+)\.)', filename)
            if match:
                number = match.group(2)
                obj_files[number] = filename
        elif filename.endswith('.xyz') or filename.endswith('.bin'):
            match = re.search(r'((\d+)\.)', filename)
            if match:
                number = match.group(2)
                res_files[number] = filename

    # Pair the files by matching numbers
    for number in obj_files:
        if number in res_files:
            obj_file_path = os.path.join(folder_path, obj_files[number])
            res_file_path = os.path.join(folder_path, res_files[number])
            file_pairs_list.append((obj_file_path, res_file_path))

    if not file_pairs_list:
        raise Exception("No data pairs found")
    return file_pairs_list

def get_meshes_list(meshes_folder_path):
    """
    Get a list of .obj file paths from the specified folder.

    Parameters:
    - meshes_folder_path: str, the folder path containing the .obj files.

    Returns:
    - obj_files_list: list of .obj file paths.
    """
    obj_files_list = []
    files = os.listdir(meshes_folder_path)

    # Select only .obj files
    for filename in files:
        if filename.endswith('.obj'):
            obj_file_path = os.path.join(meshes_folder_path, filename)
            obj_files_list.append(obj_file_path)

    return obj_files_list

# Utility function to create a directory if it doesn't exist
def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Created directory: {directory_path}")

# Utility function to load a pickle file safely
def load_pickle_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
    except pickle.UnpicklingError:
        logging.error(f"Failed to unpickle file: {filepath}")
    return None

def save_all_clusters_surface_points_image(original_points_dict, processed_points_dict,
                                           image_save_folder=IMAGE_SAVE_FOLDERPATH):
    """
    Plots and saves an image of surface points for all clusters, showing both original and processed points.

    Parameters:
    - original_points_dict (dict): Dictionary where keys are cluster labels and values are arrays of original points
      (shape: [num_points, 2] or [num_points, 3]).
    - processed_points_dict (dict): Dictionary where keys are cluster labels and values are arrays of processed points
      from the neural network for each cluster (same shape as original points).
    - image_save_folder (str): Directory to save the image.
    """
    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define a color map for distinguishing clusters
    colors = plt.cm.get_cmap('tab10', len(original_points_dict))  # Use a color map with enough unique colors

    for i, cluster in enumerate(original_points_dict.keys()):
        original_points = original_points_dict[cluster]
        processed_points = processed_points_dict[cluster]

        # Scatter plot for original points of this cluster
        ax.scatter(
            original_points[:, 0], original_points[:, 1],
            color=colors(i), label=f'Original Cluster {cluster}', alpha=0.5, s=20
        )

        # Scatter plot for processed points of this cluster
        ax.scatter(
            processed_points[:, 0], processed_points[:, 1],
            color=colors(i), edgecolor='black', marker='x', label=f'Processed Cluster {cluster}', s=30
        )

    # Labeling, legend, and grid
    ax.set_title('Surface Points for All Clusters')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend(loc='best', bbox_to_anchor=(1.05, 1), ncol=1)
    ax.grid(True)

    # Save the plot as an image
    image_path = os.path.join(image_save_folder, 'all_clusters_surface_points.png')
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close(fig)  # Close the figure to free memory

    logging.info(f"Saved surface points image for all clusters at {image_path}")

def save_combined_surface_points_images(original_points_all, processed_points_all,
                                        image_save_folder=IMAGE_SAVE_FOLDERPATH):
    # Ensure the image save folder exists
    os.makedirs(image_save_folder, exist_ok=True)

    # Extract unique time values assuming the last column contains time values
    unique_times = np.unique(original_points_all[:, 3])

    # Loop through each unique time value
    for i, time in enumerate(unique_times):
        # Create a new figure for each time slice
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Filter original points for this time slice
        original_points_slice = original_points_all[original_points_all[:, 3] == time]

        # Plot original points for this time slice
        ax.scatter(original_points_slice[:, 0], original_points_slice[:, 1], original_points_slice[:, 2],
                   color='blue', label='Original Points', alpha=0.5)

        previous_time = None
        # Determine the time range for processed points
        if i == 0:  # If it's the first time, we can't go back
            processed_points_slice = processed_points_all[processed_points_all[:, 3] == time]
        else:
            previous_time = unique_times[i - 1]
            # Filter processed points that are between previous_time and current time
            processed_points_slice = processed_points_all[
                (processed_points_all[:, 3] >= previous_time) &
                (processed_points_all[:, 3] <= time)
                ]

        # Plot processed points for this time slice
        ax.scatter(processed_points_slice[:, 0], processed_points_slice[:, 1], processed_points_slice[:, 2],
                   color='red', label='Processed Points', alpha=0.5)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title(f'3D Visualization of Original and Processed Points from Time {previous_time}_to_{time}')
        ax.legend()

        # Save the plot for the current time slice
        image_path = os.path.join(image_save_folder, f'combined_surface_points_time_{previous_time}_to_{time}.png')
        plt.savefig(image_path)
        plt.close(fig)

        print(f"Saved combined surface points image at {image_path}")

def visualize_points_with_time(original_points_all, processed_points_all, image_save_folder=IMAGE_SAVE_FOLDERPATH):
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

def visualize_original_and_processed_points(original_points_all, processed_points_all, image_save_folder=IMAGE_SAVE_FOLDERPATH):
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
