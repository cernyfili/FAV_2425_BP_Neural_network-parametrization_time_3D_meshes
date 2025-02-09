import json
import logging
import os
import pickle

import re

from matplotlib import pyplot as plt


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

def get_meshes_list(meshes_folder_path, len_clustered_data):
    """
    Get a list of .obj file paths from the specified folder.

    Parameters:
    - meshes_folder_path: str, the folder path containing the .obj files.

    Returns:
    - obj_files_list: list of .obj file paths.
    """
    obj_files_list = []
    files = os.listdir(meshes_folder_path)

    max_time_steps = min(len(files), len_clustered_data)

    files = files[:max_time_steps]

    min_index = get_file_index_from_filename(files[0])

    # Select only .obj files
    for filename in files:
        if filename.endswith('.obj'):
            obj_file_path = os.path.join(meshes_folder_path, filename)
            obj_files_list.append(obj_file_path)

            file_index = get_file_index_from_filename(obj_file_path)
            if file_index != len(obj_files_list) - 1 + min_index:
                raise Exception(f"File index mismatch: {file_index} vs {len(obj_files_list) - 1}")

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
                                           image_save_folder):
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

def init_logger(log_filepath):
    # Create a logger
    logger = logging.getLogger()
    # Set the logging level
    logger.setLevel(logging.INFO)
    # Create handlers
    console_handler = logging.StreamHandler()  # For console output
    file_handler = logging.FileHandler(log_filepath)  # For file output
    # Set the logging level for handlers
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # Add the formatter to the handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    # Example usage of the logger
    logger.info("Logging has been configured.")
    return logger

def end_logger(logger):
    # Assuming you have a logger defined as in your code
    logger.info("Ending logging and cleaning up resources.")

    # Remove all handlers
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()          # Close the handler (e.g., file handlers)
        logger.removeHandler(handler)  # Remove it from the logger


def get_file_index_from_filename(mesh_file_path, min_file_index=0):
    file_index = int(mesh_file_path.split('.')[-2][-3:])
    return file_index - min_file_index
