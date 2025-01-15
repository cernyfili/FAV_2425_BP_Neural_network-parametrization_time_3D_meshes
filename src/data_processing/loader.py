import glob
import logging
import os

import numpy as np
from sklearn.cluster import DBSCAN

from src.utils.constants import RAW_DATA_ALLOWED_FILETYPES_LIST
from utils.helpers import get_file_index_from_filename


# Restrict access to underscore-prefixed functions
def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"{name} is a private function and cannot be imported.")
    raise AttributeError(f"Module has no attribute {name}")

# Function to compute Euclidean distance between two 3D points
def _euclidean_distance(p1, p2):
    from math import sqrt
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


# Function to load .xyz files
def _load_xyz_files(filepaths):
    num_points_in_file = None
    points_allframes = []

    # find min fileindex in meshes_filepaths_list
    min_file_index = min([get_file_index_from_filename(mesh_file_path) for mesh_file_path in filepaths])

    for i, filepath in enumerate(filepaths):
        pointslist_frame = np.loadtxt(filepath, delimiter=' ')  # Adjust delimiter if needed
        if num_points_in_file is None:
            num_points_in_file = pointslist_frame.shape[0]
        elif num_points_in_file != pointslist_frame.shape[0]:
            raise ValueError("Inconsistent number of points in the files.")
        # check if index in filepath name is the same as index in list
        file_index = get_file_index_from_filename(filepath, min_file_index)

        points_allframes.append(pointslist_frame)

        if i != file_index or (len(points_allframes) - 1) != file_index:
            raise ValueError("Inconsistent index in file name.")


    return np.array(points_allframes) # Shape: (num_files, num_time_steps, num_points)

    # Function to compute max distances between pairs of points across time


def compute_max_distances_for_all_pairs(center_points_allframes):
    # todo check if it does what i want
    """
    Compute the maximum pairwise distances for all points in each file.

    Parameters:
    - data: List of arrays, each containing 3D point data for a file
    - num_points_in_file: int, number of points in each file

    Returns:
    - max_distances_mx: np.ndarray of shape (num_points_in_file, num_points_in_file)
    """

    num_points_in_file = center_points_allframes.shape[1]

    max_distances_mx = np.zeros((num_points_in_file, num_points_in_file))

    size = len(center_points_allframes)
    for i, center_points_frame in enumerate(center_points_allframes):
        # Reshape row into list of 3D points
        points = center_points_frame.reshape(-1, 3)

        # Compute pairwise distances using broadcasting
        diff = points[:, None, :] - points[None, :, :]  # Pairwise differences
        distances = np.sqrt(np.sum(diff**2, axis=-1))  # Pairwise Euclidean distances

        # Update max distances
        max_distances_mx = np.maximum(max_distances_mx, distances)

        logging.info(f"Computing max distances {i + 1} of {size}")

    return max_distances_mx


# Function for DBSCAN clustering using precomputed distances
def _dbscan_clustering_from_precomputed_distances(distances, eps=0.5, min_samples=5):
    # Extract the upper triangle of the matrix (if needed) or use the distances directly
    # distances = distances[np.triu_indices(distances.shape[0], k=1)]  # If necessary

    # Perform DBSCAN clustering with the precomputed distance matrix
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = db.fit_predict(distances)

    return labels


# Function to load .bin files and return data
def _load_bin_files(filepaths):
    logging.info("Loading .bin files...")
    data = []
    num_points_in_file = None

    min_file_index = min([get_file_index_from_filename(mesh_file_path) for mesh_file_path in filepaths])

    for i, filepath in enumerate(filepaths):
        logging.info("Loading file " + str(i) + " of " + str(len(filepaths)))
        # Read the binary file as 32-bit floats
        file_data = np.fromfile(filepath, dtype=np.float32)
        file_data = file_data[1:]  # first is size

        # Reshape the data into 3D points (3 values per point)
        points = file_data.reshape(-1, 3)

        # points_in_file
        if num_points_in_file is None:
            num_points_in_file = points.shape[0]
        elif num_points_in_file != points.shape[0]:
            raise ValueError("Inconsistent number of points in the files.")

        # check if index in filepath name is the same as index in list
        file_index = get_file_index_from_filename(filepath, min_file_index)

        data.append(points)

        if i != file_index or (len(data) - 1) != file_index:
            raise ValueError("Inconsistent index in file name.")
        # Append the points for this file to the data list

    return np.array(data) # Shape: (num_files, num_time_steps, num_points)


def load_centers_data(folder_path, time_steps):
    """
    Load .xyz files or .bin files from the specified folder and compute the maximum distances between points.
    :param folder_path: path to where is files with computed centers points
    :param file_type:
    :return:
       data: np.ndarray of shape (num_files, num_time_steps, num_points, 3)
    """
    # find file types of all files in the folder
    file_types = []
    for file in os.listdir(folder_path):
        # from the list of allowed file types
        if file.split('.')[-1] in RAW_DATA_ALLOWED_FILETYPES_LIST:
            file_types.append(file.split('.')[-1])
    # check if all files are of the same type
    if len(set(file_types)) > 1:
        raise ValueError("All files in the folder must be of the same type.")
    file_type = file_types[0]

    # Folder path containing .xyz files and the .obj file
    path = os.path.join(folder_path, '*.' + file_type)

    # Load all files in the folder
    filepaths = glob.glob(path)

    if not filepaths:
        raise ValueError("No files found in the specified folder.")

    # Limit the number of time steps to the minimum of the number of files and the specified time steps
    if time_steps is None:
        max_time_steps = len(filepaths)
    else:
        max_time_steps = max(time_steps, len(filepaths))

    filepaths = filepaths[:max_time_steps]

    # Load the .xyz files
    if file_type == 'xyz':
        points_allframes = _load_xyz_files(filepaths)
    elif file_type == 'bin':
        points_allframes = _load_bin_files(filepaths)
    else:
        raise ValueError("Invalid file type. Use 'xyz' or 'bin'.")

    return points_allframes
