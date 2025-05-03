import logging
import os
import pickle

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from src.data_processing.class_clustering import ClusteredCenterPointsAllFrames
from src.data_processing.file_loader import load_centers_files


# Restrict access to underscore-prefixed functions
def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"{name} is a private function and cannot be imported.")
    raise AttributeError(f"Module has no attribute {name}")

# class for clustered data

# region PRIVATE FUNCTIONS
# write me function which will from the variable PUB_all_center_points get points from specific time step
def _get_points_from_time_step(data, time_step):
    return data[time_step].reshape(-1, 3)

def _compute_max_distances_for_all_pairs(center_points_allframes):
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


# Function for hierarchical clustering using precomputed distances
def _hierarchical_clustering_from_precomputed_distances(distances, n_clusters=4, method='ward'):
    # condensed_distances = squareform(distances)
    condensed_distances = distances[np.triu_indices(distances.shape[0], k=1)]
    z = linkage(condensed_distances, method=method)
    labels = fcluster(z, n_clusters, criterion='maxclust')
    return labels
# endregion

def _save_clustered_data(num_clusters, raw_data_folderpath, clustered_data_filepath, time_steps, session_clustered_data_filepath):

    clustered_points = _pipeline_clustered_data_prepare(num_clusters, raw_data_folderpath, time_steps)

    # Save the clustered data
    with open(clustered_data_filepath, 'wb') as f:
        pickle.dump(clustered_points, f)

    # Save the session clustered data
    with open(session_clustered_data_filepath, 'wb') as f:
        pickle.dump(clustered_points, f)

def _pipeline_clustered_data_prepare(num_clusters, folder_path_meshes, time_steps) -> ClusteredCenterPointsAllFrames:

    center_points_allframes : np.array = load_centers_files(folder_path_meshes, time_steps) # (x,y,z)
    max_distances_mx = _compute_max_distances_for_all_pairs(center_points_allframes)

    cluster_center_labels = _hierarchical_clustering_from_precomputed_distances(max_distances_mx, n_clusters=num_clusters)
    clustered_center_points_allframes = ClusteredCenterPointsAllFrames(center_points_allframes, cluster_center_labels)
    return clustered_center_points_allframes

# Function to process and save clustered data if not already processed
def process_clustered_data(num_clusters, raw_data_folderpath, clustered_data_filepath, time_steps, session_clustered_data_filepath):
    """
    Saves all points of centers which was labeled by hierarchical clustering
    :param num_clusters:
    :param raw_data_folderpath:
    :param clustered_data_filepath:
    :param time_steps:
    :return:
    """
    if not os.path.exists(clustered_data_filepath):
        _save_clustered_data(num_clusters, raw_data_folderpath, clustered_data_filepath, time_steps, session_clustered_data_filepath)
        logging.info("Clustered data processed and saved.")
    else:
        # copy file from clustered_data_filepath to session_clustered_data_filepath
        with open(clustered_data_filepath, 'rb') as f:
            clustered_points = pickle.load(f)
        with open(session_clustered_data_filepath, 'wb') as f:
            pickle.dump(clustered_points, f)

        logging.info("Clustered data already processed.")