import logging
import os
import pickle

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from data_processing.class_clustering import ClusteredCenterPointsAllFrames
from src.data_processing.loader import load_centers_data, compute_max_distances_for_all_pairs


# Restrict access to underscore-prefixed functions
def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"{name} is a private function and cannot be imported.")
    raise AttributeError(f"Module has no attribute {name}")

# class for clustered data


def _save_clustered_data(num_clusters, raw_data_folderpath, clustered_data_filepath, time_steps):

    clustered_points = _pipeline_clustered_data_prepare(num_clusters, raw_data_folderpath, time_steps)

    # Save the clustered data
    with open(clustered_data_filepath, 'wb') as f:
        pickle.dump(clustered_points, f)

def _pipeline_clustered_data_prepare(num_clusters, folder_path_meshes, time_steps):

    center_points_allframes = load_centers_data(folder_path_meshes, time_steps)
    max_distances_mx = compute_max_distances_for_all_pairs(center_points_allframes)

    cluster_center_labels = _hierarchical_clustering_from_precomputed_distances(max_distances_mx, n_clusters=num_clusters)
    clustered_center_points_allframes = ClusteredCenterPointsAllFrames(center_points_allframes, cluster_center_labels)
    return clustered_center_points_allframes


# region Visualization


# Function to visualize points and clusters on a 3D model


# write me function which will from the variable PUB_all_center_points get points from specific time step
def _get_points_from_time_step(data, time_step):
    return data[time_step].reshape(-1, 3)


# endregion


# Function for hierarchical clustering using precomputed distances
def _hierarchical_clustering_from_precomputed_distances(distances, n_clusters=4, method='ward'):
    # condensed_distances = squareform(distances)
    condensed_distances = distances[np.triu_indices(distances.shape[0], k=1)]
    z = linkage(condensed_distances, method=method)
    labels = fcluster(z, n_clusters, criterion='maxclust')
    return labels



# Function to process and save clustered data if not already processed
def process_clustered_data(num_clusters, raw_data_folderpath, clustered_data_filepath, time_steps):
    if not os.path.exists(clustered_data_filepath):
        _save_clustered_data(num_clusters, raw_data_folderpath, clustered_data_filepath, time_steps)
        logging.info("Clustered data processed and saved.")
    else:
        logging.info("Clustered data already processed.")