import logging
import os
import pickle

import numpy as np
import trimesh
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage

from src.data_processing.loader import load_data, compute_max_distances_for_all_pairs


# Restrict access to underscore-prefixed functions
def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"{name} is a private function and cannot be imported.")
    raise AttributeError(f"Module has no attribute {name}")

# class for clustered data
class ClusteredData:
    def __init__(self, center_points_list, cluster_center_labels):
        self.points = center_points_list
        self.labels = cluster_center_labels

    def get_points_from_time_step(self, time_step):
        return self.points[time_step].reshape(-1, 3)


def _save_clustered_data(num_clusters, raw_data_folderpath, clustered_data_filepath):

    clustered_data = _pipeline_clustered_data_prepare(num_clusters, raw_data_folderpath)

    # Save the clustered data
    with open(clustered_data_filepath, 'wb') as f:
        pickle.dump(clustered_data, f)

def _pipeline_clustered_data_prepare(num_clusters, folder_path_meshes):

    center_points_list, points_in_file = load_data(folder_path_meshes)
    max_distances = compute_max_distances_for_all_pairs(center_points_list, points_in_file)

    cluster_center_labels = _hierarchical_clustering_from_precomputed_distances(max_distances, n_clusters=num_clusters)
    clustered_data = ClusteredData(center_points_list, cluster_center_labels)
    return clustered_data


# region Visualization


# Function to visualize points and clusters on a 3D model
def visualize_clusters(points, labels, image_save_folder, image_name):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels / np.max(labels), cmap='jet', s=50)
    plt.colorbar(scatter)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #
    # # rotate the axes and update
    # for angle in range(0, 360):
    #     ax.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(.001)

    # Set the initial view angle
    ax.view_init(elev=-70, azim=90)  # Change these values to rotate

    plt.title("3D Clusters with Mesh")

    image_path = os.path.join(image_save_folder, image_name)
    plt.savefig(image_path)
    plt.close(fig)


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
def process_clustered_data(num_clusters, raw_data_folderpath, clustered_data_filepath):
    if not os.path.exists(clustered_data_filepath):
        _save_clustered_data(num_clusters, raw_data_folderpath, clustered_data_filepath)
        logging.info("Clustered data processed and saved.")
    else:
        logging.info("Clustered data already processed.")