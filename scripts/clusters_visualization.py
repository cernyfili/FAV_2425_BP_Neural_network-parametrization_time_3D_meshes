import os

import numpy as np
from matplotlib import pyplot as plt

from src.utils.helpers import load_pickle_file

CLUSTERS_FILEPATH = 'data/clusters.csv'
ORIGINAL_POINTCLOUD_FILEPATH = 'data/pointcloud.csv'
PROCESSED_POINTCLOUD_FILEPATH = 'data/processed_pointcloud.csv'
IMAGE_SAVE_FOLDERPATH = 'images'

def _visualize_combined_surface_points_images(original_points_all, processed_points_all,
                                              image_save_folder, clusters=None):
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
        # Create a new figure for each time slice
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Filter original points for this time slice
        original_points_slice = original_points_all[original_points_all[:, 3] == time]

        # Plot original points for this time slice
        ax.scatter(original_points_slice[:, 0], original_points_slice[:, 1], original_points_slice[:, 2],
                   color='blue', label='Original Points', alpha=0.5)

        processed_points_slice = processed_points_all[processed_points_all[:, 3] == time]

        if clusters is not None:
            original_colors = clusters[original_points_all[:, 3] == time]
            ax.scatter(original_points_slice[:, 0], original_points_slice[:, 1], original_points_slice[:, 2],
                   c=original_colors, cmap='viridis', label='Original Points', alpha=0.5)
        else:
             ax.scatter(original_points_slice[:, 0], original_points_slice[:, 1], original_points_slice[:, 2],
                       color='blue', label='Original Points', alpha=0.5)

        if clusters is not None:
            processed_colors = clusters[processed_points_all[:, 3] == time]
            ax.scatter(processed_points_slice[:, 0], processed_points_slice[:, 1], processed_points_slice[:, 2],
                       c=processed_colors, cmap='plasma', label='Processed Points', alpha=0.5)
        else:
            ax.scatter(processed_points_slice[:, 0], processed_points_slice[:, 1], processed_points_slice[:, 2],
                       color='red', label='Processed Points', alpha=0.5)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title(f'3D Visualization of Original and Processed Points from Time {time}')
        ax.legend()

        # Save the plot for the current time slice
        image_path = os.path.join(image_save_folder, f'combined_surface_points_time_{time}.png')
        plt.savefig(image_path)
        plt.close(fig)

        print(f"Saved combined surface points image at {image_path}")



clustered_data = load_pickle_file(CLUSTERS_FILEPATH)

# load point cloud which is saved as a csv file where each row is a point without function load_picke_file
original_points = np.loadtxt(ORIGINAL_POINTCLOUD_FILEPATH, delimiter=',')
processed_points = np.loadtxt(PROCESSED_POINTCLOUD_FILEPATH, delimiter=',')

_visualize_combined_surface_points_images(original_points, processed_points, IMAGE_SAVE_FOLDERPATH, clustered_data.labels_frame)

