import logging
import os
import pickle

import numpy as np
import trimesh
from matplotlib import pyplot as plt
from scipy.spatial import KDTree

from data_processing.class_clustering import ClusteredCenterPointsAllFrames
from data_processing.class_mapping import SurfacePointsFrameList, SurfacePointsFrame
from data_processing.data_structures import MeshNDArray
from src.utils.helpers import load_pickle_file, get_meshes_list
from utils.helpers import get_file_index_from_filename


# Restrict access to underscore-prefixed functions
def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"{name} is a private function and cannot be imported.")
    raise AttributeError(f"Module has no attribute {name}")


# region PRIVATE FUNCTIONS

# def _convert_to_surface_data_list(input_list):
#     """
#     Converts a standard Python list into a SurfaceDataList.
#
#     Parameters:
#     - input_list: list of dictionaries or SurfaceData objects
#
#     Returns:
#     - SurfaceDataList instance
#     """
#     surface_data_objects = []
#
#     for item in input_list:
#         if isinstance(item, SurfacePointsFrame):
#             # Already a SurfaceData object
#             surface_data_objects.append(item)
#         else:
#             surface_data_objects.append(SurfacePointsFrame(item.points_list, item.labels_list, item.time))
#
#     return SurfacePointsFrameList(surface_data_objects)


#
# def _create_categorized_surface_points(mesh, clustered_points, cluster_labels, num_surface_points):
#     """
#     Categorize random points on the surface of a mesh based on the closest point inside the mesh.
#
#     Parameters:
#     - mesh_vertices: np.ndarray of shape (n_vertices, 3)
#     - mesh_faces: np.ndarray of shape (n_faces, 3)
#     - clustered_points: np.ndarray of shape (n_clustered_points, 3)
#     - cluster_labels: np.ndarray of shape (n_clustered_points,)
#     - num_surface_points: int, number of random points to generate on the surface
#
#     Returns:
#     - surface_points: np.ndarray of shape (num_surface_points, 3)
#     - surface_labels: np.ndarray of shape (num_surface_points,)
#     :param mesh:
#     """
#
#     mesh_vertices = mesh.vertices
#     mesh_faces = mesh.faces
#     # Generate random points on the surface of the mesh
#     surface_points = _generate_random_points_on_mesh(mesh_vertices, mesh_faces, num_surface_points)
#
#     # Build a KDTree for the clustered points
#     kdtree = KDTree(clustered_points)
#
#     # Find the closest clustered point for each surface point
#     _, indices = kdtree.query(surface_points)
#
#     # Assign the cluster label of the closest point to the surface point
#     surface_labels = cluster_labels[indices]
#
#     return surface_points, surface_labels

def _create_categorized_surface_points(mesh, centers_points_frame, centers_labels_frame, num_surface_points):
    """
    Categorize random points on the surface of a mesh based on the closest point inside the mesh.

    Parameters:
    - mesh_vertices: np.ndarray of shape (n_vertices, 3)
    - mesh_faces: np.ndarray of shape (n_faces, 3)
    - clustered_points: np.ndarray of shape (n_clustered_points, 3)
    - cluster_labels: np.ndarray of shape (n_clustered_points,)
    - num_surface_points: int, number of random points to generate on the surface

    Returns:
    - surface_points: np.ndarray of shape (num_surface_points, 3)
    - surface_labels: np.ndarray of shape (num_surface_points,)
    :param mesh:
    """

    # Extract mesh vertices and faces as NumPy arrays
    mesh_vertices = np.array(mesh.vertices)
    mesh_faces = np.array(mesh.faces)

    # Generate random points on the surface of the mesh
    surface_points = _generate_random_points_on_mesh(mesh_vertices, mesh_faces, num_surface_points)

    surface_labels = categorize_points_with_labels(centers_labels_frame, centers_points_frame, surface_points)

    # Return as NumPy arrays
    return np.array(surface_points), np.array(surface_labels)


def categorize_points_with_labels(centers_labels_frame, centers_points_frame, points):
    centers_points_frame = np.array(centers_points_frame)
    centers_labels_frame = np.array(centers_labels_frame)
    if centers_points_frame.shape[0] != centers_labels_frame.shape[0]:
        raise ValueError("Shapes of points and labels do not match.")
    # Build a KDTree for the clustered points
    kdtree = KDTree(centers_points_frame)
    # Find the closest clustered point for each surface point
    _, indices = kdtree.query(points)
    # Assign the cluster label of the closest point to the surface point
    surface_labels = centers_labels_frame[indices]
    # todo check if surface_labels is generated with same size as surface_points
    return surface_labels


def _create_surface_points_from_mesh_list(meshes_filepaths_list : list, clustered_data : ClusteredCenterPointsAllFrames,
                                          num_surface_points : int):
    """
    Create surface points for each mesh in the list.

    Parameters:
    - meshes_filepaths_list: list of str, list of file paths to the meshes

    Returns:
    - surface_points_list: list of np.ndarray, list of surface points for each mesh
    """
    surface_data_list = SurfacePointsFrameList([])

    # find min fileindex in meshes_filepaths_list
    min_file_index = min([get_file_index_from_filename(mesh_file_path) for mesh_file_path in meshes_filepaths_list])

    for i, mesh_file_path in enumerate(meshes_filepaths_list):
        logging.info("Creating surface points for mesh " + str(i + 1) + " of " + str(len(meshes_filepaths_list)))
        mesh = trimesh.load(mesh_file_path)
        centers_points_frame = clustered_data.points_allframes[i]
        center_labels_frame = clustered_data.labels_frame
        #check indexes with filepath names

        # file index is the last 3 characters before filetype
        file_index = get_file_index_from_filename(mesh_file_path, min_file_index)

        if file_index != i:
            raise ValueError("Inconsistent index in file name. Expected: " + str(i) + ", Found: " + str(file_index))

        surface_points, surface_labels = _create_categorized_surface_points(mesh, centers_points_frame,
                                                                            center_labels_frame,
                                                                            num_surface_points)

        # append both values to list with names in the list
        surface_data_list.append(SurfacePointsFrame(surface_points, surface_labels, None, mesh))

    return surface_data_list


def _prepare_surface_data(meshes_filepaths_list, clustered_data, num_surface_points):
    logging.info("Creating surface points for all meshes...")
    surface_data_list = _create_surface_points_from_mesh_list(meshes_filepaths_list, clustered_data,
                                                              num_surface_points)

    surface_data_list.assign_time_to_all_elements()
    surface_data_list.normalize_all_elements()

    return surface_data_list


def _pipeline_prepare_surface_data(clustered_data, num_surface_points, meshes_folder_path):
    # meshes_filepaths_list = get_filepaths_from_json(meshes_folder_path, json_file_path)
    meshes_filepaths_list = get_meshes_list(meshes_folder_path)
    logging.info("Creating surface points for all meshes...")
    surface_data_list = _prepare_surface_data(meshes_filepaths_list, clustered_data, num_surface_points)
    logging.info("Surface points created and normalized.")
    return surface_data_list


def _save_surface_data(clustered_data : ClusteredCenterPointsAllFrames, num_surface_points, meshes_folder_path,
                       surface_data_filepath):
    surface_data_list = _pipeline_prepare_surface_data(clustered_data, num_surface_points, meshes_folder_path)

    # save the surface data list
    with open(surface_data_filepath, 'wb') as f:
        pickle.dump(surface_data_list, f)


# def _generate_random_points_on_mesh(vertices, faces, num_points):
#     """
#     Generate random points on the surface of a mesh.
#
#     Parameters:
#     - vertices: np.ndarray of shape (n_vertices, 3)
#     - faces: np.ndarray of shape (n_faces, 3)
#     - num_points: int, number of random points to generate
#
#     Returns:
#     - points: np.ndarray of shape (num_points, 3)
#     """
#
#     # Compute the area of each face
#     def triangle_area(v0, v1, v2):
#         return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
#
#     areas = np.array([triangle_area(vertices[f[0]], vertices[f[1]], vertices[f[2]]) for f in faces])
#     total_area = np.sum(areas)
#     areas /= total_area
#
#     # Select faces based on their area
#     face_indices = np.random.choice(len(faces), size=num_points, p=areas)
#
#     # Generate random points on the selected faces
#     points = []
#     for i in face_indices:
#         f = faces[i]
#         v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
#         r1, r2 = np.random.rand(2)
#         if r1 + r2 > 1:
#             r1, r2 = 1 - r1, 1 - r2
#         point = (1 - r1 - r2) * v0 + r1 * v1 + r2 * v2
#         points.append(point)
#
#     return np.array(points)

def _generate_random_points_on_mesh(vertices, faces, num_points):
    # todo check if it is working
    """
    Generate random points on the surface of a mesh.

    Parameters:
    - vertices: np.ndarray of shape (n_vertices, 3)
    - faces: np.ndarray of shape (n_faces, 3)
    - num_points: int, number of random points to generate

    Returns:
    - points: np.ndarray of shape (num_points, 3)
    """

    # Compute the area of each face using vectorized operations
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross_products = np.cross(v1 - v0, v2 - v0) #todo check why inreachable
    areas = 0.5 * np.linalg.norm(cross_products, axis=1)
    total_area = np.sum(areas)
    probabilities = areas / total_area

    # Select faces based on their area
    face_indices = np.random.choice(len(faces), size=num_points, p=probabilities)

    # Generate random barycentric coordinates
    r1 = np.random.rand(num_points)
    r2 = np.random.rand(num_points)
    mask = r1 + r2 > 1
    r1[mask], r2[mask] = 1 - r1[mask], 1 - r2[mask]

    # Compute points on the selected faces
    selected_faces = faces[face_indices]
    v0 = vertices[selected_faces[:, 0]]
    v1 = vertices[selected_faces[:, 1]]
    v2 = vertices[selected_faces[:, 2]]
    points = (1 - r1 - r2)[:, None] * v0 + r1[:, None] * v1 + r2[:, None] * v2

    return points


# region visulization

def _visualize_surface_points(points, labels):
    """
    Visualize the categorization of surface points in 3D.

    Parameters:
    points (numpy.ndarray): Array of 3D points with shape (n_points, 3).
    labels (numpy.ndarray): Array of cluster labels for each point with shape (n_points,).

    Returns:
    None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with different colors for each cluster
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap='viridis', marker='o')

    # Add color bar to show the cluster labels
    color_bar = plt.colorbar(scatter, ax=ax, pad=0.1)
    color_bar.set_label('Cluster Label')

    # Set labels for axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set title
    ax.set_title('3D Visualization of Surface Points Categorization')

    # Show plot
    plt.show()


# endregion

# endregion

# Function to process and save neural network data if not already processed
def process_surface_data(num_surface_points, meshes_folder_path, surface_data_filepath, clustered_data_filepath):
    if not os.path.exists(surface_data_filepath):
        clustered_centers = load_pickle_file(clustered_data_filepath)
        if clustered_centers is None:
            logging.error("Clustered data could not be loaded. Exiting.")
            return
        _save_surface_data(clustered_centers, num_surface_points, meshes_folder_path, surface_data_filepath)
        logging.info("Neural network data processed and saved.")
    else:
        logging.info("Neural network data already processed.")
