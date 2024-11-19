import logging
import os
import pickle

import numpy as np
import trimesh
from matplotlib import pyplot as plt
from scipy.spatial import KDTree

from src.utils.helpers import load_pickle_file, get_meshes_list


# Restrict access to underscore-prefixed functions
def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"{name} is a private function and cannot be imported.")
    raise AttributeError(f"Module has no attribute {name}")


def convert_to_surface_data_list(input_list):
    """
    Converts a standard Python list into a SurfaceDataList.

    Parameters:
    - input_list: list of dictionaries or SurfaceData objects

    Returns:
    - SurfaceDataList instance
    """
    surface_data_objects = []

    for item in input_list:
        if isinstance(item, SurfaceData):
            # Already a SurfaceData object
            surface_data_objects.append(item)
        else:
            surface_data_objects.append(SurfaceData(item.points_list, item.labels_list, item.time))

    return SurfaceDataList(surface_data_objects)


class SurfaceDataList:
    def __init__(self, surface_data_list: list):
        if not isinstance(surface_data_list, list) or not all(
                isinstance(item, SurfaceData) for item in surface_data_list):
            raise TypeError("All items in surface_data_list must be instances of SurfaceData")
        self.list = surface_data_list
        # Initialize unique_clusters at creation
        self.unique_clusters = self.compute_unique_clusters()

    def get_unique_times(self):
        """
        Return the set of unique times.
        """
        return {surface_data.time for surface_data in self.list}

    def filter_by_time(self, time_index):
        """
        :param time_index:
        :return:
        """
        filtered_data = []
        for surface_data in self.list:
            if surface_data.time == time_index:
                filtered_data.append(surface_data)

        return SurfaceDataList(filtered_data)

    def get_cluster_labels(self):
        """
        Return the list of cluster labels.
        """
        return [label for surface_data in self.list for label in surface_data.labels_list]

    def compute_unique_clusters(self):
        """
        Private method to compute unique clusters from the surface data list.
        """
        unique_clusters = set()
        for surface_data in self.list:
            unique_clusters.update(
                int(label) for label in surface_data.labels_list)  # Convert each sub-array to a tuple
        return unique_clusters

    def get_unique_clusters(self):
        """
        Return the set of unique clusters.
        """
        self.unique_clusters = self.compute_unique_clusters()

        if self.unique_clusters is None or not self.unique_clusters:
            raise Exception("Unique clusters is Empty")

        return self.unique_clusters

    def append(self, surface_data):
        """
        Append a SurfaceData object to the list and update unique clusters.
        """
        if not isinstance(surface_data, SurfaceData):
            raise TypeError("surface_data must be an instance of SurfaceData")
        self.list.append(surface_data)
        self.unique_clusters.update(surface_data.labels_list)  # Update unique clusters

    def remove(self, surface_data):
        """
        Remove a SurfaceData object from the list and update unique clusters.
        """
        if surface_data in self.list:
            self.list.remove(surface_data)
            # Recompute unique clusters in case a label is no longer present
            self.unique_clusters = self.compute_unique_clusters()

    # append function
    def append(self, surface_data):
        # check if surface_data is instance of SurfaceData
        if not isinstance(surface_data, SurfaceData):
            raise TypeError("surface_data must be an instance of SurfaceData")
        self.list.append(surface_data)

    def filter_by_label(self, label_index):
        """
        Filter the SurfaceDataList by the given label index, keeping only the corresponding surface points.

        Parameters:
        - label_index: int, the label index to filter by

        Returns:
        - SurfaceDataList instance containing only the SurfaceData objects with the specified label index in both
          surface_labels_list and surface_points_list
        """
        filtered_data = []

        for surface_data in self.list:
            # Convert labels_list to a numpy array for efficient filtering
            labels_array = np.array(surface_data.labels_list)
            points_array = np.array(surface_data.points_list)

            # Find indices where the label matches the specified label index
            matching_indices = np.where(labels_array == label_index)[0]

            if matching_indices.size == 0:
                raise ValueError("No points with the specified label index found.")

            # Use the indices to filter points and labels
            filtered_points = points_array[matching_indices].tolist()
            filtered_labels = labels_array[matching_indices].tolist()

            # Create a new SurfaceData instance with the filtered points and labels
            filtered_data.append(SurfaceData(filtered_points, filtered_labels, surface_data.time))

        return SurfaceDataList(filtered_data)


class SurfaceData:
    """
    Class to represents points in object for one cluster and for a single time step.
    """

    def __init__(self, surface_points, surface_labels, time):
        self.points_list = surface_points
        self.labels_list = surface_labels
        self.time = time


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

def _create_categorized_surface_points(mesh, clustered_points, cluster_labels, num_surface_points):
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
    clustered_points = np.array(clustered_points)
    cluster_labels = np.array(cluster_labels)

    # Generate random points on the surface of the mesh
    surface_points = _generate_random_points_on_mesh(mesh_vertices, mesh_faces, num_surface_points)

    # Build a KDTree for the clustered points
    kdtree = KDTree(clustered_points)

    # Find the closest clustered point for each surface point
    _, indices = kdtree.query(surface_points)

    # Assign the cluster label of the closest point to the surface point
    surface_labels = cluster_labels[indices]

    # Return as NumPy arrays
    return np.array(surface_points), np.array(surface_labels)


def _assign_time_to_surfaces(surface_data_list):
    """Assign a time index to each surface data item if not already assigned."""
    for i, surface_data in enumerate(surface_data_list.list):
        if surface_data.time is not None:
            raise Exception("Time already added to surface data.")

        surface_data.time = i


def _normalize(surface_data_list):
    # todo test if it is working
    """
    Normalize the surface points for all objects in the list.
    Normalize the data to the range [0, 1] for each axis and shift it to the origin (0, 0, 0).
    :return: normalized_surface_points: SurfaceDataList
    """
    import numpy as np

    def normalize_time(surface_data_list):
        total_length = len(surface_data_list.list) - 1
        for surface_data in surface_data_list.list:
            surface_data.time /= total_length

    def compute_shift_and_scale(surface_data_list):
        # Combine all points for faster computation
        all_points = np.vstack([surface_data.points_list for surface_data in surface_data_list.list])
        min_corner = np.min(all_points, axis=0)
        max_corner = np.max(all_points, axis=0)
        shift_vector = (min_corner + max_corner) / 2
        max_norm = np.linalg.norm(all_points - shift_vector, axis=1).max()
        return shift_vector, max_norm

    def shift_and_scale_points(surface_data_list, shift_vector, max_norm):
        for surface_data in surface_data_list.list:
            surface_data.points_list = (surface_data.points_list - shift_vector) / max_norm

    # Normalize time for each object
    normalize_time(surface_data_list)

    # Compute the shift vector and max norm
    shift_vector, max_norm = compute_shift_and_scale(surface_data_list)

    # Shift points to origin and scale
    shift_and_scale_points(surface_data_list, shift_vector, max_norm)

    return surface_data_list


def _create_surface_points_from_mesh_list(meshes_filepaths_list, center_points_list, cluster_center_labels,
                                          num_surface_points):
    """
    Create surface points for each mesh in the list.

    Parameters:
    - meshes_filepaths_list: list of str, list of file paths to the meshes

    Returns:
    - surface_points_list: list of np.ndarray, list of surface points for each mesh
    """
    surface_data_list = SurfaceDataList([])
    for i, mesh_file_path in enumerate(meshes_filepaths_list):
        logging.info("Creating surface points for mesh " + str(i + 1) + " of " + str(len(meshes_filepaths_list)))
        mesh = trimesh.load(mesh_file_path)
        centers_points = center_points_list[i]
        surface_points, surface_labels = _create_categorized_surface_points(mesh, centers_points,
                                                                            cluster_center_labels,
                                                                            num_surface_points)
        # append both values to list with names in the list
        surface_data_list.append(SurfaceData(surface_points, surface_labels, None))

    return surface_data_list


def _prepare_surface_data(meshes_filepaths_list, center_points_list, cluster_center_labels, num_surface_points):
    logging.info("Creating surface points for all meshes...")
    surface_data_list = _create_surface_points_from_mesh_list(meshes_filepaths_list, center_points_list,
                                                              cluster_center_labels,
                                                              num_surface_points)

    _assign_time_to_surfaces(surface_data_list)
    _normalize(surface_data_list)

    return surface_data_list


def _pipeline_prepare_surface_data(clustered_data, num_surface_points, meshes_folder_path):
    center_points_list = clustered_data.points
    cluster_center_labels = clustered_data.labels

    # meshes_filepaths_list = get_filepaths_from_json(meshes_folder_path, json_file_path)
    meshes_filepaths_list = get_meshes_list(meshes_folder_path)
    logging.info("Creating surface points for all meshes...")
    surface_data_list = _prepare_surface_data(meshes_filepaths_list, center_points_list,
                                              cluster_center_labels, num_surface_points)
    logging.info("Surface points created and normalized.")
    return surface_data_list


def _save_surface_data(clustered_data, num_surface_points, meshes_folder_path,
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
# todo graphics max distance computing

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
    cross_products = np.cross(v1 - v0, v2 - v0)
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


# Function to process and save neural network data if not already processed
def process_surface_data(num_surface_points, meshes_folder_path, surface_data_filepath, clustered_data_filepath):
    if not os.path.exists(surface_data_filepath):
        clustered_data = load_pickle_file(clustered_data_filepath)
        if clustered_data is None:
            logging.error("Clustered data could not be loaded. Exiting.")
            return
        _save_surface_data(clustered_data, num_surface_points, meshes_folder_path, surface_data_filepath)
        logging.info("Neural network data processed and saved.")
    else:
        logging.info("Neural network data already processed.")
