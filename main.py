# Author: Filip Cerny
# Date: 2024-10-15
# Description: This file contains the main functions for the project.


import glob
import pickle
from itertools import combinations
from math import sqrt

import matplotlib.pyplot as plt
import trimesh
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN
import os

SURFACE_DATA_LIST_FILENAME = 'surface_data_list.pkl'

CLUSTERED_DATA_FILENAME = 'clustered_data.pkl'

MODEL_WEIGHTS_FILENAME = 'model_weights.pth'

NUM_CLUSTERS = 5

RAW_DATA_ALLOWED_FILETYPES_LIST = ['xyz', 'bin']

JSON_FILENAME = "center_mesh_pairs.json"

# region Constants

PUB_data_folder_path = 'data/raw/casual_man/'  # Update with the correct path
viz_obj_file_path = 'data/raw/casual_man/axyz_000001.obj'  # Path to your .obj file
PROCESSED_DATE_FOLDER = 'data/processed/casual'
MODEL_WEIGHTS_FOLDER = 'data/processed/casual/'

SURFACE_DATA_LIST_FILEPATH = os.path.join(PROCESSED_DATE_FOLDER, SURFACE_DATA_LIST_FILENAME)
CLUSTERED_DATA_FILEPATH = os.path.join(PROCESSED_DATE_FOLDER, CLUSTERED_DATA_FILENAME)
MODEL_WEIGHTS_FILEPATH = os.path.join(MODEL_WEIGHTS_FOLDER, MODEL_WEIGHTS_FILENAME)

# endregion

# region Functions
# region Load Data
# Function to compute Euclidean distance between two 3D points
def euclidean_distance(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


# Function to load .xyz files
def load_xyz_files(filepaths):
    points_in_file = None
    data = []
    for filepath in filepaths:
        points_in_time = np.loadtxt(filepath, delimiter=' ')  # Adjust delimiter if needed
        if points_in_file is None:
            points_in_file = points_in_time.shape[0]
        elif points_in_file != points_in_time.shape[0]:
            raise ValueError("Inconsistent number of points in the files.")
        data.append(points_in_time)

    return np.array(data), points_in_file  # Shape: (num_files, num_time_steps, num_points)

    # Function to compute max distances between pairs of points across time


def compute_max_distances_for_all_pairs(data, num_points_in_file):
    max_distances = np.zeros((num_points_in_file, num_points_in_file))  # Array to hold max distances

    i = 0
    for file_data in data:
        points = file_data.reshape(-1, 3)  # Reshape row into list of 3D points
        for p1, p2 in combinations(range(num_points_in_file), 2):
            distance = euclidean_distance(points[p1], points[p2])
            if distance > max_distances[p1, p2]:
                max_distances[p1, p2] = distance
        i += 1
        print("computing max distances" + str(i))
    return max_distances


# Function for DBSCAN clustering using precomputed distances
def dbscan_clustering_from_precomputed_distances(distances, eps=0.5, min_samples=5):
    # Extract the upper triangle of the matrix (if needed) or use the distances directly
    # distances = distances[np.triu_indices(distances.shape[0], k=1)]  # If necessary

    # Perform DBSCAN clustering with the precomputed distance matrix
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = db.fit_predict(distances)

    return labels


# Function to load .bin files and return data
def load_bin_files(filepaths):
    print("Loading .bin files...")
    data = []
    points_in_file = None
    i = 0
    for filepath in filepaths:

        i += 1
        print("Loading file " + str(i) + " of " + str(len(filepaths)))
        # Read the binary file as 32-bit floats
        file_data = np.fromfile(filepath, dtype=np.float32)
        file_data = file_data[1:]  # first is size

        # Reshape the data into 3D points (3 values per point)
        points = file_data.reshape(-1, 3)

        # points_in_file
        if points_in_file is None:
            points_in_file = points.shape[0]
        elif points_in_file != points.shape[0]:
            raise ValueError("Inconsistent number of points in the files.")

        # Append the points for this file to the data list
        data.append(points)

    return data, points_in_file


def load_data(folder_path):
    """
    Load .xyz files or .bin files from the specified folder and compute the maximum distances between points.
    :param folder_path: path to where is files with computed centers points
    :param file_type:
    :return: max_distances, data
        max_distances: array of maximum distances between all points of all centers
        data: array of all centers points
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

    # Load the .xyz files
    if file_type == 'xyz':
        data, points_in_file = load_xyz_files(filepaths)
    elif file_type == 'bin':
        data, points_in_file = load_bin_files(filepaths)
    else:
        raise ValueError("Invalid file type. Use 'xyz' or 'bin'.")

    # Compute the maximum distances between points
    max_distances = compute_max_distances_for_all_pairs(data, points_in_file)
    print("Max distances computed between all pairs of points.")

    # if max_distances is empty
    if not max_distances.any():
        raise ValueError("No data loaded or max distances computed.")

    return max_distances, data


# endregion

# region Clustering

# region Visualization


# Function to visualize points and clusters on a 3D model
def visualize_clusters_with_mesh(points, labels, obj_file_path):
    mesh = trimesh.load(obj_file_path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    mesh.show()  # Visualize the mesh

    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap='jet', s=50)
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
    plt.show()


# write me function which will from the variable PUB_all_center_points get points from specific time step
def get_points_from_time_step(data, time_step):
    return data[time_step].reshape(-1, 3)


# endregion
# class for clustered data
class ClusteredData:
    def __init__(self, center_points_list, cluster_center_labels):
        self.points = center_points_list
        self.labels = cluster_center_labels

    def get_points_from_time_step(self, time_step):
        return self.points[time_step].reshape(-1, 3)


# Function for hierarchical clustering using precomputed distances
def hierarchical_clustering_from_precomputed_distances(distances, n_clusters=4, method='ward'):
    # condensed_distances = squareform(distances)
    condensed_distances = distances[np.triu_indices(distances.shape[0], k=1)]
    Z = linkage(condensed_distances, method=method)
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    return labels


def hierarchical_cluster_data(folder_path, obj_file_path, file_type, num_clusters=2):
    max_distances, data = load_data(folder_path, file_type)
    # Perform hierarchical clustering using precomputed distances

    labels = hierarchical_clustering_from_precomputed_distances(max_distances, n_clusters=num_clusters)
    print("Hierarchical clustering completed.")
    # Select points from a specific time step (for example, the first time step)
    points = data[0].reshape(-1, 3)  # Reshape into list of 3D points

    # Visualize the clusters on the 3D model

    visualize_clusters_with_mesh(points, labels, obj_file_path)


# endregion
# region Mapping to surface mesh
from scipy.spatial import KDTree


def create_categorized_surface_points(mesh, clustered_points, cluster_labels, num_surface_points=1000):
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

    mesh_vertices = mesh.vertices
    mesh_faces = mesh.faces
    # Generate random points on the surface of the mesh
    surface_points = generate_random_points_on_mesh(mesh_vertices, mesh_faces, num_surface_points)

    # Build a KDTree for the clustered points
    kdtree = KDTree(clustered_points)

    # Find the closest clustered point for each surface point
    _, indices = kdtree.query(surface_points)

    # Assign the cluster label of the closest point to the surface point
    surface_labels = cluster_labels[indices]

    return surface_points, surface_labels


def generate_random_points_on_mesh(vertices, faces, num_points):
    """
    Generate random points on the surface of a mesh.

    Parameters:
    - vertices: np.ndarray of shape (n_vertices, 3)
    - faces: np.ndarray of shape (n_faces, 3)
    - num_points: int, number of random points to generate

    Returns:
    - points: np.ndarray of shape (num_points, 3)
    """

    # Compute the area of each face
    def triangle_area(v0, v1, v2):
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

    areas = np.array([triangle_area(vertices[f[0]], vertices[f[1]], vertices[f[2]]) for f in faces])
    total_area = np.sum(areas)
    areas /= total_area

    # Select faces based on their area
    face_indices = np.random.choice(len(faces), size=num_points, p=areas)

    # Generate random points on the selected faces
    points = []
    for i in face_indices:
        f = faces[i]
        v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
        r1, r2 = np.random.rand(2)
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        point = (1 - r1 - r2) * v0 + r1 * v1 + r2 * v2
        points.append(point)

    return np.array(points)


# region visulization

def visualize_surface_points(points, labels):
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
# region Neural network
# region Data preparation
import json
import os


# create surface points for all meshes

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


class SurfaceDataList:
    def __init__(self, surface_data_list):
        if not all(isinstance(item, SurfaceData) for item in surface_data_list):
            raise TypeError("All items in surface_data_list must be instances of SurfaceData")
        self.list = surface_data_list

    def create_surface_points_from_mesh_list(self, meshes_filepaths_list, center_points_list, cluster_center_labels,
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
            print("Creating surface points for mesh " + str(i + 1) + " of " + str(len(meshes_filepaths_list)))
            mesh = trimesh.load(mesh_file_path)
            centers_points = center_points_list[i]
            surface_points, surface_labels = create_categorized_surface_points(mesh, centers_points,
                                                                               cluster_center_labels,
                                                                               num_surface_points)
            # append both values to list with names in the list
            surface_data_list.append(SurfaceData(surface_points, surface_labels, None))

        self.list = surface_data_list.list  # add time function

    def add_time(surface_data_list):
        for i, surface_data in enumerate(surface_data_list.list):
            if surface_data.time is not None:
                Exception("Time already added")
                break

            surface_data.time = i
        surface_data_list.list = surface_data_list.list  # normalize the data

    def normalize(surface_data_list):
        """
        Normalize the surface points for all objects in the list.
        it will normalize the data to the range [0, 1] for each axis.
        and shifts the data to the origin of 0, 0, 0
        :return:  normalized_surface_points: SurfaceDataList
        """

        def normalize_time(time, length):
            length = length - 1
            new_time = time / length
            # print all velues
            print("Time: ", time, "Length: ", length, "New time: ", new_time)
            return new_time

        def scale_object_points(object_points, norm):
            return object_points / norm

        def shift_object_points_to_origin(object_points, shift_vector):
            return object_points - shift_vector

        def normalize_object_points(object_points, norm, shift_vector):
            object_points = shift_object_points_to_origin(object_points, shift_vector)
            object_points = scale_object_points(object_points, norm)
            return object_points

        # Calculate the bounding box for all objects combined
        all_points = np.vstack([surface_data.surface_points for surface_data in surface_data_list.list])
        min_corner = np.min(all_points, axis=0)
        shift_vector = min_corner  # Shift all objects to the position of the minimum corner

        # Calculate the maximum norm of all points across all axes for each object
        max_norm = max(
            np.linalg.norm(surface_data_element.surface_points, ord=None, axis=None).max() for surface_data_element in
            surface_data_list.list)

        normalized_surface_points = SurfaceDataList([])
        for surface_data in surface_data_list.list:
            print("Normalizing surface points for time step " + str(surface_data.time))
            normalized_time = normalize_time(surface_data.time, len(surface_data_list.list))
            normalized_points = normalize_object_points(surface_data.surface_points, max_norm, shift_vector)
            normalized_surface_points.append(
                SurfaceData(normalized_points, surface_data.surface_labels, normalized_time))

        surface_data_list.list = normalized_surface_points.list

    def prepare_data(self, meshes_filepaths_list, center_points_list, cluster_center_labels, num_surface_points):

        print("Creating surface points for all meshes...")
        self.create_surface_points_from_mesh_list(meshes_filepaths_list, center_points_list, cluster_center_labels,
                                                  num_surface_points)
        self.add_time()
        self.normalize()

    # append function
    def append(self, surface_data):
        # check if surface_data is instance of SurfaceData
        if not isinstance(surface_data, SurfaceData):
            raise TypeError("surface_data must be an instance of SurfaceData")
        self.list.append(surface_data)


class SurfaceData:
    def __init__(self, surface_points, surface_labels, time):
        self.surface_points = surface_points
        self.surface_labels = surface_labels
        self.time = time


# endregion
# region Neural network training

import torch.optim as optim

import numpy as np
import torch
from torch.utils.data import Dataset


class SurfaceDataset(Dataset):
    def __init__(self, surface_data_list):
        self.data = []
        for surface_data in surface_data_list:
            points = surface_data.surface_points
            time = np.full((points.shape[0], 1), surface_data.time)
            points_with_time = np.hstack((points, time))
            self.data.append(points_with_time)
        self.data = np.vstack(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :], self.data[idx, :]


import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        self.decoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):
        time_value = x[:, 3].unsqueeze(1)  # Extract time value and keep it as a column vector
        encoded_features = self.encoder(x)
        encoded_with_time = torch.cat((encoded_features, time_value), dim=1)  # Concatenate encoded features with time
        decoded_output = self.decoder(encoded_with_time)
        return decoded_output


def train_neural_network(data, num_epochs, model_save_path='model_weights.pth'):
    # Create dataset and dataloader
    dataset = SurfaceDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    # Initialize model, loss function, and optimizer
    model = MLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Training loop
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.float(), targets.float()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the model weights
    torch.save(model.state_dict(), model_save_path)
    print(f'Model weights saved to {model_save_path}')
# endregion
# endregion

# region Pipelines
def pipeline_clustered_data_prepare():
    num_clusters = NUM_CLUSTERS
    folder_path_meshes = PUB_data_folder_path
    max_distances, center_points_list = load_data(folder_path_meshes)

    cluster_center_labels = hierarchical_clustering_from_precomputed_distances(max_distances, n_clusters=num_clusters)
    clustered_data = ClusteredData(center_points_list, cluster_center_labels)
    return clustered_data


def pipeline_nn_data_prepare(clustered_data):
    meshes_folder_path = PUB_data_folder_path
    json_file_name = JSON_FILENAME

    json_file_path = meshes_folder_path + json_file_name
    center_points_list = clustered_data.points
    cluster_center_labels = clustered_data.labels

    meshes_filepaths_list = get_filepaths_from_json(meshes_folder_path, json_file_path)
    print("Creating surface points for all meshes...")
    PUB_surface_data_list = SurfaceDataList([])
    PUB_surface_data_list.create_surface_points_from_mesh_list(meshes_filepaths_list, center_points_list,
                                                               cluster_center_labels, 1000)
    """list of tuples with surface points and labels for all meshes specified in config json file"""
    PUB_surface_data_list.add_time()
    PUB_surface_data_list.normalize()
    print("Surface points created and normalized.")
    return PUB_surface_data_list


def save_clustered_data():
    data_processed_folder_path = PROCESSED_DATE_FOLDER

    clustered_data = pipeline_clustered_data_prepare()

    # Save the clustered data
    with open(os.path.join(data_processed_folder_path, CLUSTERED_DATA_FILENAME), 'wb') as f:
        pickle.dump(clustered_data, f)


def save_nn_data(clustered_data):
    data_processed_folder_path = PROCESSED_DATE_FOLDER

    surface_data_list = pipeline_nn_data_prepare(clustered_data)

    # save the surface data list
    with open(os.path.join(data_processed_folder_path, SURFACE_DATA_LIST_FILENAME), 'wb') as f:
        pickle.dump(surface_data_list, f)


# endregion
# endregion


# region Main
if __name__ == '__main__':
    # check if files are already processed
    data_processed_folder_path = PROCESSED_DATE_FOLDER
    if not os.path.exists(data_processed_folder_path):
        os.makedirs(data_processed_folder_path)
    # check if clustered data is already processed
    if not os.path.exists(os.path.join(data_processed_folder_path, CLUSTERED_DATA_FILENAME)):
        save_clustered_data()
    else:
        print("Clustered data already processed.")
    # check if nn data is already processed
    if not os.path.exists(os.path.join(data_processed_folder_path, SURFACE_DATA_LIST_FILENAME)):
        with open(os.path.join(data_processed_folder_path, CLUSTERED_DATA_FILENAME), 'rb') as f:
            clustered_data = pickle.load(f)
        save_nn_data(clustered_data)
    else:
        print("Neural network data already processed.")

    # pickle load surface_data
    with open(os.path.join(data_processed_folder_path, SURFACE_DATA_LIST_FILENAME), 'rb') as f:
        surface_data_list = pickle.load(f)

    # test train neural network
    surface_data_cluster_0 = [surface_data for surface_data in surface_data_list.list if
                              surface_data.surface_labels == 0]

    model_weights_path = MODEL_WEIGHTS_FILEPATH
    num_epochs = 100
    train_neural_network(surface_data_cluster_0, num_epochs, model_weights_path)

# endregion
