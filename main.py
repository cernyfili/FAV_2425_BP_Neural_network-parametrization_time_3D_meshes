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
import logging

# Configure logging for more robust output control
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# region Constants

# region relative constants
EXPORT_FOLDERNAME = "casual"
PROCESSED_FOLDERPATH = "data-main/processed"
RAW_DATA_FOLDERPATH = 'data-main/raw/casual_man/'  # Update with the correct path

VIZUALIZATION_OBJ_FILEPATH = 'data-main/raw/casual_man/axyz_000001.obj'  # Path to your .obj file

# endregion


# region static constants
MODEL_WEIGHTS_TEMPLATENAME = "model_weights_cluster_{cluster}.pth"

EXPORT_FOLDERPATH = os.path.join(PROCESSED_FOLDERPATH, EXPORT_FOLDERNAME)
os.makedirs(EXPORT_FOLDERPATH, exist_ok=True)

PROCESSED_DATA_FOLDERPATH = EXPORT_FOLDERPATH
MODEL_WEIGHTS_FOLDERPATH = EXPORT_FOLDERPATH
IMAGE_SAVE_FOLDERPATH = EXPORT_FOLDERPATH
MODEL_WEIGHTS_FILEPATH_TEMPLATE = os.path.join(EXPORT_FOLDERPATH, MODEL_WEIGHTS_TEMPLATENAME)


SURFACE_DATA_LIST_FILENAME = 'surface_data_list.pkl'

CLUSTERED_DATA_FILENAME = 'clustered_data.pkl'

MODEL_WEIGHTS_FILENAME = 'model_weights.pth'

NUM_CLUSTERS = 5

RAW_DATA_ALLOWED_FILETYPES_LIST = ['xyz', 'bin']

JSON_FILENAME = "center_mesh_pairs.json"

SURFACE_DATA_LIST_FILEPATH = os.path.join(PROCESSED_DATA_FOLDERPATH, SURFACE_DATA_LIST_FILENAME)
CLUSTERED_DATA_FILEPATH = os.path.join(PROCESSED_DATA_FOLDERPATH, CLUSTERED_DATA_FILENAME)
MODEL_WEIGHTS_FILEPATH = os.path.join(MODEL_WEIGHTS_FOLDERPATH, MODEL_WEIGHTS_FILENAME)


# endregion

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
        logging.info("computing max distances" + str(i))
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
    logging.info("Loading .bin files...")
    data = []
    points_in_file = None
    i = 0
    for filepath in filepaths:

        i += 1
        logging.info("Loading file " + str(i) + " of " + str(len(filepaths)))
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
    logging.info("Max distances computed between all pairs of points.")

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
    logging.info("Hierarchical clustering completed.")
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
            logging.info("Creating surface points for mesh " + str(i + 1) + " of " + str(len(meshes_filepaths_list)))
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
            # logging.info all velues
            logging.info("Time: ", time, "Length: ", length, "New time: ", new_time)
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
        all_points = np.vstack([surface_data.points_list for surface_data in surface_data_list.list])
        min_corner = np.min(all_points, axis=0)
        shift_vector = min_corner  # Shift all objects to the position of the minimum corner

        # Calculate the maximum norm of all points across all axes for each object
        max_norm = max(
            np.linalg.norm(surface_data_element.points_list, ord=None, axis=None).max() for surface_data_element in
            surface_data_list.list)

        normalized_surface_points = SurfaceDataList([])
        for surface_data in surface_data_list.list:
            logging.info("Normalizing surface points for time step " + str(surface_data.time))
            normalized_time = normalize_time(surface_data.time, len(surface_data_list.list))
            normalized_points = normalize_object_points(surface_data.points_list, max_norm, shift_vector)
            normalized_surface_points.append(
                SurfaceData(normalized_points, surface_data.labels_list, normalized_time))

        surface_data_list.list = normalized_surface_points.list

    def prepare_data(self, meshes_filepaths_list, center_points_list, cluster_center_labels, num_surface_points):

        logging.info("Creating surface points for all meshes...")
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
            # Find indices of points with the specified label index
            matching_indices = [i for i, label in enumerate(surface_data.labels_list) if label == label_index]
            if matching_indices is None:
                # throw error
                raise ValueError("No points with the specified label index found.")
            if matching_indices:
                # Filter surface points and labels using these indices
                filtered_points = [surface_data.points_list[i] for i in matching_indices]
                filtered_labels = [surface_data.labels_list[i] for i in matching_indices]

                # Create a new SurfaceData instance with the filtered points and labels
                filtered_data.append(SurfaceData(filtered_points, filtered_labels, surface_data.time))

        return SurfaceDataList(filtered_data)


class SurfaceData:
    def __init__(self, surface_points, surface_labels, time):
        self.points_list = surface_points
        self.labels_list = surface_labels
        self.time = time


# endregion
# region Neural network training

import torch.optim as optim

import numpy as np
import torch
from torch.utils.data import Dataset


class SurfaceDataset(Dataset):
    def __init__(self, surface_data_list):
        if surface_data_list is None:
            raise ValueError("surface_data_list must not be None")
        self.data = []
        for surface_data in surface_data_list:
            points = np.array(surface_data.points_list)  # Ensure points is a numpy array
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


# def train_neural_network(data, num_epochs, model_save_path='model_weights.pth'):
#     # Create dataset and dataloader
#     dataset = SurfaceDataset(data)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
#     # Initialize model, loss function, and optimizer
#     model = MLP()
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     # Training loop
#     for epoch in range(num_epochs):
#         for inputs, targets in dataloader:
#             inputs, targets = inputs.float(), targets.float()
#
#             # Forward pass
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#
#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
#
#     # Save the model weights
#     torch.save(model.state_dict(), model_save_path)
#     logging.info(f'Model weights saved to {model_save_path}')
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import numpy as np

# Function to get the appropriate device
def get_device():
    #return torch.device('cuda' if torch.cuda.is_available() else 'cpu') #TODO add GPU compilation support
    return torch.device('cpu')

# Function to split data and create data loaders
def create_data_loaders(data, batch_size=32):
    train_indices, val_indices = train_test_split(range(len(data)), test_size=0.2, random_state=42)
    train_dataset = Subset(data, train_indices)
    val_dataset = Subset(data, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# Function to perform one training epoch
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()  # Set model to training mode
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.float().to(device), targets.float().to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)  # Return average loss for the epoch

# Function to evaluate the model on the validation set
def evaluate(model, val_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.float().to(device), targets.float().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    return val_loss / len(val_loader)  # Return average validation loss

# Function to save the model checkpoint
def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, path)
    logging.info(f"Model saved with validation loss {val_loss:.4f} at epoch {epoch}")

# Main training function with early stopping and scheduler
def train_neural_network(data, num_epochs, patience=5, model_save_path='model_weights.pth'):
    device = get_device()
    logging.info(f"Using device: {device}")

    train_loader, val_loader = create_data_loaders(data)
    model = MLP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        # Train and evaluate for one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        # Learning rate scheduler step
        scheduler.step(val_loss)

        logging.info(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, best_val_loss, model_save_path)
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement for {epochs_no_improve} epochs")

        if epochs_no_improve >= patience:
            logging.info(f"Early stopping after {epoch} epochs (Best epoch: {best_epoch} with val loss {best_val_loss:.4f})")
            break

    logging.info(f"Training completed. Best model saved to {model_save_path}")




# endregion
# endregion

# region Pipelines
def pipeline_clustered_data_prepare():
    num_clusters = NUM_CLUSTERS
    folder_path_meshes = RAW_DATA_FOLDERPATH
    max_distances, center_points_list = load_data(folder_path_meshes)

    cluster_center_labels = hierarchical_clustering_from_precomputed_distances(max_distances, n_clusters=num_clusters)
    clustered_data = ClusteredData(center_points_list, cluster_center_labels)
    return clustered_data


def pipeline_nn_data_prepare(clustered_data):
    meshes_folder_path = RAW_DATA_FOLDERPATH
    json_file_name = JSON_FILENAME

    json_file_path = meshes_folder_path + json_file_name
    center_points_list = clustered_data.points
    cluster_center_labels = clustered_data.labels

    meshes_filepaths_list = get_filepaths_from_json(meshes_folder_path, json_file_path)
    logging.info("Creating surface points for all meshes...")
    PUB_surface_data_list = SurfaceDataList([])
    PUB_surface_data_list.create_surface_points_from_mesh_list(meshes_filepaths_list, center_points_list,
                                                               cluster_center_labels, 1000)
    """list of tuples with surface points and labels for all meshes specified in config json file"""
    PUB_surface_data_list.add_time()
    PUB_surface_data_list.normalize()
    logging.info("Surface points created and normalized.")
    return PUB_surface_data_list


def save_clustered_data():
    data_processed_folder_path = PROCESSED_DATA_FOLDERPATH

    clustered_data = pipeline_clustered_data_prepare()

    # Save the clustered data
    with open(os.path.join(data_processed_folder_path, CLUSTERED_DATA_FILENAME), 'wb') as f:
        pickle.dump(clustered_data, f)


def save_surface_data(clustered_data):
    surface_data_list = pipeline_nn_data_prepare(clustered_data)

    # save the surface data list
    with open(SURFACE_DATA_LIST_FILEPATH, 'wb') as f:
        pickle.dump(surface_data_list, f)


# endregion
# endregion


# region Main

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

# Function to process and save clustered data if not already processed
def process_clustered_data():
    if not os.path.exists(CLUSTERED_DATA_FILEPATH):
        save_clustered_data()
        logging.info("Clustered data processed and saved.")
    else:
        logging.info("Clustered data already processed.")

# Function to process and save neural network data if not already processed
def process_nn_data():
    if not os.path.exists(SURFACE_DATA_LIST_FILEPATH):
        clustered_data = load_pickle_file(CLUSTERED_DATA_FILEPATH)
        if clustered_data is not None:
            save_surface_data(clustered_data)
            logging.info("Neural network data processed and saved.")
    else:
        logging.info("Neural network data already processed.")


def save_all_clusters_surface_points_image(original_points_dict, processed_points_dict, image_save_folder=IMAGE_SAVE_FOLDERPATH):
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


# Function to train the neural network for each cluster
def train_nn_for_all_clusters(surface_data_list, max_epochs=100, patience=5):
    # Identify unique clusters in the data
    unique_clusters = surface_data_list.get_unique_labels()

    for cluster in unique_clusters:
        # Filter data for the current cluster
        surface_data_cluster = surface_data_list.filter_by_label(cluster)

        # Define a specific filepath for the model weights for this cluster
        model_weights_filepath = MODEL_WEIGHTS_FILEPATH_TEMPLATE.format(cluster=cluster)

        logging.info(f"Training neural network for cluster {cluster}...")

        # Train the neural network on the current cluster's data
        train_neural_network(surface_data_cluster.list, max_epochs, patience, model_weights_filepath)

        logging.info(f"Model weights for cluster {cluster} saved to {model_weights_filepath}")

def save_combined_surface_points_image(original_points_all, processed_points_all, image_save_folder=IMAGE_SAVE_FOLDERPATH):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot for all original points
    ax.scatter(
        original_points_all[:, 0], original_points_all[:, 1],
        color='blue', label='Original Points', alpha=0.5, s=20
    )

    # Scatter plot for all processed points
    ax.scatter(
        processed_points_all[:, 0], processed_points_all[:, 1],
        color='red', label='Processed Points', alpha=0.5, s=20
    )

    # Labels, title, and legend
    ax.set_title('Surface Points for All Clusters')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()
    ax.grid(True)

    # Save the plot
    image_path = os.path.join(image_save_folder, 'combined_surface_points.png')
    plt.savefig(image_path)
    plt.close(fig)

    print(f"Saved combined surface points image at {image_path}")

def load_trained_model(model_weights_filepath, input_size, hidden_size, output_size):
    """
    Loads the trained neural network model weights from a specified file path.

    Args:
        model_weights_filepath (str): The path to the file containing the model weights.
        input_size (int): The number of input features for the model.
        hidden_size (int): The number of hidden units in the model.
        output_size (int): The number of output features for the model.

    Returns:
        model (nn.Module): The loaded neural network model.
    """
    # Create an instance of the model
    model = MLP(input_size, hidden_size, output_size)

    # Load the model weights
    model.load_state_dict(torch.load(model_weights_filepath))

    # Set the model to evaluation mode
    model.eval()

    return model

def process_and_save_combined_image_for_all_clusters(surface_data_list):
    original_points_all = []
    processed_points_all = []

    unique_clusters = surface_data_list.get_unique_labels()

    for cluster in unique_clusters:
        # Load the original surface points for the current cluster
        surface_data_cluster = surface_data_list.filter_by_label(cluster)
        original_points = np.array(surface_data_cluster.list)  # Convert to numpy array if necessary

        # Load the trained model for the current cluster
        model_weights_filepath = MODEL_WEIGHTS_FILEPATH_TEMPLATE.format(cluster=cluster)
        model = load_trained_model(model_weights_filepath)

        # Process the original points through the model
        with torch.no_grad():
            processed_points = model(torch.tensor(original_points, dtype=torch.float32)).numpy()

        # Accumulate all original and processed points
        original_points_all.append(original_points)
        processed_points_all.append(processed_points)

    # Convert lists to numpy arrays for plotting
    original_points_all = np.vstack(original_points_all) if original_points_all else np.empty((0, 2))
    processed_points_all = np.vstack(processed_points_all) if processed_points_all else np.empty((0, 2))

    # Save the combined image
    save_combined_surface_points_image(original_points_all, processed_points_all)

# Main function to orchestrate the processing and training for each cluster
def main():
    # Ensure processed data directory exists
    ensure_directory_exists(PROCESSED_DATA_FOLDERPATH)

    # Process clustered and neural network data
    process_clustered_data()
    process_nn_data()

    # Load the processed surface data list
    surface_data_list = load_pickle_file(SURFACE_DATA_LIST_FILEPATH)
    if surface_data_list is None:
        logging.error("Surface data list could not be loaded. Exiting.")
        return

    # Train a neural network for each cluster in the data
    train_nn_for_all_clusters(surface_data_list, max_epochs=100, patience=5)

    # Process and save combined image for all clusters after training
    process_and_save_combined_image_for_all_clusters(surface_data_list)

if __name__ == '__main__':
    main()
# endregion
