import logging
import os

import numpy as np
import torch
import trimesh
from matplotlib import pyplot as plt
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from torch.utils.data import DataLoader

from data_processing.class_clustering import ClusteredCenterPointsAllFrames
from data_processing.class_mapping import SurfacePointsFrameList, SurfacePointsFrame
from data_processing.loader import load_centers_data
from data_processing.mapping import categorize_points_with_labels
from nerual_network.class_evaluation import PairPointCenterPoint, PairPointCenterPointList, DecoderElement, DecoderPairList, EvaluationResult, EvaluationResultList
from src.nerual_network.class_model import NNDataset
from utils.constants import NN_DEVICE_STR, TrainConfig
from utils.helpers import load_pickle_file
from utils.nn_config_utils import init_training_config, _add_time_column, get_loaded_meshes_list


# Restrict access to underscore-prefixed functions
def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"{name} is a private function and cannot be imported.")
    raise AttributeError(f"Module has no attribute {name}")


# region PRIVATE FUNCTIONS
def _save_pointcloud_to_file(original_points_all, processed_points_all, original_filepath, processed_filepath):
    np.savetxt(original_filepath, original_points_all, delimiter=",")
    np.savetxt(processed_filepath, processed_points_all, delimiter=",")


def _visualize_all_clusters_for_each_time(surface_data_list: SurfacePointsFrameList, image_save_folder):
    """
    Visualizes the original and processed points in 3D in one image for each time slice.
    :param original_points_all:
    :param processed_points_all:
    :param image_save_folder:
    :return:
    """
    # Ensure the image save folder exists
    os.makedirs(image_save_folder, exist_ok=True)

    # Loop through each unique time value
    for i, surface_data_frame in enumerate(surface_data_list.list):
        if surface_data_frame.time.index != i:
            raise Exception("Not same time")

        cluster_labels = surface_data_frame.labels_list
        # transfrom surface_data_slice to array with points

        points_slice = np.array(surface_data_frame.points_list)
        # transform cluster_labels to array
        cluster_labels = np.array(cluster_labels)
        _visualize_clusters(points_slice, cluster_labels, image_save_folder, f'time_{i}_clusters_time.png')


def _visualize_combined_surface_points_for_each_time(original_points_all, processed_points_all,
                                                     image_save_folder, cluster_labels):
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
        original_points_slice = original_points_all[original_points_all[:, 3] == time]
        processed_points_slice = processed_points_all[processed_points_all[:, 3] == time]
        cluster_labels_slice = [label for label in cluster_labels if label[1] == time]

        _visualize_combined_surface_points_for_one_time(image_save_folder, original_points_slice,
                                                        processed_points_slice,
                                                        f'time_{i}_combined_surface_points_time.png', time)

        # visulize clusters


def _visualize_points_with_time(original_points_all, processed_points_all, image_save_folderpath):
    """
    Visualizes the original and processed points in 3D in one image with time as color.
    :param original_points_all:
    :param processed_points_all:
    :param image_save_folder:
    :return:
    """
    os.makedirs(image_save_folderpath, exist_ok=True)

    # Create a new figure for the 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extracting x, y, z coordinates and time from original points
    original_x = original_points_all[:, 0]
    original_y = original_points_all[:, 1]
    original_z = original_points_all[:, 2]
    original_time = original_points_all[:, 3]  # Assuming the time column is the 4th column

    # Extracting x, y, z coordinates from processed points
    processed_x = processed_points_all[:, 0]
    processed_y = processed_points_all[:, 1]
    processed_z = processed_points_all[:, 2]
    processed_time = processed_points_all[:, 3]  # Assuming the time column is the 4th column

    # Scatter plot for original points (using time for color)
    scatter_original = ax.scatter(original_x, original_y, original_z,
                                  c=original_time, cmap='viridis', label='Original Points', alpha=0.5, s=50)

    # Scatter plot for processed points (using time for color)
    scatter_processed = ax.scatter(processed_x, processed_y, processed_z,
                                   c=processed_time, cmap='plasma', label='Processed Points', alpha=0.5, marker='^',
                                   s=50)

    # Setting labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('3D Visualization of Original and Processed Points with Time as Color')

    # Adding a color bar
    cbar_original = plt.colorbar(scatter_original, ax=ax, pad=0.1)
    cbar_original.set_label('Time (Original Points)')

    cbar_processed = plt.colorbar(scatter_processed, ax=ax, pad=0.1)
    cbar_processed.set_label('Time (Processed Points)')

    ax.legend()

    # Save the plot
    image_path = os.path.join(image_save_folderpath, 'combined_surface_points_time_colored.png')
    plt.savefig(image_path)
    plt.close(fig)

    print(f"Saved combined surface points image at {image_path}")


def _visualize_original_and_processed_points(original_points_all, processed_points_all, image_save_folder):
    """
    Visualizes the original and processed points in 3D in two separate images with time as color.
    :param original_points_all:
    :param processed_points_all:
    :param image_save_folder:
    :return:
    """
    # Ensure the save folder exists
    os.makedirs(image_save_folder, exist_ok=True)

    # Extracting x, y, z coordinates and time from original points
    original_x = original_points_all[:, 0]
    original_y = original_points_all[:, 1]
    original_z = original_points_all[:, 2]
    original_time = original_points_all[:, 3]  # Assuming the time column is the 4th column

    # Create a new figure for the original points 3D plot
    fig_original = plt.figure(figsize=(12, 8))
    ax_original = fig_original.add_subplot(111, projection='3d')

    # Scatter plot for original points (using time for color)
    scatter_original = ax_original.scatter(original_x, original_y, original_z,
                                           c=original_time, cmap='viridis', label='Original Points', alpha=1, s=70)

    # Setting labels
    ax_original.set_xlabel('X Label')
    ax_original.set_ylabel('Y Label')
    ax_original.set_zlabel('Z Label')
    ax_original.set_title('3D Visualization of Original Points with Time as Color')

    # Adding a color bar for original points
    cbar_original = plt.colorbar(scatter_original, ax=ax_original, pad=0.1)
    cbar_original.set_label('Time (Original Points)')

    # Save the original points plot
    original_image_path = os.path.join(image_save_folder, 'original_surface_points_time_colored.png')
    plt.savefig(original_image_path)
    plt.close(fig_original)

    print(f"Saved original surface points image at {original_image_path}")

    # Extracting x, y, z coordinates and time from processed points
    processed_x = processed_points_all[:, 0]
    processed_y = processed_points_all[:, 1]
    processed_z = processed_points_all[:, 2]
    processed_time = processed_points_all[:, 3]  # Assuming the time column is the 4th column

    # Create a new figure for the processed points 3D plot
    fig_processed = plt.figure(figsize=(12, 8))
    ax_processed = fig_processed.add_subplot(111, projection='3d')

    # Scatter plot for processed points (using time for color)
    scatter_processed = ax_processed.scatter(processed_x, processed_y, processed_z,
                                             c=processed_time, cmap='plasma', label='Processed Points', alpha=1, s=70)

    # Setting labels
    ax_processed.set_xlabel('X Label')
    ax_processed.set_ylabel('Y Label')
    ax_processed.set_zlabel('Z Label')
    ax_processed.set_title('3D Visualization of Processed Points with Time as Color')

    # Adding a color bar for processed points
    cbar_processed = plt.colorbar(scatter_processed, ax=ax_processed, pad=0.1)
    cbar_processed.set_label('Time (Processed Points)')

    # Save the processed points plot
    processed_image_path = os.path.join(image_save_folder, 'processed_surface_points_time_colored.png')
    plt.savefig(processed_image_path)
    plt.close(fig_processed)

    print(f"Saved processed surface points image at {processed_image_path}")


def _prepare_export_data(surface_data_list, train_config: TrainConfig):
    model_weights_template = train_config.file_path_config.model_weights_folderpath_template
    batch_size = train_config.nn_config.batch_size

    original_points_all = []
    processed_points_all = []
    cluster_labels = []
    unique_clusters = surface_data_list.get_unique_clusters()
    for i, cluster in enumerate(unique_clusters):
        if i >= train_config.num_clusters:
            break

        # Load the original surface points for the current cluster
        surface_data_cluster = surface_data_list.filter_by_label(cluster)

        # Create a SurfaceDataset instance with the filtered surface data
        original_points_dataset = NNDataset(surface_data_cluster)

        # Load the trained model for the current cluster
        model_weights_filepath = model_weights_template.format(cluster=cluster)
        model = _load_trained_model(model_weights_filepath, train_config)
        device = torch.device(NN_DEVICE_STR)

        model.to(device)

        # Prepare a DataLoader for original points
        original_points_loader = DataLoader(original_points_dataset, batch_size=batch_size, shuffle=False)

        # Process the original points through the model
        processed_points = []
        with torch.no_grad():
            for batch in original_points_loader:
                inputs = batch[0]  # Get only the points with time
                inputs = inputs.float().to(device)
                encoder_inputs = NNDataset.get_encoder_input(inputs)

                outputs = model(encoder_inputs)  # Forward pass through the model
                processed_points.append(outputs)

        # Convert processed points to a single numpy array
        processed_points = torch.cat(processed_points).cpu().numpy()

        # Accumulate all original and processed points
        original_points_all.append(original_points_dataset.data)  # You can store the numpy array directly
        processed_points_all.append(processed_points)
        # cluster labels store
        cluster_labels.extend([(cluster, point[-1]) for point in original_points_dataset.data])
        """ saved cluster list with id same as points and its structure is (cluster_id, time) """
    # Convert lists to numpy arrays for plotting
    original_points_all = np.vstack(original_points_all) if original_points_all else np.empty((0, 4))
    processed_points_all = np.vstack(processed_points_all) if processed_points_all else np.empty((0, 4))

    processed_points_all = np.hstack((processed_points_all, original_points_all[:, 3].reshape(-1, 1)))

    return original_points_all, processed_points_all, cluster_labels


def _load_trained_model(model_weights_filepath: str, train_config: TrainConfig):
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

    # Load the checkpoint
    model, optimizer, loss_function = init_training_config(train_config)

    checkpoint = torch.load(model_weights_filepath)  # Load the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])  # Load the model state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load the optimizer state
    epoch = checkpoint['epoch']  # Get the epoch number
    val_loss = checkpoint['val_loss']  # Get the validation loss

    # Set the model to evaluation mode
    model.eval()

    return model


def _visualize_clusters(points, labels, image_save_folder, image_name):
    """
    Visualizes 3D points with cluster labels in distinct colors.

    Parameters:
    - points: np.ndarray of shape (N, 3), where N is the number of points.
    - labels: np.ndarray of shape (N,), cluster labels for each point.
    - image_save_folder: str, folder path to save the image.
    - image_name: str, name of the image file.
    """
    # Check input shapes
    if points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")
    if points.shape[0] != labels.shape[0]:
        raise ValueError(f"Number of points ({points.shape[0]}) must match number of labels ({labels.shape[0]})")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Generate distinct colors for each cluster
    unique_labels = np.unique(labels)
    colormap = plt.get_cmap('jet')
    colors = colormap(np.linspace(0, 1, len(unique_labels)))

    # Map labels to colors
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    point_colors = np.array([label_to_color[label] for label in labels])

    # Create scatter plot
    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=point_colors, s=50
    )

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the initial view angle
    ax.view_init(elev=-70, azim=90)  # Adjust these values as needed

    # Set plot title
    plt.title("3D Clusters with Mesh")

    # Save the image
    image_path = os.path.join(image_save_folder, image_name)
    plt.savefig(image_path)
    plt.close(fig)


def run_model_decoder_all_times_with_selected_encoder_time(surface_data_list: SurfacePointsFrameList,
                                                           time: float, train_config: TrainConfig):
    model_weights_template = train_config.file_path_config.model_weights_folderpath_template
    batch_size = train_config.nn_config.batch_size

    original_points_all = []
    processed_points_all = []
    cluster_labels = []

    unique_clusters = surface_data_list.get_unique_clusters()

    # select original points where time is 0
    original_points_frame = surface_data_list.get_element_by_time_index(time)

    for i, cluster in enumerate(unique_clusters):
        if i >= train_config.num_clusters:
            break
        # Load the original surface points for the current cluster
        surface_data_cluster_timeframe = original_points_frame.filter_by_label(cluster)

        # Create a SurfaceDataset instance with the filtered surface data
        original_points_dataset = NNDataset(SurfacePointsFrameList([surface_data_cluster_timeframe]))

        # Prepare a DataLoader for original points
        original_points_loader = DataLoader(original_points_dataset, batch_size=batch_size, shuffle=False)

        # Load the trained model for the current cluster
        model_weights_filepath = model_weights_template.format(cluster=cluster)
        model = _load_trained_model(model_weights_filepath, train_config)
        device = torch.device(NN_DEVICE_STR)

        model.to(device)

        encoded_features = []

        # Step 1: Encode the original data
        with torch.no_grad():  # No need to calculate gradients during evaluation
            for inputs in original_points_loader:
                inputs = inputs[0].float().to(device)
                encoder_input_data = NNDataset.get_encoder_input(inputs)
                encoded_features_element = model.encoder(encoder_input_data)
                encoded_features.append(encoded_features_element)

        # Process the original points through the model
        encoded_features = torch.cat(encoded_features).to(device)
        # tenosor column vector with the same value which is 0
        for i, surface_points_frame in enumerate(surface_data_list.list):
            if surface_points_frame.time.index != i:
                raise Exception("Not same time")

            time_value = surface_points_frame.time.value
            encoded_with_time = _add_time_column(encoded_features, time_value)

            # Pass through the decoder
            decoded_output = model.decoder(encoded_with_time)
            # Add the time_value as a column to decoded_output

            decoded_output = decoded_output.cpu().detach().numpy()
            time_column = np.full((decoded_output.shape[0], 1), time_value)
            decoded_output_with_time = np.hstack((decoded_output, time_column))

            # Append the modified decoded output
            processed_points_all.append(decoded_output_with_time)

        # Convert processed points to a single numpy array

        # Accumulate all original and processed points
        original_points_all.append(np.array(original_points_dataset.data))  # You can store the numpy array directly
        # cluster labels store
        cluster_labels.extend([(cluster, point[-1]) for point in original_points_dataset.data])
        """ saved cluster list with id same as points and its structure is (cluster_id, time) """

    original_points_all = np.vstack(original_points_all) if original_points_all else np.empty((0, 4))

    processed_points_all = np.vstack(processed_points_all) if processed_points_all else np.empty((0, 4))

    return original_points_all, processed_points_all, cluster_labels


def _visualize_uv_points_in_3d(surface_data_list: SurfacePointsFrameList, images_save_folderpath: str,
                               time: float, train_config: TrainConfig):
    def visualize_for_eachtime(original_points_all, processed_points_all):

        # Funkce pro normalizaci pro jednotlivé osy
        def normalize_slices(data):

            normalized = np.zeros_like(data)

            min_val = np.min(data, axis=0)
            max_val = np.max(data, axis=0)

            for index, value in enumerate(data):
                normalized_value = (value - min_val) / (max_val - min_val)

                normalized[index] = normalized_value

            return normalized

        # create folder if not created
        os.makedirs(images_save_folderpath, exist_ok=True)

        # Normalizace podle osy X, Y, Z
        normalized_x = normalize_slices(original_points_all[:, 0])
        normalized_y = normalize_slices(original_points_all[:, 1])
        normalized_z = normalize_slices(original_points_all[:, 2])

        # Spojení barev
        rgb_colors = np.stack([normalized_x, normalized_y, normalized_z], axis=1)

        # Extract unique time values assuming the last column contains time values
        unique_times = np.unique(processed_points_all[:, 3])

        # Loop through each unique time value
        for i, time in enumerate(unique_times):
            processed_points_slice = processed_points_all[processed_points_all[:, 3] == time]

            visualize_for_one_time(images_save_folderpath, rgb_colors,
                                   processed_points_slice,
                                   f'time_{i}_uv_color_representation.png', time)

    def visualize_for_one_time(images_save_folderpath, rgb_colors, processed_points_slice, image_name, time):
        # visualize point cloud for original point and make the color of the point based on the processed_points
        # where the x,y,z is r,g,b values

        # Create a new figure for each time slice
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Přidání bodů do 3D grafu
        for i, point in enumerate(processed_points_slice):
            ax.scatter(point[0], point[1], point[2], color=rgb_colors[i])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title(f'3D Visualization of Original and Processed Points from Time {time}')
        ax.legend()
        # Save the plot for the current time slice
        image_path = os.path.join(images_save_folderpath, image_name)
        plt.savefig(image_path)
        plt.close(fig)
        print(f"Saved combined surface points image at {image_path}")

    original_points_all, processed_points_all, cluster_labels = run_model_decoder_all_times_with_selected_encoder_time(
        surface_data_list=surface_data_list, time=time, train_config=train_config)
    visualize_for_eachtime(original_points_all, processed_points_all)


def _visualize_combined_surface_points_for_one_time(image_save_folder, original_points_slice, processed_points_slice,
                                                    image_name, time):
    # Create a new figure for each time slice
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Filter original points for this time slice

    # Plot original points for this time slice
    ax.scatter(original_points_slice[:, 0], original_points_slice[:, 1], original_points_slice[:, 2],
               color='blue', label='Original Points', alpha=0.5)

    # Plot processed points for this time slice
    ax.scatter(processed_points_slice[:, 0], processed_points_slice[:, 1], processed_points_slice[:, 2],
               color='red', label='Processed Points', alpha=0.5)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title(f'3D Visualization of Original and Processed Points from Time {time}')
    ax.legend()
    # Save the plot for the current time slice
    image_path = os.path.join(image_save_folder, image_name)
    plt.savefig(image_path)
    plt.close(fig)
    print(f"Saved combined surface points image at {image_path}")


# endregion


def compute_variance(evaluation_results_list):
    variance_list = []
    for evaluation_result in evaluation_results_list.list:
        variance_list_list = []
        for decoder_element in evaluation_result.decoder_pair_list.list:
            unique_ids = decoder_element.pair_processed_center.get_unique_ids()
            for id in unique_ids:
                pair_list = decoder_element.pair_processed_center.get_decoder_element_by_id(id)
                distances = [pair.distance for pair in pair_list]
                # compute statistical dispersion
                variance = np.var(distances)
                variance_list_list.append({"variance": variance, "id": id})
        variance_list.append({"time": evaluation_result.encoder_time, "variance_list": variance_list_list})

    return variance_list


def _compute_centers_metrics(surface_data_list, train_config, num_points, nn_lr):
    """
    Computes metrics which:
    1. for every sequenco of points
    1.a selects num_points from the sequence
    2. finds closest center point loaded from files
    3. puts original points through the encoder
    4. puts ouput throug the decoder in every time
    5. computes the distances between the ceneter point again
    6. outputs statistical dispersion for every point
    7. do this for all sequences of points
    :param surface_data_list:
    :param train_config:
    :param num_points:
    :return:
    """

    def load_centers_data_from_files(folder_path, time_steps):
        center_points, num_points_in_file = load_centers_data(folder_path, time_steps)
        # make it a Surface data list where each SurfacePoint is one num_points_in_file slice of center_points

        # itarate over center_points and create SurfacePoint with num_points_in_file points
        center_points_list_current = SurfacePointsFrameList([])
        for i in range(0, center_points.shape[0]):
            center_points_list_current.append(SurfacePointsFrame(center_points[i]))

        center_points_list_current.assign_time_to_all_elements()
        center_points_list_current.normalize_all_elements()
        return center_points_list_current

    def compute_distance(first_point, second_point):
        return np.linalg.norm(first_point - second_point)

    def find_closest_centers(center_points_list: SurfacePointsFrame, points_list: SurfacePointsFrame):
        pair_original_center = []
        index = 0
        for index_point, point in enumerate(points_list.points_list):
            closest_center_point = None
            min_distance = float('inf')
            for center_point in center_points_list.points_list:
                distance = compute_distance(point, center_point)
                if distance < min_distance:
                    closest_center_point = center_point
                    min_distance = distance

            if points_list.time != center_points_list.time:
                raise Exception("Not same time")
            time = points_list.time

            label = points_list.labels_list[index_point]

            pair_original_center.append(
                PairPointCenterPoint(point, closest_center_point, min_distance, index + 1, time, label))
        return PairPointCenterPointList(pair_original_center)

    def run_through_model(center_points_list: SurfacePointsFrameList, surface_data_list: SurfacePointsFrameList,
                          model_weights_template: str, batch_size: int, num_points: int, nn_lr):
        # todo check if should be normalized
        if len(center_points_list.list) != len(surface_data_list.list):
            raise Exception("Not same number of center points and surface data")
        pair_list_len = len(surface_data_list.list)

        unique_clusters = surface_data_list.get_unique_clusters()
        unique_times = surface_data_list.get_unique_times()

        evaluation_result_list = EvaluationResultList([])

        for i in range(0, pair_list_len):

            # select original points where time is 0
            surface_data_timeframe = surface_data_list.list[i]
            if surface_data_timeframe.time.index != i:
                raise Exception("Not same time")

            encoder_time = surface_data_timeframe.time.value

            center_points_timeframe = center_points_list.list[i]
            if center_points_timeframe.time.index != i:
                raise Exception("Not same time")

            surface_data_timeframe = surface_data_list.select_random_points(num_points)

            pair_original_center = find_closest_centers(center_points_timeframe, surface_data_timeframe)

            for cluster in unique_clusters:

                # Load the original surface points for the current cluster
                pair_original_center_cluster = pair_original_center.filter_by_point_clusterlabel(cluster)
                surface_data_cluster = pair_original_center_cluster.get_points_list()
                surface_data_cluster = convert_to_surfacepointsframelist(surface_data_cluster)

                # Create a SurfaceDataset instance with the filtered surface data
                original_points_dataset = NNDataset(surface_data_cluster)
                # Prepare a DataLoader for original points
                original_points_loader = DataLoader(original_points_dataset, batch_size=batch_size, shuffle=False)

                # Load the trained model for the current cluster
                model_weights_filepath = model_weights_template.format(cluster=cluster)
                model = _load_trained_model(model_weights_filepath, train_config)
                device = torch.device(NN_DEVICE_STR)

                model.to(device)

                # Step 1: Encode the original data
                with torch.no_grad():  # No need to calculate gradients during evaluation
                    encoded_features = model.encoder(original_points_loader)

                decoder_pair_list = DecoderPairList([])

                # iterate to decoder over all times
                for decoder_time in unique_times:
                    # Create a tensor of the same shape as the time feature in the input
                    time_tensor = torch.full((encoded_features.size(0), 1), decoder_time, dtype=torch.float32)
                    # Concatenate the encoded features with the time tensor
                    encoded_with_time = torch.cat((encoded_features, time_tensor), dim=1)
                    # Pass through the decoder
                    decoded_output = model.decoder(encoded_with_time)

                    decoder_processed_points = decoded_output
                    decoder_processed_points_timeframe = SurfacePointsFrame([], None, decoder_time)
                    # convert to
                    for point in decoder_processed_points:
                        decoder_processed_points_timeframe.points_list.append(point)

                    decoder_center_points_timeframe = center_points_timeframe.get_element_by_time_index(decoder_time)

                    decoder_pair_processed_center = find_closest_centers(decoder_center_points_timeframe,
                                                                         decoder_processed_points_timeframe)

                    decoder_pair_list.append(DecoderElement(decoder_pair_processed_center, decoder_time))

                evaluation_result_list.append(
                    EvaluationResult(pair_original_center_cluster, encoder_time, decoder_pair_list))

        return evaluation_result_list

        # load center points

    center_points_list = load_centers_data_from_files(train_config.file_path_config.raw_data_folderpath,
                                                      train_config.time_steps)

    # run through the model
    evaluation_results_list = run_through_model(center_points_list,
                                                surface_data_list,
                                                train_config.file_path_config.model_weights_folderpath_template,
                                                train_config.nn_config.batch_size, num_points, nn_lr)

    variance_list = compute_variance(evaluation_results_list)

    return variance_list, evaluation_results_list


def _compute_mesh_shape_metrics(surface_data_list: SurfacePointsFrameList, train_config: TrainConfig,
                                clustered_data: ClusteredCenterPointsAllFrames, nn_lr):
    """
    Computes metrics which:
    1. loads mesh in one specified time
    2. runs these mesh points through the encoder in specified time
    3. runs output through the decoder in all times
    4. compares the output of an time with a mesh in the same time
    :param surface_data_list:
    :param train_config:
    :return:
    """

    def compute_laplacian_eigenvalues(mesh, k=10):
        """
        Compute the first k eigenvalues of the Laplace-Beltrami operator for a mesh.

        Args:
            mesh (trimesh.Trimesh): Trimesh mesh object.
            k (int): Number of smallest eigenvalues to compute.

        Returns:
            np.ndarray: Sorted eigenvalues.
        """
        # Get the adjacency matrix of the mesh
        adjacency_matrix = mesh.vertex_adjacency_matrix

        # Compute the Laplacian matrix
        laplacian = csgraph.laplacian(adjacency_matrix, normed=True)

        # Compute the smallest k eigenvalues of the Laplacian matrix
        eigenvalues, _ = eigsh(laplacian, k=k, which='SM')
        return np.sort(eigenvalues)

    def compute_similarity(original_mesh, processed_mesh):
        # Compute eigenvalues
        eigenvalues1 = compute_laplacian_eigenvalues(original_mesh)
        eigenvalues2 = compute_laplacian_eigenvalues(processed_mesh)
        # Compare eigenvalues (e.g., L2 distance)
        return np.linalg.norm(eigenvalues1 - eigenvalues2)

    def label_points_by_clustered_data(surface_data_list: SurfacePointsFrameList,
                                       clustered_data: ClusteredCenterPointsAllFrames):
        final_surface_data_list = SurfacePointsFrameList([])

        for i, surface_data_frame in enumerate(surface_data_list.list):
            mesh_points = surface_data_frame.points_list
            centers_points_frame = clustered_data.points_allframes[i]
            centers_labels_frame = clustered_data.labels_frame
            # check indexes with filepath names

            surface_labels = categorize_points_with_labels(centers_labels_frame, centers_points_frame, mesh_points)
            # append both values to list with names in the list
            final_surface_data_list.append(SurfacePointsFrame(mesh_points, surface_labels, None))
        return final_surface_data_list

    mesh_folder_path = train_config.file_path_config.raw_data_folderpath
    loaded_meshes_list = get_loaded_meshes_list(mesh_folder_path)

    all_vertices = []
    for mesh in loaded_meshes_list:
        # Access vertices as a NumPy array
        vertices = mesh.vertices
        all_vertices.append(vertices)

    mesh_points_allframes = convert_to_surfacepointsframelist(all_vertices)
    mesh_points_allframes = label_points_by_clustered_data(mesh_points_allframes, clustered_data)

    mesh_points_allframes.assign_time_to_all_elements()
    mesh_points_allframes.normalize_all_elements()

    time = 0
    original_points_all, processed_points_all, cluster_labels = run_model_decoder_all_times_with_selected_encoder_time(
        surface_data_list=mesh_points_allframes, time=time, train_config=train_config)

    if len(loaded_meshes_list) != len(surface_data_list.list) and len(mesh_points_allframes.list) != len(
            surface_data_list.list):
        raise Exception("Not same number of loaded meshes and surface data")

    similarity_list = []
    # compare the output of the decoder with the mesh in the same time
    for i, surface_data_frame in enumerate(surface_data_list.list):
        if surface_data_frame.time.index != i:
            raise Exception("Not same time")
        time = surface_data_frame.time.value

        loaded_mesh_timeframe = loaded_meshes_list[i]

        mesh_points_timeframe = mesh_points_allframes.list[i]
        if mesh_points_timeframe.time.index != i:
            raise Exception("Not same time")
        mesh_points_timeframe_points = mesh_points_timeframe.points_list

        processed_points_timeframe = processed_points_all[processed_points_all[:, 3] == time]

        # convert to meshes
        original_mesh = trimesh.Trimesh(vertices=mesh_points_timeframe_points, faces=loaded_mesh_timeframe.faces)
        processed_mesh = trimesh.Trimesh(vertices=processed_points_timeframe, faces=loaded_mesh_timeframe.faces)

        similarity = compute_similarity(original_mesh, processed_mesh)
        similarity_list.append({"time": time, "similarity": similarity})

    return similarity_list


def convert_to_surfacepointsframelist(all_vertices):
    # transform mesh vertices to  and normalize them and add time
    mesh_points_list = SurfacePointsFrameList([])
    for vertices in all_vertices:
        mesh_points = SurfacePointsFrame(vertices)
        mesh_points_list.append(mesh_points)
    return mesh_points_list


def evaluate(train_config: TrainConfig):
    clustered_data = load_pickle_file(train_config.file_path_config.clustered_data_filepath)
    if clustered_data is None:
        logging.error("Clustered data could not be loaded. Exiting.")
        return

    surface_data_list = load_pickle_file(train_config.file_path_config.surface_data_filepath)

    if surface_data_list is None or surface_data_list.list is None:
        logging.error("Surface data list could not be loaded. Exiting.")
        return
    # surface_data_list = convert_to_surface_data_list(surface_data_list)

    images_save_folderpath = train_config.file_path_config.images_save_folderpath
    model_weights_template = train_config.file_path_config.model_weights_folderpath_template
    point_cloud_original_filepath = train_config.file_path_config.point_cloud_original_filepath
    point_cloud_processed_filepath = train_config.file_path_config.point_cloud_processed_filepath
    batch_size = train_config.nn_config.batch_size
    nn_lr = train_config.nn_config.nn_lr

    # region Metrics
    # varience_list, evaluation_list = _compute_centers_metrics(surface_data_list, train_config,
    #                                                           num_points=EVAL_NUM_SURFACE_POINTS)
    # # save varience_list string represenatation and evaluation_list reprezentetion to file
    # with open(train_config.file_path_config.center_metric_variances_filepath, "w") as file:
    #     file.write(str(varience_list))
    # with open(train_config.file_path_config.center_metric_eval_filepath, "w") as file:
    #     file.write(str(evaluation_list))

    # mesh_shape_metrics = _compute_mesh_shape_metrics(surface_data_list, train_config, clustered_data)
    # # save mesh_shape_metrics to file
    # with open(train_config.file_path_config.mesh_shape_metrics_filepath, "w") as file:
    #     file.write(str(mesh_shape_metrics))
    # endregion

    # region Visulize


    # original_points_all, processed_points_all, cluster_labels = _prepare_export_data(
    #     surface_data_list=surface_data_list, train_config=train_config)
    #
    # _visualize_combined_surface_points_for_each_time(original_points_all, processed_points_all,
    #                                                  os.path.join(images_save_folderpath,
    #                                                               "time_combined_original_processed"), cluster_labels)
    # _visualize_all_clusters_for_each_time(surface_data_list, os.path.join(images_save_folderpath, "time_clusters"))
    #
    # _visualize_points_with_time(original_points_all, processed_points_all, images_save_folderpath)
    # # Save the combined image
    # _visualize_original_and_processed_points(original_points_all, processed_points_all, images_save_folderpath)
    #
    # _save_pointcloud_to_file(original_points_all, processed_points_all, point_cloud_original_filepath,
    #                          point_cloud_processed_filepath)
    #
    # _visualize_uv_points_in_3d(surface_data_list=surface_data_list,
    #                            images_save_folderpath=os.path.join(images_save_folderpath, "time_uv_points_0"), time=0,
    #                            train_config=train_config)

    _visualize_uv_points_in_3d(surface_data_list=surface_data_list,
                               images_save_folderpath=os.path.join(images_save_folderpath, "time_uv_points_59"),
                               time=59, train_config=train_config)

    # endregion
