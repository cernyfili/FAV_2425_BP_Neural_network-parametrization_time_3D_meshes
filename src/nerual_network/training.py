import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

from data_processing.class_mapping import SurfacePointsFrameList
from nerual_network.data_structures import LossFunctionInfo
from src.nerual_network.class_model import NNDataset
from src.utils.helpers import load_pickle_file
from utils.constants import NN_DEVICE_STR, TrainConfig
from utils.nn_config_utils import init_training_config


# Restrict access to underscore-prefixed functions
def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"{name} is a private function and cannot be imported.")
    raise AttributeError(f"Module has no attribute {name}")


# region PRIVATE FUNCTIONS
#
# # Function to split data and create data loaders
# def _create_data_loaders(surface_data_list: SurfacePointsFrameList, batch_size: int):
#     # Create an instance of SurfaceDataset using the provided surface_data_list
#     dataset = NNDataset(surface_data_list)
#
#     # Split indices for training and validation
#     train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
#
#     # Create subsets for training and validation
#     train_dataset = Subset(dataset, train_indices)
#     val_dataset = Subset(dataset, val_indices)
#
#     # Create data loaders for training and validation
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#
#     return train_loader, val_loader

# Function to split data and create data loaders
def _create_data_loaders(surface_data_list: SurfacePointsFrameList, batch_size: int):
    # Create an instance of SurfaceDataset using the provided surface_data_list
    dataset = NNDataset(surface_data_list)

    # Split indices for training and validation
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

    # Create subsets for training and validation
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    #Sort each subset by the time index column
    train_dataset.indices = sorted(train_dataset.indices)
    val_dataset.indices = sorted(val_dataset.indices)

    # Create data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


#
# def _create_data_loaders(surface_data_list: SurfacePointsFrameList, batch_size: int):
#     # Create an instance of SurfaceDataset using the provided surface_data_list
#     dataset = NNDataset(surface_data_list)
#
#     # Split indices for training and validation
#     split_idx = int(0.8 * len(dataset))
#     train_indices = list(range(split_idx))  # First 80%
#     val_indices = list(range(split_idx, len(dataset)))  # Last 20%
#
#     # Create subsets for training and validation
#     train_dataset = Subset(dataset, train_indices)
#     val_dataset = Subset(dataset, val_indices)
#
#     # Sort each subset by the time index column
#     train_dataset.indices = sorted(train_dataset.indices)
#     val_dataset.indices = sorted(val_dataset.indices)
#
#     # Create data loaders for training and validation
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#
#     return train_loader, val_loader


# Function to evaluate the model on the validation set
def _evaluate(model, val_loader, loss_function, device, loss_function_info : LossFunctionInfo):
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    # with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.float().to(device), targets.float().to(device)

        loss = loss_function(inputs, targets, model, loss_function_info)
        val_loss += loss.item()
    return val_loss / len(val_loader)  # Return average validation loss


# Function to save the model checkpoint
def _save_checkpoint(model, optimizer, epoch, val_loss):
    logging.info(f"Model saved with validation loss {val_loss:.4f} at epoch {epoch}")
    return {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
# endregion

# Function to train the neural network for each cluster
def _train_nn_for_all_clusters(surface_data_list: SurfacePointsFrameList, train_config : TrainConfig):
    max_epochs = train_config.nn_config.max_epochs
    model_weights_template = train_config.file_path_config.model_weights_folderpath_template

    if max_epochs == 0:
        return

    logging.info("Starting Training Neural network")
    # Identify unique clusters in the data
    unique_clusters = surface_data_list.get_unique_clusters()
    meshes_list = surface_data_list.get_normalized_meshes_list()

    loss_function_info = LossFunctionInfo()
    loss_function_info.meshes_list = meshes_list
    loss_function_info.device = torch.device(NN_DEVICE_STR)
    loss_function_info.time_list = surface_data_list.get_time_list()
    loss_function_info.data = surface_data_list

    all_frames_closest_list = surface_data_list.create_all_frames_all_points_closest_centers_indices()
    loss_function_info.closest_centers_indicies_all_frames = np.array(all_frames_closest_list, dtype=int)

    for cluster in unique_clusters:
        # Filter data for the current cluster
        surface_data_cluster = surface_data_list.filter_by_label(cluster)

        # Define a specific filepath for the model weights for this cluster
        model_weights_filepath = model_weights_template.format(cluster=cluster)

        logging.info(f"--------------------Training neural network for cluster {cluster}...")

        # Train the neural network on the current cluster's data
        _train_neural_network(data_cluster=surface_data_cluster, model_save_path=model_weights_filepath,
                              loss_function_info=loss_function_info, train_config=train_config)

        logging.info(f"Model weights for cluster {cluster} saved to {model_weights_filepath}")

# Main training function with early stopping and scheduler
def _train_neural_network(data_cluster: SurfacePointsFrameList, model_save_path, loss_function_info : LossFunctionInfo, train_config: TrainConfig):
    device = torch.device(NN_DEVICE_STR)
    logging.info(f"Using device: {device}")

    batch_size = train_config.nn_config.batch_size
    num_epochs = train_config.nn_config.max_epochs
    patience = train_config.nn_config.patience

    # if not data.is_normalized:
    #     logging.warning("Data is not normalized. Neural network training may not be effective.")

    train_loader, val_loader = _create_data_loaders(data_cluster, batch_size)

    model, optimizer, loss_function = init_training_config(train_config)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_loss = float('inf')
    best_train_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    best_checkpoint = None

    loss_function_info.data_cluster = data_cluster

    for epoch in range(1, num_epochs + 1):
        # Train and evaluate for one epoch

        train_loss = _train_one_epoch(model, train_loader, loss_function, optimizer, device, loss_function_info)
        val_loss = _evaluate(model, val_loader, loss_function, device, loss_function_info)

        # Learning rate scheduler step
        # scheduler.step(val_loss)

        logging.info(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.10f}, Val Loss: {val_loss:.10f}")

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_checkpoint = _save_checkpoint(model, optimizer, epoch, best_val_loss)
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement for {epochs_no_improve} epochs")

        if epochs_no_improve >= patience:
            logging.info(
                f"Early stopping after {epoch} epochs (Best epoch: {best_epoch} with train loss: {best_train_loss:.10f}, val loss: {best_val_loss:.10f})")
            break

    torch.save(best_checkpoint, model_save_path)
    logging.info(f"Training completed. Best model saved to {model_save_path}")

# Function to perform one training epoch
def _train_one_epoch(model, train_loader, loss_function, optimizer, device, loss_function_info : LossFunctionInfo):
    model.to(device)
    model.train()  # Set model to training mode
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.float().to(device), targets.float().to(device)

        # Forward pass
        loss = loss_function(inputs, targets, model, loss_function_info)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)  # Return average loss for the epoch

def train_nn(train_config: TrainConfig):
    logging.info("------------------TRAINING STARTED------------------")
    logging.info(f"Number of clusters: {train_config.num_clusters}")
    logging.info(f"Number of surface points: {train_config.num_surface_points}")
    logging.info(f"Maximum number of epochs: {train_config.nn_config.max_epochs}")
    logging.info(f"Patience for early stopping: {train_config.nn_config.patience}")
    logging.info(f"Batch size: {train_config.nn_config.batch_size}")
    logging.info(f"Raw data folder: {train_config.file_path_config.raw_data_folderpath}")

    model, optimizer, loss_function = init_training_config(train_config)
    # print model, optimizer, loss_function class or function names
    logging.info(f"Model: {model}")
    logging.info(f"Optimizer: {optimizer}")
    logging.info(f"Loss function: {loss_function}")

    surface_data_list = load_pickle_file(train_config.file_path_config.surface_data_filepath)
    if surface_data_list is None or surface_data_list.public_list is None or not isinstance(surface_data_list,
                                                                                     SurfacePointsFrameList):
        logging.error("Surface data list could not be loaded. Exiting.")
        return

    _train_nn_for_all_clusters(surface_data_list=surface_data_list, train_config=train_config)
    logging.info("------------------TRAINING ENDED------------------")
