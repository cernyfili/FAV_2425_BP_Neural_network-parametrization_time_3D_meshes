import logging

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

from src.data_processing.clustering import process_clustered_data
from src.data_processing.mapping import SurfaceDataList, process_surface_data
from src.nerual_network.model import NNDataset
from src.utils.constants import nn_optimizer, nn_model, TrainConfig
from src.utils.helpers import load_pickle_file


# Restrict access to underscore-prefixed functions
def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"{name} is a private function and cannot be imported.")
    raise AttributeError(f"Module has no attribute {name}")


# Function to perform one training epoch
def _train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.to(device)
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


# Main training function with early stopping and scheduler
def _train_neural_network(data, num_epochs, patience, model_save_path, batch_size):
    device = get_device()
    logging.info(f"Using device: {device}")

    train_loader, val_loader = _create_data_loaders(data, batch_size)
    model = nn_model
    criterion = nn.MSELoss()
    optimizer = nn_optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    best_checkpoint = None

    for epoch in range(1, num_epochs + 1):
        # Train and evaluate for one epoch
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _evaluate(model, val_loader, criterion, device)

        # Learning rate scheduler step
        scheduler.step(val_loss)

        logging.info(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.10f}, Val Loss: {val_loss:.10f}")

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_checkpoint = _save_checkpoint(model, optimizer, epoch, best_val_loss)
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement for {epochs_no_improve} epochs")

        if epochs_no_improve >= patience:

            logging.info(
                f"Early stopping after {epoch} epochs (Best epoch: {best_epoch} with val loss {best_val_loss:.4f})")
            break

    torch.save(best_checkpoint, model_save_path)
    logging.info(f"Training completed. Best model saved to {model_save_path}")


# Function to train the neural network for each cluster
def _train_nn_for_all_clusters(surface_data_list: SurfaceDataList, max_epochs, patience, batch_size,
                               model_weights_template):
    logging.info("Starting Training Neural network")
    # Identify unique clusters in the data
    unique_clusters = surface_data_list.get_unique_clusters()

    for cluster in unique_clusters:
        # Filter data for the current cluster
        surface_data_cluster = surface_data_list.filter_by_label(cluster)

        # Define a specific filepath for the model weights for this cluster
        model_weights_filepath = model_weights_template.format(cluster=cluster)

        logging.info(f"--------------------Training neural network for cluster {cluster}...")

        # Train the neural network on the current cluster's data
        _train_neural_network(surface_data_cluster.list, max_epochs, patience, model_weights_filepath, batch_size)

        logging.info(f"Model weights for cluster {cluster} saved to {model_weights_filepath}")


# Function to get the appropriate device
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # return torch.device('cpu')


# Function to split data and create data loaders
def _create_data_loaders(surface_data_list, batch_size):
    # Create an instance of SurfaceDataset using the provided surface_data_list
    dataset = NNDataset(surface_data_list)

    # Split indices for training and validation
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

    # Create subsets for training and validation
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Create data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# Function to evaluate the model on the validation set
def _evaluate(model, val_loader, criterion, device):
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
def _save_checkpoint(model, optimizer, epoch, val_loss):
    logging.info(f"Model saved with validation loss {val_loss:.4f} at epoch {epoch}")
    return {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }



def train_nn_for_object(train_config: TrainConfig):
    logging.info("------------------TRAINING STARTED------------------")
    logging.info(f"Number of clusters: {train_config.num_clusters}")
    logging.info(f"Number of surface points: {train_config.num_surface_points}")
    logging.info(f"Maximum number of epochs: {train_config.nn_config.max_epochs}")
    logging.info(f"Patience for early stopping: {train_config.nn_config.patience}")
    logging.info(f"Batch size: {train_config.nn_config.batch_size}")
    logging.info(f"Raw data folder: {train_config.file_path_config.raw_data_folderpath}")

    process_clustered_data(train_config.num_clusters, train_config.file_path_config.raw_data_folderpath,
                           train_config.file_path_config.clustered_data_filepath)
    process_surface_data(train_config.num_surface_points, train_config.file_path_config.raw_data_folderpath,
                         train_config.file_path_config.surface_data_filepath,
                         train_config.file_path_config.clustered_data_filepath)

    surface_data_list = load_pickle_file(train_config.file_path_config.surface_data_filepath)
    if surface_data_list is None or surface_data_list.list is None or not isinstance(surface_data_list,
                                                                                     SurfaceDataList):
        logging.error("Surface data list could not be loaded. Exiting.")
        return

    _train_nn_for_all_clusters(surface_data_list, max_epochs=train_config.nn_config.max_epochs,
                               patience=train_config.nn_config.patience,
                               batch_size=train_config.nn_config.batch_size,
                               model_weights_template=train_config.file_path_config.model_weights_folderpath)
    logging.info("------------------TRAINING ENDED------------------")
