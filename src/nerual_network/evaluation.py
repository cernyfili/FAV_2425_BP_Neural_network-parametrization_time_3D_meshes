import numpy as np
from torch.utils.data import DataLoader

from nerual_network.model import NNDataset
from utils.constants import NN_OPTIMIZER, NN_MODEL, MODEL_WEIGHTS_FILEPATH_TEMPLATE
from utils.helpers import visualize_original_and_processed_points, visualize_points_with_time, \
    save_combined_surface_points_images


# Restrict access to underscore-prefixed functions
def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"{name} is a private function and cannot be imported.")
    raise AttributeError(f"Module has no attribute {name}")

def process_and_save_combined_image_for_all_clusters(surface_data_list):
    original_points_all = []
    processed_points_all = []

    unique_clusters = surface_data_list.get_unique_clusters()

    for cluster in unique_clusters:
        # Load the original surface points for the current cluster
        surface_data_cluster = surface_data_list.filter_by_label(cluster)

        # Create a SurfaceDataset instance with the filtered surface data
        original_points_dataset = NNDataset(surface_data_cluster.list)

        # Load the trained model for the current cluster
        model_weights_filepath = MODEL_WEIGHTS_FILEPATH_TEMPLATE.format(cluster=cluster)
        model = _load_trained_model(model_weights_filepath)

        # Prepare a DataLoader for original points
        original_points_loader = DataLoader(original_points_dataset, batch_size=32, shuffle=True)

        # Process the original points through the model
        processed_points = []
        with torch.no_grad():
            for batch in original_points_loader:
                inputs = batch[0]  # Get only the points with time
                inputs = inputs.float()

                outputs = model(inputs)  # Forward pass through the model
                processed_points.append(outputs)

        # Convert processed points to a single numpy array
        processed_points = torch.cat(processed_points).numpy()

        # Accumulate all original and processed points
        original_points_all.append(original_points_dataset.data)  # You can store the numpy array directly
        processed_points_all.append(processed_points)

    # Convert lists to numpy arrays for plotting
    original_points_all = np.vstack(original_points_all) if original_points_all else np.empty((0, 4))
    processed_points_all = np.vstack(processed_points_all) if processed_points_all else np.empty((0, 4))

    # Save the combined image
    save_combined_surface_points_images(original_points_all, processed_points_all)
    visualize_points_with_time(original_points_all, processed_points_all)
    visualize_original_and_processed_points(original_points_all, processed_points_all)


def _load_trained_model(model_weights_filepath):
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
    model = NN_MODEL
    optimizer = NN_OPTIMIZER

    checkpoint = torch.load(model_weights_filepath)  # Load the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])  # Load the model state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load the optimizer state
    epoch = checkpoint['epoch']  # Get the epoch number
    val_loss = checkpoint['val_loss']  # Get the validation loss

    # Set the model to evaluation mode
    model.eval()

    return model
