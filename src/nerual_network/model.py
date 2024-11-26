import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# Restrict access to underscore-prefixed functions
def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"{name} is a private function and cannot be imported.")
    raise AttributeError(f"Module has no attribute {name}")


class NNDataset(Dataset):
    def __init__(self, surface_data_list: list):
        if surface_data_list is None:
            raise ValueError("surface_data_list must not be None")

        self.data = []
        for surface_data in surface_data_list:
            points = surface_data.points_list
            points = np.array(points)
            time = np.full((points.shape[0], 1), surface_data.time)
            points_with_time = np.hstack((points, time))
            self.data.append(points_with_time)

        self.data = np.vstack(self.data)  # Shape: [total_points, feature_count]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Separate data based on the target and input requirements
        targets = self.data[idx, :3]   # First 3 columns as targets
        inputs = self.data[idx, :4]    # All 4 columns as inputs, including "time" as the last one
        return inputs, targets


class Simple_MLP_01(nn.Module):
    def __init__(self):
        super(Simple_MLP_01, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.decoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):
        time_value = x[:, 3].unsqueeze(1)  # Extract time value and keep it as a column vector
        encoded_features = self.encoder(x)
        encoded_with_time = torch.cat((encoded_features, time_value), dim=1)  # Concatenate encoded features with time
        decoded_output = self.decoder(encoded_with_time)
        return decoded_output


class Simple_MLP_02(nn.Module):
    def __init__(self):
        super(Simple_MLP_02, self).__init__()

        # Updated encoder with an extra layer, max neurons is 512
        self.encoder = nn.Sequential(
            nn.Linear(4, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),  # Added layer with 64 neurons
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        # Updated decoder with an extra layer, max neurons is 512
        self.decoder = nn.Sequential(
            nn.Linear(3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),  # Added layer with 64 neurons
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x, time_value=None):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor.
            time_value (torch.Tensor, optional): Custom time value as a column vector.
                                                 If None, it will be extracted from `x`.

        Returns:
            torch.Tensor: Decoded output.
        """
        if time_value is None:
            # Extract time value from the input tensor if not provided
            time_value = x[:, 3].unsqueeze(1)

        # Encode the input features
        encoded_features = self.encoder(x)

        # Concatenate the encoded features with the time value
        encoded_with_time = torch.cat((encoded_features, time_value), dim=1)

        # Decode the combined encoded features and time
        decoded_output = self.decoder(encoded_with_time)

        return decoded_output

