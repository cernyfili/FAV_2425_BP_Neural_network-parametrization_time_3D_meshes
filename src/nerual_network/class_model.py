from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from data_processing.class_mapping import SurfacePointsFrameList
from utils.constants import ModelType


# Restrict access to underscore-prefixed functions
def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"{name} is a private function and cannot be imported.")
    raise AttributeError(f"Module has no attribute {name}")


class NNDataset(Dataset):
    def __init__(self, surface_data_list: SurfacePointsFrameList):
        if surface_data_list is None:
            raise ValueError("surface_data_list must not be None")

        self.data = []
        for surface_data in surface_data_list.public_list:
            points_all = surface_data.normalized_points_list
            points_all = np.array(points_all)

            time_values = np.full((points_all.shape[0], 1), surface_data.time.value)
            time_index = np.full((points_all.shape[0], 1), surface_data.time.index)
            point_indices = np.arange(points_all.shape[0]).reshape(-1, 1)

            points_with_info = np.hstack((points_all, time_values, time_index, point_indices))
            self.data.append(points_with_info)

        self.data = np.vstack(self.data)  # Shape: [total_points, feature_count]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Separate data based on the target and input requirements
        targets = self.data[idx, :3]  # First 3 columns as targets
        inputs = self.data[idx]  # All columns as inputsis
        return inputs, targets

    @staticmethod
    def get_time_indices_column(input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor[:, 4].unsqueeze(1)

    @staticmethod
    def get_time_values_column(input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor[:, 3].unsqueeze(1)

    @staticmethod
    def get_point_indices_column(input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor[:, 5].unsqueeze(1)

    @staticmethod
    def get_encoder_input(input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor[:, :4]

    @staticmethod
    def get_points_columns(input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor[:, :3]

    @staticmethod
    def get_unique_time_indices_list(input_tensor: torch.Tensor) -> torch.Tensor:
        time_indices = NNDataset.get_time_indices_column(input_tensor)
        return torch.unique(time_indices)

    @staticmethod
    def filter_by_time_index(input_tensor: torch.Tensor, time_index: int) -> torch.Tensor:
        time_indices = NNDataset.get_time_indices_column(input_tensor)
        # convert time_indices to a int tensor
        time_indices = time_indices.int()
        mask = (time_indices == time_index).squeeze()

        return_tensor = input_tensor[mask]
        return return_tensor

    @staticmethod
    def split_by_time_value(input_tensor: np.ndarray) -> list[np.ndarray]:
        time_values = input_tensor[:, 3]
        unique_time_values = np.unique(time_values)
        return [input_tensor[time_values == time_index] for time_index in unique_time_values]

    @staticmethod
    def select_random_values(input_tensor: torch.Tensor, num_values: int) -> torch.Tensor:
        """
        Select a random subset of values from the input tensor.

        Args:
            input_tensor (torch.Tensor): The input tensor.
            num_values (int): The number of random values to select.

        Returns:
            torch.Tensor: A tensor containing the selected random values.
        """
        if num_values > input_tensor.size(0):
            raise ValueError("num_values must be less than or equal to the size of the input tensor.")

        indices = torch.randperm(input_tensor.size(0))[:num_values]
        return input_tensor[indices]


class Simple_MLP_01(nn.Module):
    """
    deprecated not use
    """

    def __init__(self):
        super(Simple_MLP_01, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    # def forward(self, x):
    #     time_value = x[:, 3].unsqueeze(1)  # Extract time value and keep it as a column vector
    #     encoded_features = self.encoder(x)
    #     encoded_with_time = torch.cat((encoded_features, time_value), dim=1)  # Concatenate encoded features with time
    #     decoded_output = self.decoder(encoded_with_time)
    #     return decoded_output


class Simple_MLP_02(nn.Module):
    """
    Changes:
    - Added an extra layer to the encoder and decoder
    """

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

    # def forward(self, x, time_value: int = None):
    #     """
    #     Forward pass for the model.
    #
    #     Args:
    #         x (torch.Tensor): Input tensor.
    #         time_value (torch.Tensor, optional): Custom time value as a column vector.
    #                                              If None, it will be extracted from `x`.
    #
    #     Returns:
    #         torch.Tensor: Decoded output.
    #     """
    #     time = x[:, 3].unsqueeze(1)
    #     if time_value is not None:
    #         # change all the time_value so all elements have the value of time_value
    #         time = torch.full_like(time, time_value)
    #
    #     # Encode the input features
    #     encoded_features = self.encoder(x)
    #
    #     # Concatenate the encoded features with the time value
    #     encoded_with_time = torch.cat((encoded_features, time), dim=1)
    #
    #     # Decode the combined encoded features and time
    #     decoded_output = self.decoder(encoded_with_time)
    #
    #     return decoded_output


class Simple_MLP_03(nn.Module):
    """
    Changes:
    - at the end of encoder added Tanh activation function
    used for loss function: uv streach
    """

    def __init__(self):
        super(Simple_MLP_03, self).__init__()

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
            nn.Linear(64, 2),
            nn.Tanh()
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

    # def forward(self, x, time_value: int = None):
    #     """
    #     Forward pass for the model.
    #
    #     Args:
    #         x (torch.Tensor): Input tensor.
    #         time_value (torch.Tensor, optional): Custom time value as a column vector.
    #                                              If None, it will be extracted from `x`.
    #
    #     Returns:
    #         torch.Tensor: Decoded output.
    #     """
    #     time = x[:, 3].unsqueeze(1)
    #     if time_value is not None:
    #         # change all the time_value so all elements have the value of time_value
    #         time = torch.full_like(time, time_value)
    #
    #     # Ensure the input tensor has requires_grad=True
    #     x.requires_grad_(True)
    #
    #     # Encode the input features
    #     encoded_features = self.encoder(x)
    #
    #     # Concatenate the encoded features with the time value
    #     encoded_with_time = torch.cat((encoded_features, time), dim=1)
    #
    #     # Decode the combined encoded features and time
    #     decoded_output = self.decoder(encoded_with_time)
    #
    #     return decoded_output


class Simple_MLP_04(nn.Module):
    """
    Changes:
    - simplier architecture to reduce overfitting
    used for loss function: centers
    """

    def __init__(self):
        super(Simple_MLP_04, self).__init__()

        # Updated encoder with an extra layer, max neurons is 512
        self.encoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),  # Added layer with 64 neurons
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        # Updated decoder with an extra layer, max neurons is 512
        self.decoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),  # Added layer with 64 neurons
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    # def forward(self, x, time_value: int = None):
    #     """
    #     Forward pass for the model.
    #
    #     Args:
    #         x (torch.Tensor): Input tensor.
    #         time_value (torch.Tensor, optional): Custom time value as a column vector.
    #                                              If None, it will be extracted from `x`.
    #
    #     Returns:
    #         torch.Tensor: Decoded output.
    #     """
    #     time = x[:, 3].unsqueeze(1)
    #     if time_value is not None:
    #         # change all the time_value so all elements have the value of time_value
    #         time = torch.full_like(time, time_value)
    #
    #     # Encode the input features
    #     encoded_features = self.encoder(x)
    #
    #     # Concatenate the encoded features with the time value
    #     encoded_with_time = torch.cat((encoded_features, time), dim=1)
    #
    #     # Decode the combined encoded features and time
    #     decoded_output = self.decoder(encoded_with_time)
    #
    #     return decoded_output



MODELS_LIST: dict[ModelType: callable] ={
    ModelType.SIMPLE_MLP_01: Simple_MLP_01,
    ModelType.SIMPLE_MLP_02: Simple_MLP_02,
    ModelType.SIMPLE_MLP_03: Simple_MLP_03,
    ModelType.SIMPLE_MLP_04: Simple_MLP_04
}