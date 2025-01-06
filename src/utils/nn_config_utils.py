#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File name: nn_config_utils.py
Author: Filip Cerny
Created: 06.01.2025
Version: 1.0
Description: 
"""
import torch
from torch import optim, nn

from nerual_network.class_model import Simple_MLP_02
from nerual_network.evaluation import get_loaded_meshes_list, get_time_specific_decoder_data

def __get_random_time(inputs):
    # Extract the time column (assuming it's the 4th column)
    time_column = inputs[:, 3]

    # Select a random index
    random_index = torch.randint(0, time_column.size(0), (1,)).item()

    # Retrieve the time value at the selected index
    random_time = time_column[random_index].item()

    return random_time

def __one_way_chamfer_distance_loss(original_mesh_v : torch.Tensor, original_mesh_f : torch.Tensor, decoded_mesh_v : torch.Tensor):
    """
    Computes the one-way Chamfer Distance loss (decoded → original)
    for vertices outside the original mesh.

    Args:
        original_mesh_v (torch.Tensor): Vertices of the original mesh (N x 3).
        original_mesh_f (torch.Tensor): Faces of the original mesh (M x 3).
        decoded_mesh_v (torch.Tensor): Vertices of the decoded mesh (K x 3).

    Returns:
        torch.Tensor: One-way Chamfer Distance loss.
    """
    # Convert PyTorch tensors to numpy arrays for IGL
    original_mesh_v_np = original_mesh_v.detach().cpu().numpy()
    original_mesh_f_np = original_mesh_f.detach().cpu().numpy()
    decoded_mesh_v_np = decoded_mesh_v.detach().cpu().numpy()

    # Compute signed distances and closest points (Decoded → Original)
    distances, _, closest_points = igl.signed_distance(
        decoded_mesh_v_np, original_mesh_v_np, original_mesh_f_np
    )

    # Convert distances to a PyTorch tensor
    distances = torch.tensor(distances, device=decoded_mesh_v.device)

    # Filter: Only consider points with positive signed distances (outside the mesh)
    outside_indices = distances > 0

    # Select corresponding points from decoded_mesh_v and original_mesh_v
    selected_decoded_points = decoded_mesh_v[outside_indices]
    selected_original_points = torch.tensor(closest_points[outside_indices], device=decoded_mesh_v.device)

    # Compute MSE between selected decoded_mesh points and original_mesh points
    mse_loss = nn.MSELoss(selected_decoded_points, original_mesh_v)

    return mse_loss

# region LOSS FUNCTIONS
def _loss_function_standard(inputs, targets, model, loss_info):
    outputs = model(inputs)
    loss = nn.MSELoss(outputs, targets)
    return loss

def _loss_function_chamfer(inputs, targets, model, loss_info):
    raw_data_folder = loss_info['raw_data_folder']
    time_list = loss_info['time_list']
    device = loss_info['device']

    # select one random time from inputs
    time = __get_random_time(inputs)
    # Forward pass: Encoder and Decoder
    encoded = model.encoder(inputs)
    decoder_data = get_time_specific_decoder_data(device=device, encoded_features=encoded, time_value=time)
    decoded_mesh_v = model.decoder(decoder_data)  # Decodes to 3D
    time_index = next((time_element.index for time_element in time_list if time_element.value == time), None)
    meshes_list = get_loaded_meshes_list(raw_data_folder)
    selected_mesh = meshes_list[time_index]
    original_mesh_v = selected_mesh.vertices
    original_mesh_f = selected_mesh.faces
    # Compute one-way Chamfer Distance loss
    loss_chamfer = __one_way_chamfer_distance_loss(original_mesh_v, original_mesh_f, decoded_mesh_v)
    loss_standard = _loss_function_standard(inputs, targets, model, loss_info)
    return loss_chamfer + loss_standard
# endregion

# Configuration function to initialize model, optimizer, and criterion
def get_training_config(nn_lr) -> (nn.Module, optim.Optimizer, nn.Module):
    model = Simple_MLP_02()
    optimizer = optim.Adam(model.parameters(), lr=nn_lr)
    loss_function = _loss_function_chamfer
    return model, optimizer, loss_function