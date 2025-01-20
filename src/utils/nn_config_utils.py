#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File name: nn_config_utils.py
Author: Filip Cerny
Created: 06.01.2025
Version: 1.0
Description: 
"""
import logging

import igl
import numpy as np
import torch
import trimesh
from torch import optim, nn

from data_processing.class_mapping import MeshList
from nerual_network.class_model import Simple_MLP_02
from utils.helpers import get_meshes_list


def __get_random_time(inputs: torch.Tensor) -> tuple[float, int]:
    # Extract the time value column (assuming it's the 4th column)
    time_value_column = inputs[:, 3]

    # Extract the time index column (assuming it's the 5th column)
    time_index_column = inputs[:, 4]

    # Select a random index
    random_index = torch.randint(0, time_value_column.size(0), (1,)).item()

    # Retrieve the time value and time index at the selected index
    random_time_value = time_value_column[random_index].item()
    random_time_index = time_index_column[random_index].item()

    return random_time_value, int(random_time_index)


# region COMPUTE LOSS VALUE FUNCTIONS
def __compute_loss_one_way_chamfer_distance(original_mesh_v: np, original_mesh_f: np,
                                            decoded_mesh_v: torch.Tensor):
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
    # original_mesh_v_np = original_mesh_v.detach().cpu().numpy()
    # original_mesh_f_np = original_mesh_f.detach().cpu().numpy()

    original_mesh_v_np = original_mesh_v
    original_mesh_f_np = original_mesh_f
    decoded_mesh_v_np = decoded_mesh_v.detach().cpu().numpy()

    # Compute signed distances and closest points (Decoded → Original)
    distances, _, closest_points = igl.signed_distance(
        decoded_mesh_v_np, original_mesh_v_np, original_mesh_f_np
    )

    # Convert distances to a PyTorch tensor
    distances = torch.tensor(distances, device=decoded_mesh_v.device)

    # Filter: Only consider points with positive signed distances (outside the mesh)
    outside_indices = distances > 0

    # if no outside indices, return 0
    if not torch.any(outside_indices):
        return torch.tensor(0.0, device=decoded_mesh_v.device)

    # Create tensors of the same size as decoded_mesh_v
    selected_decoded_points = torch.zeros_like(decoded_mesh_v)
    selected_closest_points = torch.zeros_like(decoded_mesh_v)

    # Convert closest_points to the same dtype as decoded_mesh_v
    closest_points_tensor = torch.tensor(closest_points, device=decoded_mesh_v.device, dtype=decoded_mesh_v.dtype)

    # Assign values only for points outside the mesh
    selected_decoded_points[outside_indices] = decoded_mesh_v[outside_indices]
    selected_closest_points[outside_indices] = closest_points_tensor[outside_indices]

    # Compute MSE between selected decoded_mesh points and original_mesh points
    mse_loss = nn.MSELoss()(selected_decoded_points, selected_closest_points)

    return mse_loss


def __compute_loss_uv_streach(inputs, model, targets):
    area_coefficient = 1.0
    mse_area_loss = mse_area(area_coefficient)
    # outputs = model(inputs)
    encoder_input = prepare_encoder_input_data(inputs)
    loss = mse_area_loss(targets, encoder_input, model.encoder)
    return loss


def mse_area(area_coefficient):
    def mse_area(targets, inputs, encoder):
        inputs_without_time = inputs[:, :-1]
        diff_pos = torch.sum(torch.square(targets - inputs_without_time), dim=1).mean()

        # requires_grad = True
        encoder_input = inputs.requires_grad_(True)
        encoder_output = encoder(encoder_input)

        uv_pred = encoder_output
        u_pred = uv_pred[::, 0]
        v_pred = uv_pred[::, 1]
        # x_var = encoder_input[:, :-1]

        g_u_all = torch.autograd.grad(outputs=u_pred, inputs=encoder_input, grad_outputs=torch.ones_like(u_pred),
                                      retain_graph=True, allow_unused=True, create_graph=True)
        g_u = g_u_all[0]
        g_v_all = torch.autograd.grad(outputs=v_pred, inputs=encoder_input, grad_outputs=torch.ones_like(v_pred),
                                      retain_graph=True, allow_unused=True, create_graph=True)
        g_v = g_v_all[0]

        e = torch.sum(g_u * g_u, dim=1)
        f = torch.sum(g_u * g_v, dim=1)
        g = torch.sum(g_v * g_v, dim=1)
        det_i = e * g - f * f
        diff_area = torch.abs(det_i).mean()  # Use negative mean to maximize det_i

        return diff_pos + area_coefficient * diff_area

    return mse_area


# endregion

# region LOSS FUNCTIONS
def _loss_function_standard(inputs, targets, model, loss_info):
    model_input_data = prepare_encoder_input_data(inputs)
    outputs = model(model_input_data)
    loss = nn.MSELoss()(outputs, targets)
    return loss


def _loss_function_chamfer(inputs, targets, model, loss_info):
    def __prepare_data(device, inputs, model, meshes_list: MeshList):
        # select one random time from inputs
        time_value, time_index = __get_random_time(inputs)

        # remove last columen (time_index) from inputs
        inputs_encoder = prepare_encoder_input_data(inputs)

        # Forward pass: Encoder and Decoder
        encoded = model.encoder(inputs_encoder)

        decoder_input_data = prepare_decoder_input_data(device=device, encoded_features=encoded, time_value=time_value)

        decoded_mesh_v = model.decoder(decoder_input_data)  # Decodes to 3D

        selected_mesh = meshes_list.get_mesh_by_time_index(time_index)
        original_mesh_v = selected_mesh.vertices
        original_mesh_f = selected_mesh.faces

        return decoded_mesh_v, original_mesh_f, original_mesh_v

    meshes_list = loss_info['meshes_list']
    device = loss_info['device']

    decoded_mesh_v, original_mesh_f, original_mesh_v = __prepare_data(device, inputs, model, meshes_list)
    # Compute one-way Chamfer Distance loss
    # original_mesh_v = torch.tensor(original_mesh_v, device=device, requires_grad=False)
    # original_mesh_f = torch.tensor(original_mesh_f, device=device, requires_grad=False)

    loss_chamfer = __compute_loss_one_way_chamfer_distance(original_mesh_v, original_mesh_f, decoded_mesh_v)
    # if f"{loss_chamfer}" != '0.0':
    #     logging.info(f"Chamfer loss: {loss_chamfer}")
    loss_standard = _loss_function_standard(inputs, targets, model, loss_info)

    combined_loss = loss_chamfer + loss_standard

    logging.info(f"Chamfer loss: {loss_chamfer}, Standard loss: {loss_standard}, Combined loss: {combined_loss}")

    return combined_loss

# function which will return torch with time values from argument inputs
def __select_unique_time_values(time_value_tensor: torch.Tensor) -> torch.Tensor:

    # get unique time values
    unique_time_values = torch.unique(time_value_tensor)

    return unique_time_values

# function which will return time_values list where is the two most common time values
def __select_most_common_time_values(time_value_tensor: torch.Tensor, num_values = 2) -> torch.Tensor:
    # Count occurrences of each value in the tensor
    counts = torch.bincount(time_value_tensor)

    # Get the indices of the two most common values
    top_two_indices = torch.topk(counts, num_values).indices

    # Return the two most common values
    return top_two_indices

def _compute_loss_chamfer_distance_with_time_tensor(decoder_data: torch.Tensor, meshes_list: MeshList, decoder,
                                                    select_times_function : callable) -> int:
    """

    :param decoder_data: tensor 4 columns [x_value, y_value, time_value, time_index]
    :param meshes_list:
    :param decoder:
    :param select_times_function:
    :return:
    """
    # for each unique time in decoder_input_data run this throug decoder and then find with coresponding index from time_index_tensor where
    # decoder_input_data 3th column is equal to time_index_tensor and then coresponding mesh from meshes_list with that index and compute chamfer distance

    loss_combined = 0

    time_index_tensor = decoder_data[:, 3]
    # get unique time values from decoder_input_data from 3th column
    time_index_selection = select_times_function(time_index_tensor)

    # for each unique time value
    for time_index in time_index_selection:
        # get all rows where 4th column is equal to time_value
        decoder_data_select_time_index = decoder_data[decoder_data[:, 3] == time_index]

        meshe = meshes_list.get_mesh_by_time_index(int(time_index))

        # run decoder on decoder_data_time_value
        decoder_input_data_time_value = decoder_data_select_time_index[:, :3]  # remove last column (time_index)
        decoded_mesh_v = decoder(decoder_input_data_time_value)

        # Compute one-way Chamfer Distance loss
        loss_chamfer = __compute_loss_one_way_chamfer_distance(original_mesh_v=meshe.vertices,
                                                               original_mesh_f=meshe.faces,
                                                               decoded_mesh_v=decoded_mesh_v)

        loss_combined += loss_chamfer

    return loss_combined


def _loss_function_chamfer_better_random_dist(inputs, targets, model, loss_info):
    def __get_time_tensor(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Extract the time value column (assuming it's the 4th column)
        time_value_column = inputs[:, 3]

        # Extract the time index column (assuming it's the 5th column)
        time_index_column = inputs[:, 4]

        # generate tensor of with 2 columns where you randomly select time values from time_value_column and add corresponding time_index
        # random_index = torch.randint(0, time_value_column.size(0), (1,)).item()
        # random_time_value_tensor = time_value_column[random_index]
        #
        # # Retrieve the time value and time index at the selected index
        # random_time_index_tensor = time_index_column[random_index]
        time_value_tensor = time_value_column
        time_index_tensor = time_index_column

        return time_value_tensor, time_index_tensor

    def __prepare_data(device, inputs, model):
        # select one random time from inputs
        time_value_tensor, time_index_tensor = __get_time_tensor(inputs)

        # remove last columen (time_index) from inputs
        inputs_encoder = prepare_encoder_input_data(inputs)

        # Forward pass: Encoder and Decoder
        encoded = model.encoder(inputs_encoder)

        # add time_value_tensor to encoded features
        decoder_input_data = torch.cat((encoded, time_value_tensor.unsqueeze(1)), dim=1).to(device)

        #add time_index_tensor to decoder_input_data
        decoder_input_data = torch.cat((decoder_input_data, time_index_tensor.unsqueeze(1)), dim=1).to(device)

        return decoder_input_data

    select_function = __select_unique_time_values

    meshes_list = loss_info['meshes_list']
    device = loss_info['device']

    decoder_data = __prepare_data(device, inputs, model)
    loss_chamfer = _compute_loss_chamfer_distance_with_time_tensor(decoder_data=decoder_data, meshes_list=meshes_list,
                                                                   decoder=model.decoder, select_times_function=select_function)

    loss_standard = _loss_function_standard(inputs, targets, model, loss_info)

    combined_loss = loss_chamfer + loss_standard

    logging.info(f"Chamfer loss: {loss_chamfer}, Standard loss: {loss_standard}, Combined loss: {combined_loss}")

    return combined_loss


def _loss_function_uv_streach(inputs, targets, model, loss_info):
    loss_uv_streach = __compute_loss_uv_streach(inputs, model, targets)
    loss_standard = _loss_function_standard(inputs, targets, model, loss_info)

    return loss_uv_streach + loss_standard


# endregion

# Configuration function to initialize model, optimizer, and criterion
def get_training_config(nn_lr) -> (nn.Module, optim.Optimizer, nn.Module):
    model = Simple_MLP_02()
    optimizer = optim.Adam(model.parameters(), lr=nn_lr)
    loss_function = _loss_function_chamfer_better_random_dist
    return model, optimizer, loss_function


def prepare_encoder_input_data(inputs):
    # get first 4 columns from inputs
    return inputs[:, :4]


def prepare_decoder_input_data(device, encoded_features, time_value):
    # Create a tensor of the same shape as the time feature in the input
    time_tensor = torch.full((encoded_features.size(0), 1), time_value, dtype=torch.float32).to(device)
    # Concatenate the encoded features with the time tensor
    encoded_with_time = torch.cat((encoded_features, time_tensor), dim=1).to(device)
    return encoded_with_time


def get_loaded_meshes_list(meshes_folder_path: str):
    meshes_filepaths_list = get_meshes_list(meshes_folder_path)
    loaded_meshes_list = []
    for mesh_filepath in meshes_filepaths_list:
        mesh = trimesh.load(mesh_filepath)
        loaded_meshes_list.append(mesh)
    return loaded_meshes_list
