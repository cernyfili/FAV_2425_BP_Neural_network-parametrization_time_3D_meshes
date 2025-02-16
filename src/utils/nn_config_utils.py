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
from attr import dataclass
from numpy import ndarray
from scipy.spatial import KDTree
from torch import optim, nn, tensor

from data_processing.class_mapping import MeshList, TimeFrame, SurfacePointsFrameList, SurfacePointsFrame, \
     find_closest_centers, \
    compute_distances_from_centers, CentersInfo
from nerual_network.class_model import Simple_MLP_02
from utils.constants import TrainConfig, loss_function_name, CDataPreprocessing, LOSS_FUNC_NORMAL_DIST_MEAN, LOSS_FUNC_NORMAL_DIST_STD
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
    def __get_random_time_from_time_list(time_list : list[TimeFrame]) -> tuple[float, int]:
        random_time_element = np.random.choice(time_list)
        return random_time_element.value, random_time_element.index

    def __prepare_data(device, inputs, model, meshes_list: MeshList, time_list : list[TimeFrame]):
        # select one random time from inputs
        time_value, time_index = __get_random_time_from_time_list(time_list)

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
    time_list : list[TimeFrame] = loss_info['time_list']

    decoded_mesh_v, original_mesh_f, original_mesh_v = __prepare_data(device, inputs, model, meshes_list, time_list)
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
def __select_unique_time(time_tensor: torch.Tensor) -> torch.Tensor:

    # get unique time values
    unique_times = torch.unique(time_tensor)

    return unique_times

# function which will return time_values list where is the two most common time values
def __select_most_common_time_values(time_value_tensor: torch.Tensor, num_values = 2) -> torch.Tensor:
    # convert all values in tensor to int
    time_value_tensor = time_value_tensor.int()

    # Count occurrences of each value in the tensor
    counts = torch.bincount(time_value_tensor)

    # ratio = 0.5
    # # get number of half of all count of unique values
    # num_select = int(counts.size(0) * ratio)

    # if there is less unique values than num_values, return 1
    if counts.size(0) < num_values:
        num_values = 1

    # Get the indices of the two most common values
    top_two_indices = torch.topk(counts, num_values).indices

    # Return the two most common values
    return top_two_indices

def _compute_loss_chamfer_distance_with_time_tensor(decoder_data: torch.Tensor, meshes_list: MeshList, decoder,
                                                    select_times_function : callable) -> list[tensor]:
    """

    :param decoder_data: tensor 4 columns [x_value, y_value, time_value, time_index]
    :param meshes_list:
    :param decoder:
    :param select_times_function:
    :return:
    """
    # for each unique time in decoder_input_data run this throug decoder and then find with coresponding index from time_index_tensor where
    # decoder_input_data 3th column is equal to time_index_tensor and then coresponding mesh from meshes_list with that index and compute chamfer distance

    loss_list = []

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

        loss_list.append(loss_chamfer)

    return loss_list

def _get_time_tensor_from_input(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

def _get_decoder_input_data(device, inputs, model):
    # select one random time from inputs
    time_value_tensor, time_index_tensor = _get_time_tensor_from_input(inputs)

    # remove last columen (time_index) from inputs
    inputs_encoder = prepare_encoder_input_data(inputs)

    # Forward pass: Encoder and Decoder
    encoded = model.encoder(inputs_encoder)

    # add time_value_tensor to encoded features
    decoder_input_data = torch.cat((encoded, time_value_tensor.unsqueeze(1)), dim=1).to(device)

    #add time_index_tensor to decoder_input_data
    decoder_input_data = torch.cat((decoder_input_data, time_index_tensor.unsqueeze(1)), dim=1).to(device)

    return decoder_input_data

def _loss_function_chamfer_better_random_dist(inputs, targets, model, loss_info):


    select_function = __select_unique_time

    meshes_list = loss_info['meshes_list']
    device = loss_info['device']

    decoder_data = _get_decoder_input_data(device, inputs, model)
    loss_chamfer_list = _compute_loss_chamfer_distance_with_time_tensor(decoder_data=decoder_data, meshes_list=meshes_list,
                                                                   decoder=model.decoder, select_times_function=select_function)

    #compute avrage of all chamfer distances
    loss_chamfer = torch.stack(loss_chamfer_list).mean()

    loss_standard = _loss_function_standard(inputs, targets, model, loss_info)

    combined_loss = loss_chamfer + loss_standard

    logging.info(f"Chamfer loss: {loss_chamfer}, Standard loss: {loss_standard}, Combined loss: {combined_loss}")

    return combined_loss


def _loss_function_uv_streach(inputs, targets, model, loss_info):
    loss_uv_streach = __compute_loss_uv_streach(inputs, model, targets)
    loss_standard = _loss_function_standard(inputs, targets, model, loss_info)

    return loss_uv_streach + loss_standard


def __compute_center_distance_loss(input_points : torch.Tensor, decoded_points : torch.Tensor, centers_info : CentersInfo) -> torch.tensor:
    """

    :param input_points: input points and decoded points is same points on same index
    :param decoded_points:
    :param centers_points:
    :return:
    """
    # region SANITY CHECKS
    if len(input_points) != len(decoded_points):
        raise ValueError("Length of input_points and decoded_data must be the same")

    if input_points.shape[1] != 3:
        raise ValueError("Input points must have 3 columns")
    if decoded_points.shape[1] != 3:
        raise ValueError("Decoded data must have 3 columns")
    if centers_info.points.shape[1] != 3:
        raise ValueError("Center points must have 3 columns")
    if not isinstance(centers_info, CentersInfo):
        raise ValueError("Center points must be a numpy array ")
    # endregion

    num_closest_centers = CDataPreprocessing.NUM_CLOSEST_CENTERS_TO_POINT

    centers_points_tensor = torch.tensor(centers_info.points, device=input_points.device)
    centers_points_tensor.requires_grad_(True)
    closest_centers_tensor = find_closest_centers(input_points, centers_points_tensor, num_closest_centers, centers_info.kd_tree)

    input_centers_distances = compute_distances_from_centers(input_points, closest_centers_tensor, num_closest_centers)
    decoded_centers_distances = compute_distances_from_centers(decoded_points, closest_centers_tensor, num_closest_centers)

    # check
    if input_centers_distances.shape != decoded_centers_distances.shape:
        raise ValueError("Length of closest_centers and original_centers must be the same")

    # Create a normal distribution with mean and std of original distances
    mean_original = LOSS_FUNC_NORMAL_DIST_MEAN
    std_original = LOSS_FUNC_NORMAL_DIST_STD
    normal_dist = torch.distributions.Normal(mean_original, std_original)

    # Convert distances to probabilities
    prob_input = normal_dist.cdf(input_centers_distances)
    prob_decoded = normal_dist.cdf(decoded_centers_distances)

    # Compute the loss as the mean squared error between the probabilities
    loss = nn.MSELoss()(prob_input, prob_decoded)
    return loss


# dataclass


def _compute_loss_function_centers(inputs, targets, model, loss_info) -> list:
    data : SurfacePointsFrameList = loss_info['data']
    device = loss_info['device']
    time_list = loss_info['time_list'] # list of TimeFrame

    # from dat

    decoder_input_data_all = _get_decoder_input_data(device, inputs, model)
    time_index_tensor = decoder_input_data_all[:, 3]

    decoder_input_data = decoder_input_data_all[:, :3]  # remove last column (time_index)
    decoded_data = model.decoder(decoder_input_data)

    unique_time_elements = __select_unique_time(time_index_tensor)

    loss_list = []

    for time_index in unique_time_elements:

        # indices from time_index_tensor where is the time_index
        indices_time_index_tensor = time_index_tensor == time_index

        filtered_decoded_data = decoded_data[indices_time_index_tensor]
        filtered_decoded_data.requires_grad_(True)

        # original data
        original_data = data.get_element_by_time_index(time_index)
        centers_info = original_data.centers_info

        # get first 3 columns from inputs
        filtered_input_data = inputs[:, :3]

        filtered_input_data = filtered_input_data[indices_time_index_tensor]
        filtered_input_data.requires_grad_(True)

        # find indices from original_points which are the same point values as in filtered_input_data

        # indices_original_points_np = np.where(np.isin(original_points_np, filtered_input_data_np).all(axis=1))[0]
        # 
        # filtered_original_centers = original_centers_np[indices_original_points_np]

        # get first 3 columns

        loss_distances = __compute_center_distance_loss(filtered_input_data, filtered_decoded_data, centers_info)
        loss_list.append(loss_distances)

    return loss_list

def _loss_function_centers(inputs, targets, model, loss_info):
    loss_centers_list = _compute_loss_function_centers(inputs, targets, model, loss_info)
    loss_centers = torch.stack(loss_centers_list).sum()
    batch_size = inputs.size(0)
    loss_centers = loss_centers / batch_size

    loss_standard = _loss_function_standard(inputs, targets, model, loss_info)

    loss = loss_centers + loss_standard

    logging.info(f"Center loss: {loss_centers}, Standard loss: {loss_standard}, Combined loss: {loss}")
    return loss



# endregion




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



LOSS_FUNCTIONS_LIST : dict[str : callable] = {
    'standard': _loss_function_standard,
    'chamfer': _loss_function_chamfer,
    'chamfer_better_random_dist': _loss_function_chamfer_better_random_dist,
    'uv_streach': _loss_function_uv_streach,
    'centers': _loss_function_centers
}


# Configuration function to initialize model, optimizer, and criterion
def init_training_config(train_config : TrainConfig) -> (nn.Module, optim.Optimizer, nn.Module):
    nn_lr = train_config.nn_config.nn_lr
    loss_function_name = train_config.nn_config.loss_function_name

    model = Simple_MLP_02()
    optimizer = optim.Adam(model.parameters(), lr=nn_lr)
    loss_function = LOSS_FUNCTIONS_LIST[loss_function_name]
    return model, optimizer, loss_function