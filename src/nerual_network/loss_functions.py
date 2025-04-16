#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File name: loss_function.py
Author: Filip Cerny
Created: 17.03.2025
Version: 1.0
Description: 
"""
import logging

import igl
import numpy as np
import torch
from torch import nn, tensor

from data_processing.class_mapping import TimeFrame, MeshList, CentersInfo, SurfacePointsFrame, SurfacePointsFrameList, \
    LossFunctionInfo
from nerual_network.class_model import NNDataset
from utils.constants import LOSS_FUNC_NORMAL_DIST_MEAN, LOSS_FUNC_NORMAL_DIST_STD, CDataPreprocessing, LossFunctionType


# region PRIVATE FUNCTIONS
def _add_time_column(encoded_features, time_value):
    device = encoded_features.device
    # Create a tensor of the same shape as the time feature in the input
    time_tensor = torch.full((encoded_features.size(0), 1), time_value, dtype=torch.float32).to(device)
    # Concatenate the encoded features with the time tensor
    encoded_with_time = torch.cat((encoded_features, time_tensor), dim=1)
    return encoded_with_time


# def __get_random_time(inputs: torch.Tensor) -> tuple[float, int]:
#     # Extract the time value column (assuming it's the 4th column)
#     time_value_column = inputs[:, 3]
#
#     # Extract the time index column (assuming it's the 5th column)
#     time_index_column = inputs[:, 4]
#
#     # Select a random index
#     random_index = torch.randint(0, time_value_column.size(0), (1,)).item()
#
#     # Retrieve the time value and time index at the selected index
#     random_time_value = time_value_column[random_index].item()
#     random_time_index = time_index_column[random_index].item()
#
#     return random_time_value, int(random_time_index)


def __select_unique_time(time_tensor: torch.Tensor) -> torch.Tensor:
    # get unique time values
    unique_times = torch.unique(time_tensor)

    return unique_times


def __select_most_common_time_values(time_value_tensor: torch.Tensor, num_values=2) -> torch.Tensor:
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

#
# def _get_time_tensor_from_input(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#     # Extract the time value column (assuming it's the 4th column)
#     time_value_column = inputs[:, 3]
#
#     # Extract the time index column (assuming it's the 5th column)
#     time_index_column = inputs[:, 4]
#
#     # generate tensor of with 2 columns where you randomly select time values from time_value_column and add corresponding time_index
#     # random_index = torch.randint(0, time_value_column.size(0), (1,)).item()
#     # random_time_value_tensor = time_value_column[random_index]
#     #
#     # # Retrieve the time value and time index at the selected index
#     # random_time_index_tensor = time_index_column[random_index]
#     time_value_tensor = time_value_column
#     time_index_tensor = time_index_column
#
#     return time_value_tensor, time_index_tensor


def _run_through_encoder(inputs, encoder):
    # remove last columen (time_index) from inputs
    inputs_encoder = NNDataset.get_encoder_input(inputs)


    encoded = encoder(inputs_encoder)

    return encoded


def _run_through_decoder_at_time(encoded_output : torch.tensor, decoder : callable, time : TimeFrame) -> torch.tensor:
    decoder_input_data = _add_time_column(encoded_output, time.value)
    decoder_output = decoder(decoder_input_data)
    return decoder_output


def run_through_nn_at_decoder_time(inputs : torch.tensor, model : callable, decoder_time: TimeFrame) -> torch.tensor:
    encoder_output_data = _run_through_encoder(inputs, model.encoder)
    decoder_output_data = _run_through_decoder_at_time(encoder_output_data, model.decoder, decoder_time)
    return decoder_output_data

def run_through_nn_at_same_time(inputs : torch.tensor, model : callable) -> torch.tensor:
    encoder_output_data = _run_through_encoder(inputs, model.encoder)

    time_column = NNDataset.get_time_values_column(inputs)
    encoder_output_data_with_time = torch.cat((encoder_output_data, time_column), dim=1)

    decoder_output_data = model.decoder(encoder_output_data_with_time)

    return decoder_output_data

# endregion


def loss_function_standard(inputs, targets, model, loss_info):
    outputs = run_through_nn_at_same_time(inputs, model)
    loss = nn.MSELoss()(outputs, targets)
    return loss


# region LOSS FUNCTION CHAMFER
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


def __compute_loss_chamfer_distance_with_time_tensor(inputs: torch.Tensor, meshes_list: MeshList, model,
                                                     select_times_function: callable, time_list : list[TimeFrame]) -> list[tensor]:
    """

    :param decoder_data: tensor 4 columns [x_value, y_value, time_value, time_index]
    :param meshes_list:
    :param decoder:
    :param select_times_function:
    :return:
    """
    time_list_dict = {time.index: time for time in time_list}

    encoded_features = _run_through_encoder(inputs, model.encoder)

    time_index_tensor = NNDataset.get_time_indices_column(inputs)

    if encoded_features.shape[0] != time_index_tensor.shape[0]:
        raise ValueError("Encoded features and time index tensor must have the same number of rows")

    # for each unique time in decoder_input_data run this throug decoder and then find with coresponding index from time_index_tensor where
    # decoder_input_data 3th column is equal to time_index_tensor and then coresponding mesh from meshes_list with that index and compute chamfer distance

    loss_list = []

    # get unique time values from decoder_input_data from 3th column
    time_index_selection = select_times_function(time_index_tensor)

    # for each unique time value
    for time_index in time_index_selection:
        # get all rows where 4th column is equal to time_value
        filtered_encoded_features = encoded_features[time_index_tensor == time_index]

        meshe = meshes_list.get_mesh_by_time_index(int(time_index))


        decoded_mesh_v = _run_through_decoder_at_time(encoded_output=filtered_encoded_features, decoder= model.decoder, time=time_list_dict[time_index])

        # Compute one-way Chamfer Distance loss
        loss_chamfer = __compute_loss_one_way_chamfer_distance(original_mesh_v=meshe.vertices,
                                                               original_mesh_f=meshe.faces,
                                                               decoded_mesh_v=decoded_mesh_v)

        loss_list.append(loss_chamfer)

    return loss_list


def loss_function_chamfer(inputs, targets, model, loss_info):
    def __get_random_time_from_time_list(time_list: list[TimeFrame]) -> TimeFrame:
        random_time_element = np.random.choice(time_list)
        return random_time_element

    def __prepare_data(inputs, model, meshes_list: MeshList, time_list: list[TimeFrame]):
        # select one random time from inputs
        time_frame = __get_random_time_from_time_list(time_list)

        decoded_mesh_v = run_through_nn_at_decoder_time(inputs, model, time_frame)

        selected_mesh = meshes_list.get_mesh_by_time_index(time_frame.index)
        original_mesh_v = selected_mesh.vertices
        original_mesh_f = selected_mesh.faces

        return decoded_mesh_v, original_mesh_f, original_mesh_v

    meshes_list = loss_info.meshes_list
    device = loss_info.device
    time_list: list[TimeFrame] = loss_info.time_list

    decoded_mesh_v, original_mesh_f, original_mesh_v = __prepare_data(inputs, model, meshes_list, time_list)
    # Compute one-way Chamfer Distance loss
    # original_mesh_v = torch.tensor(original_mesh_v, device=device, requires_grad=False)
    # original_mesh_f = torch.tensor(original_mesh_f, device=device, requires_grad=False)

    loss_chamfer = __compute_loss_one_way_chamfer_distance(original_mesh_v, original_mesh_f, decoded_mesh_v)
    # if f"{loss_chamfer}" != '0.0':
    #     logging.info(f"Chamfer loss: {loss_chamfer}")
    loss_standard = loss_function_standard(inputs, targets, model, loss_info)

    combined_loss = loss_chamfer + loss_standard

    logging.info(f"Chamfer loss: {loss_chamfer}, Standard loss: {loss_standard}, Combined loss: {combined_loss}")

    return combined_loss


def loss_function_chamfer_better_random_dist(inputs, targets, model, loss_info):
    select_function = __select_unique_time

    meshes_list = loss_info.meshes_list
    device = loss_info.device
    time_list: list[TimeFrame] = loss_info.time_list

    loss_chamfer_list = __compute_loss_chamfer_distance_with_time_tensor(inputs=inputs,
                                                                         meshes_list=meshes_list,
                                                                         model=model,
                                                                         select_times_function=select_function,
                                                                         time_list=time_list)

    # compute avrage of all chamfer distances
    loss_chamfer = torch.stack(loss_chamfer_list).mean()

    loss_standard = loss_function_standard(inputs, targets, model, loss_info)

    combined_loss = loss_chamfer + loss_standard

    logging.info(f"Chamfer loss: {loss_chamfer}, Standard loss: {loss_standard}, Combined loss: {combined_loss}")

    return combined_loss
# endregion


# region LOSS FUNCTION UV STREACH
def __compute_loss_uv_streach(inputs, model, targets):
    area_coefficient = 1.0
    mse_area_loss = mse_area(area_coefficient)
    # outputs = model(inputs)
    encoder_input = NNDataset.get_encoder_input(inputs)
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


def loss_function_uv_streach(inputs, targets, model, loss_info):
    loss_uv_streach = __compute_loss_uv_streach(inputs, model, targets)
    loss_standard = loss_function_standard(inputs, targets, model, loss_info)

    return loss_uv_streach + loss_standard
# endregion


# region LOSS FUNCTION CENTERS
def __compute_center_distance_loss(input_points: torch.Tensor, decoded_points: torch.Tensor,
                                   closest_centers_indices_to_points: np.ndarray, centers_points_at_input_time: CentersInfo,
                                   centers_points_at_decoded_time: CentersInfo) -> torch.tensor:
    """
    :param input_points: tensor with 3 columns (x, y, z)
    :param decoded_points:  tensor with 3 columns (x, y, z)
    :param closest_centers_indices_to_points: indexes of closest centers to input_points
    :param centers_points_at_input_time: centers info in input_points time
    :param centers_points_at_decoded_time: centers info in decoded_points time
    :return:
    """

    # region INLINED FUNCTIONS
    def run_through_gaussian(input_centers_distances, decoded_centers_distances):
        # Create a normal distribution with mean and std of original distances
        mean_original = LOSS_FUNC_NORMAL_DIST_MEAN
        std_original = LOSS_FUNC_NORMAL_DIST_STD
        normal_dist = torch.distributions.Normal(mean_original, std_original)
        # Convert distances to probabilities
        prob_input = normal_dist.cdf(input_centers_distances)
        prob_decoded = normal_dist.cdf(decoded_centers_distances)
        return prob_decoded, prob_input

    # endregion

    # region SANITY CHECKS
    if len(input_points) != len(decoded_points):
        raise ValueError("Length of input_points and decoded_data must be the same")
    len_points = len(input_points)
    if len(closest_centers_indices_to_points) != len_points:
        raise ValueError("Length of closest_centers and input_points must be the same")

    if input_points.shape[1] != 3:
        raise ValueError("Input points must have 3 columns")
    if decoded_points.shape[1] != 3:
        raise ValueError("Decoded data must have 3 columns")

    # cloest_center_indices
    if closest_centers_indices_to_points is None:
        raise ValueError("Closest centers indices are empty.")
    if closest_centers_indices_to_points.shape[1] != CDataPreprocessing.NUM_CLOSEST_CENTERS_TO_POINT:
        raise ValueError("Closest centers indices must have 3 columns.")
    # endregion

    num_closest_centers = CDataPreprocessing.NUM_CLOSEST_CENTERS_TO_POINT

    # make tensor from list of lists to list with only one element
    closest_centers_indices_tensor = torch.tensor(closest_centers_indices_to_points, device=input_points.device)
    """ """

    all_input_centers_points = centers_points_at_input_time.points
    all_input_centers_points = torch.tensor(all_input_centers_points, device=input_points.device)
    input_centers_points = all_input_centers_points[closest_centers_indices_tensor]
    """ points of closest centers (with indexes closest_center_indices) to input_points in input time """

    all_decoded_centers_points = centers_points_at_decoded_time.points
    all_decoded_centers_points = torch.tensor(all_decoded_centers_points, device=input_points.device)
    decoded_centers_points = all_decoded_centers_points[closest_centers_indices_tensor]
    """ points of closest centers (with indexes closest_center_indices) to decoded_points in decoded time """

    if input_centers_points.shape != decoded_centers_points.shape:
        raise ValueError("Input centers points and decoded centers points must have the same number of columns")

    input_centers_distances = compute_distances_from_point_to_multiple_centers(points=input_points,
                                                                               closest_centers_points=input_centers_points)

    decoded_centers_distances = compute_distances_from_point_to_multiple_centers(points=decoded_points,
                                                                                 closest_centers_points=decoded_centers_points)

    # check
    if input_centers_distances.shape != decoded_centers_distances.shape:
        raise ValueError("Length of closest_centers and original_centers must be the same")

    # todo change
    # decoded_values, input_values = run_through_gaussian(input_centers_distances, decoded_centers_distances)

    input_values = input_centers_distances
    decoded_values = decoded_centers_distances
    # Compute the loss as the mean squared error between the probabilities
    loss = nn.MSELoss()(input_values, decoded_values)
    return loss


def __compute_loss_function_centers(inputs, model, loss_info : LossFunctionInfo) -> list:
    """

    :param inputs: where columns mean (x_value, y_value, z_value, time_value, time_index, point_index)
    :param model:
    :param loss_info:
    :return:
    """

    # region INLINE FUNCTIONS

    # def get_closest_centers_indicies(data_at_input_time : SurfacePointsFrame, inputs_points_index_column : list[int]):
    #
    #     input_all_labeled_points = data_at_input_time.normalized_labeled_points_list
    #     filtered_input_labeled_points = input_all_labeled_points.filter_by_points_indices(inputs_points_index_column)
    #
    #     closest_centers_points_list = filtered_input_labeled_points.get_closest_centers()
    #     closest_centers_indices = [element.get_centers_indices() for element in closest_centers_points_list]
    #     """ indexes of closest centers to input_points """
    #     if len(closest_centers_indices) != len(inputs_points_index_column):
    #         raise ValueError("Length of closest_centers_indices and input_points must be the same")
    #     return closest_centers_indices

    data: SurfacePointsFrameList = loss_info.data
    time_list = loss_info.time_list  # list of TimeFrame

    # region Decoded data
    decoder_time = np.random.choice(time_list)

    # VAR centers_points_at_decoded_time - get all center points from decoded time
    data_frame_at_decoded_time = data.get_element_by_time_index(decoder_time.index)
    centers_points_at_decoded_time = data_frame_at_decoded_time.normalized_centers_info

    # endregion

    unique_time_elements = NNDataset.get_unique_time_indices_list(inputs)
    loss_list = []
    for input_time_index in unique_time_elements:
        input_time_index = int(input_time_index)
        inputs_at_time = NNDataset.filter_by_time_index(inputs, input_time_index)

        # VAR inputs_points - get all points from inputs
        inputs_points = NNDataset.get_points_columns(inputs_at_time)

        # VAR decoded_points - decoded points from decoder_time
        decoded_points = run_through_nn_at_decoder_time(inputs_at_time, model, decoder_time)

        # VAR closest_centers_indices - get closest centers to input points
        inputs_points_index_column = NNDataset.get_point_indices_column(inputs_at_time)
        inputs_points_index_column = [int(element) for element in inputs_points_index_column]
        closest_centers_indices = loss_info.closest_centers_indicies_all_frames[input_time_index][inputs_points_index_column]

        # VAR input_centers_info - get all center points from input time
        data_frame_at_input_time = data.get_element_by_time_index(input_time_index)
        centers_points_at_input_time = data_frame_at_input_time.normalized_centers_info
        if not centers_points_at_input_time:
            raise ValueError("Input data must have centers info")


        loss_distances = __compute_center_distance_loss(input_points=inputs_points, decoded_points=decoded_points,
                                                        closest_centers_indices_to_points=closest_centers_indices,
                                                        centers_points_at_input_time=centers_points_at_input_time,
                                                        centers_points_at_decoded_time=centers_points_at_decoded_time)
        loss_list.append(loss_distances)

    return loss_list


def loss_function_centers(inputs, targets, model, loss_info):
    loss_centers_list = __compute_loss_function_centers(inputs, model, loss_info)
    loss_centers = torch.stack(loss_centers_list).mean()
    batch_size = inputs.size(0)
    loss_centers = loss_centers

    loss_standard = loss_function_standard(inputs, targets, model, loss_info)

    loss = loss_centers + loss_standard

    #logging.info(f"Center loss: {loss_centers}, Standard loss: {loss_standard}, Combined loss: {loss}")
    return loss
# endregion


# region LOSS FUNCTION PCA PREPROCESS

class LossFunctionPCAPrepocess():
    def __init__(self):
        pass

    def __compute_loss_pca_preprocess(self, inputs, targets, model, data : SurfacePointsFrameList):
        # #compute loss
        # loss = loss_function_standard(inputs, targets, model, loss_info)
        # return loss
        pass

    def __call__(self, inputs, targets, model, loss_info):
        data : SurfacePointsFrameList = loss_info.data_cluster
        loss = self.__compute_loss_pca_preprocess(inputs, targets, model, data)
        return loss

# endregion


LOSS_FUNCTIONS_LIST: dict[LossFunctionType: callable] = {
    LossFunctionType.STANDARD: loss_function_standard,
    LossFunctionType.CHAMFER: loss_function_chamfer,
    LossFunctionType.CHAMFER2: loss_function_chamfer_better_random_dist,
    LossFunctionType.UV_STREACH: loss_function_uv_streach,
    LossFunctionType.CENTERS: loss_function_centers,
}


def compute_distances_from_point_to_multiple_centers(points: torch.Tensor, closest_centers_points: torch.Tensor) -> torch.Tensor:
    # region SANITY CHECKS
    if points is None:
        raise AssertionError("Points are empty.")
    # check if points elements shape is (x,y,z) and it is float numbers
    if points.shape[1] != 3:
        raise AssertionError("Points must have 3 coordinates.")

    if closest_centers_points is None:
        raise AssertionError("Centers points are empty.")
    # check if centers_points elements shape is (x,y,z) and it is float numbers
    if closest_centers_points.shape[2] != 3:
        raise AssertionError("Centers points must have 3 coordinates.")

    if closest_centers_points.shape[0] != points.shape[0]:
        raise AssertionError("Centers points must have the same number of points")

    # endregion

    num_closest_centers = closest_centers_points.shape[1]

    # compute distances
    points_in_row = points.unsqueeze(1).repeat(1, num_closest_centers, 1).view(-1, 3)
    closest_centers_points_in_row = closest_centers_points.view(-1, 3)
    distances_tensor = torch.norm(points_in_row - closest_centers_points_in_row, dim=1)

    return distances_tensor
