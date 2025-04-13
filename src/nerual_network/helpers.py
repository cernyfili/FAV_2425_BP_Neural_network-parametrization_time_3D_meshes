#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File name: helpers.py
Author: Filip Cerny
Created: 10.04.2025
Version: 1.0
Description: 
"""
import numpy as np
import torch

from nerual_network.class_model import NNDataset


def get_closest_centers_indices(closest_centers_indicies_all_frames : np.ndarray, inputs : torch.tensor, input_time_index : int) -> np.ndarray:
    inputs_points_index_column = NNDataset.get_point_indices_column(inputs)
    inputs_points_index_column = [int(element) for element in inputs_points_index_column]
    closest_centers_indices = closest_centers_indicies_all_frames[input_time_index][inputs_points_index_column]

    return closest_centers_indices