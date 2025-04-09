#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File name: class_training.py
Author: Filip Cerny
Created: 09.04.2025
Version: 1.0
Description: 
"""
from dataclasses import dataclass

import numpy as np

from data_processing.class_mapping import MeshList, TimeFrame, SurfacePointsFrameList

import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class LossFunctionInfo:
    """Class to hold information about a loss function."""
    meshes_list: MeshList | None = None
    device: any = None
    time_list: list[TimeFrame] | None = None
    data_cluster: SurfacePointsFrameList | None = None
    data : SurfacePointsFrameList | None = None
    closest_centers_indicies_all_frames : np.ndarray | None = None
    """in the shape of (number_of_frames, number_of_points, number_of_closest_centers) int"""


