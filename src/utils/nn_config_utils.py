#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File name: nn_config_utils.py
Author: Filip Cerny
Created: 06.01.2025
Version: 1.0
Description: 
"""

from torch import optim, nn

from nerual_network.class_model import Simple_MLP_04
from nerual_network.loss_functions import LOSS_FUNCTIONS_LIST
from utils.constants import TrainConfig


# Configuration function to initialize model, optimizer, and criterion
def init_training_config(train_config: TrainConfig) -> (nn.Module, optim.Optimizer, nn.Module):
    nn_lr = train_config.nn_config.nn_lr
    loss_function_name = train_config.nn_config.loss_function_name

    model = Simple_MLP_04()
    optimizer = optim.Adam(model.parameters(), lr=nn_lr)
    loss_function = LOSS_FUNCTIONS_LIST[loss_function_name]
    return model, optimizer, loss_function
