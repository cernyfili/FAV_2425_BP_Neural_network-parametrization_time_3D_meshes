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

from nerual_network.class_model import Simple_MLP_04, Simple_MLP_01, Simple_MLP_02
from nerual_network.loss_functions import LOSS_FUNCTIONS_LIST
from nerual_network.class_model import MODELS_LIST
from utils.constants import TrainConfig, ModelType


# Configuration function to initialize model, optimizer, and criterion
def init_training_config(train_config: TrainConfig) -> (nn.Module, optim.Optimizer, nn.Module):
    nn_lr = train_config.nn_config.nn_lr
    loss_function_type = train_config.nn_config.loss_function_type
    model_type = train_config.nn_config.model_type

    model = init_model(model_type)
    optimizer = optim.Adam(model.parameters(), lr=nn_lr)
    loss_function = LOSS_FUNCTIONS_LIST[loss_function_type]
    return model, optimizer, loss_function

def init_model(model_type: ModelType) -> nn.Module:
    """
    Initialize the model based on the model type.
    """
    return MODELS_LIST[model_type]()
