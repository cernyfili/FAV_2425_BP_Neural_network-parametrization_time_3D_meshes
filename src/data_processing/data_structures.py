#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File name: modes.py
Author: Filip Cerny
Created: 16.01.2025
Version: 1.0
Description: 
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class MeshNDArray:
    vertices: np.ndarray
    faces: np.ndarray