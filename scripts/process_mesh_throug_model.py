#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File name: process_mesh_throug_model.py
Author: Filip Cerny
Created: 10.03.2025
Version: 1.0
Description: 
"""
import logging
import os

from nerual_network.evaluation.meshes import MeshDataVisualizer, process_mesh_through_model_pipeline
from nerual_network.helpers import load_trained_nn_from_files, MeshData
from utils.constants import FilePathConfig, DEFAULT_TRAIN_CONFIG

DATA_FOLDERNAME: str = 'ball_test'
SESSION_FOLDERNAME: str = "ball_test_20250309_234312"

MESH_TIME_INDEX: int = 0
#MESH_FILENAME: str = "ball000.obj"

OUTPUT_folderpath: str = "out"


# create data class for mesh where is mesh time index and mesh file path


# def process_mesh_through_model(mesh_filepath, train_config, output_filepath):


def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = "data"
    processed_folder = "processed"
    raw_folder = "raw"

    processed_session_folderpath = os.path.join(data_folder, processed_folder, DATA_FOLDERNAME, SESSION_FOLDERNAME)
    raw_folderpath = os.path.join(data_folder, raw_folder)
    processed_folderpath = os.path.join(data_folder, processed_folder)

    train_config = DEFAULT_TRAIN_CONFIG
    train_config.file_path_config = FilePathConfig.create_test_mode(
        data_foldername=DATA_FOLDERNAME,
        processed_session_folderpath=processed_session_folderpath,
        raw_folderpath=raw_folderpath,
        processed_folderpath=processed_folderpath
    )

    folder_name = "test"
    output_folderpath = os.path.join(OUTPUT_folderpath, folder_name)

    #mesh_filepath = os.path.join(train_config.file_path_config.raw_data_folderpath, MESH_FILENAME)

    loaded_models = load_trained_nn_from_files(train_config)

    processed_data = process_mesh_through_model_pipeline(MeshData(time_index=MESH_TIME_INDEX), train_config, loaded_models)

    visualizer = MeshDataVisualizer(processed_data)
    visualizer.save_as_obj_file(output_folderpath)


if __name__ == '__main__':
    main()
