#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File name: cmd_app.py
Author: Filip Cerny
Created: 02.05.2025
Version: 1.0
Description: 
"""
import logging
import os
from typing import List

import typer

from src.data_processing.clustering import process_clustered_data
from src.data_processing.mapping import process_surface_data
from src.nerual_network.evaluation.evaluation import save_centers_img_pipeline
from src.nerual_network.evaluation.meshes import process_mesh_through_model_pipeline, MeshDataVisualizer
from src.nerual_network.evaluation.metrics import compute_save_centers_metrics, compute_save_mesh_shape_metrics
from src.nerual_network.evaluation.visualization import _prepare_export_data, \
    _visualize_combined_surface_points_for_each_time, _visualize_all_clusters_for_each_time, \
    _visualize_points_with_time, _visualize_original_and_processed_points, _save_pointcloud_to_file, \
    visualize_uv_points_in_3d, _create_pointclouds_from_time_to_all_times
from src.nerual_network.helpers import load_trained_nn_from_files_code, MeshData, create_timestemp_dir, CentersMetricsInfo
from src.nerual_network.training import train_nn
from src.utils.constants import FilePathConfig, TrainConfig, CONFIG_JSON_FILENAME
from src.utils.helpers import init_logger, end_logger, load_pickle_file


def preprocess_data(train_config):
    process_clustered_data(train_config.num_clusters, train_config.file_path_config.raw_data_folderpath,
                           train_config.file_path_config.clustered_data_filepath, train_config.max_time_steps,
                           train_config.file_path_config.session_clustered_data_filepath)

    process_surface_data(train_config.num_surface_points, train_config.file_path_config.raw_data_folderpath,
                         train_config.file_path_config.surface_data_filepath,
                         train_config.file_path_config.session_clustered_data_filepath,
                         train_config.file_path_config.session_surface_data_filepath)


def train_pipeline(json_config_filepath: str, raw_data_folderpath: str, processed_folderpath: str):
    file_path_config = FilePathConfig.create_main(raw_data_folderpath, processed_folderpath)

    train_config = TrainConfig.from_json(json_config_filepath, file_path_config)

    train_config.save_to_json(
        os.path.join(train_config.file_path_config.processed_session_folderpath, CONFIG_JSON_FILENAME))

    logger = init_logger(train_config.file_path_config.log_filepath)

    data_foldername = train_config.file_path_config.raw_data_folderpath.split("/")[-1]
    logging.info("---------------------START OBJECT-------------------")
    logging.info(f"MAIN - Processing data for {data_foldername}")
    preprocess_data(train_config)
    logging.info(f"MAIN - Training neural network for {data_foldername}")
    train_nn(train_config)
    logging.info("---------------------END OBJECT-------------------")

    end_logger(logger)


def process_mesh_pipeline(processed_folderpath: str, evaluation_folderpath: str, mesh_time_index: int,
                          format: List[str]):
    logging.basicConfig(level=logging.INFO)

    file_path_config = FilePathConfig.create_main("", processed_folderpath)
    config_json_filepath = os.path.join(processed_folderpath, CONFIG_JSON_FILENAME)
    train_config = TrainConfig.from_json(config_json_filepath, file_path_config)

    surface_data_list = load_pickle_file(train_config.file_path_config.session_surface_data_filepath)

    if surface_data_list is None or surface_data_list.public_list is None:
        logging.error("Surface data list could not be loaded. Exiting.")
        return

    loaded_models = load_trained_nn_from_files_code(train_config)

    logging.info("START: Mesh though model")
    processed_data = process_mesh_through_model_pipeline(MeshData(time_index=mesh_time_index), train_config,
                                                         loaded_models)
    visualizer = MeshDataVisualizer(processed_data)
    mesh_files_folderpath = os.path.join(evaluation_folderpath, "mesh_files")
    os.makedirs(mesh_files_folderpath, exist_ok=True)

    for fmt in format:
        if fmt == "obj":
            visualizer.save_as_obj_file(mesh_files_folderpath)
            # Add logic to handle OBJ format
        elif fmt == "ply":
            visualizer.save_as_ply_file(mesh_files_folderpath)
            # Add logic to handle PLY format
        elif fmt == "png":
            visualizer.save_img_of_meshes(mesh_files_folderpath)
            # Add logic to handle PNG format
        else:
            typer.echo(f"Unknown format: {fmt}")

    logging.info("END: Mesh though model")


def evaluate_metrics_pipeline(processed_folderpath: str, evaluation_folderpath: str, metrics_center_eval_points: int,
                              metrics_mesh_shape_mesh_time_index: int):
    logging.basicConfig(level=logging.INFO)

    file_path_config = FilePathConfig.create_main("", processed_folderpath)
    config_json_filepath = os.path.join(processed_folderpath, CONFIG_JSON_FILENAME)
    train_config = TrainConfig.from_json(config_json_filepath, file_path_config)

    surface_data_list = load_pickle_file(train_config.file_path_config.session_surface_data_filepath)

    if surface_data_list is None or surface_data_list.public_list is None:
        logging.error("Surface data list could not be loaded. Exiting.")
        return

    loaded_models = load_trained_nn_from_files_code(train_config)

    # METRICS CENTERS

    logging.info(f"START: Metric - Centers for {metrics_center_eval_points} points")
    folder_path = os.path.join(evaluation_folderpath, "metric_centers")
    os.makedirs(folder_path, exist_ok=True)
    folder_path = create_timestemp_dir(folder_path)

    compute_save_centers_metrics(
        CentersMetricsInfo(surface_data_list, loaded_models, metrics_center_eval_points),
        folder_path)
    logging.info("END: Metric - Centers")

    # METRICS SIMILIAR SHAPE
    logging.info("START: Metric - Mesh shape")
    folderpath = os.path.join(evaluation_folderpath, "metric_mesh_shape")
    os.makedirs(folderpath, exist_ok=True)
    folderpath = create_timestemp_dir(folderpath)

    compute_save_mesh_shape_metrics(folderpath, surface_data_list, train_config, loaded_models,
                                    metrics_mesh_shape_mesh_time_index)
    logging.info("END: Metric - Mesh shape")


def visualize_pipeline(processed_folderpath: str, evaluation_folderpath: str):
    logging.basicConfig(level=logging.INFO)

    file_path_config = FilePathConfig.create_main("", processed_folderpath)
    config_json_filepath = os.path.join(processed_folderpath, CONFIG_JSON_FILENAME)
    train_config = TrainConfig.from_json(config_json_filepath, file_path_config)

    surface_data_list = load_pickle_file(train_config.file_path_config.session_surface_data_filepath)

    if surface_data_list is None or surface_data_list.public_list is None:
        logging.error("Surface data list could not be loaded. Exiting.")
        return

    loaded_models = load_trained_nn_from_files_code(train_config)


    save_centers_img_pipeline(evaluation_folderpath, surface_data_list)

    # region Bundle
    original_points_all, processed_points_all = _prepare_export_data(
        surface_data_list=surface_data_list, loaded_models=loaded_models)

    _visualize_combined_surface_points_for_each_time(original_points_all, processed_points_all,
                                                     os.path.join(evaluation_folderpath,
                                                                  "img_processed_points"))
    _visualize_all_clusters_for_each_time(surface_data_list, os.path.join(evaluation_folderpath, "img_clusters"))

    _visualize_points_with_time(original_points_all, processed_points_all, evaluation_folderpath)
    # Save the combined image
    _visualize_original_and_processed_points(original_points_all, processed_points_all, evaluation_folderpath)

    point_cloud_original_filepath = train_config.file_path_config.point_cloud_original_filepath
    point_cloud_processed_filepath = train_config.file_path_config.point_cloud_processed_filepath
    _save_pointcloud_to_file(original_points_all, processed_points_all, point_cloud_original_filepath,
                             point_cloud_processed_filepath)
    # endregion

    visualize_uv_points_in_3d(surface_data_list=surface_data_list,
                              images_save_folderpath=os.path.join(evaluation_folderpath, "img_time_uv_points_0"),
                              time_index=0, loaded_models=loaded_models, modulo=1)

    _create_pointclouds_from_time_to_all_times(surface_data_list=surface_data_list,
                                               images_save_folderpath=os.path.join(evaluation_folderpath,
                                                                                   "point_clouds_all_times_from_time_0"),
                                               time_index=0, loaded_models=loaded_models)
