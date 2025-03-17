#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File name: process_mesh_throug_model.py
Author: Filip Cerny
Created: 10.03.2025
Version: 1.0
Description: 
"""
import copy
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import TypeAlias

import numpy as np
import trimesh
from trimesh import Trimesh

from data_processing.class_clustering import ClusteredCenterPointsAllFrames
from data_processing.class_mapping import SurfacePointsFrame, TimeFrame, SurfacePointsFrameList
from data_processing.mapping import categorize_points_with_labels
from nerual_network.evaluation.evaluation import _create_pointclouds_from_time_to_all_times, \
    _run_model_with_one_encoder_time_to_all_decoder_times_prepare_for_visualization
from utils.constants import FilePathConfig, DEFAULT_TRAIN_CONFIG, TrainConfig
from utils.helpers import load_pickle_file

DATA_FOLDERNAME: str = 'ball_test'
SESSION_FOLDERNAME: str = "ball_test_20250309_234312"

MESH_TIME_INDEX: int = 0
#MESH_FILENAME: str = "ball000.obj"

OUTPUT_folderpath: str = "out"


# create data class for mesh where is mesh time index and mesh file path
@dataclass
class MeshData:
    time_index: int

RGBColorArray : TypeAlias = np.ndarray
ProcessedPointsListSplitByTimeValue : TypeAlias = list[np.ndarray]

MeshStruct : TypeAlias = Trimesh

Folderpath : TypeAlias = str

@dataclass
class NNOutputForVisualization:
    rgb_colors: RGBColorArray
    processed_points: ProcessedPointsListSplitByTimeValue


class DataVisualizer:
    def __init__(self, processed_data: NNOutputForVisualization):
        self.processed_data = processed_data

    @staticmethod
    def create_dir(output_folderpath: Folderpath) -> Folderpath:
        # create output folder if not exists
        # add current time to folder name
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folderpath = os.path.join(output_folderpath, f"mesh_{current_time_str}")
        os.makedirs(output_folderpath, exist_ok=True)
        return output_folderpath

    def save_as_pointcloud_to_file(self, save_folderpath : str):
        rgb_colors = self.processed_data.rgb_colors
        processed_points_split_by_time_value = self.processed_data.processed_points

        # make dir if not made
        save_folderpath = self.create_dir(save_folderpath)

        # Save the RGB values to another file
        rgb_colors_filepath = os.path.join(save_folderpath, 'rgb_colors.txt')
        np.savetxt(rgb_colors_filepath, rgb_colors, delimiter=",")

        logging.info(f"Saved RGB colors to {rgb_colors_filepath}")
        # todo finish - add denormalization

        for i, processed_points_one_time_value in enumerate(processed_points_split_by_time_value):
            processed_points_filepath = os.path.join(save_folderpath, f'processed_points_{i}.xyz')
            np.savetxt(processed_points_filepath, processed_points_one_time_value, delimiter=",")
            logging.info(f"Saved processed points to {processed_points_filepath}")

    def save_as_meshes_to_file(self, save_folderpath: str, origin_mesh : MeshStruct):
        processed_points_split_by_time_value = self.processed_data.processed_points

        # make dir if not made
        save_folderpath = self.create_dir(save_folderpath)

        for i, processed_points_one_time_value in enumerate(processed_points_split_by_time_value):
            processed_points_filepath = os.path.join(save_folderpath, f'processed_points_{i}.obj')
            origin_mesh_filepath = os.path.join(save_folderpath, f'origin_mesh_{i}.obj')

            # check if origin_mesh vertices is the same number of points as processed points
            if len(origin_mesh.vertices) != len(processed_points_one_time_value):
                raise ValueError(f"Number of vertices in origin mesh is not the same as number of processed points. Origin mesh vertices: {len(origin_mesh.vertices)}, processed points: {len(processed_points_one_time_value)}")

            # test_origin_mesh_verticies = np.random.rand(len(origin_mesh.vertices), 3)
            #
            # test_origin_mesh = trimesh.Trimesh(vertices=test_origin_mesh_verticies, faces=origin_mesh.faces)
            # test_origin_mesh.export(origin_mesh_filepath)

            mesh = trimesh.Trimesh(vertices=processed_points_one_time_value, faces=origin_mesh.faces)
            mesh.export(processed_points_filepath)
            logging.info(f"Saved processed points to {processed_points_filepath}")


def __save_pointcloud_to_file(processed_data : NNOutputForVisualization, images_save_folderpath : str):
    rgb_colors = processed_data.rgb_colors
    processed_points_split_by_time_value = processed_data.processed_points

    # make dir if not made
    os.makedirs(images_save_folderpath, exist_ok=True)

    # Save the RGB values to another file
    rgb_colors_filepath = os.path.join(images_save_folderpath, 'rgb_colors.txt')
    np.savetxt(rgb_colors_filepath, rgb_colors, delimiter=",")

    logging.info(f"Saved RGB colors to {rgb_colors_filepath}")
    # todo finish - add denormalization

    for i, processed_points_one_time_value in enumerate(processed_points_split_by_time_value):
        processed_points_filepath = os.path.join(images_save_folderpath, f'processed_points_{i}.xyz')
        np.savetxt(processed_points_filepath, processed_points_one_time_value, delimiter=",")
        logging.info(f"Saved processed points to {processed_points_filepath}")

@dataclass
class ProcessedMeshData:
    processed_data: NNOutputForVisualization
    origin_mesh: MeshStruct

# def process_mesh_through_model(mesh_filepath, train_config, output_filepath):
def process_mesh_through_model(origin_mesh_data: MeshData, train_config: TrainConfig) -> ProcessedMeshData | None:
    """
    Function to process mesh through model
    :param origin_mesh_data:
    :param train_config:
    :return:
    """



    # region PREPARE MESH FOR MODEL
    # region STEP Read clustered data, surface data
    clustered_data: ClusteredCenterPointsAllFrames = load_pickle_file(
        train_config.file_path_config.clustered_data_filepath)
    if clustered_data is None:
        logging.error("Clustered data could not be loaded. Exiting.")
        return None

    surface_data_list: SurfacePointsFrameList = load_pickle_file(train_config.file_path_config.surface_data_filepath)
    if surface_data_list is None or surface_data_list.list is None:
        logging.error("Surface data list could not be loaded. Exiting.")
        return None
    # endregion

    # region STEP Create Surface data from this

    # create surface data list where input vertices are meshes vertices and they are clustered by labels

    original_loaded_meshes = surface_data_list.get_original_meshes_list()

    mesh_surface_points_frame_list = SurfacePointsFrameList([])

    for original_mesh in original_loaded_meshes:
        time_index = original_mesh[0]
        mesh = original_mesh[1]

        mesh_vertices = np.array(mesh.vertices)
        mesh_faces = np.array(mesh.faces)

        ## Categorize points
        centers_labels_frame = clustered_data.labels_frame
        centers_points_frame = clustered_data.points_allframes[time_index]
        labels = categorize_points_with_labels(centers_labels_frame, centers_points_frame, mesh_vertices)

        ## Create Surface data
        mesh_surface_points_frame = SurfacePointsFrame.create_instance(surface_points=mesh_vertices,
                                                                       surface_labels=labels,
                                                                       time=None, mesh=mesh,
                                                                       centers_points=centers_points_frame)

        ## region get time value
        surface_data_frame = surface_data_list.get_element_by_time_index(time_index)
        if surface_data_frame is None:
            logging.error(f"Surface data frame for time index {time_index} could not be found. Exiting.")
            return

        time_value = surface_data_frame.time.value
        ## endregion

        mesh_surface_points_frame.time = TimeFrame(index=time_index, value=time_value)
        # endregion

        mesh_surface_points_frame_list.append(mesh_surface_points_frame)

    # endregion

    # region PROCESS MESH THROUGH MODEL

    # deep copy of mesh_surface_points_frame_list
    normalized_mesh_surface_points_frame_list: SurfacePointsFrameList = copy.deepcopy(mesh_surface_points_frame_list)
    normalized_mesh_surface_points_frame_list.normalize_labeled_points_by_values(surface_data_list.normalize_values)
    input_model_data: SurfacePointsFrameList = normalized_mesh_surface_points_frame_list

    rgb_colors, processed_points_split_by_time_value = _run_model_with_one_encoder_time_to_all_decoder_times_prepare_for_visualization(
        surface_data_list=input_model_data, time_index=origin_mesh_data.time_index, train_config=train_config)

    # _create_pointclouds_from_time_to_all_times(surface_data_list=input_model_data,
    #                                            images_save_folderpath=os.path.join(output_folderpath,
    #                                                                                "point_clouds_all_times"),
    #                                            time_index=origin_mesh_data.time_index, train_config=train_config)

    denormalized_points_split_by_time_value = []
    for i, processed_points_one_time_value in enumerate(processed_points_split_by_time_value):
        denormalized_points = SurfacePointsFrameList.denormalize_points(surface_data_list.normalize_values, processed_points_one_time_value)
        denormalized_points_split_by_time_value.append(denormalized_points)

    origin_mesh = original_loaded_meshes[origin_mesh_data.time_index]
    if origin_mesh[0] != origin_mesh_data.time_index:
        raise ValueError("Time index of origin mesh does not match the time index of the mesh data")
    return ProcessedMeshData(NNOutputForVisualization(rgb_colors=rgb_colors, processed_points=denormalized_points_split_by_time_value), origin_mesh[1])


def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = "data"
    processed_folder = "processed"
    raw_folder = "raw"

    processed_session_folderpath = os.path.join(data_folder, processed_folder, DATA_FOLDERNAME, SESSION_FOLDERNAME)
    raw_folderpath = os.path.join(data_folder, raw_folder)
    processed_folderpath = os.path.join(data_folder, processed_folder)

    train_config = DEFAULT_TRAIN_CONFIG
    train_config.file_path_config = FilePathConfig(
        data_foldername=DATA_FOLDERNAME,
        processed_session_folderpath=processed_session_folderpath,
        raw_folderpath=raw_folderpath,
        processed_folderpath=processed_folderpath
    )

    folder_name = "test"
    output_folderpath = os.path.join(OUTPUT_folderpath, folder_name)

    #mesh_filepath = os.path.join(train_config.file_path_config.raw_data_folderpath, MESH_FILENAME)

    processed_data = process_mesh_through_model(MeshData(time_index=MESH_TIME_INDEX), train_config)

    visualizer = DataVisualizer(processed_data.processed_data)
    visualizer.save_as_meshes_to_file(output_folderpath, processed_data.origin_mesh)


if __name__ == '__main__':
    main()
