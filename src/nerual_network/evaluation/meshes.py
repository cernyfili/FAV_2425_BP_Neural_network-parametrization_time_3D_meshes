#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File name: meshes.py
Author: Filip Cerny
Created: 17.04.2025
Version: 1.0
Description: 
"""
import copy
import logging
import os
from typing import TypeAlias
import numpy as np
import trimesh
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from trimesh import Trimesh

from data_processing.class_clustering import ClusteredCenterPointsAllFrames
from data_processing.class_mapping import SurfacePointsFrameList, SurfacePointsFrame, TimeFrame
from data_processing.mapping import categorize_points_with_labels

from nerual_network.helpers import ProcessedMeshData, MeshData, MeshFilepathsDic, \
    _run_model_with_one_encoder_time_to_all_decoder_times_prepare_for_visualization, \
    ProcessedPointsListSplitByTimeValue, NNOutputForVisualization, create_timestemp_dir, FilePath, TimeIndex, \
    LoadedModelDic
from utils.constants import TrainConfig
from utils.helpers import load_pickle_file


TriMeshDict : TypeAlias = dict[TimeIndex, Trimesh]


class MeshDataVisualizer:
    def __init__(self, processed_data: ProcessedMeshData):
        self.processed_mesh_data : ProcessedMeshData = processed_data


    def save_as_pointcloud_to_file(self, save_folderpath : str):
        rgb_colors = self.processed_mesh_data.processed_visualization_data.rgb_colors
        processed_points_split_by_time_value = self.processed_mesh_data.processed_visualization_data.processed_points

        # make dir if not made
        save_folderpath = create_timestemp_dir(save_folderpath)

        # Save the RGB values to another file
        rgb_colors_filepath = os.path.join(save_folderpath, 'rgb_colors.txt')
        np.savetxt(rgb_colors_filepath, rgb_colors, delimiter=",")

        logging.info(f"Saved RGB colors to {rgb_colors_filepath}")
        # todo finish - add denormalization

        for i, processed_points_one_time_value in processed_points_split_by_time_value.items():
            processed_points_filepath = os.path.join(save_folderpath, f'processed_points_{i}.xyz')
            np.savetxt(processed_points_filepath, processed_points_one_time_value, delimiter=",")
            logging.info(f"Saved processed points to {processed_points_filepath}")

    def _get_trimesh_dict(self) -> TriMeshDict:
        processed_points_split_by_time_value = self.processed_mesh_data.processed_visualization_data.processed_points
        rgb_colors = self.processed_mesh_data.processed_visualization_data.rgb_colors
        rgb_colors = (rgb_colors * 255).astype(np.uint8)

        origin_mesh = self.processed_mesh_data.origin_mesh

        trimesh_dict = dict()

        for time_index, processed_points_one_time_value in processed_points_split_by_time_value.items():

            # check if origin_mesh vertices is the same number of points as processed points
            if len(origin_mesh.vertices) != len(processed_points_one_time_value):
                raise ValueError(f"Number of vertices in origin mesh is not the same as number of processed points. Origin mesh vertices: {len(origin_mesh.vertices)}, processed points: {len(processed_points_one_time_value)}")

            mesh = trimesh.Trimesh(vertices=processed_points_one_time_value, faces=origin_mesh.faces, vertex_colors=rgb_colors)
            trimesh_dict[time_index] = mesh

        return trimesh_dict


    def save_as_obj_file(self, save_folderpath: str) -> MeshFilepathsDic:
        trimesh_dict = self._get_trimesh_dict()

        filepaths_dict = MeshFilepathsDic()
        for time_index, mesh in trimesh_dict.items():
            processed_points_filepath = os.path.join(save_folderpath, f'processed_mesh_{time_index}.obj')

            filepaths_dict[TimeIndex(time_index)] = FilePath(processed_points_filepath)
            mesh.export(processed_points_filepath)
            logging.info(f"Saved processed points to {processed_points_filepath}")

        return filepaths_dict

    def save_as_ply_file(self, save_folderpath: str):
        trimesh_dict = self._get_trimesh_dict()

        # make dir if not made
        save_folderpath = create_timestemp_dir(save_folderpath)

        for time_index, mesh in trimesh_dict.items():
            processed_points_filepath = os.path.join(save_folderpath, f'processed_points_{time_index}.ply')

            mesh.export(processed_points_filepath)
            logging.info(f"Saved processed points to {processed_points_filepath}")

    def save_img_of_meshes(self, save_folderpath: str):
        trimesh_dict = self._get_trimesh_dict()

        # make dir if not made
        save_folderpath = create_timestemp_dir(save_folderpath)

        rgb_colors = self.processed_mesh_data.processed_visualization_data.rgb_colors

        # find min and max in all trimesh points
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        for time_index, mesh in trimesh_dict.items():
            vertices = mesh.vertices
            min_x = min(min_x, vertices[:, 0].min())
            max_x = max(max_x, vertices[:, 0].max())

            min_y = min(min_y, vertices[:, 1].min())
            max_y = max(max_y, vertices[:, 1].max())

            min_z = min(min_z, vertices[:, 2].min())
            max_z = max(max_z, vertices[:, 2].max())

        min_axis = min(min_x, min_y, min_z)
        max_axis = max(max_x, max_y, max_z)


        for time_index, mesh in trimesh_dict.items():
            processed_points_filepath = os.path.join(save_folderpath, f'processed_points_{time_index}.png')

            # Extract vertices and faces
            vertices = mesh.vertices
            faces = mesh.faces
            # Save the plot to a PNG file
            # Create a basic plot
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            # Apply face colors
            if mesh.visual.kind == 'vertex':
                # Ensure colors are normalized and include alpha
                colors_vertices = mesh.visual.vertex_colors
                if colors_vertices.max() > 1.0:  # Check if colors are in 0-255 range
                    colors_vertices = colors_vertices / 255.0
            else:
                raise ValueError("Mesh does not have vertex colors. Please provide a color map or use a different method to visualize the mesh.")

            # Create a Poly3DCollection for the mesh
            length = len(vertices[faces])
            face_colors_red = [(1.0, 0.0, 0.0)] * length
            face_colors = np.mean(colors_vertices[faces], axis=1)

            poly_collection = Poly3DCollection(vertices[faces], facecolors=face_colors, edgecolor='none', linewidths=0.2)
            ax.add_collection3d(poly_collection)

            # Set axis limits
            ax.set_xlim(min_axis, max_axis)
            ax.set_ylim(min_axis, max_axis)
            ax.set_zlim(min_axis, max_axis)

            # Show the plot
            # Save the plot to a PNG file
            plt.savefig(processed_points_filepath, format='png', dpi=300)
            plt.close()
            logging.info(f"Saved processed points to {processed_points_filepath}")

    # def save_img_of_meshes_with_ply(self, save_folderpath: str):
    #     trimesh_dict = self._get_trimesh_dict()
    #
    #     # make dir if not made
    #     save_folderpath = create_timestemp_dir(save_folderpath)
    #
    #     for time_index, mesh in trimesh_dict.items():
    #         processed_points_filepath = os.path.join(save_folderpath, f'processed_points_{time_index}.ply')
    #         img_filepath = os.path.join(save_folderpath, f'processed_points_{time_index}.png')
    #
    #         mesh.export(processed_points_filepath)
    #         logging.info(f"Saved processed points to {processed_points_filepath}")
    #
    #         mesh = o3d.io.read_triangle_mesh(processed_points_filepath)
    #         if not mesh.has_vertex_normals():
    #             mesh.compute_vertex_normals()
    #
    #         # Set up the offscreen renderer
    #         width, height = 1024, 768
    #         renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    #
    #         # Set background color (white RGBA)
    #         renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    #
    #         # Set up material
    #         material = o3d.visualization.rendering.MaterialRecord()
    #         material.shader = "defaultLit"
    #
    #         # Add mesh to scene
    #         renderer.scene.add_geometry("mesh", mesh, material)
    #
    #         # Center the camera on the mesh
    #         bbox = mesh.get_axis_aligned_bounding_box()
    #         renderer.setup_camera(60.0, bbox, bbox.get_center())
    #
    #         # Render to image
    #         image = renderer.render_to_image()
    #
    #         # Save the image
    #         o3d.io.write_image(img_filepath, image)
    #         print("Saved image to rendered_mesh.png")


def _create_mesh_surfacedatalist(clustered_data : ClusteredCenterPointsAllFrames, surface_data_list : SurfacePointsFrameList) -> SurfacePointsFrameList:
    logging.info("creating mesh surfacedatalist")
    original_loaded_meshes = surface_data_list.get_original_meshes_list()

    mesh_surface_points_frame_list = SurfacePointsFrameList([])

    for original_mesh in original_loaded_meshes:
        time_index = original_mesh[0]
        mesh = original_mesh[1]

        mesh_vertices = np.array(mesh.vertices)

        ## Categorize points
        centers_labels_frame = clustered_data.labels_frame
        centers_points_frame = clustered_data.points_allframes[time_index]
        labels = categorize_points_with_labels(centers_labels_frame, centers_points_frame, mesh_vertices)

        ## Create Surface data
        mesh_surface_points_frame = SurfacePointsFrame.create_instance(surface_points=mesh_vertices,
                                                                       surface_labels=labels, mesh=mesh,
                                                                       centers_points=centers_points_frame)

        ## region get time value
        surface_data_frame = surface_data_list.get_element_by_time_index(time_index)
        if surface_data_frame is None:
            logging.error(f"Surface data frame for time index {time_index} could not be found. Exiting.")
            raise ValueError(f"Surface data frame for time index {time_index} could not be found. Exiting.")

        time_value = surface_data_frame.time.value
        ## endregion

        mesh_surface_points_frame.time = TimeFrame(index=time_index, value=time_value)
        # endregion

        mesh_surface_points_frame_list.append(mesh_surface_points_frame)

    return mesh_surface_points_frame_list


def process_mesh_through_model_pipeline(origin_mesh_data: MeshData, train_config: TrainConfig,
                                        loaded_models : LoadedModelDic) -> ProcessedMeshData | None:
    # region STEP Read clustered data, surface data
    clustered_data: ClusteredCenterPointsAllFrames = load_pickle_file(
        train_config.file_path_config.session_clustered_data_filepath)
    if clustered_data is None:
        logging.error("Clustered data could not be loaded. Exiting.")
        return None

    surface_data_list: SurfacePointsFrameList = load_pickle_file(train_config.file_path_config.session_surface_data_filepath)
    if surface_data_list is None or surface_data_list.public_list is None:
        logging.error("Surface data list could not be loaded. Exiting.")
        return None
    # endregion

    return process_mesh_through_model(origin_mesh_data=origin_mesh_data, loaded_models=loaded_models, clustered_data=clustered_data,
                                      surface_data_list=surface_data_list)


def process_mesh_through_model(origin_mesh_data: MeshData, loaded_models : LoadedModelDic, clustered_data : ClusteredCenterPointsAllFrames, surface_data_list: SurfacePointsFrameList) -> ProcessedMeshData | None:
    """
    Function to process mesh through model
    :param surface_data_list:
    :param clustered_data:
    :param loaded_models:
    :param origin_mesh_data:
    :return:
    """

    logging.info("Processing mesh through model")

    # region PREPARE MESH FOR MODEL


    # region STEP Create Surface data from this

    # create surface data list where input vertices are meshes vertices and they are clustered by labels

    mesh_surface_points_frame_list: SurfacePointsFrameList = _create_mesh_surfacedatalist(clustered_data=
        clustered_data, surface_data_list=surface_data_list)

    # endregion

    # region PROCESS MESH THROUGH MODEL

    # deep copy of mesh_surface_points_frame_list
    mesh_surface_points_frame_list.normalize_labeled_points_by_values(surface_data_list.normalize_values)

    visualization_data = _run_model_with_one_encoder_time_to_all_decoder_times_prepare_for_visualization(
        surface_data_list=mesh_surface_points_frame_list, time_index=origin_mesh_data.time_index, loaded_models=loaded_models)
    processed_points_split_by_time_value = visualization_data.processed_points


    # _create_pointclouds_from_time_to_all_times(surface_data_list=input_model_data,
    #                                            images_save_folderpath=os.path.join(output_folderpath,
    #                                                                                "point_clouds_all_times"),
    #                                            time_index=origin_mesh_data.time_index, train_config=train_config)

    logging.info("Denormalizing points")
    denormalized_points_split_by_time_value : ProcessedPointsListSplitByTimeValue = dict()
    for time_index, processed_points_one_time_value in processed_points_split_by_time_value.items():
        denormalized_points = SurfacePointsFrameList.denormalize_points(surface_data_list.normalize_values, processed_points_one_time_value)
        denormalized_points_split_by_time_value[time_index] = denormalized_points

    origin_mesh = mesh_surface_points_frame_list.get_element_by_time_index(origin_mesh_data.time_index).original_mesh
    return ProcessedMeshData(
        NNOutputForVisualization(rgb_colors=visualization_data.rgb_colors, processed_points=denormalized_points_split_by_time_value), origin_mesh)
