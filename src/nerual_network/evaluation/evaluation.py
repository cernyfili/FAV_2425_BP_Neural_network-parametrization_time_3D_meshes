import logging
import os

from src.data_processing.class_mapping import SurfacePointsFrameList
from src.nerual_network.evaluation.meshes import process_mesh_through_model_pipeline, MeshDataVisualizer
from src.nerual_network.evaluation.metrics import compute_save_centers_metrics, compute_save_mesh_shape_metrics
from src.nerual_network.evaluation.visualization import _visualize_all_clusters_for_each_time, \
    _visualize_combined_surface_points_for_each_time, _visualize_points_with_time, \
    _visualize_original_and_processed_points, _prepare_export_data, visualize_uv_points_in_3d, save_visualize_centers, \
    _save_pointcloud_to_file, _create_pointclouds_from_time_to_all_times
from src.nerual_network.helpers import load_trained_nn_from_files_code, MeshData, CentersMetricsInfo, create_timestemp_dir, \
    LoadedModelDic
from src.utils.constants import TrainConfig
from src.utils.helpers import load_pickle_file


# Restrict access to underscore-prefixed functions
def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"{name} is a private function and cannot be imported.")
    raise AttributeError(f"Module has no attribute {name}")


# create an alias type for an model dictionary where key is cluster index
def evaluate(train_config: TrainConfig):
    """
    Evaluates the model by loading the trained model and running it through the data.
    :param train_config:
    :return:
    """
    evaluate_partial(train_config)
    #evaluate_full(train_config)

def evaluate_partial(train_config: TrainConfig):
    surface_data_list = load_pickle_file(train_config.file_path_config.session_surface_data_filepath)

    if surface_data_list is None or surface_data_list.public_list is None:
        logging.error("Surface data list could not be loaded. Exiting.")
        return

    evaluation_folderpath = train_config.file_path_config.evaluation_folderpath

    loaded_models = load_trained_nn_from_files_code(train_config)

    save_metric_mesh_shape_pipeline(evaluation_folderpath=evaluation_folderpath,
                                    surface_data_list=surface_data_list,
                                    train_config=train_config,
                                    loaded_models=loaded_models,
                                    mesh_time_index=0)

    save_mesh_thrugh_model_pipeline(evaluation_folderpath, loaded_models, train_config, 0)

    #save_metrics_centers_pipeline(evaluation_folderpath, loaded_models, surface_data_list)

def evaluate_partial_2(train_config: TrainConfig):
    surface_data_list = load_pickle_file(train_config.file_path_config.session_surface_data_filepath)

    if surface_data_list is None or surface_data_list.public_list is None:
        logging.error("Surface data list could not be loaded. Exiting.")
        return

    evaluation_folderpath = train_config.file_path_config.evaluation_folderpath

    loaded_models = load_trained_nn_from_files_code(train_config)

    save_metrics_centers_pipeline(evaluation_folderpath, loaded_models, surface_data_list)



def evaluate_full(train_config: TrainConfig):
    surface_data_list = load_pickle_file(train_config.file_path_config.session_surface_data_filepath)

    if surface_data_list is None or surface_data_list.public_list is None:
        logging.error("Surface data list could not be loaded. Exiting.")
        return

    evaluation_folderpath = train_config.file_path_config.evaluation_folderpath

    loaded_models = load_trained_nn_from_files_code(train_config)


    # region Save Evaluation files
    save_mesh_thrugh_model_pipeline(evaluation_folderpath, loaded_models, train_config, 0)

    save_centers_img_pipeline(evaluation_folderpath, surface_data_list)

    # region Bundle
    original_points_all, processed_points_all = _prepare_export_data(
         surface_data_list=surface_data_list, loaded_models=loaded_models)

    _visualize_combined_surface_points_for_each_time(original_points_all, processed_points_all,
                                                     os.path.join(evaluation_folderpath,
                                                                  "img_time_combined_only_processed"))
    _visualize_all_clusters_for_each_time(surface_data_list, os.path.join(evaluation_folderpath, "img_time_clusters"))

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
                              time_index=0, loaded_models=loaded_models, modulo=5)

    visualize_uv_points_in_3d(surface_data_list=surface_data_list,
                              images_save_folderpath=os.path.join(evaluation_folderpath, "img_time_uv_points_59"),
                              time_index=59, loaded_models=loaded_models, modulo=5)

    _create_pointclouds_from_time_to_all_times(surface_data_list=surface_data_list,
                                               images_save_folderpath=os.path.join(evaluation_folderpath,
                                                                                   "point_clouds_all_times_time_0"),
                                               time_index=0, loaded_models=loaded_models)

    # endregion

    # region Save Metrics
    save_metrics_centers_pipeline(evaluation_folderpath, loaded_models, surface_data_list)
    save_metric_mesh_shape_pipeline(evaluation_folderpath, surface_data_list, train_config, loaded_models, 0)


def save_centers_img_pipeline(evaluation_folderpath, surface_data_list):
    centers_image_foldername = "img_centers"
    centers_image_folderpath = os.path.join(evaluation_folderpath, centers_image_foldername)
    os.makedirs(centers_image_folderpath, exist_ok=True)

    centers_image_folderpath = create_timestemp_dir(centers_image_folderpath)

    save_visualize_centers(surface_data_list, centers_image_folderpath)


def save_mesh_thrugh_model_pipeline(evaluation_folderpath, loaded_models, train_config, mesh_time_index : int) -> None:
    logging.info("START: Mesh though model")
    processed_data = process_mesh_through_model_pipeline(MeshData(time_index=mesh_time_index), train_config, loaded_models)
    visualizer = MeshDataVisualizer(processed_data)
    mesh_files_folderpath = os.path.join(evaluation_folderpath, "mesh_files")
    os.makedirs(mesh_files_folderpath, exist_ok=True)
    visualizer.save_as_ply_file(mesh_files_folderpath)
    visualizer.save_img_of_meshes(mesh_files_folderpath)
    logging.info("END: Mesh though model")


def save_metrics_centers_pipeline(evaluation_folderpath, loaded_models, surface_data_list):
    eval_surface_points_num = 10000
    logging.info(f"START: Metric - Centers for {eval_surface_points_num} points")
    folder_path = os.path.join(evaluation_folderpath, "metric_centers")
    os.makedirs(folder_path, exist_ok=True)
    folder_path = create_timestemp_dir(folder_path)

    compute_save_centers_metrics(
        CentersMetricsInfo(surface_data_list, loaded_models, eval_surface_points_num),
        folder_path)
    logging.info("END: Metric - Centers")

def save_metric_mesh_shape_pipeline(evaluation_folderpath : str, surface_data_list : SurfacePointsFrameList, train_config : TrainConfig, loaded_models : LoadedModelDic, mesh_time_index : int):
    logging.info("START: Metric - Mesh shape")
    folderpath = os.path.join(evaluation_folderpath, "metric_mesh_shape")
    os.makedirs(folderpath, exist_ok=True)
    folderpath = create_timestemp_dir(folderpath)

    compute_save_mesh_shape_metrics(folderpath, surface_data_list, train_config, loaded_models, mesh_time_index)
    logging.info("END: Metric - Mesh shape")

