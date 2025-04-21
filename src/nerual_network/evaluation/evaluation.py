import logging
import os

from data_processing.class_mapping import SurfacePointsFrameList
from nerual_network.evaluation.meshes import process_mesh_through_model_pipeline, MeshDataVisualizer
from nerual_network.evaluation.metrics import compute_save_centers_metrics, compute_save_mesh_shape_metrics
from nerual_network.evaluation.visualization import _visualize_all_clusters_for_each_time, \
    _visualize_combined_surface_points_for_each_time, _visualize_points_with_time, \
    _visualize_original_and_processed_points, _prepare_export_data, visualize_uv_points_in_3d, save_visualize_centers, \
    _save_pointcloud_to_file, _create_pointclouds_from_time_to_all_times
from nerual_network.helpers import load_trained_nn_from_files, MeshData, CentersMetricsInfo, create_timestemp_dir, \
    LoadedModelDic
from utils.constants import TrainConfig
from utils.helpers import load_pickle_file


# Restrict access to underscore-prefixed functions
def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"{name} is a private function and cannot be imported.")
    raise AttributeError(f"Module has no attribute {name}")


# region PRIVATE FUNCTIONS


# endregion


# def _compute_centers_metrics(surface_data_list, train_config, num_points, nn_lr):
#     """
#     Computes metrics which:
#     1. for every sequenco of points
#     1.a selects num_points from the sequence
#     2. finds closest center point loaded from files
#     3. puts original points through the encoder
#     4. puts ouput throug the decoder in every time
#     5. computes the distances between the ceneter point again
#     6. outputs statistical dispersion for every point
#     7. do this for all sequences of points
#     :param surface_data_list:
#     :param train_config:
#     :param num_points:
#     :return:
#     """
#
#     def load_centers_data_from_files(folder_path, time_steps):
#         raise NotImplementedError("Not implemented yet")
#
#         center_points, num_points_in_file = load_centers_files(folder_path, time_steps)
#         # make it a Surface data list where each SurfacePoint is one num_points_in_file slice of center_points
#
#         # itarate over center_points and create SurfacePoint with num_points_in_file points
#         center_points_list_current = SurfacePointsFrameList([])
#         for i in range(0, center_points.shape[0]):
#             center_points_list_current.append(SurfacePointsFrame(center_points[i]))
#
#         center_points_list_current.assign_time_to_all_elements()
#         center_points_list_current.normalize_all_elements()
#         return center_points_list_current
#
#     def compute_distance(first_point, second_point):
#         return np.linalg.norm(first_point - second_point)
#
#     def find_closest_centers(center_points_list: SurfacePointsFrame, points_list: SurfacePointsFrame):
#         pair_original_center = []
#         index = 0
#         for index_point, point in enumerate(points_list.normalized_points_list):
#             closest_center_point = None
#             min_distance = float('inf')
#             for center_point in center_points_list.normalized_points_list:
#                 distance = compute_distance(point, center_point)
#                 if distance < min_distance:
#                     closest_center_point = center_point
#                     min_distance = distance
#
#             if points_list.time != center_points_list.time:
#                 raise Exception("Not same time")
#             time = points_list.time
#
#             label = points_list.labels_list[index_point]
#
#             pair_original_center.append(
#                 PairPointCenterPoint(point, closest_center_point, min_distance, index + 1, time, label))
#         return PairPointCenterPointList(pair_original_center)
#
#     def run_through_model(center_points_list: SurfacePointsFrameList, surface_data_list: SurfacePointsFrameList,
#                           model_weights_template: str, batch_size: int, num_points: int, nn_lr):
#         raise NotImplementedError("Not implemented yet")
#
#         # todo check if should be normalized
#         if len(center_points_list.list) != len(surface_data_list.list):
#             raise Exception("Not same number of center points and surface data")
#         pair_list_len = len(surface_data_list.list)
#
#         unique_clusters = surface_data_list.get_unique_clusters_indexes()
#         unique_times = surface_data_list.get_unique_times()
#
#         evaluation_result_list = EvaluationResultList([])
#
#         for i in range(0, pair_list_len):
#
#             # select original points where time is 0
#             surface_data_timeframe = surface_data_list.list[i]
#             if surface_data_timeframe.time.index != i:
#                 raise Exception("Not same time")
#
#             encoder_time = surface_data_timeframe.time.value
#
#             center_points_timeframe = center_points_list.list[i]
#             if center_points_timeframe.time.index != i:
#                 raise Exception("Not same time")
#
#             surface_data_timeframe = surface_data_list.select_random_points(num_points)
#
#             pair_original_center = find_closest_centers(center_points_timeframe, surface_data_timeframe)
#
#             for cluster in unique_clusters:
#
#                 # Load the original surface points for the current cluster
#                 pair_original_center_cluster = pair_original_center.filter_by_point_clusterlabel(cluster)
#                 surface_data_cluster = pair_original_center_cluster.get_points_list()
#                 surface_data_cluster = _convert_to_surfacepointsframelist(surface_data_cluster)
#
#                 # Create a SurfaceDataset instance with the filtered surface data
#                 original_points_dataset = NNDataset(surface_data_cluster)
#                 # Prepare a DataLoader for original points
#                 original_points_loader = DataLoader(original_points_dataset, batch_size=batch_size, shuffle=False)
#
#                 # Load the trained model for the current cluster
#                 model_weights_filepath = model_weights_template.format(cluster=cluster)
#                 model = _load_trained_model(model_weights_filepath, train_config)
#                 device = torch.device(NN_DEVICE_STR)
#
#                 model.to(device)
#
#                 # Step 1: Encode the original data
#                 with torch.no_grad():  # No need to calculate gradients during evaluation
#                     encoded_features = model.encoder(original_points_loader)
#
#                 decoder_pair_list = DecoderPairList([])
#
#                 # iterate to decoder over all times
#                 for decoder_time in unique_times:
#                     # Create a tensor of the same shape as the time feature in the input
#                     time_tensor = torch.full((encoded_features.size(0), 1), decoder_time, dtype=torch.float32)
#                     # Concatenate the encoded features with the time tensor
#                     encoded_with_time = torch.cat((encoded_features, time_tensor), dim=1)
#                     # Pass through the decoder
#                     decoded_output = model.decoder(encoded_with_time)
#
#                     decoder_processed_points = decoded_output
#                     decoder_processed_points_timeframe = SurfacePointsFrame([], None, decoder_time)
#                     # convert to
#                     for point in decoder_processed_points:
#                         decoder_processed_points_timeframe.points_list.append(point)
#
#                     decoder_center_points_timeframe = center_points_timeframe.get_element_by_time_index(decoder_time)
#
#                     decoder_pair_processed_center = find_closest_centers(decoder_center_points_timeframe,
#                                                                          decoder_processed_points_timeframe)
#
#                     decoder_pair_list.append(DecoderElement(decoder_pair_processed_center, decoder_time))
#
#                 evaluation_result_list.append(
#                     EvaluationResult(pair_original_center_cluster, encoder_time, decoder_pair_list))
#
#         return evaluation_result_list
#
#         # load center points
#
#     center_points_list = load_centers_data_from_files(train_config.file_path_config.raw_data_folderpath,
#                                                       train_config.max_time_steps)
#     # just loads the center points from files (already loaded in surface_data_list
#
#     # run through the model
#     evaluation_results_list = run_through_model(center_points_list,
#                                                 surface_data_list,
#                                                 train_config.file_path_config.model_weights_folderpath_template,
#                                                 train_config.nn_config.batch_size, num_points, nn_lr)
#
#     variance_list = _compute_variance(evaluation_results_list)
#
#     return variance_list, evaluation_results_list


# def get_centers_points_by_time_and_closestcentersindicies(data, closest_centers_indices_tensor, time, device):


# def get_centers_matrix_by_cluster_frame(closest_centers_matrix, input_tensor, device, time_index):


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

    loaded_models = load_trained_nn_from_files(train_config)

    save_metric_mesh_shape_pipeline(evaluation_folderpath=evaluation_folderpath,
                                    surface_data_list=surface_data_list,
                                    train_config=train_config,
                                    loaded_models=loaded_models,
                                    mesh_time_index=0)

def evaluate_full(train_config: TrainConfig):
    surface_data_list = load_pickle_file(train_config.file_path_config.session_surface_data_filepath)

    if surface_data_list is None or surface_data_list.public_list is None:
        logging.error("Surface data list could not be loaded. Exiting.")
        return

    evaluation_folderpath = train_config.file_path_config.evaluation_folderpath

    loaded_models = load_trained_nn_from_files(train_config)


    # region Save Evaluation files
    save_mesh_thrugh_model_pipeline(evaluation_folderpath, loaded_models, train_config, 0)

    save_centers_pipeline(evaluation_folderpath, surface_data_list)

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


def save_centers_pipeline(evaluation_folderpath, surface_data_list):
    centers_image_foldername = "img_centers"
    centers_image_folderpath = os.path.join(evaluation_folderpath, centers_image_foldername)
    os.makedirs(centers_image_folderpath, exist_ok=True)

    centers_image_folderpath = create_timestemp_dir(centers_image_folderpath)

    save_visualize_centers(surface_data_list, centers_image_folderpath)


def save_mesh_thrugh_model_pipeline(evaluation_folderpath, loaded_models, train_config, mesh_time_index : int) -> None:
    processed_data = process_mesh_through_model_pipeline(MeshData(time_index=mesh_time_index), train_config, loaded_models)
    visualizer = MeshDataVisualizer(processed_data)
    mesh_files_folderpath = os.path.join(evaluation_folderpath, "mesh_files")
    os.makedirs(mesh_files_folderpath, exist_ok=True)
    visualizer.save_img_of_meshes(mesh_files_folderpath)


def save_metrics_centers_pipeline(evaluation_folderpath, loaded_models, surface_data_list):
    eval_surface_points_num = 10
    folder_path = os.path.join(evaluation_folderpath, "metric_centers")
    os.makedirs(folder_path, exist_ok=True)
    folder_path = create_timestemp_dir(folder_path)

    compute_save_centers_metrics(
        CentersMetricsInfo(surface_data_list, loaded_models, eval_surface_points_num),
        folder_path)

def save_metric_mesh_shape_pipeline(evaluation_folderpath : str, surface_data_list : SurfacePointsFrameList, train_config : TrainConfig, loaded_models : LoadedModelDic, mesh_time_index : int):
    folderpath = os.path.join(evaluation_folderpath, "metric_mesh_shape")
    os.makedirs(folderpath, exist_ok=True)
    folderpath = create_timestemp_dir(folderpath)

    compute_save_mesh_shape_metrics(folderpath, surface_data_list, train_config, loaded_models, mesh_time_index)

