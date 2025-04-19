# Main function to orchestrate the processing and training for each cluster
import logging
import os

from nerual_network.evaluation.evaluation import evaluate
from nerual_network.training import train_nn
from src.data_processing.clustering import process_clustered_data
from src.data_processing.mapping import process_surface_data
from utils.constants import DEFAULT_NN_CONFIG, FilePathConfig, TrainConfig, \
    TEST_MODE, CDataPreprocessing, ModelType, LossFunctionType, NNConfig, nn_max_epochs, nn_patience, nn_batch_size, \
    nn_lr
from utils.helpers import init_logger, end_logger


def preprocess_data(train_config):
    process_clustered_data(train_config.num_clusters, train_config.file_path_config.raw_data_folderpath,
                           train_config.file_path_config.clustered_data_filepath, train_config.max_time_steps, train_config.file_path_config.session_clustered_data_filepath)

    process_surface_data(train_config.num_surface_points, train_config.file_path_config.raw_data_folderpath,
                         train_config.file_path_config.surface_data_filepath,
                         train_config.file_path_config.session_clustered_data_filepath, train_config.file_path_config.session_surface_data_filepath)


def main():

    # data_folders = ["ball", "casual_man_1000", "casual_man_4000", "vr_take"]

    #data_folders = ["ball"]
    #data_folders = ["casual_man_1000",  "vr_take", "casual_man_4000"]

    train_config_list = [
        TrainConfig(nn_config=NNConfig(nn_max_epochs=nn_max_epochs, nn_patience=nn_patience, nn_batch_size=nn_batch_size,
                                       nn_model=ModelType.SIMPLE_MLP_04, nn_lr=nn_lr, loss_function_type=LossFunctionType.CENTERS),
                    file_path_config=FilePathConfig.create_test_mode(data_foldername="casual_man_1000"),
                    num_clusters=CDataPreprocessing.NUM_CLUSTERS, num_surface_points=CDataPreprocessing.NUM_SURFACE_POINTS,
                    time_steps=CDataPreprocessing.MAX_TIME_STEPS),

        # chamfer2
        TrainConfig(nn_config=NNConfig(nn_max_epochs=nn_max_epochs, nn_patience=nn_patience, nn_batch_size=nn_batch_size,
                                       nn_model=ModelType.SIMPLE_MLP_02, nn_lr=nn_lr, loss_function_type=LossFunctionType.CHAMFER2),
                    file_path_config=FilePathConfig.create_test_mode(data_foldername="ball"),
                    num_clusters=CDataPreprocessing.NUM_CLUSTERS, num_surface_points=CDataPreprocessing.NUM_SURFACE_POINTS,
                    time_steps=CDataPreprocessing.MAX_TIME_STEPS),
        TrainConfig(nn_config=NNConfig(nn_max_epochs=nn_max_epochs, nn_patience=nn_patience, nn_batch_size=nn_batch_size,
                                       nn_model=ModelType.SIMPLE_MLP_02, nn_lr=nn_lr, loss_function_type=LossFunctionType.CHAMFER2),
                    file_path_config=FilePathConfig.create_test_mode(data_foldername="casual_man_1000"),
                    num_clusters=CDataPreprocessing.NUM_CLUSTERS, num_surface_points=CDataPreprocessing.NUM_SURFACE_POINTS,
                    time_steps=CDataPreprocessing.MAX_TIME_STEPS),

        # uv streach
        TrainConfig(nn_config=NNConfig(nn_max_epochs=nn_max_epochs, nn_patience=nn_patience, nn_batch_size=nn_batch_size,
                                       nn_model=ModelType.SIMPLE_MLP_03, nn_lr=nn_lr, loss_function_type=LossFunctionType.UV_STREACH),
                    file_path_config=FilePathConfig.create_test_mode(data_foldername="ball"),
                    num_clusters=CDataPreprocessing.NUM_CLUSTERS, num_surface_points=CDataPreprocessing.NUM_SURFACE_POINTS,
                    time_steps=CDataPreprocessing.MAX_TIME_STEPS),
        TrainConfig(nn_config=NNConfig(nn_max_epochs=nn_max_epochs, nn_patience=nn_patience, nn_batch_size=nn_batch_size,
                                       nn_model=ModelType.SIMPLE_MLP_03, nn_lr=nn_lr, loss_function_type=LossFunctionType.UV_STREACH),
                    file_path_config=FilePathConfig.create_test_mode(data_foldername="casual_man_1000"),
                    num_clusters=CDataPreprocessing.NUM_CLUSTERS, num_surface_points=CDataPreprocessing.NUM_SURFACE_POINTS,
                    time_steps=CDataPreprocessing.MAX_TIME_STEPS),

        # standard
        TrainConfig(nn_config=NNConfig(nn_max_epochs=nn_max_epochs, nn_patience=nn_patience, nn_batch_size=nn_batch_size,
                                       nn_model=ModelType.SIMPLE_MLP_02, nn_lr=nn_lr, loss_function_type=LossFunctionType.STANDARD),
                    file_path_config=FilePathConfig.create_test_mode(data_foldername="ball"),
                    num_clusters=CDataPreprocessing.NUM_CLUSTERS, num_surface_points=CDataPreprocessing.NUM_SURFACE_POINTS,
                    time_steps=CDataPreprocessing.MAX_TIME_STEPS),
        TrainConfig(nn_config=NNConfig(nn_max_epochs=nn_max_epochs, nn_patience=nn_patience, nn_batch_size=nn_batch_size,
                                       nn_model=ModelType.SIMPLE_MLP_02, nn_lr=nn_lr, loss_function_type=LossFunctionType.STANDARD),
                    file_path_config=FilePathConfig.create_test_mode(data_foldername="casual_man_1000"),
                    num_clusters=CDataPreprocessing.NUM_CLUSTERS, num_surface_points=CDataPreprocessing.NUM_SURFACE_POINTS,
                    time_steps=CDataPreprocessing.MAX_TIME_STEPS),

    ]

    if TEST_MODE:
        train_config_list = [
            TrainConfig(nn_config=NNConfig(nn_max_epochs=5, nn_patience=nn_patience, nn_batch_size=nn_batch_size,
                                           nn_model=ModelType.SIMPLE_MLP_04, nn_lr=nn_lr, loss_function_type=LossFunctionType.STANDARD),
                        file_path_config=FilePathConfig.create_test_mode(data_foldername="ball_test"),
                        num_clusters=CDataPreprocessing.NUM_CLUSTERS, num_surface_points=500,
                        time_steps=CDataPreprocessing.MAX_TIME_STEPS)]

    for train_config in train_config_list:

        train_config.save_to_json(os.path.join(train_config.file_path_config.processed_session_folderpath, "config.json"))

        logger = init_logger(train_config.file_path_config.log_filepath)

        data_foldername = train_config.file_path_config.raw_data_folderpath.split("/")[-1]
        logging.info("---------------------START OBJECT-------------------")
        logging.info(f"MAIN - Processing data for {data_foldername}")
        preprocess_data(train_config)
        logging.info(f"MAIN - Training neural network for {data_foldername}")
        train_nn(train_config)
        logging.info(f"MAIN - Evaluating neural network for {data_foldername}")
        evaluate(train_config)
        logging.info("---------------------END OBJECT-------------------")

        end_logger(logger)


if __name__ == '__main__':
    main()
