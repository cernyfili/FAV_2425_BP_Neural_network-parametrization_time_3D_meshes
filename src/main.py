# Main function to orchestrate the processing and training for each cluster
import logging

from nerual_network.training import train_nn
from src.data_processing.clustering import process_clustered_data
from src.data_processing.mapping import process_surface_data
from utils.constants import DEFAULT_NN_CONFIG, NUM_CLUSTERS, \
    NUM_SURFACE_POINTS, MAX_TIME_STEPS, FilePathConfig, TrainConfig
from utils.helpers import init_logger, end_logger


def preprocess_data(train_config):
    process_clustered_data(train_config.num_clusters, train_config.file_path_config.raw_data_folderpath,
                           train_config.file_path_config.clustered_data_filepath, train_config.time_steps)
    process_surface_data(train_config.num_surface_points, train_config.file_path_config.raw_data_folderpath,
                         train_config.file_path_config.surface_data_filepath,
                         train_config.file_path_config.clustered_data_filepath)


def main():
    # data_folders = ["ball", "casual_man_1000", "casual_man_4000", "vr_take"]

    data_folders = ["ball", "casual_man_1000"]

    for data_foldername in data_folders:
        train_config = TrainConfig(nn_config=DEFAULT_NN_CONFIG,
                                   file_path_config=FilePathConfig(data_foldername=data_foldername),
                                   num_clusters=NUM_CLUSTERS, num_surface_points=NUM_SURFACE_POINTS,
                                   time_steps=MAX_TIME_STEPS)

        train_config.nn_config.max_epochs = 10
        train_config.num_surface_points = 1000
        logger = init_logger(train_config.file_path_config.log_filepath)

        logging.info("---------------------START OBJECT-------------------")
        logging.info(f"MAIN - Processing data for {data_foldername}")
        preprocess_data(train_config)
        logging.info(f"MAIN - Training neural network for {data_foldername}")
        train_nn(train_config)
        logging.info(f"MAIN - Evaluating neural network for {data_foldername}")
        #evaluate(train_config)
        logging.info("---------------------END OBJECT-------------------")

        end_logger(logger)


if __name__ == '__main__':
    main()
