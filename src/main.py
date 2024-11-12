# Main function to orchestrate the processing and training for each cluster
from nerual_network.training import train_nn_for_object
from utils.constants import DEFAULT_TRAIN_CONFIG, TrainConfig, FilePathConfig, DEFAULT_NN_CONFIG


# def train_nn_for_object(num_clusters, num_surface_points, nn_max_epochs, nn_patience, nn_batch_size, raw_data_folderpath):
#
#     logging.info("------------------TRAINING STARTED------------------")
#     # basic constants log
#     logging.info(f"Number of clusters: {NUM_CLUSTERS}")
#     logging.info(f"Number of surface points: {NUM_SURFACE_POINTS}")
#     logging.info(f"Maximum number of epochs: {NN_MAX_EPOCHS}")
#     logging.info(f"Patience for early stopping: {NN_PATIENCE}")
#     logging.info(f"Batch size: {NN_BATCH_SIZE}")
#     # raw data folder
#     logging.info(f"Raw data folder: {RAW_DATA_FOLDERPATH}")
#
#
#     # Process clustered and neural network data
#     process_clustered_data(NUM_CLUSTERS)
#     process_surface_data(NUM_SURFACE_POINTS)
#
#     # Load the processed surface data list
#     surface_data_list = load_pickle_file(SURFACE_DATA_LIST_FILEPATH)
#     if surface_data_list is None or surface_data_list.list is None or not isinstance(surface_data_list, SurfaceDataList) :
#         logging.error("Surface data list could not be loaded. Exiting.")
#         return
#
#     # Train a neural network for each cluster in the data
#     train_nn_for_all_clusters(surface_data_list, max_epochs=NN_MAX_EPOCHS, patience=NN_PATIENCE, batch_size=NN_BATCH_SIZE)
#
#     # Process and save combined image for all clusters after training
#     process_and_save_combined_image_for_all_clusters(surface_data_list)

def main():
    #data_folders = ["ball", "casual_man_1000", "casual_man_4000", "vr_take"]
    data_folders = ["ball"]

    for data_foldername in data_folders:
        train_config = TrainConfig(nn_config=DEFAULT_NN_CONFIG,
                                   file_path_config=FilePathConfig(data_foldername=data_foldername))
        train_nn_for_object(train_config)

if __name__ == '__main__':
    main()