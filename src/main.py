# Main function to orchestrate the processing and training for each cluster
import logging
from data_processing.clustering import process_clustered_data
from data_processing.mapping import process_surface_data, SurfaceDataList
from nerual_network.evaluation import process_and_save_combined_image_for_all_clusters
from nerual_network.training import train_nn_for_all_clusters
from utils.constants import NN_MAX_EPOCHS, NN_PATIENCE, SURFACE_DATA_LIST_FILEPATH, NN_BATCH_SIZE, NUM_CLUSTERS, \
    NUM_SURFACE_POINTS
from utils.helpers import load_pickle_file


def main():

    # Process clustered and neural network data
    process_clustered_data(NUM_CLUSTERS)
    process_surface_data(NUM_SURFACE_POINTS)

    # Load the processed surface data list
    surface_data_list = load_pickle_file(SURFACE_DATA_LIST_FILEPATH)
    if surface_data_list is None or surface_data_list.list is None or not isinstance(surface_data_list, SurfaceDataList) :
        logging.error("Surface data list could not be loaded. Exiting.")
        return

    # Train a neural network for each cluster in the data
    train_nn_for_all_clusters(surface_data_list, max_epochs=NN_MAX_EPOCHS, patience=NN_PATIENCE, batch_size=NN_BATCH_SIZE)

    # Process and save combined image for all clusters after training
    process_and_save_combined_image_for_all_clusters(surface_data_list)


if __name__ == '__main__':
    main()