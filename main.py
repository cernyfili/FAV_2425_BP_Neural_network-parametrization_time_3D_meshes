# Main function to orchestrate the processing and training for each cluster
import logging
import os
from typing import List

import typer

from src.nerual_network.training import train_nn
from src.utils.cmd_app import preprocess_data, train_pipeline, process_mesh_pipeline, evaluate_metrics_pipeline, \
    visualize_pipeline
from src.utils.constants import FilePathConfig, TrainConfig, \
    TEST_MODE, CDataPreprocessing, ModelType, LossFunctionType, NNConfig, nn_patience, nn_batch_size, \
    nn_lr
from src.utils.helpers import init_logger, end_logger

app = typer.Typer(
    help="Neural Network CLI app for creating TMV (time varying mashes) parametrization, training, evaluation, and visualization.")


@app.command(help="Preprocess and train the neural network on prepared data.")
def train(
        nn_config_file: str = typer.Option(...,
                                           help="JSON file in specified format in documentation with settings for neural network"),
        raw_data_folderpath: str = typer.Option(...,
                                                help="Path to an folder where are saved raw train data: \n 1) *[time_index].obj files for all times in sequance\n 2) *[time_index].xyz or *[time_index].bin with position of generated centers with ARAP method"),
        output_folderpath: str = typer.Option(...,
                                              help="Path to an folder where to save the trained model and processed data"),
):
    train_pipeline(nn_config_file, raw_data_folderpath, output_folderpath)


@app.command(help="Run a selected mesh from original data through the trained neural network and export results.")
def process(
        processed_folder_path: str = typer.Option(...,
                                                  help="Path to the folder with processed nn data from train command."),
        output_folder_path: str = typer.Option(..., help="Path to the folder where to save the processed mesh data."),
        mesh_time_index: int = typer.Option(...,
                                            help="Time index of the mesh to process from original data which was the neural network trained on."),
        format: List[str] = typer.Option(...,
                                         help="Output formats: obj, ply (with UV colors), png (with UV colors) - you can specify multiple separating with spaces.")
):
    process_mesh_pipeline(processed_folderpath=processed_folder_path, evaluation_folderpath=output_folder_path,
                          mesh_time_index=mesh_time_index, format=format)


@app.command(
    help="Evaluate a trained model using custom metrics (metric of closest centers, metric of similar shape).")
def evaluate(
        processed_folder_path: str = typer.Option(...,help="Path to the folder with processed nn data from train command."),
        output_folder_path: str = typer.Option(...,help="Path to the folder where to save the metrics outputs."),
        metrics_center_eval_points: int = typer.Option(...,help="Number of points on surface of each mesh in each time to evaluate the metric of closest centers."),
        metrics_mesh_shape_mesh_time_index: int = typer.Option(...,help="Time index of the mesh to process from original data which was the neural network trained on and then the metric of similar shape are computed on.")
):
    evaluate_metrics_pipeline(processed_folderpath=processed_folder_path, evaluation_folderpath=output_folder_path, metrics_center_eval_points=metrics_center_eval_points,
                              metrics_mesh_shape_mesh_time_index=metrics_mesh_shape_mesh_time_index)



@app.command(help="Visualize trained data as images and point clouds for neural network settings improvement (detailed in documentation).")
def visualize(
        processed_folder_path: str = typer.Option(...,help="Path to the folder with processed nn data from train command."),
        output_folder_path: str = typer.Option(...,help="Path to the folder where to save the outputs."),
):
    visualize_pipeline(processed_folderpath=processed_folder_path, evaluation_folderpath=output_folder_path)


def run_from_project():

    train_config_list = [
        TrainConfig(
            nn_config=NNConfig(nn_max_epochs=25, nn_patience=3, nn_batch_size=128,
                               nn_model=ModelType.SIMPLE_MLP_04, nn_lr=nn_lr,
                               loss_function_type=LossFunctionType.CHAMFER2),
            file_path_config=FilePathConfig.create_test_mode(data_foldername="casual_man_1000"),
            num_clusters=CDataPreprocessing.NUM_CLUSTERS, num_surface_points=CDataPreprocessing.NUM_SURFACE_POINTS,
            time_steps=CDataPreprocessing.MAX_TIME_STEPS),

    ]

    if TEST_MODE:
        train_config_list = [
            TrainConfig(nn_config=NNConfig(nn_max_epochs=5, nn_patience=nn_patience, nn_batch_size=nn_batch_size,
                                           nn_model=ModelType.SIMPLE_MLP_04, nn_lr=nn_lr,
                                           loss_function_type=LossFunctionType.STANDARD),
                        file_path_config=FilePathConfig.create_test_mode(data_foldername="ball_test"),
                        num_clusters=CDataPreprocessing.NUM_CLUSTERS, num_surface_points=500,
                        time_steps=CDataPreprocessing.MAX_TIME_STEPS)]

    for train_config in train_config_list:
        train_config.save_to_json(
            os.path.join(train_config.file_path_config.processed_session_folderpath, "config.json"))

        logger = init_logger(train_config.file_path_config.log_filepath)

        data_foldername = train_config.file_path_config.raw_data_folderpath.split("/")[-1]
        logging.info("---------------------START OBJECT-------------------")
        logging.info(f"MAIN - Processing data for {data_foldername}")
        preprocess_data(train_config)
        logging.info(f"MAIN - Training neural network for {data_foldername}")
        train_nn(train_config)
        logging.info(f"MAIN - Evaluating neural network for {data_foldername}")
        # evaluate(train_config)
        logging.info("---------------------END OBJECT-------------------")

        end_logger(logger)


if __name__ == '__main__':
    app()
