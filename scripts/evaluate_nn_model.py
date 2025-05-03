import logging
import os
import sys

from src.utils.cmd_app import preprocess_data
from src.nerual_network.evaluation.evaluation import evaluate, evaluate_partial, evaluate_partial_2
from src.utils.constants import FilePathConfig, DEFAULT_TRAIN_CONFIG, ModelType, LossFunctionType, TrainConfig, \
    CONFIG_JSON_FILENAME

# "C:\Users\RDClient\Documents\GitHub\FAV_BP_24_25_Parametrization\data\processed\ball\ball_20241210"
# init logger just to console

logging.basicConfig(level=logging.INFO)

data_folder = "data"
processed_folder = "processed"
raw_folder = "raw"

data_foldername_list = [
                        #("ball", "_FINAL_chamfer_ball_20250430_153457_324517")
]

for element in data_foldername_list:
    data_foldername = element[0]
    session_foldername = element[1]

    processed_session_folderpath = os.path.join(data_folder, processed_folder, data_foldername, session_foldername)
    raw_folderpath = os.path.join(data_folder, raw_folder)
    processed_folderpath = os.path.join(data_folder, processed_folder)

    config_json_filepath = os.path.join(processed_session_folderpath, CONFIG_JSON_FILENAME)

    file_path_config = FilePathConfig.create_test_mode(
        data_foldername=data_foldername,
        processed_session_folderpath=processed_session_folderpath,
        raw_folderpath=raw_folderpath,
        processed_folderpath=processed_folderpath
    )

    train_config = TrainConfig.from_json(config_json_filepath, file_path_config)

    preprocess_data(train_config)

    evaluate_partial(train_config)

data_foldername_list = [
                        # ("ball", "_FINAL_chamfer_ball_20250430_153457_324517"),
                        ("casual_man_1000", "_FINAL_casual_man_1000_20250418_173451_centers"),
                        ("casual_man_1000", "_FINAL_casual_man_1000_20250422_154326_standard")
                        ]

for element in data_foldername_list:
    data_foldername = element[0]
    session_foldername = element[1]

    processed_session_folderpath = os.path.join(data_folder, processed_folder, data_foldername, session_foldername)
    raw_folderpath = os.path.join(data_folder, raw_folder)
    processed_folderpath = os.path.join(data_folder, processed_folder)

    config_json_filepath = os.path.join(processed_session_folderpath, CONFIG_JSON_FILENAME)

    file_path_config = FilePathConfig.create_test_mode(
        data_foldername=data_foldername,
        processed_session_folderpath=processed_session_folderpath,
        raw_folderpath=raw_folderpath,
        processed_folderpath=processed_folderpath
    )

    train_config = TrainConfig.from_json(config_json_filepath, file_path_config)

    preprocess_data(train_config)

    evaluate_partial_2(train_config)
