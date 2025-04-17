import logging
import os

from main import preprocess_data
from nerual_network.evaluation.evaluation import evaluate
from utils.constants import FilePathConfig, DEFAULT_TRAIN_CONFIG, ModelType, LossFunctionType, TrainConfig

# "C:\Users\RDClient\Documents\GitHub\FAV_BP_24_25_Parametrization\data\processed\ball\ball_20241210"
# init logger just to console

logging.basicConfig(level=logging.INFO)


data_folder = "data"
processed_folder = "processed"
raw_folder = "raw"

DATA_FOLDERNAME = 'ball'
SESSION_FOLDERNAME = "ball_20250410_132313"
CONFIG_JSON_FILENAME = "config.json"


processed_session_folderpath = os.path.join(data_folder, processed_folder, DATA_FOLDERNAME, SESSION_FOLDERNAME)
raw_folderpath = os.path.join(data_folder, raw_folder)
processed_folderpath = os.path.join(data_folder, processed_folder)

config_json_filepath = os.path.join(processed_session_folderpath, CONFIG_JSON_FILENAME)


file_path_config = FilePathConfig.create_test_mode(
                               data_foldername=DATA_FOLDERNAME,
                               processed_session_folderpath=processed_session_folderpath,
                               raw_folderpath=raw_folderpath,
                               processed_folderpath=processed_folderpath
                           )

train_config = TrainConfig.from_json(config_json_filepath, file_path_config)

preprocess_data(train_config)

evaluate(train_config)
