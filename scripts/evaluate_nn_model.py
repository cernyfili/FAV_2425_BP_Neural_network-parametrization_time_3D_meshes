import logging
import os

from nerual_network.evaluation.evaluation import evaluate
from utils.constants import FilePathConfig, DEFAULT_TRAIN_CONFIG

# "C:\Users\RDClient\Documents\GitHub\FAV_BP_24_25_Parametrization\data\processed\ball\ball_20241210"
# init logger just to console

logging.basicConfig(level=logging.INFO)


data_folder = "data"
processed_folder = "processed"
raw_folder = "raw"

DATA_FOLDERNAME = 'ball_test'
SESSION_FOLDERNAME = "ball_test_20250309_234312"

processed_session_folderpath = os.path.join(data_folder, processed_folder, DATA_FOLDERNAME, SESSION_FOLDERNAME)
raw_folderpath = os.path.join(data_folder, raw_folder)
processed_folderpath = os.path.join(data_folder, processed_folder)

TRAIN_CONFIG = DEFAULT_TRAIN_CONFIG
TRAIN_CONFIG.file_path_config = FilePathConfig(
                               data_foldername=DATA_FOLDERNAME,
                               processed_session_folderpath=processed_session_folderpath,
                               raw_folderpath=raw_folderpath,
                               processed_folderpath=processed_folderpath
                           )
# TRAIN_CONFIG.num_clusters = 4

evaluate(TRAIN_CONFIG)
