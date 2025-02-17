from nerual_network.evaluation import evaluate
from utils.constants import DEFAULT_NN_CONFIG, FilePathConfig, TrainConfig, DEFAULT_TRAIN_CONFIG

# "C:\Users\RDClient\Documents\GitHub\FAV_BP_24_25_Parametrization\data\processed\ball\ball_20241210"
PROCESSED_SESSION_FOLDERPATH = 'data\\processed\\ball\\ball_20250217_010431'
RAW_FOLDERPATH = "data\\raw"
PROCESSED_FOLDERPATH = "data\\processed"
DATA_FOLDERNAME = 'ball'

TRAIN_CONFIG = DEFAULT_TRAIN_CONFIG
TRAIN_CONFIG.file_path_config = FilePathConfig(
                               data_foldername=DATA_FOLDERNAME,
                               processed_session_folderpath=PROCESSED_SESSION_FOLDERPATH,
                               raw_folderpath=RAW_FOLDERPATH,
                               processed_folderpath=PROCESSED_FOLDERPATH
                           )
TRAIN_CONFIG.num_clusters = 2

evaluate(TRAIN_CONFIG)
