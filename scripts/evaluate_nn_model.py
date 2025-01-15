from nerual_network.evaluation import evaluate
from utils.constants import DEFAULT_NN_CONFIG, FilePathConfig, TrainConfig

# "C:\Users\RDClient\Documents\GitHub\FAV_BP_24_25_Parametrization\data\processed\ball\ball_20241210"
PROCESSED_SESSION_FOLDERPATH = 'data\\processed\\ball\\ball_20250107_222419'
RAW_FOLDERPATH = "data\\raw"
PROCESSED_FOLDERPATH = "data\\processed"
DATA_FOLDERNAME = 'ball'

TRAIN_CONFIG = TrainConfig(nn_config=DEFAULT_NN_CONFIG,
                           file_path_config=FilePathConfig(
                               data_foldername=DATA_FOLDERNAME,
                               processed_session_folderpath=PROCESSED_SESSION_FOLDERPATH,
                               raw_folderpath=RAW_FOLDERPATH,
                               processed_folderpath=PROCESSED_FOLDERPATH
                           )
                           )

evaluate(TRAIN_CONFIG)
