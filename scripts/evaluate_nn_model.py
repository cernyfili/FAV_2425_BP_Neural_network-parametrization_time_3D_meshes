from nerual_network.evaluation import evaluate
from utils.constants import TrainConfig, DEFAULT_NN_CONFIG, FilePathConfig

PROCESSED_SESSION_FOLDERPATH = 'models/nn_model_weights'
DATA_FOLDERNAME = 'data'

TRAIN_CONFIG = TrainConfig(nn_config=DEFAULT_NN_CONFIG,
                           file_path_config=FilePathConfig(data_foldername=DATA_FOLDERNAME,
                                                           processed_session_folderpath=PROCESSED_SESSION_FOLDERPATH))



evaluate(TRAIN_CONFIG)
