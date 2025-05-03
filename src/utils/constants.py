import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from jsonschema import validate, ValidationError

TEST_MODE = True

# Restrict access to only uppercase constants
def __getattr__(name):
    if not name.isupper():
        raise AttributeError(f"{name} is a private variable and cannot be imported.")
    raise AttributeError(f"Module has no attribute {name}")

class ModelType(str, Enum):
    SIMPLE_MLP_01 = "simple_mlp_01"
    SIMPLE_MLP_02 = "simple_mlp_02"
    SIMPLE_MLP_03 = "simple_mlp_03"
    SIMPLE_MLP_04 = "simple_mlp_04"

class LossFunctionType(str, Enum):
    STANDARD = "standard"
    CENTERS = "centers"
    CHAMFER = "chamfer"
    CHAMFER2 = "chamfer2"
    UV_STREACH = "uv_streach"

# region CONSTANTS
# region Neural network constants
nn_max_epochs = 10_000
nn_patience = 5
nn_batch_size = 128
nn_lr = 1e-4

loss_function_name : LossFunctionType = LossFunctionType.STANDARD
model_type : ModelType = ModelType.SIMPLE_MLP_04
#nn_model = Simple_MLP_02()
#nn_optimizer = optim.Adam(nn_model.parameters(), lr=nn_lr)

if TEST_MODE:
    NN_DEVICE_STR = 'cpu'
else:
    NN_DEVICE_STR = 'cuda'

LOSS_FUNC_NORMAL_DIST_MEAN = 0.0
LOSS_FUNC_NORMAL_DIST_STD = 1.0
# endregion

# region Data processing constants
@dataclass(frozen=True)
class CDataPreprocessing:
    NUM_CLOSEST_CENTERS_TO_POINT = 3
    NUM_CLUSTERS = 5
    NUM_SURFACE_POINTS = 20_000
    MAX_TIME_STEPS = 100
    RAW_DATA_ALLOWED_FILETYPES_LIST = ['xyz', 'bin']

# endregion

# region Filepaths constants
# region filenames
current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

model_weights_templatename = "model_weights_cluster_{cluster}.pth"
surface_data_list_filename = 'surface_data_list.pkl'
clustered_data_filename = 'clustered_data.pkl'
log_file_filename = 'application.log'
point_cloud_original_filename = f"original_points_all_{current_time_str}.csv"
point_cloud_processed_filename = f"processed_points_all_{current_time_str}.csv"
center_metric_eval = "center_metric_eval"
center_metric_variances = "center_metric_variances"
mesh_shape_metrics = "mesh_shape_metrics.txt"
evaluation_folder_name = "evaluation"
CONFIG_JSON_FILENAME = "config.json"

current_absolutepath = os.path.dirname(os.path.abspath(__file__))
data_absolutepath = os.path.join(current_absolutepath, "..", "..", "data")
raw_foldername = "raw"
processed_foldername = "processed"

default_raw_folderpath = os.path.join(data_absolutepath, raw_foldername)
default_processed_folderpath = os.path.join(data_absolutepath, processed_foldername)
os.makedirs(default_processed_folderpath, exist_ok=True)


# endregion



# RAW_DATA_FOLDERPATH = os.path.join(raw_data_folderpath, data_foldername)  # Update with the correct path
# IMAGE_SAVE_FOLDERPATH = processed_session_folderpath


#
# SURFACE_DATA_EXPORT_FILE_FILEPATH = os.path.join(processed_data_folderpath, surface_data_list_filename)
# CLUSTERED_DATA_FILEPATH = os.path.join(processed_data_folderpath, clustered_data_filename)
# MODEL_WEIGHTS_FILEPATH_TEMPLATE = os.path.join(model_weights_folderpath, model_weights_templatename)
#
# POINT_CLOUD_ORIGINAL_FILEPATH = os.path.join(processed_session_folderpath, point_cloud_original_filename)
# POINT_CLOUD_PROCESSED_FILEPATH = os.path.join(processed_session_folderpath, point_cloud_processed_filename)


# endregion
# endregion



# region DATA STRUCTURES
class NNConfig:
    def __init__(self, nn_max_epochs: int, nn_patience: int, nn_batch_size: int, nn_model: ModelType, nn_lr: float,
                 loss_function_type: LossFunctionType):
        self.max_epochs : int = nn_max_epochs
        self.patience : int = nn_patience
        self.batch_size : int = nn_batch_size
        self.model_type : ModelType = nn_model
        self.nn_lr : float = nn_lr
        self.loss_function_type : LossFunctionType = loss_function_type

    def to_dict(self):
        return {
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "model_type": self.model_type.value,
            "nn_lr": self.nn_lr,
            "loss_function_name": self.loss_function_type.value
        }

    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            nn_max_epochs=config_dict["max_epochs"],
            nn_patience=config_dict["patience"],
            nn_batch_size=config_dict["batch_size"],
            nn_model=ModelType(config_dict["model_type"]),
            nn_lr=config_dict["nn_lr"],
            loss_function_type=LossFunctionType(config_dict["loss_function_name"])
        )

    @staticmethod
    def get_json_schema():
        return {
            "type": "object",
            "properties": {
                "max_epochs": {"type": "integer"},
                "patience": {"type": "integer"},
                "batch_size": {"type": "integer"},
                "model_type": {"type": "string", "enum": [str(model.value) for model in ModelType]},
                "nn_lr": {"type": "number"},
                "loss_function_name": {"type": "string", "enum": [str(loss_function.value) for loss_function in LossFunctionType]}
            },
            "required": ["max_epochs", "patience", "batch_size", "model_type", "nn_lr", "loss_function_name"]
        }

    def __str__(self):
        return f"NNConfig(nn_max_epochs={self.max_epochs}, nn_patience={self.patience}, nn_batch_size={self.batch_size}, nn_model={self.model_type})"

    def __repr__(self):
        return f"NNConfig(nn_max_epochs={self.max_epochs}, nn_patience={self.patience}, nn_batch_size={self.batch_size}, nn_model={self.model_type})"

class FilePathConfig:
    def __init__(self):

        self.raw_data_folderpath = None
        self.processed_session_folderpath = None

        self.surface_data_filepath = None
        self.session_surface_data_filepath = None

        self.clustered_data_filepath = None
        self.session_clustered_data_filepath = None

        self.log_filepath = None

        self.model_weights_folderpath_template = None

        # EVAL VALUES

        # EVAL IMAGES
        self.evaluation_folderpath = None

        self.point_cloud_original_filepath = None
        self.point_cloud_processed_filepath = None

        # EVAL METRICS
        self.center_metric_eval_filepath = None
        self.mesh_shape_metrics_filepath = None

        self.metrics_mesh_shape_metro_filepath = os.path.join(os.getcwd(), "bin", "metro.exe")
        
    @classmethod
    def create_test_mode(cls, data_foldername, processed_session_folderpath=None, raw_folderpath=None,
    processed_folderpath=None):
        instance = cls()
        
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Create a folder name based on the current date and time
        timestamped_foldername = f"{data_foldername}_{current_time_str}"

        if raw_folderpath is None:
            raw_folderpath = default_raw_folderpath

        if processed_folderpath is None:
            processed_folderpath = default_processed_folderpath
        processed_data_folderpath = str(os.path.join(processed_folderpath, data_foldername))

        if processed_session_folderpath is None:
            processed_session_folderpath = str(os.path.join(str(processed_data_folderpath), timestamped_foldername))
        os.makedirs(processed_session_folderpath, exist_ok=True)

        instance.processed_session_folderpath = processed_session_folderpath

        instance.raw_data_folderpath = os.path.join(raw_folderpath, data_foldername)

        instance.surface_data_filepath = os.path.join(processed_data_folderpath, surface_data_list_filename)
        #instance.session_surface_data_filepath = os.path.join(processed_session_folderpath, surface_data_list_filename)
        instance.session_surface_data_filepath = instance.surface_data_filepath

        instance.clustered_data_filepath = os.path.join(processed_data_folderpath, clustered_data_filename)
        #instance.session_clustered_data_filepath = os.path.join(processed_session_folderpath, clustered_data_filename)
        instance.session_clustered_data_filepath = instance.clustered_data_filepath

        instance.log_filepath = os.path.join(instance.processed_session_folderpath,
                                         log_file_filename)  # Specify your log file path here

        instance.model_weights_folderpath_template = os.path.join(instance.processed_session_folderpath, model_weights_templatename)

        # EVAL VALUES

        # EVAL IMAGES
        instance.evaluation_folderpath = os.path.join(instance.processed_session_folderpath, evaluation_folder_name)
        os.makedirs(instance.evaluation_folderpath, exist_ok=True)


        instance.point_cloud_original_filepath = os.path.join(instance.evaluation_folderpath, point_cloud_original_filename)
        instance.point_cloud_processed_filepath = os.path.join(instance.evaluation_folderpath, point_cloud_processed_filename)

        # EVAL METRICS
        instance.center_metric_eval_filepath = os.path.join(instance.evaluation_folderpath, center_metric_eval)
        instance.mesh_shape_metrics_filepath = os.path.join(instance.evaluation_folderpath, mesh_shape_metrics)

        return instance
        
    @classmethod
    def create_main(cls, raw_data_folderpath : str, processed_folderpath : str):
        instance = cls()

        instance.processed_session_folderpath = processed_folderpath
        instance.raw_data_folderpath = raw_data_folderpath

        instance.surface_data_filepath = os.path.join(instance.processed_session_folderpath, surface_data_list_filename)
        instance.session_surface_data_filepath = instance.surface_data_filepath

        instance.clustered_data_filepath = os.path.join(instance.processed_session_folderpath, clustered_data_filename)
        instance.session_clustered_data_filepath = instance.clustered_data_filepath

        instance.log_filepath = os.path.join(instance.processed_session_folderpath, log_file_filename)

        instance.model_weights_folderpath_template = os.path.join(instance.processed_session_folderpath, model_weights_templatename)

        # EVAL VALUES

        instance.evaluation_folderpath = os.path.join(instance.processed_session_folderpath, evaluation_folder_name)
        os.makedirs(instance.evaluation_folderpath, exist_ok=True)

        # EVAL IMAGES
        instance.point_cloud_original_filepath = os.path.join(instance.evaluation_folderpath, point_cloud_original_filename)
        instance.point_cloud_processed_filepath = os.path.join(instance.evaluation_folderpath, point_cloud_processed_filename)

        # EVAL METRICS
        instance.center_metric_eval_filepath = os.path.join(instance.evaluation_folderpath, center_metric_eval)
        instance.mesh_shape_metrics_filepath = os.path.join(instance.evaluation_folderpath, mesh_shape_metrics)

        return instance

    def __str__(self):
        return f"FilePathConfig(raw_data_folderpath={self.raw_data_folderpath}, image_save_folderpath={self.evaluation_folderpath}, surface_data_filepath={self.surface_data_filepath}, clustered_data_filepath={self.clustered_data_filepath}, model_weights_template={self.model_weights_folderpath_template}, point_cloud_original_filename={self.point_cloud_original_filepath}, point_cloud_processed_filename={self.point_cloud_processed_filepath})"

    def __repr__(self):
        return f"FilePathConfig(raw_data_folderpath={self.raw_data_folderpath}, image_save_folderpath={self.evaluation_folderpath}, surface_data_filepath={self.surface_data_filepath}, clustered_data_filepath={self.clustered_data_filepath}, model_weights_template={self.model_weights_folderpath_template}, point_cloud_original_filename={self.point_cloud_original_filepath}, point_cloud_processed_filename={self.point_cloud_processed_filepath})"

class TrainConfig:
    def __init__(self, nn_config: NNConfig, file_path_config: FilePathConfig, num_clusters,
                 num_surface_points, time_steps):
        self.num_clusters = num_clusters
        self.num_surface_points = num_surface_points
        self.nn_config : NNConfig = nn_config
        self.file_path_config : FilePathConfig = file_path_config
        self.max_time_steps = time_steps

    def to_dict(self):
        return {
            "num_clusters": self.num_clusters,
            "num_surface_points": self.num_surface_points,
            "max_time_steps": self.max_time_steps,
            "nn_config": self.nn_config.to_dict(),
        }

    @staticmethod
    def json_schema():
        return {
            "type": "object",
            "properties": {
                "num_clusters": {"type": "integer"},
                "num_surface_points": {"type": "integer"},
                "max_time_steps": {"type": "integer"},
                "nn_config": NNConfig.get_json_schema(),
            },
            "required": ["num_clusters", "num_surface_points", "max_time_steps", "nn_config"]
        }

    @classmethod
    def from_dict(cls, config_dict : dict, file_path_config: FilePathConfig):
        return cls(
            nn_config=NNConfig.from_dict(config_dict["nn_config"]),
            file_path_config=file_path_config,
            num_clusters=config_dict["num_clusters"],
            num_surface_points=config_dict["num_surface_points"],
            time_steps=config_dict["max_time_steps"]
        )

    @classmethod
    def from_json(cls, json_filepath: str, file_path_config: FilePathConfig):

        with open(json_filepath, 'r') as f:
            config_dict = json.load(f)

        schema = cls.json_schema()

        try:
            validate(instance=config_dict, schema=schema)
            logging.info("JSON is valid")
        except ValidationError as e:
            raise ValueError(f"JSON is invalid: {e.message}")

        train_config = cls.from_dict(config_dict, file_path_config)

        return train_config

    def save_to_json(self, filepath: str):
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def __str__(self):
        return f"TrainConfig(num_clusters={self.num_clusters}, num_surface_points={self.num_surface_points}, nn_config={self.nn_config}, file_path_config={self.file_path_config}, time_steps={self.max_time_steps})"

    def __repr__(self):
        return f"TrainConfig(num_clusters={self.num_clusters}, num_surface_points={self.num_surface_points}, nn_config={self.nn_config}, file_path_config={self.file_path_config}, time_steps={self.max_time_steps})"
# endregion


DEFAULT_NN_CONFIG = NNConfig(nn_max_epochs=nn_max_epochs, nn_patience=nn_patience, nn_batch_size=nn_batch_size,
                             nn_model=model_type, nn_lr=nn_lr, loss_function_type=loss_function_name)

DEFAULT_TRAIN_CONFIG = TrainConfig(nn_config=DEFAULT_NN_CONFIG,
                                   file_path_config=FilePathConfig.create_test_mode(data_foldername="default"),
                                   num_clusters=CDataPreprocessing.NUM_CLUSTERS, num_surface_points=CDataPreprocessing.NUM_SURFACE_POINTS, time_steps=CDataPreprocessing.MAX_TIME_STEPS)

