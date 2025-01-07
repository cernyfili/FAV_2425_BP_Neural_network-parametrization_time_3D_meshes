import os
from datetime import datetime


# Restrict access to only uppercase constants
def __getattr__(name):
    if not name.isupper():
        raise AttributeError(f"{name} is a private variable and cannot be imported.")
    raise AttributeError(f"Module has no attribute {name}")



# region CONSTANTS
# region Neural network constants
nn_max_epochs = 10_000
nn_patience = 5
nn_batch_size = 128
nn_lr = 1e-4
#nn_model = Simple_MLP_02()
#nn_optimizer = optim.Adam(nn_model.parameters(), lr=nn_lr)

NN_DEVICE_STR = 'cuda'
# endregion

# region Data processing constants
NUM_CLUSTERS = 5

NUM_SURFACE_POINTS = 10_000

MAX_TIME_STEPS = 100

RAW_DATA_ALLOWED_FILETYPES_LIST = ['xyz', 'bin']
# endregion


# region Data constants
# endregion

# region Filepaths constants
# region filenames
model_weights_templatename = "model_weights_cluster_{cluster}.pth"
surface_data_list_filename = 'surface_data_list.pkl'
clustered_data_filename = 'clustered_data.pkl'
log_file_filename = 'application.log'
point_cloud_original_filename = "original_points_all.csv"
point_cloud_processed_filename = "processed_points_all.csv"
center_metric_eval = "center_metric_eval.txt"
center_metric_variances = "center_metric_variances.txt"
mesh_shape_metrics = "mesh_shape_metrics.txt"

current_absolutepath = os.path.dirname(os.path.abspath(__file__))
data_absolutepath = os.path.join(current_absolutepath, "..", "..", "data")
raw_foldername = "raw"
processed_foldername = "processed"

default_raw_folderpath = os.path.join(data_absolutepath, raw_foldername)
default_processed_folderpath = os.path.join(data_absolutepath, processed_foldername)
os.makedirs(default_processed_folderpath, exist_ok=True)


# endregion


# region public filepaths
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
# endregion


# region DATA STRUCTURES


# endregion


class NNConfig:
    def __init__(self, nn_max_epochs: int, nn_patience: int, nn_batch_size: int, nn_model, nn_optimizer, nn_lr : float):
        self.max_epochs = nn_max_epochs
        self.patience = nn_patience
        self.batch_size = nn_batch_size
        self.model = nn_model
        self.optimizer = nn_optimizer
        self.nn_lr : float = nn_lr

    def __str__(self):
        return f"NNConfig(nn_max_epochs={self.max_epochs}, nn_patience={self.patience}, nn_batch_size={self.batch_size}, nn_model={self.model}, nn_optimizer={self.optimizer})"

    def __repr__(self):
        return f"NNConfig(nn_max_epochs={self.max_epochs}, nn_patience={self.patience}, nn_batch_size={self.batch_size}, nn_model={self.model}, nn_optimizer={self.optimizer})"


DEFAULT_NN_CONFIG = NNConfig(nn_max_epochs=nn_max_epochs, nn_patience=nn_patience, nn_batch_size=nn_batch_size,
                             nn_model=None, nn_optimizer=None, nn_lr=nn_lr)


class FilePathConfig:
    def __init__(self, data_foldername, processed_session_folderpath=None, raw_folderpath=None,
                 processed_folderpath=None):
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
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

        self.raw_data_folderpath = os.path.join(raw_folderpath, data_foldername)

        self.surface_data_filepath = os.path.join(processed_data_folderpath, surface_data_list_filename)
        self.clustered_data_filepath = os.path.join(processed_data_folderpath, clustered_data_filename)

        self.log_filepath = os.path.join(processed_session_folderpath,
                                         log_file_filename)  # Specify your log file path here

        # EVAL IMAGES
        self.images_save_folderpath = os.path.join(processed_session_folderpath)
        self.model_weights_folderpath = os.path.join(processed_session_folderpath, model_weights_templatename)
        self.point_cloud_original_filepath = os.path.join(processed_session_folderpath, point_cloud_original_filename)
        self.point_cloud_processed_filepath = os.path.join(processed_session_folderpath, point_cloud_processed_filename)

        # EVAL METRICS
        self.center_metric_eval_filepath = os.path.join(processed_session_folderpath, center_metric_eval)
        self.center_metric_variances_filepath = os.path.join(processed_session_folderpath, center_metric_variances)
        self.mesh_shape_metrics_filepath = os.path.join(processed_session_folderpath, mesh_shape_metrics)

    def __str__(self):
        return f"FilePathConfig(raw_data_folderpath={self.raw_data_folderpath}, image_save_folderpath={self.images_save_folderpath}, surface_data_filepath={self.surface_data_filepath}, clustered_data_filepath={self.clustered_data_filepath}, model_weights_template={self.model_weights_folderpath}, point_cloud_original_filename={self.point_cloud_original_filepath}, point_cloud_processed_filename={self.point_cloud_processed_filepath})"

    def __repr__(self):
        return f"FilePathConfig(raw_data_folderpath={self.raw_data_folderpath}, image_save_folderpath={self.images_save_folderpath}, surface_data_filepath={self.surface_data_filepath}, clustered_data_filepath={self.clustered_data_filepath}, model_weights_template={self.model_weights_folderpath}, point_cloud_original_filename={self.point_cloud_original_filepath}, point_cloud_processed_filename={self.point_cloud_processed_filepath})"


class TrainConfig:
    def __init__(self, nn_config: NNConfig, file_path_config: FilePathConfig, num_clusters=NUM_CLUSTERS,
                 num_surface_points=NUM_SURFACE_POINTS, time_steps=None):
        self.num_clusters = num_clusters
        self.num_surface_points = num_surface_points
        self.nn_config = nn_config
        self.file_path_config = file_path_config
        self.time_steps = time_steps

    def __str__(self):
        return f"TrainConfig(num_clusters={self.num_clusters}, num_surface_points={self.num_surface_points}, nn_config={self.nn_config}, file_path_config={self.file_path_config}, time_steps={self.time_steps})"

    def __repr__(self):
        return f"TrainConfig(num_clusters={self.num_clusters}, num_surface_points={self.num_surface_points}, nn_config={self.nn_config}, file_path_config={self.file_path_config}, time_steps={self.time_steps})"


DEFAULT_TRAIN_CONFIG = TrainConfig(nn_config=DEFAULT_NN_CONFIG,
                                   file_path_config=FilePathConfig(data_foldername="default"),
                                   num_clusters=NUM_CLUSTERS, num_surface_points=NUM_SURFACE_POINTS, time_steps=MAX_TIME_STEPS)
EVAL_NUM_SURFACE_POINTS = 1000
