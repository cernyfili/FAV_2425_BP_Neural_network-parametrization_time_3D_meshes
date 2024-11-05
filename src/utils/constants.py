import os
from datetime import datetime
from nerual_network.model import MLP
import torch.optim as optim

# Restrict access to only uppercase constants
def __getattr__(name):
    if not name.isupper():
        raise AttributeError(f"{name} is a private variable and cannot be imported.")
    raise AttributeError(f"Module has no attribute {name}")

# region Neural network constants
NN_MAX_EPOCHS = 100
NN_PATIENCE = 5
NN_BATCH_SIZE = 32

NN_MODEL = MLP()
NN_OPTIMIZER = optim.Adam(NN_MODEL.parameters(), lr=0.001)
# endregion

# region Data processing constants
NUM_CLUSTERS = 5
NUM_SURFACE_POINTS = 1000

RAW_DATA_ALLOWED_FILETYPES_LIST = ['xyz', 'bin']
# endregion


# region Data constants
data_foldername = "ball"

VIZUALIZATION_OBJ_FILEPATH = 'data/raw/ball/ball000.obj'  # Path to your .obj file
# endregion

# region Filepaths constants
# region filenames
model_weights_templatename = "model_weights_cluster_{cluster}.pth"
surface_data_list_filename = 'surface_data_list.pkl'
clustered_data_filename = 'clustered_data.pkl'
JSON_FILENAME = "center_mesh_pairs.json"
log_file_filename = 'application.log'

raw_data_folderpath = 'data/raw'
processed_folderpath = "data/processed"
os.makedirs(processed_folderpath, exist_ok=True)
# endregion

# region private variables
# Get the current date and time in a formatted string
current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
# Create a folder name based on the current date and time
timestamped_foldername = f"{data_foldername}_{current_time_str}"
processed_data_folderpath = os.path.join(processed_folderpath, data_foldername)
processed_session_folderpath = os.path.join(processed_data_folderpath, timestamped_foldername)
os.makedirs(processed_session_folderpath, exist_ok=True)

model_weights_folderpath = processed_session_folderpath
# endregion


# region public filepaths
RAW_DATA_FOLDERPATH = os.path.join(raw_data_folderpath, data_foldername)# Update with the correct path
IMAGE_SAVE_FOLDERPATH = processed_session_folderpath

LOG_FILE_FILEPATH = os.path.join(processed_session_folderpath, log_file_filename)  # Specify your log file path here

SURFACE_DATA_LIST_FILEPATH = os.path.join(processed_data_folderpath, surface_data_list_filename)
CLUSTERED_DATA_FILEPATH = os.path.join(processed_data_folderpath, clustered_data_filename)
MODEL_WEIGHTS_FILEPATH_TEMPLATE = os.path.join(model_weights_folderpath, model_weights_templatename)
# endregion

# endregion

