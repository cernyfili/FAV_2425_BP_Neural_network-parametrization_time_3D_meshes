import logging
import os
import pickle


# Restrict access to underscore-prefixed functions
def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"{name} is a private function and cannot be imported.")
    raise AttributeError(f"Module has no attribute {name}")

def get_meshes_list(meshes_folder_path, len_clustered_data):
    """
    Get a list of .obj file paths from the specified folder.

    Parameters:
    - meshes_folder_path: str, the folder path containing the .obj files.

    Returns:
    - obj_files_list: list of .obj file paths.
    """
    obj_files_list = []
    files = os.listdir(meshes_folder_path)

    max_time_steps = min(len(files), len_clustered_data)

    files = files[:max_time_steps]

    min_index = get_file_index_from_filename(files[0])

    # Select only .obj files
    for filename in files:
        if filename.endswith('.obj'):
            obj_file_path = os.path.join(meshes_folder_path, filename)
            obj_files_list.append(obj_file_path)

            file_index = get_file_index_from_filename(obj_file_path)
            if file_index != len(obj_files_list) - 1 + min_index:
                raise Exception(f"File index mismatch: {file_index} vs {len(obj_files_list) - 1}")

    return obj_files_list


# Utility function to load a pickle file safely
def load_pickle_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
    except pickle.UnpicklingError:
        logging.error(f"Failed to unpickle file: {filepath}")
    return None


def init_logger(log_filepath):
    # Create a logger
    logger = logging.getLogger()
    # Set the logging level
    logger.setLevel(logging.INFO)
    # Create handlers
    console_handler = logging.StreamHandler()  # For console output
    file_handler = logging.FileHandler(log_filepath)  # For file output
    # Set the logging level for handlers
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # Add the formatter to the handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    # Example usage of the logger
    logger.info("Logging has been configured.")
    return logger

def end_logger(logger):
    # Assuming you have a logger defined as in your code
    logger.info("Ending logging and cleaning up resources.")

    # Remove all handlers
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()          # Close the handler (e.g., file handlers)
        logger.removeHandler(handler)  # Remove it from the logger


def get_file_index_from_filename(mesh_file_path, min_file_index=0):
    file_index = int(mesh_file_path.split('.')[-2][-3:])
    return file_index - min_file_index
