import logging

from utils.constants import LOG_FILE_FILEPATH

# region Logger

# Configure logging for more robust output control


# Create a logger
logger = logging.getLogger()

# Set the logging level
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()  # For console output
file_handler = logging.FileHandler(LOG_FILE_FILEPATH)  # For file output

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

# endregion