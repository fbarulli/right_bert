# logging_config.py
# project_structure.py
import os
import logging
import yaml
from typing import Dict, Any
import traceback

def setup_logging(log_config: Dict[str, Any], default_level: int = logging.INFO) -> None:
    """
    Sets up logging based on the provided configuration.

    Args:
        log_config: A dictionary containing logging configuration.  Expected keys:
            'level': (str) The desired logging level (e.g., "DEBUG", "INFO", "WARNING").
            'file': (str, optional) The file to write logs to. If None, logs to console.
        default_level: The default logging level if not specified in the config.
    """
    try:
        level_str = log_config.get('level', '').upper()
        level = getattr(logging, level_str, default_level)

        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=level, format=log_format)

        if 'file' in log_config:
            file_handler = logging.FileHandler(log_config['file'])
            file_handler.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(file_handler)

    except Exception as e:
        print(f"Error setting up logging: {e}")
        traceback.print_exc()
        # Fallback to basic console logging
        logging.basicConfig(level=default_level, format='%(asctime)s - %(levelname)s - %(message)s')



def load_yaml_config(filepath: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.

    Args:
        filepath: The path to the YAML configuration file.

    Returns:
        A dictionary representing the YAML configuration.  Returns an empty
        dictionary if any error occurs during loading.
    """
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
            # Add basic validation, check if config is None or empty
            if config is None:
                logging.warning(f"YAML file at {filepath} is empty.")
                return {}
            return config
    except FileNotFoundError:
        logging.error(f"YAML file not found: {filepath}")
        traceback.print_exc()
        return {}  # Return an empty dict to signal failure
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {filepath}: {e}")
        traceback.print_exc()
        return {}
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading YAML: {filepath}: {e}")
        traceback.print_exc()
        return {}


def create_directory(dir_path: str) -> None:
    """Creates a directory if it doesn't exist.  Logs errors."""
    try:
        os.makedirs(dir_path, exist_ok=True)
        logging.debug(f"Directory created: {dir_path}")
    except OSError as e:
        logging.error(f"Error creating directory {dir_path}: {e}")
        traceback.print_exc()