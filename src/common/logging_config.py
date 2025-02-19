
# logging_config.py
# src/common/logging_config.py
import os
import logging
import yaml
import traceback
from typing import Dict, Any

def setup_logging(config: Dict[str, Any], default_level: int = logging.INFO) -> None:
    try:
        level_str = log_config['level'].upper()
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
        logging.basicConfig(level=default_level, format='%(asctime)s - %(levelname)s - %(message)s')



def load_yaml_config(filepath: str) -> Dict[str, Any]:
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
            if config is None:
                logging.warning(f"YAML file at {filepath} is empty.")
                return {}
            return config
    except FileNotFoundError:
        logging.error(f"YAML file not found: {filepath}")
        traceback.print_exc()
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {filepath}: {e}")
        traceback.print_exc()
        return {}
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading YAML: {filepath}: {e}")
        traceback.print_exc()
        return {}


def create_directory(dir_path: str) -> None:
    try:
        os.makedirs(dir_path, exist_ok=True)
        logging.debug(f"Directory created: {dir_path}")
    except OSError as e:
        logging.error(f"Error creating directory {dir_path}: {e}")
        traceback.print_exc()