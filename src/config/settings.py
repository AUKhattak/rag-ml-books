# config/settings.py (concept, not final code)
import logging.config
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent


def load_logging():
    with open(CONFIG_DIR / "logging_config.yaml") as f:
        logging.config.dictConfig(yaml.safe_load(f))


load_logging()
