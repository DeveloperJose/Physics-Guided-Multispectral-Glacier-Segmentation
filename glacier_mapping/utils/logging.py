import logging
import sys
from pathlib import Path

LOGGER = logging.getLogger("glacier_mapping")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

if not LOGGER.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)


def configure_file_logging(log_file_path: str):
    file_path = Path(log_file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(file_path))
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(file_handler)
    LOGGER.info(f"File logging enabled: {log_file_path}")


def log(level, message):
    LOGGER.log(level, message)


def info(message):
    log(logging.INFO, message)


def warning(message):
    log(logging.WARNING, message)


def error(message):
    log(logging.ERROR, message)


def debug(message):
    log(logging.DEBUG, message)


def print_conf(conf):
    for k, v in conf.items():
        log(logging.INFO, f"{k} = {v}")
