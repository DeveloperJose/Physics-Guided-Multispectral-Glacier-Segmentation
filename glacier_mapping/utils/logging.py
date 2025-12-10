#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logging utilities for glacier mapping project.

This module provides a custom timestamped logger for internal application logging.
It uses a named logger 'glacier_mapping' to avoid duplicate logs from root logger.
"""

import datetime
import logging
import sys
from pathlib import Path

# Create named logger to avoid interference with root logger
LOGGER = logging.getLogger("glacier_mapping")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False  # Prevent duplication by not propagating to root

# Setup console handler if not present
if not LOGGER.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    # Use a simple formatter that passes the message through
    # The timestamping is handled by the log() wrapper for legacy compatibility
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(console_handler)


def configure_file_logging(log_file_path: str):
    """Configure logger to write to a file in addition to console.

    Args:
        log_file_path: Path to the log file
    """
    file_path = Path(log_file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(str(file_path))
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(file_handler)

    # Log that file logging is enabled
    log(logging.INFO, f"File logging enabled: {log_file_path}")


def log(level, message):
    """Timestamped logger used throughout the project.

    Args:
        level: Logging level (logging.INFO, logging.WARNING, etc.)
        message: Message string to log
    """
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y, %H:%M:%S")
    level_name = logging.getLevelName(level)

    # Format: "DD-MM-YYYY, HH:MM:SS    LEVEL   Message"
    formatted_message = f"{timestamp}\t{level_name}   {message}"

    LOGGER.log(level, formatted_message)


def info(message):
    """Log an info message with timestamp."""
    log(logging.INFO, message)


def warning(message):
    """Log a warning message with timestamp."""
    log(logging.WARNING, message)


def error(message):
    """Log an error message with timestamp."""
    log(logging.ERROR, message)


def debug(message):
    """Log a debug message with timestamp."""
    log(logging.DEBUG, message)


def print_conf(conf):
    """Pretty-print config dictionary to logger.

    Args:
        conf: Dictionary of configuration key-value pairs
    """
    for k, v in conf.items():
        log(logging.INFO, f"{k} = {v}")
