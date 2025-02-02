# src/app/utils/logger.py
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from app.config import BaseConfig


def setup_logger():
    """Configure application logging

    Sets up logging with:
    - Console output for all logs
    - File output with rotation
    - Formatting with timestamp, level, and message
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(BaseConfig.LOG_LEVEL)

    # Create formatters
    formatter = logging.Formatter(BaseConfig.LOG_FORMAT)

    # Ensure log directory exists
    log_dir = BaseConfig.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    # Create and configure file handler with rotation
    file_handler = RotatingFileHandler(
        filename=log_dir / 'app.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(BaseConfig.LOG_LEVEL)

    # Create and configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(BaseConfig.LOG_LEVEL)

    # Remove existing handlers to avoid duplication
    logger.handlers.clear()

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log initial message
    logger.info('Logging system initialized')

    return logger


def get_logger(name):
    """Get a logger instance for a specific module

    Args:
        name: The name of the module requesting the logger

    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)