"""
Centralized logging utility for the application.
"""

import logging
import sys
from typing import Optional


def get_logger(name: str = "CRAG", level: Optional[int] = logging.INFO) -> logging.Logger:
    """
    Get a configured logger with standard formatting.

    Args:
        name: Name of the logger module.
        level: Logging level (defaults to logging.INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
