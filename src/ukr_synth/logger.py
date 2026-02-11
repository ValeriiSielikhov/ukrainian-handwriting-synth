"""Logger configuration for ukr_synth."""

import logging
from typing import Optional

_logger: Optional[logging.Logger] = None


def logger_setup(
    level: int | str = logging.INFO,
    format: str | None = None,
    datefmt: str | None = None,
    logger_name: str = "ukr_synth",
) -> logging.Logger:
    """Configure and return the logger.

    Parameters
    ----------
    level
        Logging level (default: INFO)
    format
        Log message format. If None, uses default format.
    datefmt
        Date format. If None, uses default format.
    logger_name
        Name of the logger instance.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    global _logger

    if format is None:
        format = "%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d:%(funcName)s(): %(message)s"

    if datefmt is None:
        datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=level,
        format=format,
        datefmt=datefmt,
        force=True,
    )

    _logger = logging.getLogger(logger_name)
    return _logger


def get_logger(logger_name: str | None = None) -> logging.Logger:
    """Get the logger instance.

    If logger hasn't been setup yet, initializes it with default settings.

    Parameters
    ----------
    logger_name
        Name of the logger. If None and logger not setup, uses "ukr_synth".

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    global _logger

    if _logger is None:
        name = logger_name or "ukr_synth"
        return logger_setup(logger_name=name)

    if logger_name is not None:
        return logging.getLogger(logger_name)

    return _logger
