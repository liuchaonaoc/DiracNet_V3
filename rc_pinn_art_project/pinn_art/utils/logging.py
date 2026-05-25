"""Simple logging."""

from __future__ import annotations

import logging


def get_logger(name: str = "pinn_art") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger
