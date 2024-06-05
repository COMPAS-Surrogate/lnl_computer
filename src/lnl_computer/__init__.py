import os

from .logger import logger

HERE = os.path.dirname(os.path.abspath(__file__))

logger.info(f"source-path: {HERE}")
