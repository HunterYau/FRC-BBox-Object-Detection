"""TensorFlow object detector for COCO-format datasets."""

from .config import CONFIG, ExperimentConfig
from .model import build_detector

__all__ = ["CONFIG", "ExperimentConfig", "build_detector"]

