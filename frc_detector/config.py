"""Training settings used when ``train.py`` is run from an IDE.

Edit the values in the clearly marked section below, then run ``train.py``.
The dataclasses farther down provide the complete set of advanced settings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# =============================================================================
# EDIT THESE SETTINGS, THEN CLICK RUN ON train.py
# =============================================================================

# Dataset paths (COCO JSON annotations and their matching image folders).
TRAIN_ANNOTATIONS = "data/train/_annotations.coco.json"
TRAIN_IMAGES = "data/train"
VALIDATION_ANNOTATIONS = "data/valid/_annotations.coco.json"
VALIDATION_IMAGES = "data/valid"

# Data loading. -1 lets TensorFlow choose the number of workers/prefetch batches.
BATCH_SIZE = 2
NUM_WORKERS = -1
PREFETCH_BATCHES = -1

# Main training choices.
EPOCHS = 100
INITIAL_LEARNING_RATE = 2e-4
OUTPUT_DIR = "artifacts/frc_detector"
AUTO_RESUME = True
MIXED_PRECISION = False

# Use "imagenet" for pretrained weights or None to start with random weights.
BACKBONE_WEIGHTS: str | None = "imagenet"

# Input resolutions. Images keep their aspect ratio.
TRAIN_SHORT_SIDES = (480, 512, 544, 576, 608, 640, 704, 768)
VALIDATION_SHORT_SIDE = 640
MAX_LONG_SIDE = 1280

# =============================================================================
# ADVANCED SETTINGS
# =============================================================================


@dataclass
class DataConfig:
    # COCO paths. Image file names are resolved relative to the matching folder.
    train_annotations: str = TRAIN_ANNOTATIONS
    train_images: str = TRAIN_IMAGES
    validation_annotations: str = VALIDATION_ANNOTATIONS
    validation_images: str = VALIDATION_IMAGES

    batch_size: int = BATCH_SIZE
    # -1 means tf.data.AUTOTUNE. Set a positive value to cap parallel mapping.
    num_workers: int = NUM_WORKERS
    prefetch_batches: int = PREFETCH_BATCHES
    shuffle_buffer: int = 2048
    cache_validation: bool = False

    # The aspect ratio is never distorted. A short-side size is sampled each
    # training image; images are then padded to the largest H/W in that batch.
    train_short_sides: tuple[int, ...] = TRAIN_SHORT_SIDES
    validation_short_side: int = VALIDATION_SHORT_SIDE
    max_long_side: int = MAX_LONG_SIDE
    pad_to_multiple: int = 32

    include_crowd: bool = False
    seed: int = 42


@dataclass
class AugmentationConfig:
    enabled: bool = True
    horizontal_flip_probability: float = 0.5
    random_crop_probability: float = 0.25
    minimum_crop_scale: float = 0.65
    zoom_out_probability: float = 0.20
    maximum_zoom_out: float = 1.35
    rotate_90_probability: float = 0.0

    color_jitter_probability: float = 0.8
    brightness_delta: float = 0.15
    contrast_range: tuple[float, float] = (0.75, 1.25)
    saturation_range: tuple[float, float] = (0.70, 1.30)
    hue_delta: float = 0.05
    gamma_probability: float = 0.20
    gamma_range: tuple[float, float] = (0.75, 1.35)
    grayscale_probability: float = 0.05
    jpeg_probability: float = 0.15
    jpeg_quality_range: tuple[int, int] = (65, 100)
    gaussian_blur_probability: float = 0.12
    gaussian_noise_probability: float = 0.15
    gaussian_noise_stddev: float = 0.025
    cutout_probability: float = 0.20
    maximum_cutout_fraction: float = 0.25


@dataclass
class ModelConfig:
    backbone: str = "mobilenet_v2"
    backbone_weights: str | None = BACKBONE_WEIGHTS
    fpn_channels: int = 128
    head_depth: int = 3
    head_channels: int = 128
    l2_regularization: float = 1e-5
    strides: tuple[int, ...] = (8, 16, 32, 64)
    size_ranges: tuple[tuple[float, float], ...] = (
        (0.0, 64.0),
        (64.0, 128.0),
        (128.0, 256.0),
        (256.0, 1e8),
    )
    center_sampling_radius: float = 1.5


@dataclass
class TrainingConfig:
    epochs: int = EPOCHS
    initial_learning_rate: float = INITIAL_LEARNING_RATE
    minimum_learning_rate_ratio: float = 0.03
    warmup_epochs: int = 3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 10.0
    mixed_precision: bool = MIXED_PRECISION
    early_stopping_patience: int = 15
    auto_resume: bool = AUTO_RESUME

    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    box_loss_weight: float = 2.0
    centerness_loss_weight: float = 1.0

    validation_score_threshold: float = 0.05
    summary_score_threshold: float = 0.25
    nms_iou_threshold: float = 0.60
    max_detections: int = 100

    output_dir: str = OUTPUT_DIR
    checkpoint_keep: int = 5


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def output_dir(self) -> Path:
        return Path(self.training.output_dir)


CONFIG = ExperimentConfig()
