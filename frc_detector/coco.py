"""COCO JSON parsing and TensorFlow input-pipeline construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow as tf

from .augment import augment_sample, resize_preserving_aspect_ratio
from .config import AugmentationConfig, DataConfig


@dataclass(frozen=True)
class CocoRecord:
    image_id: int
    image_path: str
    width: int
    height: int
    boxes: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class CocoIndex:
    records: tuple[CocoRecord, ...]
    category_ids: tuple[int, ...]
    class_names: tuple[str, ...]
    category_id_to_label: dict[int, int]

    def __len__(self) -> int:
        return len(self.records)


def load_coco_index(
    annotation_file: str | Path,
    image_dir: str | Path,
    *,
    category_ids: tuple[int, ...] | None = None,
    include_crowd: bool = False,
) -> CocoIndex:
    """Load and validate a COCO file, converting category IDs to 0..C-1."""
    annotation_path = Path(annotation_file).expanduser().resolve()
    image_root = Path(image_dir).expanduser().resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    with annotation_path.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    categories = {int(item["id"]): str(item["name"]) for item in coco.get("categories", [])}
    if not categories:
        raise ValueError(f"No categories were found in {annotation_path}")

    if category_ids is None:
        ordered_ids = tuple(sorted(categories))
    else:
        ordered_ids = tuple(int(value) for value in category_ids)
        missing = [value for value in ordered_ids if value not in categories]
        if missing:
            raise ValueError(f"Validation COCO file is missing category IDs: {missing}")
    category_to_label = {category_id: label for label, category_id in enumerate(ordered_ids)}

    images: dict[int, dict] = {}
    for item in coco.get("images", []):
        image_id = int(item["id"])
        if image_id in images:
            raise ValueError(f"Duplicate COCO image id: {image_id}")
        images[image_id] = item
    if not images:
        raise ValueError(f"No images were found in {annotation_path}")

    annotations_by_image: dict[int, list[dict]] = {image_id: [] for image_id in images}
    for annotation in coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if image_id not in images or category_id not in category_to_label:
            continue
        if not include_crowd and int(annotation.get("iscrowd", 0)):
            continue
        annotations_by_image[image_id].append(annotation)

    records: list[CocoRecord] = []
    missing_images: list[str] = []
    for image_id in sorted(images):
        item = images[image_id]
        image_path = image_root / str(item["file_name"])
        if not image_path.is_file():
            missing_images.append(str(image_path))
            continue

        width = int(item.get("width", 0))
        height = int(item.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions for COCO image id {image_id}: {width}x{height}")

        boxes: list[list[float]] = []
        labels: list[int] = []
        for annotation in annotations_by_image[image_id]:
            x, y, box_width, box_height = map(float, annotation["bbox"])
            x1 = float(np.clip(x, 0.0, width))
            y1 = float(np.clip(y, 0.0, height))
            x2 = float(np.clip(x + box_width, 0.0, width))
            y2 = float(np.clip(y + box_height, 0.0, height))
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(category_to_label[int(annotation["category_id"])])

        records.append(
            CocoRecord(
                image_id=image_id,
                image_path=str(image_path),
                width=width,
                height=height,
                boxes=np.asarray(boxes, dtype=np.float32).reshape((-1, 4)),
                labels=np.asarray(labels, dtype=np.int32),
            )
        )

    if missing_images:
        preview = "\n".join(missing_images[:10])
        suffix = "" if len(missing_images) <= 10 else f"\n... and {len(missing_images) - 10} more"
        raise FileNotFoundError(f"COCO JSON references missing images:\n{preview}{suffix}")
    if not records:
        raise ValueError("The COCO dataset contains no usable image records.")

    return CocoIndex(
        records=tuple(records),
        category_ids=ordered_ids,
        class_names=tuple(categories[value] for value in ordered_ids),
        category_id_to_label=category_to_label,
    )


def _record_generator(index: CocoIndex) -> Iterator[dict[str, object]]:
    for record in index.records:
        yield {
            "image_path": record.image_path,
            "image_id": np.int64(record.image_id),
            "boxes": record.boxes,
            "labels": record.labels,
        }


def _decode_record(sample: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    encoded = tf.io.read_file(sample["image_path"])
    image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    image = tf.image.convert_image_dtype(image, tf.float32)
    return {**sample, "image": image}


def _pad_batch_to_multiple(batch: dict[str, tf.Tensor], multiple: int) -> dict[str, tf.Tensor]:
    height = tf.shape(batch["image"])[1]
    width = tf.shape(batch["image"])[2]
    target_height = tf.cast(tf.math.ceil(tf.cast(height, tf.float32) / multiple), tf.int32) * multiple
    target_width = tf.cast(tf.math.ceil(tf.cast(width, tf.float32) / multiple), tf.int32) * multiple
    pad_height = target_height - height
    pad_width = target_width - width
    image = tf.pad(batch["image"], [[0, 0], [0, pad_height], [0, pad_width], [0, 0]])
    return {**batch, "image": image}


def build_dataset(
    index: CocoIndex,
    data_config: DataConfig,
    augmentation_config: AugmentationConfig,
    *,
    training: bool,
) -> tf.data.Dataset:
    """Build a padded, variable-resolution tf.data pipeline."""
    output_signature = {
        "image_path": tf.TensorSpec((), tf.string),
        "image_id": tf.TensorSpec((), tf.int64),
        "boxes": tf.TensorSpec((None, 4), tf.float32),
        "labels": tf.TensorSpec((None,), tf.int32),
    }
    dataset = tf.data.Dataset.from_generator(
        lambda: _record_generator(index), output_signature=output_signature
    )
    if training:
        dataset = dataset.shuffle(
            min(data_config.shuffle_buffer, max(len(index), 1)),
            seed=data_config.seed,
            reshuffle_each_iteration=True,
        )

    workers = tf.data.AUTOTUNE if data_config.num_workers < 0 else data_config.num_workers
    dataset = dataset.map(_decode_record, num_parallel_calls=workers)

    def prepare(sample: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        image, boxes, labels = sample["image"], sample["boxes"], sample["labels"]
        if training and augmentation_config.enabled:
            image, boxes, labels = augment_sample(image, boxes, labels, augmentation_config)
            short_sides = tf.constant(data_config.train_short_sides, dtype=tf.int32)
            target_short_side = tf.random.shuffle(short_sides)[0]
        else:
            target_short_side = tf.constant(data_config.validation_short_side, tf.int32)
        image, boxes = resize_preserving_aspect_ratio(
            image, boxes, target_short_side, data_config.max_long_side
        )
        image_shape = tf.shape(image)[:2]
        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "image_shape": image_shape,
            "image_id": sample["image_id"],
            "image_path": sample["image_path"],
        }

    dataset = dataset.map(prepare, num_parallel_calls=workers)
    if not training and data_config.cache_validation:
        dataset = dataset.cache()

    dataset = dataset.padded_batch(
        data_config.batch_size,
        padded_shapes={
            "image": (None, None, 3),
            "boxes": (None, 4),
            "labels": (None,),
            "image_shape": (2,),
            "image_id": (),
            "image_path": (),
        },
        padding_values={
            "image": np.float32(0.0),
            "boxes": np.float32(0.0),
            "labels": np.int32(-1),
            "image_shape": np.int32(0),
            "image_id": np.int64(-1),
            "image_path": "",
        },
        drop_remainder=False,
    )
    dataset = dataset.map(
        lambda sample: _pad_batch_to_multiple(sample, data_config.pad_to_multiple),
        num_parallel_calls=workers,
    )
    options = tf.data.Options()
    options.experimental_deterministic = not training
    dataset = dataset.with_options(options)
    prefetch = tf.data.AUTOTUNE if data_config.prefetch_batches < 0 else data_config.prefetch_batches
    return dataset.prefetch(prefetch)

