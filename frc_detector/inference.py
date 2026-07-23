"""Keras/TFLite decoding, prediction, matching, and visualization helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.patches import Rectangle

from .coco import CocoIndex
from .losses import make_points


@dataclass
class Detections:
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


def decode_predictions(
    predictions: dict[str, tf.Tensor],
    image_shapes: tf.Tensor,
    num_classes: int,
    strides: tuple[int, ...],
    *,
    score_threshold: float = 0.05,
    iou_threshold: float = 0.60,
    max_detections: int = 100,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Decode raw P3-P6 outputs and run class-aware NMS."""
    all_boxes: list[tf.Tensor] = []
    all_scores: list[tf.Tensor] = []
    for prediction, stride in zip(
        [predictions[key] for key in sorted(predictions)], strides, strict=True
    ):
        shape = tf.shape(prediction)
        points = make_points(shape[1], shape[2], stride)
        flattened = tf.reshape(prediction, (shape[0], -1, num_classes + 5))
        class_scores = tf.sigmoid(tf.cast(flattened[:, :, :num_classes], tf.float32))
        distances = (
            tf.nn.softplus(tf.cast(flattened[:, :, num_classes : num_classes + 4], tf.float32))
            * float(stride)
        )
        center_scores = tf.sigmoid(tf.cast(flattened[:, :, num_classes + 4 :], tf.float32))
        scores = class_scores * center_scores
        px = points[None, :, 0]
        py = points[None, :, 1]
        boxes = tf.stack(
            [
                py - distances[:, :, 1],
                px - distances[:, :, 0],
                py + distances[:, :, 3],
                px + distances[:, :, 2],
            ],
            axis=-1,
        )
        valid = tf.logical_and(
            points[None, :, 0] < tf.cast(image_shapes[:, None, 1], tf.float32),
            points[None, :, 1] < tf.cast(image_shapes[:, None, 0], tf.float32),
        )
        scores = tf.where(valid[:, :, None], scores, 0.0)
        all_boxes.append(boxes)
        all_scores.append(scores)

    boxes = tf.concat(all_boxes, axis=1)
    scores = tf.concat(all_scores, axis=1)
    boxes = boxes[:, :, None, :]
    nms_boxes, nms_scores, nms_labels, valid_detections = tf.image.combined_non_max_suppression(
        boxes=boxes,
        scores=scores,
        max_output_size_per_class=max_detections,
        max_total_size=max_detections,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        pad_per_class=False,
        clip_boxes=False,
    )
    # combined_non_max_suppression uses YXYX; the public API uses XYXY.
    nms_boxes = tf.gather(nms_boxes, [1, 0, 3, 2], axis=-1)
    widths = tf.cast(image_shapes[:, 1], tf.float32)
    heights = tf.cast(image_shapes[:, 0], tf.float32)
    x1 = tf.clip_by_value(nms_boxes[:, :, 0], 0.0, widths[:, None])
    y1 = tf.clip_by_value(nms_boxes[:, :, 1], 0.0, heights[:, None])
    x2 = tf.clip_by_value(nms_boxes[:, :, 2], 0.0, widths[:, None])
    y2 = tf.clip_by_value(nms_boxes[:, :, 3], 0.0, heights[:, None])
    nms_boxes = tf.stack([x1, y1, x2, y2], axis=-1)
    return nms_boxes, nms_scores, tf.cast(nms_labels, tf.int32), valid_detections


def prepare_image(
    image_path: str | Path,
    *,
    short_side: int = 640,
    maximum_long_side: int = 1280,
    pad_multiple: int = 32,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float], tuple[int, int]]:
    encoded = tf.io.read_file(str(image_path))
    original = tf.io.decode_image(encoded, channels=3, expand_animations=False)
    original = tf.image.convert_image_dtype(original, tf.float32)
    original_shape = tf.shape(original)[:2]
    shape_float = tf.cast(original_shape, tf.float32)
    scale = tf.cast(short_side, tf.float32) / tf.reduce_min(shape_float)
    scale = tf.minimum(scale, tf.cast(maximum_long_side, tf.float32) / tf.reduce_max(shape_float))
    resized_shape = tf.maximum(tf.cast(tf.round(shape_float * scale), tf.int32), 1)
    resized = tf.image.resize(original, resized_shape, antialias=True)
    target_height = tf.cast(tf.math.ceil(tf.cast(resized_shape[0], tf.float32) / pad_multiple), tf.int32) * pad_multiple
    target_width = tf.cast(tf.math.ceil(tf.cast(resized_shape[1], tf.float32) / pad_multiple), tf.int32) * pad_multiple
    padded = tf.image.pad_to_bounding_box(resized, 0, 0, target_height, target_width)
    scale_y = float(resized_shape[0].numpy()) / float(original_shape[0].numpy())
    scale_x = float(resized_shape[1].numpy()) / float(original_shape[1].numpy())
    return (
        padded.numpy()[None],
        original.numpy(),
        (scale_x, scale_y),
        (int(resized_shape[0].numpy()), int(resized_shape[1].numpy())),
    )


def predict_keras_image(
    model: tf.keras.Model,
    image_path: str | Path,
    num_classes: int,
    strides: tuple[int, ...],
    *,
    short_side: int = 640,
    maximum_long_side: int = 1280,
    score_threshold: float = 0.25,
    iou_threshold: float = 0.60,
    max_detections: int = 100,
) -> tuple[Detections, np.ndarray]:
    image, original, (scale_x, scale_y), resized_shape = prepare_image(
        image_path, short_side=short_side, maximum_long_side=maximum_long_side
    )
    predictions = model(image, training=False)
    boxes, scores, labels, valid = decode_predictions(
        predictions,
        tf.constant([resized_shape], tf.int32),
        num_classes,
        strides,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )
    count = int(valid[0].numpy())
    result_boxes = boxes[0, :count].numpy()
    result_boxes /= np.asarray([scale_x, scale_y, scale_x, scale_y], np.float32)
    return (
        Detections(
            boxes=result_boxes,
            scores=scores[0, :count].numpy(),
            labels=labels[0, :count].numpy(),
        ),
        original,
    )


def _numpy_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), np.float32)
    intersection_x1 = np.maximum(box[0], boxes[:, 0])
    intersection_y1 = np.maximum(box[1], boxes[:, 1])
    intersection_x2 = np.minimum(box[2], boxes[:, 2])
    intersection_y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(intersection_x2 - intersection_x1, 0.0) * np.maximum(
        intersection_y2 - intersection_y1, 0.0
    )
    box_area = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
    boxes_area = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0.0
    )
    return intersection / np.maximum(box_area + boxes_area - intersection, 1e-7)


def match_detections(
    detections: Detections,
    true_boxes: np.ndarray,
    true_labels: np.ndarray,
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    matched_truth: set[int] = set()
    matched_ious: list[float] = []
    order = np.argsort(-detections.scores)
    for prediction_index in order:
        candidate_indices = np.flatnonzero(true_labels == detections.labels[prediction_index])
        candidate_indices = np.asarray(
            [index for index in candidate_indices if int(index) not in matched_truth], dtype=np.int32
        )
        if not candidate_indices.size:
            continue
        ious = _numpy_iou(detections.boxes[prediction_index], true_boxes[candidate_indices])
        best_local = int(np.argmax(ious))
        if float(ious[best_local]) >= iou_threshold:
            truth_index = int(candidate_indices[best_local])
            matched_truth.add(truth_index)
            matched_ious.append(float(ious[best_local]))
    true_positive = len(matched_truth)
    false_positive = len(detections.boxes) - true_positive
    false_negative = len(true_boxes) - true_positive
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
    }


def find_ground_truth(index: CocoIndex, image_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    target = Path(image_path).resolve()
    exact = [record for record in index.records if Path(record.image_path).resolve() == target]
    if not exact:
        exact = [record for record in index.records if Path(record.image_path).name == target.name]
    if len(exact) != 1:
        raise ValueError(f"Could not uniquely match {image_path} to one COCO image record.")
    return exact[0].boxes.copy(), exact[0].labels.copy()


def visualize_prediction(
    model: tf.keras.Model,
    image_path: str | Path,
    class_names: tuple[str, ...] | list[str],
    strides: tuple[int, ...],
    *,
    true_boxes: np.ndarray | None = None,
    true_labels: np.ndarray | None = None,
    score_threshold: float = 0.25,
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[Detections, dict[str, float | int] | None]:
    """Show predictions and optional truth; return proper detection metrics.

    Green dashed boxes are ground truth. Solid boxes are model predictions.
    Unlike classification accuracy, bounding-box quality is reported as
    precision, recall, F1 and matched IoU at IoU >= 0.50.
    """
    detections, image = predict_keras_image(
        model,
        image_path,
        len(class_names),
        strides,
        score_threshold=score_threshold,
    )
    figure, axis = plt.subplots(figsize=(12, 8))
    axis.imshow(image)
    axis.axis("off")
    colors = plt.get_cmap("tab20")
    if true_boxes is not None and true_labels is not None:
        for box, label in zip(true_boxes, true_labels, strict=True):
            x1, y1, x2, y2 = box
            axis.add_patch(
                Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2, linestyle="--"
                )
            )
            axis.text(x1, y1, f"GT {class_names[int(label)]}", color="black", backgroundcolor="lime")

    for box, score, label in zip(detections.boxes, detections.scores, detections.labels, strict=True):
        x1, y1, x2, y2 = box
        color = colors(int(label) % 20)
        axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
        axis.text(
            x1,
            y2,
            f"{class_names[int(label)]} {float(score):.2f}",
            color="white",
            backgroundcolor=color,
        )

    metrics = None
    if true_boxes is not None and true_labels is not None:
        metrics = match_detections(detections, true_boxes, true_labels)
        axis.set_title(
            "IoU@0.50 — "
            f"precision {metrics['precision']:.3f}, recall {metrics['recall']:.3f}, "
            f"F1 {metrics['f1']:.3f}, matched IoU {metrics['mean_matched_iou']:.3f}"
        )
    else:
        axis.set_title(f"{len(detections.boxes)} detections at score >= {score_threshold:.2f}")
    figure.tight_layout()
    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
    return detections, metrics


class TFLiteDetector:
    """Small TFLite runtime wrapper with dynamic input resizing and NumPy NMS."""

    def __init__(self, model_path: str | Path, metadata_path: str | Path | None = None):
        self.model_path = Path(model_path)
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        default_metadata = self.model_path.with_suffix(self.model_path.suffix + ".json")
        metadata_file = Path(metadata_path) if metadata_path else default_metadata
        if not metadata_file.is_file():
            raise FileNotFoundError(f"TFLite metadata not found: {metadata_file}")
        self.metadata: dict[str, Any] = json.loads(metadata_file.read_text(encoding="utf-8"))
        self.strides = tuple(int(value) for value in self.metadata["strides"])
        self.class_names = tuple(self.metadata["class_names"])

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
        order = np.argsort(-scores)
        keep: list[int] = []
        while order.size:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break
            ious = _numpy_iou(boxes[current], boxes[order[1:]])
            order = order[1:][ious <= threshold]
        return keep

    def predict(
        self,
        image_path: str | Path,
        *,
        short_side: int = 640,
        maximum_long_side: int = 1280,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.60,
        max_detections: int = 100,
    ) -> tuple[Detections, np.ndarray]:
        image, original, (scale_x, scale_y), resized_shape = prepare_image(
            image_path, short_side=short_side, maximum_long_side=maximum_long_side
        )
        input_details = self.interpreter.get_input_details()[0]
        self.interpreter.resize_tensor_input(input_details["index"], image.shape, strict=False)
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()[0]
        input_scale, input_zero = input_details["quantization"]
        input_value = image
        if input_details["dtype"] != np.float32:
            input_value = np.round(image / input_scale + input_zero).astype(input_details["dtype"])
        self.interpreter.set_tensor(input_details["index"], input_value)
        self.interpreter.invoke()

        outputs: list[np.ndarray] = []
        for detail in self.interpreter.get_output_details():
            value = self.interpreter.get_tensor(detail["index"])
            scale, zero = detail["quantization"]
            if detail["dtype"] != np.float32 and scale > 0:
                value = (value.astype(np.float32) - zero) * scale
            outputs.append(value)
        outputs.sort(key=lambda value: int(value.shape[1]) * int(value.shape[2]), reverse=True)

        candidate_boxes: list[np.ndarray] = []
        candidate_scores: list[np.ndarray] = []
        candidate_labels: list[np.ndarray] = []
        num_classes = len(self.class_names)
        for output, stride in zip(outputs, self.strides, strict=True):
            _, height, width, _ = output.shape
            flattened = output.reshape((-1, num_classes + 5)).astype(np.float32)
            class_scores = 1.0 / (1.0 + np.exp(-flattened[:, :num_classes]))
            distances = np.logaddexp(0.0, flattened[:, num_classes : num_classes + 4]) * stride
            center = 1.0 / (1.0 + np.exp(-flattened[:, num_classes + 4]))
            scores = class_scores * center[:, None]
            grid_x, grid_y = np.meshgrid(
                (np.arange(width, dtype=np.float32) + 0.5) * stride,
                (np.arange(height, dtype=np.float32) + 0.5) * stride,
            )
            points = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)
            valid_points = np.logical_and(points[:, 0] < resized_shape[1], points[:, 1] < resized_shape[0])
            point_indices, class_indices = np.nonzero(np.logical_and(scores >= score_threshold, valid_points[:, None]))
            if not point_indices.size:
                continue
            selected_distances = distances[point_indices]
            selected_points = points[point_indices]
            boxes = np.stack(
                [
                    selected_points[:, 0] - selected_distances[:, 0],
                    selected_points[:, 1] - selected_distances[:, 1],
                    selected_points[:, 0] + selected_distances[:, 2],
                    selected_points[:, 1] + selected_distances[:, 3],
                ],
                axis=-1,
            )
            candidate_boxes.append(boxes)
            candidate_scores.append(scores[point_indices, class_indices])
            candidate_labels.append(class_indices.astype(np.int32))

        if not candidate_boxes:
            return Detections(np.empty((0, 4), np.float32), np.empty((0,), np.float32), np.empty((0,), np.int32)), original
        boxes = np.concatenate(candidate_boxes)
        scores = np.concatenate(candidate_scores)
        labels = np.concatenate(candidate_labels)
        kept: list[int] = []
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            kept.extend(indices[self._nms(boxes[indices], scores[indices], iou_threshold)])
        kept_array = np.asarray(kept, np.int32)
        kept_array = kept_array[np.argsort(-scores[kept_array])[:max_detections]]
        boxes = boxes[kept_array]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, resized_shape[1]) / scale_x
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, resized_shape[0]) / scale_y
        return Detections(boxes, scores[kept_array], labels[kept_array]), original

