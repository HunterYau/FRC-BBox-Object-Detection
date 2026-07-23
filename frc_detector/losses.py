"""FCOS target assignment, focal loss, centerness, and generalized IoU."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from .config import ModelConfig, TrainingConfig


@dataclass
class LossOutput:
    total: tf.Tensor
    classification: tf.Tensor
    box: tf.Tensor
    centerness: tf.Tensor
    regularization: tf.Tensor
    positive_locations: tf.Tensor


def make_points(height: tf.Tensor, width: tf.Tensor, stride: int) -> tf.Tensor:
    x = (tf.cast(tf.range(width), tf.float32) + 0.5) * float(stride)
    y = (tf.cast(tf.range(height), tf.float32) + 0.5) * float(stride)
    grid_x, grid_y = tf.meshgrid(x, y)
    return tf.stack([tf.reshape(grid_x, (-1,)), tf.reshape(grid_y, (-1,))], axis=-1)


def _assign_level(
    points: tf.Tensor,
    boxes: tf.Tensor,
    labels: tf.Tensor,
    image_shapes: tf.Tensor,
    size_range: tuple[float, float],
    stride: int,
    center_radius: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    # Add an invalid dummy box so all-background batches remain well-defined.
    batch_size = tf.shape(boxes)[0]
    boxes = tf.concat([boxes, tf.zeros((batch_size, 1, 4), boxes.dtype)], axis=1)
    labels = tf.concat([labels, tf.fill((batch_size, 1), tf.constant(-1, labels.dtype))], axis=1)

    px = points[None, None, :, 0]
    py = points[None, None, :, 1]
    x1 = boxes[:, :, None, 0]
    y1 = boxes[:, :, None, 1]
    x2 = boxes[:, :, None, 2]
    y2 = boxes[:, :, None, 3]
    left = px - x1
    top = py - y1
    right = x2 - px
    bottom = y2 - py
    distances = tf.stack([left, top, right, bottom], axis=-1)

    inside_box = tf.reduce_min(distances, axis=-1) > 0.0
    maximum_distance = tf.reduce_max(distances, axis=-1)
    in_size_range = tf.logical_and(
        maximum_distance >= float(size_range[0]), maximum_distance < float(size_range[1])
    )
    valid_ground_truth = labels[:, :, None] >= 0

    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    radius = float(stride) * center_radius
    center_x1 = tf.maximum(x1, center_x - radius)
    center_y1 = tf.maximum(y1, center_y - radius)
    center_x2 = tf.minimum(x2, center_x + radius)
    center_y2 = tf.minimum(y2, center_y + radius)
    inside_center = tf.logical_and(
        tf.logical_and(px > center_x1, px < center_x2),
        tf.logical_and(py > center_y1, py < center_y2),
    )
    candidates = tf.logical_and(
        tf.logical_and(inside_box, inside_center), tf.logical_and(in_size_range, valid_ground_truth)
    )

    areas = tf.maximum(x2 - x1, 0.0) * tf.maximum(y2 - y1, 0.0)
    infinity = tf.constant(float("inf"), tf.float32)
    candidate_areas = tf.where(candidates, areas, infinity)
    matched_indices = tf.argmin(candidate_areas, axis=1, output_type=tf.int32)
    matched_area = tf.reduce_min(candidate_areas, axis=1)

    valid_points = tf.logical_and(
        points[None, :, 0] < tf.cast(image_shapes[:, None, 1], tf.float32),
        points[None, :, 1] < tf.cast(image_shapes[:, None, 0], tf.float32),
    )
    positive = tf.logical_and(tf.math.is_finite(matched_area), valid_points)
    matched_boxes = tf.gather(boxes, matched_indices, axis=1, batch_dims=1)
    matched_labels = tf.gather(labels, matched_indices, axis=1, batch_dims=1)
    target_left = points[None, :, 0] - matched_boxes[:, :, 0]
    target_top = points[None, :, 1] - matched_boxes[:, :, 1]
    target_right = matched_boxes[:, :, 2] - points[None, :, 0]
    target_bottom = matched_boxes[:, :, 3] - points[None, :, 1]
    target_distances = tf.stack([target_left, target_top, target_right, target_bottom], axis=-1)
    target_distances = tf.where(positive[:, :, None], target_distances, 0.0)
    matched_labels = tf.where(positive, matched_labels, -1)

    horizontal = tf.minimum(target_left, target_right) / tf.maximum(
        tf.maximum(target_left, target_right), 1e-6
    )
    vertical = tf.minimum(target_top, target_bottom) / tf.maximum(
        tf.maximum(target_top, target_bottom), 1e-6
    )
    centerness = tf.sqrt(tf.maximum(horizontal * vertical, 0.0))
    centerness = tf.where(positive, centerness, 0.0)
    return matched_labels, target_distances, centerness, positive, valid_points


def _focal_loss(
    logits: tf.Tensor,
    targets: tf.Tensor,
    valid_mask: tf.Tensor,
    alpha: float,
    gamma: float,
) -> tf.Tensor:
    logits = tf.cast(logits, tf.float32)
    targets = tf.cast(targets, tf.float32)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits)
    probabilities = tf.sigmoid(logits)
    probability_t = targets * probabilities + (1.0 - targets) * (1.0 - probabilities)
    alpha_factor = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    modulating_factor = tf.pow(1.0 - probability_t, gamma)
    loss = alpha_factor * modulating_factor * cross_entropy
    return tf.reduce_sum(loss * tf.cast(valid_mask[:, :, None], tf.float32))


def _generalized_iou_from_distances(
    predicted: tf.Tensor, target: tf.Tensor, points: tf.Tensor
) -> tf.Tensor:
    predicted = tf.cast(predicted, tf.float32)
    target = tf.cast(target, tf.float32)
    px = points[None, :, 0]
    py = points[None, :, 1]
    predicted_box = tf.stack(
        [px - predicted[:, :, 0], py - predicted[:, :, 1], px + predicted[:, :, 2], py + predicted[:, :, 3]],
        axis=-1,
    )
    target_box = tf.stack(
        [px - target[:, :, 0], py - target[:, :, 1], px + target[:, :, 2], py + target[:, :, 3]],
        axis=-1,
    )
    intersection_x1 = tf.maximum(predicted_box[:, :, 0], target_box[:, :, 0])
    intersection_y1 = tf.maximum(predicted_box[:, :, 1], target_box[:, :, 1])
    intersection_x2 = tf.minimum(predicted_box[:, :, 2], target_box[:, :, 2])
    intersection_y2 = tf.minimum(predicted_box[:, :, 3], target_box[:, :, 3])
    intersection = tf.maximum(intersection_x2 - intersection_x1, 0.0) * tf.maximum(
        intersection_y2 - intersection_y1, 0.0
    )
    predicted_area = tf.maximum(predicted_box[:, :, 2] - predicted_box[:, :, 0], 0.0) * tf.maximum(
        predicted_box[:, :, 3] - predicted_box[:, :, 1], 0.0
    )
    target_area = tf.maximum(target_box[:, :, 2] - target_box[:, :, 0], 0.0) * tf.maximum(
        target_box[:, :, 3] - target_box[:, :, 1], 0.0
    )
    union = predicted_area + target_area - intersection
    iou = intersection / tf.maximum(union, 1e-6)

    enclosing_x1 = tf.minimum(predicted_box[:, :, 0], target_box[:, :, 0])
    enclosing_y1 = tf.minimum(predicted_box[:, :, 1], target_box[:, :, 1])
    enclosing_x2 = tf.maximum(predicted_box[:, :, 2], target_box[:, :, 2])
    enclosing_y2 = tf.maximum(predicted_box[:, :, 3], target_box[:, :, 3])
    enclosing_area = tf.maximum(enclosing_x2 - enclosing_x1, 0.0) * tf.maximum(
        enclosing_y2 - enclosing_y1, 0.0
    )
    return iou - (enclosing_area - union) / tf.maximum(enclosing_area, 1e-6)


def compute_detection_loss(
    predictions: dict[str, tf.Tensor],
    boxes: tf.Tensor,
    labels: tf.Tensor,
    image_shapes: tf.Tensor,
    num_classes: int,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    regularization_losses: list[tf.Tensor] | None = None,
) -> LossOutput:
    classification_sums: list[tf.Tensor] = []
    box_sums: list[tf.Tensor] = []
    center_sums: list[tf.Tensor] = []
    center_weight_sums: list[tf.Tensor] = []
    positive_counts: list[tf.Tensor] = []

    ordered_predictions = [predictions[key] for key in sorted(predictions)]
    for prediction, stride, size_range in zip(
        ordered_predictions, model_config.strides, model_config.size_ranges, strict=True
    ):
        shape = tf.shape(prediction)
        points = make_points(shape[1], shape[2], stride)
        flattened = tf.reshape(prediction, (shape[0], -1, num_classes + 5))
        class_logits = flattened[:, :, :num_classes]
        raw_distances = flattened[:, :, num_classes : num_classes + 4]
        center_logits = flattened[:, :, num_classes + 4]
        predicted_distances = tf.nn.softplus(tf.cast(raw_distances, tf.float32)) * float(stride)

        target_labels, target_distances, target_center, positive, valid = _assign_level(
            points,
            boxes,
            labels,
            image_shapes,
            size_range,
            stride,
            model_config.center_sampling_radius,
        )
        safe_labels = tf.maximum(target_labels, 0)
        one_hot = tf.one_hot(safe_labels, num_classes, dtype=tf.float32)
        one_hot = tf.where(positive[:, :, None], one_hot, 0.0)
        classification_sums.append(
            _focal_loss(
                class_logits,
                one_hot,
                valid,
                training_config.focal_alpha,
                training_config.focal_gamma,
            )
        )

        giou = _generalized_iou_from_distances(predicted_distances, target_distances, points)
        positive_float = tf.cast(positive, tf.float32)
        box_sums.append(tf.reduce_sum((1.0 - giou) * target_center * positive_float))
        center_weight_sums.append(tf.reduce_sum(target_center * positive_float))
        center_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(target_center, tf.float32), logits=tf.cast(center_logits, tf.float32)
        )
        center_sums.append(tf.reduce_sum(center_loss * positive_float))
        positive_counts.append(tf.reduce_sum(positive_float))

    positive_count = tf.maximum(tf.add_n(positive_counts), 1.0)
    center_weight = tf.maximum(tf.add_n(center_weight_sums), 1.0)
    classification_loss = tf.add_n(classification_sums) / positive_count
    box_loss = tf.add_n(box_sums) / center_weight
    centerness_loss = tf.add_n(center_sums) / positive_count
    if regularization_losses:
        regularization_loss = tf.add_n([tf.cast(value, tf.float32) for value in regularization_losses])
    else:
        regularization_loss = tf.constant(0.0, tf.float32)
    total = (
        classification_loss
        + training_config.box_loss_weight * box_loss
        + training_config.centerness_loss_weight * centerness_loss
        + regularization_loss
    )
    return LossOutput(
        total=total,
        classification=classification_loss,
        box=box_loss,
        centerness=centerness_loss,
        regularization=regularization_loss,
        positive_locations=positive_count,
    )
