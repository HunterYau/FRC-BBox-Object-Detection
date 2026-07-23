"""Bounding-box-aware TensorFlow image augmentations."""

from __future__ import annotations

import tensorflow as tf

from .config import AugmentationConfig


def _coin_flip(probability: float) -> tf.Tensor:
    return tf.random.uniform(()) < probability


def _horizontal_flip(image: tf.Tensor, boxes: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    width = tf.cast(tf.shape(image)[1], tf.float32)
    x1, y1, x2, y2 = tf.unstack(boxes, axis=-1)
    boxes = tf.stack([width - x2, y1, width - x1, y2], axis=-1)
    return tf.image.flip_left_right(image), boxes


def _random_crop(
    image: tf.Tensor, boxes: tf.Tensor, labels: tf.Tensor, minimum_scale: float
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    scale = tf.random.uniform((), minimum_scale, 1.0)
    crop_height = tf.maximum(tf.cast(tf.cast(height, tf.float32) * scale, tf.int32), 2)
    crop_width = tf.maximum(tf.cast(tf.cast(width, tf.float32) * scale, tf.int32), 2)
    offset_y = tf.random.uniform((), 0, tf.maximum(height - crop_height + 1, 1), dtype=tf.int32)
    offset_x = tf.random.uniform((), 0, tf.maximum(width - crop_width + 1, 1), dtype=tf.int32)

    cropped = tf.image.crop_to_bounding_box(image, offset_y, offset_x, crop_height, crop_width)
    offset = tf.cast(tf.stack([offset_x, offset_y, offset_x, offset_y]), tf.float32)
    shifted = boxes - offset
    x1, y1, x2, y2 = tf.unstack(shifted, axis=-1)
    centers_x = (x1 + x2) * 0.5
    centers_y = (y1 + y2) * 0.5
    keep = tf.logical_and(
        tf.logical_and(centers_x >= 0.0, centers_x <= tf.cast(crop_width, tf.float32)),
        tf.logical_and(centers_y >= 0.0, centers_y <= tf.cast(crop_height, tf.float32)),
    )
    clipped = tf.stack(
        [
            tf.clip_by_value(x1, 0.0, tf.cast(crop_width, tf.float32)),
            tf.clip_by_value(y1, 0.0, tf.cast(crop_height, tf.float32)),
            tf.clip_by_value(x2, 0.0, tf.cast(crop_width, tf.float32)),
            tf.clip_by_value(y2, 0.0, tf.cast(crop_height, tf.float32)),
        ],
        axis=-1,
    )
    valid_size = tf.logical_and(clipped[:, 2] - clipped[:, 0] >= 2.0, clipped[:, 3] - clipped[:, 1] >= 2.0)
    keep = tf.logical_and(keep, valid_size)
    kept_boxes = tf.boolean_mask(clipped, keep)
    kept_labels = tf.boolean_mask(labels, keep)

    # Do not turn an annotated image into a background-only image by accident.
    use_crop = tf.logical_or(tf.size(labels) == 0, tf.size(kept_labels) > 0)
    return tf.cond(
        use_crop,
        lambda: (cropped, kept_boxes, kept_labels),
        lambda: (image, boxes, labels),
    )


def _zoom_out(image: tf.Tensor, boxes: tf.Tensor, maximum_scale: float) -> tuple[tf.Tensor, tf.Tensor]:
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    scale = tf.random.uniform((), 1.0, maximum_scale)
    canvas_height = tf.maximum(tf.cast(tf.cast(height, tf.float32) * scale, tf.int32), height)
    canvas_width = tf.maximum(tf.cast(tf.cast(width, tf.float32) * scale, tf.int32), width)
    offset_y = tf.random.uniform((), 0, canvas_height - height + 1, dtype=tf.int32)
    offset_x = tf.random.uniform((), 0, canvas_width - width + 1, dtype=tf.int32)
    image = tf.image.pad_to_bounding_box(image, offset_y, offset_x, canvas_height, canvas_width)
    offset = tf.cast(tf.stack([offset_x, offset_y, offset_x, offset_y]), tf.float32)
    return image, boxes + offset


def _rotate_90(image: tf.Tensor, boxes: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    k = tf.random.uniform((), 1, 4, dtype=tf.int32)
    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)
    x1, y1, x2, y2 = tf.unstack(boxes, axis=-1)
    rotated_boxes = tf.switch_case(
        k - 1,
        branch_fns=(
            lambda: tf.stack([y1, width - x2, y2, width - x1], axis=-1),
            lambda: tf.stack([width - x2, height - y2, width - x1, height - y1], axis=-1),
            lambda: tf.stack([height - y2, x1, height - y1, x2], axis=-1),
        ),
    )
    return tf.image.rot90(image, k), rotated_boxes


def _color_jitter(image: tf.Tensor, config: AugmentationConfig) -> tf.Tensor:
    image = tf.image.random_brightness(image, config.brightness_delta)
    image = tf.image.random_contrast(image, *config.contrast_range)
    image = tf.image.random_saturation(image, *config.saturation_range)
    image = tf.image.random_hue(image, config.hue_delta)
    return tf.clip_by_value(image, 0.0, 1.0)


def _gaussian_blur(image: tf.Tensor) -> tf.Tensor:
    kernel_1d = tf.constant([1.0, 4.0, 6.0, 4.0, 1.0], tf.float32)
    kernel = tf.tensordot(kernel_1d, kernel_1d, axes=0)
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = tf.tile(kernel[:, :, None, None], [1, 1, 3, 1])
    return tf.nn.depthwise_conv2d(image[None], kernel, strides=[1, 1, 1, 1], padding="SAME")[0]


def _cutout(image: tf.Tensor, maximum_fraction: float) -> tf.Tensor:
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    cut_height = tf.maximum(
        tf.cast(tf.cast(height, tf.float32) * tf.random.uniform((), 0.05, maximum_fraction), tf.int32), 1
    )
    cut_width = tf.maximum(
        tf.cast(tf.cast(width, tf.float32) * tf.random.uniform((), 0.05, maximum_fraction), tf.int32), 1
    )
    offset_y = tf.random.uniform((), 0, tf.maximum(height - cut_height + 1, 1), dtype=tf.int32)
    offset_x = tf.random.uniform((), 0, tf.maximum(width - cut_width + 1, 1), dtype=tf.int32)
    hole = tf.ones((cut_height, cut_width, 1), tf.float32)
    mask = tf.image.pad_to_bounding_box(hole, offset_y, offset_x, height, width)
    mask = 1.0 - mask
    return image * mask


def augment_sample(
    image: tf.Tensor,
    boxes: tf.Tensor,
    labels: tf.Tensor,
    config: AugmentationConfig,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Apply geometry-safe and photometric augmentations."""
    image, boxes = tf.cond(
        _coin_flip(config.horizontal_flip_probability),
        lambda: _horizontal_flip(image, boxes),
        lambda: (image, boxes),
    )
    image, boxes, labels = tf.cond(
        _coin_flip(config.random_crop_probability),
        lambda: _random_crop(image, boxes, labels, config.minimum_crop_scale),
        lambda: (image, boxes, labels),
    )
    image, boxes = tf.cond(
        _coin_flip(config.zoom_out_probability),
        lambda: _zoom_out(image, boxes, config.maximum_zoom_out),
        lambda: (image, boxes),
    )
    image, boxes = tf.cond(
        _coin_flip(config.rotate_90_probability),
        lambda: _rotate_90(image, boxes),
        lambda: (image, boxes),
    )

    image = tf.cond(
        _coin_flip(config.color_jitter_probability),
        lambda: _color_jitter(image, config),
        lambda: image,
    )
    image = tf.cond(
        _coin_flip(config.gamma_probability),
        lambda: tf.pow(tf.clip_by_value(image, 0.0, 1.0), tf.random.uniform((), *config.gamma_range)),
        lambda: image,
    )
    image = tf.cond(
        _coin_flip(config.grayscale_probability),
        lambda: tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image)),
        lambda: image,
    )
    image = tf.cond(
        _coin_flip(config.jpeg_probability),
        lambda: tf.image.convert_image_dtype(
            tf.image.random_jpeg_quality(
                tf.image.convert_image_dtype(tf.clip_by_value(image, 0.0, 1.0), tf.uint8),
                config.jpeg_quality_range[0],
                config.jpeg_quality_range[1],
            ),
            tf.float32,
        ),
        lambda: image,
    )
    image = tf.cond(
        _coin_flip(config.gaussian_blur_probability), lambda: _gaussian_blur(image), lambda: image
    )
    image = tf.cond(
        _coin_flip(config.gaussian_noise_probability),
        lambda: image + tf.random.normal(tf.shape(image), stddev=config.gaussian_noise_stddev),
        lambda: image,
    )
    image = tf.cond(
        _coin_flip(config.cutout_probability),
        lambda: _cutout(image, config.maximum_cutout_fraction),
        lambda: image,
    )
    return tf.clip_by_value(image, 0.0, 1.0), boxes, labels


def resize_preserving_aspect_ratio(
    image: tf.Tensor,
    boxes: tf.Tensor,
    target_short_side: tf.Tensor | int,
    maximum_long_side: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    shape = tf.cast(tf.shape(image)[:2], tf.float32)
    short_side = tf.reduce_min(shape)
    long_side = tf.reduce_max(shape)
    scale = tf.cast(target_short_side, tf.float32) / short_side
    scale = tf.minimum(scale, tf.cast(maximum_long_side, tf.float32) / long_side)
    new_shape = tf.maximum(tf.cast(tf.round(shape * scale), tf.int32), 1)
    image = tf.image.resize(image, new_shape, method=tf.image.ResizeMethod.BILINEAR, antialias=True)
    scale_y = tf.cast(new_shape[0], tf.float32) / shape[0]
    scale_x = tf.cast(new_shape[1], tf.float32) / shape[1]
    box_scale = tf.stack([scale_x, scale_y, scale_x, scale_y])
    return image, boxes * box_scale
