"""MobileNetV2-FPN anchor-free detector architecture."""

from __future__ import annotations

import math

import tensorflow as tf

from .config import ModelConfig


@tf.keras.utils.register_keras_serializable(package="FRCDetector")
class ResizeLike(tf.keras.layers.Layer):
    """Resize the first feature map to the spatial shape of the second."""

    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        source, target = inputs
        return tf.image.resize(source, tf.shape(target)[1:3], method="nearest")

    def compute_output_shape(self, input_shape):
        source_shape, target_shape = input_shape
        return (source_shape[0], target_shape[1], target_shape[2], source_shape[3])


def _conv_regularizer(config: ModelConfig) -> tf.keras.regularizers.Regularizer:
    return tf.keras.regularizers.L2(config.l2_regularization)


def _separable_block(name: str, channels: int, config: ModelConfig) -> tf.keras.Sequential:
    regularizer = _conv_regularizer(config)
    return tf.keras.Sequential(
        [
            tf.keras.layers.SeparableConv2D(
                channels,
                3,
                padding="same",
                use_bias=False,
                depthwise_regularizer=regularizer,
                pointwise_regularizer=regularizer,
            ),
            tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5),
            tf.keras.layers.Activation("swish"),
        ],
        name=name,
    )


def _prediction_head(
    features: list[tf.Tensor], num_classes: int, config: ModelConfig
) -> dict[str, tf.Tensor]:
    class_tower = [
        _separable_block(f"class_tower_{index}", config.head_channels, config)
        for index in range(config.head_depth)
    ]
    box_tower = [
        _separable_block(f"box_tower_{index}", config.head_channels, config)
        for index in range(config.head_depth)
    ]
    prior_probability = 0.01
    class_bias = -math.log((1.0 - prior_probability) / prior_probability)
    regularizer = _conv_regularizer(config)
    class_output = tf.keras.layers.Conv2D(
        num_classes,
        3,
        padding="same",
        bias_initializer=tf.keras.initializers.Constant(class_bias),
        kernel_regularizer=regularizer,
        name="class_logits",
    )
    box_output = tf.keras.layers.Conv2D(
        4,
        3,
        padding="same",
        bias_initializer=tf.keras.initializers.Constant(1.0),
        kernel_regularizer=regularizer,
        name="box_distances_raw",
    )
    center_output = tf.keras.layers.Conv2D(
        1,
        3,
        padding="same",
        kernel_regularizer=regularizer,
        name="centerness_logits",
    )

    predictions: dict[str, tf.Tensor] = {}
    for level, feature in enumerate(features, start=3):
        class_feature = feature
        box_feature = feature
        for layer in class_tower:
            class_feature = layer(class_feature)
        for layer in box_tower:
            box_feature = layer(box_feature)
        prediction = tf.keras.layers.Concatenate(axis=-1, name=f"p{level}")(
            [class_output(class_feature), box_output(box_feature), center_output(box_feature)]
        )
        predictions[f"p{level}"] = prediction
    return predictions


def build_detector(
    num_classes: int,
    config: ModelConfig,
    *,
    backbone_weights: str | None = None,
) -> tf.keras.Model:
    """Build a dynamic-resolution detector returning raw P3-P6 predictions.

    Each output channel layout is ``[class logits, raw LTRB, centerness]``.
    Bounding-box distances become positive during loss/inference using softplus.
    """
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    if config.backbone.lower() != "mobilenet_v2":
        raise ValueError(f"Unsupported backbone: {config.backbone}")
    if len(config.strides) != 4 or len(config.size_ranges) != 4:
        raise ValueError("The P3-P6 detector requires exactly four strides and size ranges.")

    selected_weights = config.backbone_weights if backbone_weights is None else backbone_weights
    images = tf.keras.Input(shape=(None, None, 3), dtype=tf.float32, name="images")
    normalized = tf.keras.layers.Rescaling(2.0, offset=-1.0, name="mobilenet_normalization")(images)
    backbone = tf.keras.applications.MobileNetV2(
        input_tensor=normalized,
        include_top=False,
        weights=selected_weights,
        alpha=1.0,
    )
    backbone.name = "mobilenet_v2_backbone"
    backbone_outputs = [
        backbone.get_layer("block_5_add").output,
        backbone.get_layer("block_12_add").output,
        backbone.get_layer("out_relu").output,
    ]
    feature_extractor = tf.keras.Model(images, backbone_outputs, name="backbone_features")
    c3, c4, c5 = feature_extractor(images)

    regularizer = _conv_regularizer(config)
    lateral3 = tf.keras.layers.Conv2D(
        config.fpn_channels, 1, kernel_regularizer=regularizer, name="lateral_c3"
    )(c3)
    lateral4 = tf.keras.layers.Conv2D(
        config.fpn_channels, 1, kernel_regularizer=regularizer, name="lateral_c4"
    )(c4)
    lateral5 = tf.keras.layers.Conv2D(
        config.fpn_channels, 1, kernel_regularizer=regularizer, name="lateral_c5"
    )(c5)

    p5 = lateral5
    p4 = tf.keras.layers.Add(name="fpn_merge_p4")([lateral4, ResizeLike(name="resize_p5_to_p4")([p5, lateral4])])
    p3 = tf.keras.layers.Add(name="fpn_merge_p3")([lateral3, ResizeLike(name="resize_p4_to_p3")([p4, lateral3])])
    p3 = tf.keras.layers.Conv2D(
        config.fpn_channels, 3, padding="same", kernel_regularizer=regularizer, name="fpn_p3"
    )(p3)
    p4 = tf.keras.layers.Conv2D(
        config.fpn_channels, 3, padding="same", kernel_regularizer=regularizer, name="fpn_p4"
    )(p4)
    p5 = tf.keras.layers.Conv2D(
        config.fpn_channels, 3, padding="same", kernel_regularizer=regularizer, name="fpn_p5"
    )(p5)
    p6 = tf.keras.layers.Conv2D(
        config.fpn_channels,
        3,
        strides=2,
        padding="same",
        kernel_regularizer=regularizer,
        name="fpn_p6",
    )(p5)

    predictions = _prediction_head([p3, p4, p5, p6], num_classes, config)
    return tf.keras.Model(images, predictions, name="frc_fcos_detector")


def get_backbone(model: tf.keras.Model) -> tf.keras.Model:
    return model.get_layer("backbone_features").get_layer("mobilenet_v2_backbone")

