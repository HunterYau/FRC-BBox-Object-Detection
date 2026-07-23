"""Custom TensorFlow training loop with resume, validation, and best-model saves."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from .coco import CocoIndex
from .config import ExperimentConfig
from .inference import Detections, decode_predictions
from .losses import LossOutput, compute_detection_loss
from .metrics import DetectionEvaluator


@tf.keras.utils.register_keras_serializable(package="FRCDetector")
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        peak_learning_rate: float,
        total_steps: int,
        warmup_steps: int,
        minimum_ratio: float,
        name: str = "warmup_cosine",
    ):
        super().__init__()
        self.peak_learning_rate = float(peak_learning_rate)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.minimum_ratio = float(minimum_ratio)
        self.name = name

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        with tf.name_scope(self.name):
            step = tf.cast(step, tf.float32)
            peak = tf.cast(self.peak_learning_rate, tf.float32)
            warmup_steps = tf.cast(max(self.warmup_steps, 1), tf.float32)
            warmup_rate = peak * (step + 1.0) / warmup_steps
            progress = (step - float(self.warmup_steps)) / float(
                max(self.total_steps - self.warmup_steps, 1)
            )
            progress = tf.clip_by_value(progress, 0.0, 1.0)
            cosine = 0.5 * (1.0 + tf.cos(tf.constant(math.pi) * progress))
            decay_rate = peak * (self.minimum_ratio + (1.0 - self.minimum_ratio) * cosine)
            return tf.where(step < float(self.warmup_steps), warmup_rate, decay_rate)

    def get_config(self) -> dict[str, Any]:
        return {
            "peak_learning_rate": self.peak_learning_rate,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "minimum_ratio": self.minimum_ratio,
            "name": self.name,
        }


def configure_tensorflow(config: ExperimentConfig) -> None:
    tf.keras.utils.set_random_seed(config.data.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    if config.training.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")


class Trainer:
    def __init__(
        self,
        model: tf.keras.Model,
        train_dataset: tf.data.Dataset,
        validation_dataset: tf.data.Dataset,
        train_index: CocoIndex,
        validation_index: CocoIndex,
        config: ExperimentConfig,
        *,
        resume: bool | None = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.train_index = train_index
        self.validation_index = validation_index
        self.config = config
        self.num_classes = len(train_index.class_names)
        self.output_dir = config.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.steps_per_epoch = math.ceil(len(train_index) / config.data.batch_size)
        total_steps = max(self.steps_per_epoch * config.training.epochs, 1)
        schedule = WarmupCosine(
            config.training.initial_learning_rate,
            total_steps,
            self.steps_per_epoch * config.training.warmup_epochs,
            config.training.minimum_learning_rate_ratio,
        )
        optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.AdamW(
            learning_rate=schedule,
            weight_decay=config.training.weight_decay,
            global_clipnorm=config.training.gradient_clip_norm,
        )
        if config.training.mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        self.optimizer = optimizer

        self.epoch = tf.Variable(0, dtype=tf.int64, trainable=False, name="completed_epoch")
        self.best_map = tf.Variable(-1.0, dtype=tf.float32, trainable=False, name="best_map")
        self.checkpoint = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.epoch,
            best_map=self.best_map,
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            str(self.output_dir / "checkpoints"),
            max_to_keep=config.training.checkpoint_keep,
        )
        should_resume = config.training.auto_resume if resume is None else resume
        if should_resume and self.checkpoint_manager.latest_checkpoint:
            status = self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            status.expect_partial()
            print(
                f"Resumed {self.checkpoint_manager.latest_checkpoint} "
                f"at completed epoch {int(self.epoch.numpy())}."
            )

        self.history_path = self.output_dir / "history.json"
        self.history: list[dict[str, Any]] = []
        if self.history_path.is_file() and int(self.epoch.numpy()) > 0:
            try:
                self.history = json.loads(self.history_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.history = []

    def _loss(self, batch: dict[str, tf.Tensor], training: bool) -> LossOutput:
        predictions = self.model(batch["image"], training=training)
        return compute_detection_loss(
            predictions,
            batch["boxes"],
            batch["labels"],
            batch["image_shape"],
            self.num_classes,
            self.config.model,
            self.config.training,
            self.model.losses,
        )

    @tf.function(reduce_retracing=True)
    def _train_step(self, batch: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            losses = self._loss(batch, training=True)
            optimized_loss = losses.total
            if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                optimized_loss = self.optimizer.scale_loss(optimized_loss)
        variables = self.model.trainable_variables
        gradients = tape.gradient(optimized_loss, variables)
        gradient_pairs = [(gradient, variable) for gradient, variable in zip(gradients, variables) if gradient is not None]
        self.optimizer.apply_gradients(gradient_pairs)
        return {
            "total": losses.total,
            "classification": losses.classification,
            "box": losses.box,
            "centerness": losses.centerness,
            "regularization": losses.regularization,
        }

    @tf.function(reduce_retracing=True)
    def _validation_step(
        self, batch: dict[str, tf.Tensor]
    ) -> tuple[dict[str, tf.Tensor], dict[str, tf.Tensor]]:
        predictions = self.model(batch["image"], training=False)
        losses = compute_detection_loss(
            predictions,
            batch["boxes"],
            batch["labels"],
            batch["image_shape"],
            self.num_classes,
            self.config.model,
            self.config.training,
            self.model.losses,
        )
        return predictions, {
            "total": losses.total,
            "classification": losses.classification,
            "box": losses.box,
            "centerness": losses.centerness,
            "regularization": losses.regularization,
        }

    @staticmethod
    def _mean_losses(values: list[dict[str, float]]) -> dict[str, float]:
        if not values:
            return {}
        return {key: float(np.mean([item[key] for item in values])) for key in values[0]}

    def train_epoch(self, epoch_index: int) -> dict[str, float]:
        epoch_losses: list[dict[str, float]] = []
        started = time.perf_counter()
        for step, batch in enumerate(self.train_dataset, start=1):
            losses = self._train_step(batch)
            values = {key: float(value.numpy()) for key, value in losses.items()}
            epoch_losses.append(values)
            if step == 1 or step % 20 == 0 or step == self.steps_per_epoch:
                print(
                    f"\rEpoch {epoch_index + 1}/{self.config.training.epochs} "
                    f"step {step}/{self.steps_per_epoch} loss={values['total']:.4f}",
                    end="",
                    flush=True,
                )
        print(f" - {time.perf_counter() - started:.1f}s")
        return {f"train_{key}": value for key, value in self._mean_losses(epoch_losses).items()}

    def validate(self) -> dict[str, float]:
        evaluator = DetectionEvaluator(
            self.num_classes, self.config.training.summary_score_threshold
        )
        validation_losses: list[dict[str, float]] = []
        for batch in self.validation_dataset:
            predictions, losses = self._validation_step(batch)
            validation_losses.append({key: float(value.numpy()) for key, value in losses.items()})
            boxes, scores, labels, valid = decode_predictions(
                predictions,
                batch["image_shape"],
                self.num_classes,
                self.config.model.strides,
                score_threshold=self.config.training.validation_score_threshold,
                iou_threshold=self.config.training.nms_iou_threshold,
                max_detections=self.config.training.max_detections,
            )
            for index in range(int(tf.shape(batch["image"])[0].numpy())):
                count = int(valid[index].numpy())
                true_mask = batch["labels"][index].numpy() >= 0
                evaluator.update(
                    int(batch["image_id"][index].numpy()),
                    Detections(
                        boxes[index, :count].numpy(),
                        scores[index, :count].numpy(),
                        labels[index, :count].numpy(),
                    ),
                    batch["boxes"][index].numpy()[true_mask],
                    batch["labels"][index].numpy()[true_mask],
                )
        results = {f"validation_{key}": value for key, value in self._mean_losses(validation_losses).items()}
        results.update({f"validation_{key}": value for key, value in evaluator.result().items()})
        return results

    def _learning_rate(self) -> float:
        optimizer = self.optimizer
        if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            optimizer = optimizer.inner_optimizer
        learning_rate = optimizer.learning_rate
        if callable(learning_rate):
            learning_rate = learning_rate(optimizer.iterations)
        return float(tf.keras.backend.get_value(learning_rate))

    def _write_metadata(self) -> None:
        metadata = {
            "format_version": 1,
            "model_type": "MobileNetV2-FPN-FCOS",
            "num_classes": self.num_classes,
            "class_names": list(self.train_index.class_names),
            "category_ids": list(self.train_index.category_ids),
            "strides": list(self.config.model.strides),
            "size_ranges": [list(value) for value in self.config.model.size_ranges],
            "input": {
                "dtype": "float32",
                "range": [0.0, 1.0],
                "layout": "NHWC",
                "dynamic_height_width": True,
                "pad_to_multiple": self.config.data.pad_to_multiple,
            },
            "prediction_channels": "[class_logits, raw_ltrb, centerness_logit]",
            "config": self.config.to_dict(),
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    def fit(self) -> Path:
        start_epoch = int(self.epoch.numpy())
        patience_count = 0
        best_weights_path = self.output_dir / "best.weights.h5"
        self._write_metadata()
        if start_epoch >= self.config.training.epochs:
            print(
                f"Checkpoint already completed {start_epoch} epochs; increase EPOCHS or pass --epochs."
            )

        for epoch_index in range(start_epoch, self.config.training.epochs):
            train_results = self.train_epoch(epoch_index)
            validation_results = self.validate()
            current_map = validation_results["validation_map"]
            improved = current_map > float(self.best_map.numpy()) + 1e-6
            if improved:
                self.best_map.assign(current_map)
                self.model.save_weights(best_weights_path)
                patience_count = 0
            else:
                patience_count += 1

            self.epoch.assign(epoch_index + 1)
            checkpoint_path = self.checkpoint_manager.save(checkpoint_number=epoch_index + 1)
            epoch_result: dict[str, Any] = {
                "epoch": epoch_index + 1,
                "learning_rate": self._learning_rate(),
                **train_results,
                **validation_results,
                "best": improved,
                "checkpoint": checkpoint_path,
            }
            self.history.append(epoch_result)
            self.history_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
            print(
                f"val loss={validation_results['validation_total']:.4f}, "
                f"mAP={current_map:.4f}, AP50={validation_results['validation_ap50']:.4f}, "
                f"precision={validation_results['validation_precision50']:.4f}, "
                f"recall={validation_results['validation_recall50']:.4f}"
                + (" (new best)" if improved else "")
            )
            if patience_count >= self.config.training.early_stopping_patience:
                print(f"Early stopping after {patience_count} epochs without mAP improvement.")
                break

        if best_weights_path.is_file():
            self.model.load_weights(best_weights_path)
        model_path = self.output_dir / "trained_detector.keras"
        self.model.save(model_path, include_optimizer=False)
        self._write_metadata()
        print(f"Saved reloadable best model to {model_path}")
        print(f"Latest resumable training state: {self.checkpoint_manager.latest_checkpoint}")
        return model_path

