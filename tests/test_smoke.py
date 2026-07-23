from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from frc_detector.coco import build_dataset, load_coco_index
from frc_detector.config import (
    AugmentationConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)
from frc_detector.inference import decode_predictions
from frc_detector.losses import compute_detection_loss
from frc_detector.model import build_detector
from frc_detector.trainer import Trainer
from frc_detector.augment import augment_sample


class DetectorSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_config = ModelConfig(
            backbone_weights=None,
            fpn_channels=32,
            head_channels=32,
            head_depth=1,
        )
        cls.model = build_detector(2, cls.model_config)

    def test_variable_resolution_forward_loss_and_decode(self) -> None:
        for height, width in ((128, 160), (160, 128)):
            images = tf.random.uniform((1, height, width, 3))
            boxes = tf.constant([[[20.0, 18.0, 82.0, 91.0]]])
            labels = tf.constant([[1]], tf.int32)
            image_shapes = tf.constant([[height, width]], tf.int32)
            predictions = self.model(images, training=True)
            self.assertEqual(tuple(predictions), ("p3", "p4", "p5", "p6"))
            losses = compute_detection_loss(
                predictions,
                boxes,
                labels,
                image_shapes,
                2,
                self.model_config,
                TrainingConfig(),
                self.model.losses,
            )
            self.assertTrue(np.isfinite(float(losses.total.numpy())))
            decoded = decode_predictions(
                predictions, image_shapes, 2, self.model_config.strides, max_detections=10
            )
            self.assertEqual(decoded[0].shape, (1, 10, 4))

    def test_backpropagation_and_keras_reload(self) -> None:
        images = tf.random.uniform((1, 128, 160, 3))
        boxes = tf.constant([[[20.0, 18.0, 82.0, 91.0]]])
        labels = tf.constant([[1]], tf.int32)
        image_shapes = tf.constant([[128, 160]], tf.int32)
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            losses = compute_detection_loss(
                predictions,
                boxes,
                labels,
                image_shapes,
                2,
                self.model_config,
                TrainingConfig(),
                self.model.losses,
            )
        gradients = tape.gradient(losses.total, self.model.trainable_variables)
        pairs = [(gradient, variable) for gradient, variable in zip(gradients, self.model.trainable_variables) if gradient is not None]
        self.assertGreater(len(pairs), 0)
        tf.keras.optimizers.Adam(1e-5).apply_gradients(pairs)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "detector.keras"
            self.model.save(path)
            reloaded = tf.keras.models.load_model(path, compile=False)
            outputs = reloaded(images, training=False)
            self.assertEqual(tuple(outputs), ("p3", "p4", "p5", "p6"))

    def test_dynamic_tflite_conversion_and_invoke(self) -> None:
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        model_bytes = converter.convert()
        self.assertGreater(len(model_bytes), 1_000_000)
        interpreter = tf.lite.Interpreter(model_content=model_bytes)
        input_detail = interpreter.get_input_details()[0]
        interpreter.resize_tensor_input(input_detail["index"], (1, 128, 160, 3), strict=False)
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        interpreter.set_tensor(input_detail["index"], np.zeros((1, 128, 160, 3), np.float32))
        interpreter.invoke()
        outputs = [interpreter.get_tensor(item["index"]) for item in interpreter.get_output_details()]
        self.assertEqual(len(outputs), 4)
        self.assertTrue(all(np.all(np.isfinite(value)) for value in outputs))

    def test_coco_pipeline_preserves_valid_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "sample.png"
            Image.new("RGB", (96, 64), (128, 64, 32)).save(image_path)
            annotation_path = root / "annotations.json"
            annotation_path.write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "sample.png", "width": 96, "height": 64}],
                        "annotations": [
                            {"id": 1, "image_id": 1, "category_id": 5, "bbox": [10, 8, 30, 20], "area": 600}
                        ],
                        "categories": [{"id": 5, "name": "game_piece"}],
                    }
                ),
                encoding="utf-8",
            )
            index = load_coco_index(annotation_path, root)
            data_config = DataConfig(
                batch_size=1,
                validation_short_side=64,
                max_long_side=128,
                num_workers=1,
                prefetch_batches=1,
            )
            dataset = build_dataset(
                index, data_config, replace(AugmentationConfig(), enabled=False), training=False
            )
            batch = next(iter(dataset))
            self.assertEqual(int(batch["labels"][0, 0]), 0)
            self.assertTrue(np.all(batch["boxes"][0, 0].numpy() > 0))
            self.assertEqual(int(batch["image_id"][0]), 1)

    def test_all_augmentation_paths_keep_valid_geometry(self) -> None:
        image = tf.ones((80, 120, 3), tf.float32) * 0.5
        boxes = tf.constant([[12.0, 10.0, 70.0, 60.0]], tf.float32)
        labels = tf.constant([0], tf.int32)
        config = AugmentationConfig(
            horizontal_flip_probability=1.0,
            random_crop_probability=1.0,
            zoom_out_probability=1.0,
            rotate_90_probability=1.0,
            color_jitter_probability=1.0,
            gamma_probability=1.0,
            grayscale_probability=1.0,
            jpeg_probability=1.0,
            gaussian_blur_probability=1.0,
            gaussian_noise_probability=1.0,
            cutout_probability=1.0,
        )
        augmented_image, augmented_boxes, augmented_labels = augment_sample(
            image, boxes, labels, config
        )
        self.assertEqual(augmented_image.shape[-1], 3)
        self.assertGreaterEqual(float(tf.reduce_min(augmented_image)), 0.0)
        self.assertLessEqual(float(tf.reduce_max(augmented_image)), 1.0)
        self.assertEqual(len(augmented_boxes), len(augmented_labels))
        if len(augmented_boxes):
            self.assertTrue(np.all(augmented_boxes[:, 2:].numpy() > augmented_boxes[:, :2].numpy()))

    def test_one_epoch_training_checkpoint_and_final_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = []
            annotations = []
            for image_id in (1, 2):
                name = f"sample_{image_id}.png"
                Image.new("RGB", (96, 64), (32 * image_id, 64, 128)).save(root / name)
                images.append({"id": image_id, "file_name": name, "width": 96, "height": 64})
                annotations.append(
                    {
                        "id": image_id,
                        "image_id": image_id,
                        "category_id": 5,
                        "bbox": [10 + image_id, 8, 30, 20],
                        "area": 600,
                    }
                )
            annotation_path = root / "annotations.json"
            annotation_path.write_text(
                json.dumps(
                    {
                        "images": images,
                        "annotations": annotations,
                        "categories": [{"id": 5, "name": "game_piece"}],
                    }
                ),
                encoding="utf-8",
            )
            index = load_coco_index(annotation_path, root)
            data_config = DataConfig(
                batch_size=1,
                train_short_sides=(64,),
                validation_short_side=64,
                max_long_side=128,
                num_workers=1,
                prefetch_batches=1,
            )
            training_config = TrainingConfig(
                epochs=1,
                warmup_epochs=0,
                output_dir=str(root / "run"),
                early_stopping_patience=2,
                max_detections=10,
            )
            experiment = ExperimentConfig(
                data=data_config,
                augmentation=replace(AugmentationConfig(), enabled=False),
                model=self.model_config,
                training=training_config,
            )
            training_dataset = build_dataset(
                index, data_config, experiment.augmentation, training=True
            )
            validation_dataset = build_dataset(
                index, data_config, experiment.augmentation, training=False
            )
            model = build_detector(1, self.model_config)
            trainer = Trainer(
                model,
                training_dataset,
                validation_dataset,
                index,
                index,
                experiment,
                resume=False,
            )
            model_path = trainer.fit()
            self.assertTrue(model_path.is_file())
            self.assertTrue((root / "run" / "metadata.json").is_file())
            self.assertTrue((root / "run" / "history.json").is_file())
            self.assertIsNotNone(trainer.checkpoint_manager.latest_checkpoint)


if __name__ == "__main__":
    unittest.main()
