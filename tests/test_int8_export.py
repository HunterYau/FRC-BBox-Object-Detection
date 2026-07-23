from __future__ import annotations

import unittest

import numpy as np
import tensorflow as tf

from frc_detector.config import ModelConfig
from frc_detector.model import build_detector


class Int8ExportTest(unittest.TestCase):
    def test_calibrated_int8_conversion(self) -> None:
        model = build_detector(
            1,
            ModelConfig(
                backbone_weights=None,
                fpn_channels=16,
                head_channels=16,
                head_depth=1,
            ),
        )

        def representative_dataset():
            for value in (0.0, 0.5, 1.0):
                yield [tf.ones((1, 96, 96, 3), tf.float32) * value]

        fixed_input = tf.keras.Input(
            batch_shape=(1, 96, 96, 3), dtype=tf.float32, name="images"
        )
        fixed_model = tf.keras.Model(fixed_input, model(fixed_input, training=False))
        converter = tf.lite.TFLiteConverter.from_keras_model(fixed_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        model_bytes = converter.convert()
        self.assertGreater(len(model_bytes), 1_000_000)

        interpreter = tf.lite.Interpreter(model_content=model_bytes)
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        interpreter.set_tensor(input_detail["index"], np.zeros((1, 96, 96, 3), np.float32))
        interpreter.invoke()
        self.assertEqual(len(interpreter.get_output_details()), 4)


if __name__ == "__main__":
    unittest.main()
