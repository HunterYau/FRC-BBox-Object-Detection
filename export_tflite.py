"""Export a trained Keras detector to a portable TensorFlow Lite model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import tensorflow as tf

# Importing the package registers its custom serializable Keras layers.
import frc_detector  # noqa: F401
from frc_detector.coco import load_coco_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/frc_detector/trained_detector.keras")
    parser.add_argument("--output", default="artifacts/frc_detector/detector_float16.tflite")
    parser.add_argument(
        "--quantization",
        choices=("float32", "float16", "dynamic", "int8"),
        default="float16",
    )
    parser.add_argument("--representative-json", default=None, help="COCO JSON required for int8.")
    parser.add_argument("--representative-images", default=None, help="Image folder required for int8.")
    parser.add_argument("--representative-count", type=int, default=200)
    parser.add_argument(
        "--int8-input-size",
        type=int,
        default=640,
        help="INT8 calibration/export is fixed square; other modes remain dynamic.",
    )
    return parser.parse_args()


def representative_data(
    annotation_file: str,
    image_dir: str,
    count: int,
    input_size: int,
) -> Iterator[list[tf.Tensor]]:
    index = load_coco_index(annotation_file, image_dir)
    for record in index.records[:count]:
        encoded = tf.io.read_file(record.image_path)
        image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_with_pad(image, input_size, input_size, antialias=True)
        yield [image[None]]


def make_converter(model: tf.keras.Model, args: argparse.Namespace) -> tf.lite.TFLiteConverter:
    if args.quantization == "int8":
        if not args.representative_json or not args.representative_images:
            raise ValueError("INT8 export requires --representative-json and --representative-images.")
        fixed_input = tf.keras.Input(
            batch_shape=(1, args.int8_input_size, args.int8_input_size, 3),
            dtype=tf.float32,
            name="images",
        )
        fixed_model = tf.keras.Model(
            fixed_input,
            model(fixed_input, training=False),
            name="fixed_input_detector",
        )
        converter = tf.lite.TFLiteConverter.from_keras_model(fixed_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_data(
            args.representative_json,
            args.representative_images,
            args.representative_count,
            args.int8_input_size,
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Float boundaries keep application preprocessing/decoding simple while
        # convolution weights and activations inside the graph are INT8.
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        return converter

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if args.quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif args.quantization == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter


def verify_tflite(model_bytes: bytes, dynamic: bool) -> list[dict[str, object]]:
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    input_detail = interpreter.get_input_details()[0]
    if dynamic:
        interpreter.resize_tensor_input(input_detail["index"], (1, 320, 416, 3), strict=False)
    interpreter.allocate_tensors()
    output_summary: list[dict[str, object]] = []
    for detail in interpreter.get_output_details():
        output_summary.append(
            {
                "name": detail["name"],
                "shape": [int(value) for value in detail["shape"]],
                "shape_signature": [int(value) for value in detail["shape_signature"]],
                "dtype": detail["dtype"].__name__,
            }
        )
    return output_summary


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    converter = make_converter(model, args)
    print(f"Converting {model_path} ({args.quantization}) ...")
    model_bytes = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(model_bytes)

    source_metadata_path = model_path.parent / "metadata.json"
    metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    dynamic = args.quantization != "int8"
    metadata["tflite"] = {
        "quantization": args.quantization,
        "dynamic_height_width": dynamic,
        "fixed_input_size": None if dynamic else args.int8_input_size,
        "outputs": verify_tflite(model_bytes, dynamic),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {output_path} ({len(model_bytes) / (1024 * 1024):.2f} MiB)")
    print(f"Wrote decoder metadata to {metadata_path}")


if __name__ == "__main__":
    main()
