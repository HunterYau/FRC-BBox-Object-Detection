"""Visualize a trained Keras detector prediction and optional COCO truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from frc_detector.coco import load_coco_index
from frc_detector.inference import find_ground_truth, visualize_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Image to inspect.")
    parser.add_argument("--model", default="artifacts/frc_detector/trained_detector.keras")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--annotations", default=None, help="Optional COCO JSON containing this image.")
    parser.add_argument("--image-dir", default=None, help="Required with --annotations.")
    parser.add_argument("--score", type=float, default=0.25)
    parser.add_argument("--save", default=None, help="Optional output plot path.")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    metadata_path = Path(args.metadata).resolve() if args.metadata else model_path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    class_names = tuple(metadata["class_names"])
    category_ids = tuple(metadata["category_ids"])
    strides = tuple(metadata["strides"])
    model = tf.keras.models.load_model(model_path, compile=False)

    true_boxes = true_labels = None
    if args.annotations:
        if not args.image_dir:
            raise ValueError("--image-dir is required when --annotations is used.")
        index = load_coco_index(args.annotations, args.image_dir, category_ids=category_ids)
        true_boxes, true_labels = find_ground_truth(index, args.image)
    detections, metrics = visualize_prediction(
        model,
        args.image,
        class_names,
        strides,
        true_boxes=true_boxes,
        true_labels=true_labels,
        score_threshold=args.score,
        save_path=args.save,
        show=not args.no_show,
    )
    print(f"Detections: {len(detections.boxes)}")
    if metrics:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
