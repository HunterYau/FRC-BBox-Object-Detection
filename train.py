"""Train the variable-resolution TensorFlow detector on a COCO JSON dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from frc_detector.coco import build_dataset, load_coco_index
from frc_detector.config import CONFIG
from frc_detector.model import build_detector
from frc_detector.trainer import Trainer, configure_tensorflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", default=CONFIG.data.train_annotations)
    parser.add_argument("--train-images", default=CONFIG.data.train_images)
    parser.add_argument("--val-json", default=CONFIG.data.validation_annotations)
    parser.add_argument("--val-images", default=CONFIG.data.validation_images)
    parser.add_argument("--epochs", type=int, default=CONFIG.training.epochs)
    parser.add_argument("--batch-size", type=int, default=CONFIG.data.batch_size)
    parser.add_argument("--output-dir", default=CONFIG.training.output_dir)
    parser.add_argument("--no-resume", action="store_true", help="Ignore an existing training checkpoint.")
    parser.add_argument("--no-imagenet", action="store_true", help="Initialize the backbone randomly.")
    parser.add_argument("--mixed-precision", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CONFIG.data.train_annotations = args.train_json
    CONFIG.data.train_images = args.train_images
    CONFIG.data.validation_annotations = args.val_json
    CONFIG.data.validation_images = args.val_images
    CONFIG.data.batch_size = args.batch_size
    CONFIG.training.epochs = args.epochs
    CONFIG.training.output_dir = args.output_dir
    CONFIG.training.mixed_precision = args.mixed_precision
    if args.no_imagenet:
        CONFIG.model.backbone_weights = None
    configure_tensorflow(CONFIG)

    train_index = load_coco_index(
        CONFIG.data.train_annotations,
        CONFIG.data.train_images,
        include_crowd=CONFIG.data.include_crowd,
    )
    validation_index = load_coco_index(
        CONFIG.data.validation_annotations,
        CONFIG.data.validation_images,
        category_ids=train_index.category_ids,
        include_crowd=CONFIG.data.include_crowd,
    )
    print(
        f"Training images: {len(train_index)} | validation images: {len(validation_index)} | "
        f"classes: {', '.join(train_index.class_names)}"
    )
    train_dataset = build_dataset(
        train_index, CONFIG.data, CONFIG.augmentation, training=True
    )
    validation_dataset = build_dataset(
        validation_index, CONFIG.data, CONFIG.augmentation, training=False
    )
    model = build_detector(len(train_index.class_names), CONFIG.model)
    model.summary(line_length=120)
    trainer = Trainer(
        model,
        train_dataset,
        validation_dataset,
        train_index,
        validation_index,
        CONFIG,
        resume=not args.no_resume,
    )
    model_path = trainer.fit()
    print(f"Training complete: {model_path}")

    # Optional manual inspection after training:
    # from frc_detector.coco import load_coco_index
    # from frc_detector.inference import find_ground_truth, visualize_prediction
    # image_path = r"C:\path\to\your\image.jpg"
    # truth_boxes, truth_labels = find_ground_truth(validation_index, image_path)
    # visualize_prediction(
    #     model, image_path, validation_index.class_names, CONFIG.model.strides,
    #     true_boxes=truth_boxes, true_labels=truth_labels,
    # )


if __name__ == "__main__":
    main()

