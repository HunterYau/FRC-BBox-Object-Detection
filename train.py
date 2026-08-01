"""IDE-friendly training entry point.

Edit ``frc_detector/config.py``, then run this file with the IDE's Run button.
No command-line arguments are required or read.
"""

from __future__ import annotations

from frc_detector.coco import build_dataset, load_coco_index
from frc_detector.config import CONFIG, ExperimentConfig
from frc_detector.model import build_detector
from frc_detector.trainer import Trainer, configure_tensorflow


def main(config: ExperimentConfig = CONFIG) -> None:
    """Train using values from ``frc_detector/config.py``."""
    print(
        "Using frc_detector/config.py: "
        f"batch_size={config.data.batch_size}, "
        f"num_workers={config.data.num_workers}, "
        f"epochs={config.training.epochs}, "
        f"auto_resume={config.training.auto_resume}"
    )
    configure_tensorflow(config)

    train_index = load_coco_index(
        config.data.train_annotations,
        config.data.train_images,
        include_crowd=config.data.include_crowd,
    )
    validation_index = load_coco_index(
        config.data.validation_annotations,
        config.data.validation_images,
        category_ids=train_index.category_ids,
        include_crowd=config.data.include_crowd,
    )
    print(
        f"Training images: {len(train_index)} | validation images: {len(validation_index)} | "
        f"classes: {', '.join(train_index.class_names)}"
    )
    train_dataset = build_dataset(
        train_index, config.data, config.augmentation, training=True
    )
    validation_dataset = build_dataset(
        validation_index, config.data, config.augmentation, training=False
    )
    model = build_detector(len(train_index.class_names), config.model)
    model.summary(line_length=120)
    trainer = Trainer(
        model,
        train_dataset,
        validation_dataset,
        train_index,
        validation_index,
        config,
    )
    model_path = trainer.fit()
    print(f"Training complete: {model_path}")

    # Optional manual inspection after training:
    # from frc_detector.coco import load_coco_index
    # from frc_detector.inference import find_ground_truth, visualize_prediction
    # image_path = r"C:\path\to\your\image.jpg"
    # truth_boxes, truth_labels = find_ground_truth(validation_index, image_path)
    # visualize_prediction(
    #     model, image_path, validation_index.class_names, config.model.strides,
    #     true_boxes=truth_boxes, true_labels=truth_labels,
    # )


if __name__ == "__main__":
    main()
