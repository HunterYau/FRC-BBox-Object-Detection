"""Dependency-free COCO-style AP and thresholded detection metrics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .inference import Detections, _numpy_iou


class DetectionEvaluator:
    """Compute 101-point interpolated AP over IoU 0.50:0.05:0.95.

    This intentionally omits COCO's crowd, area-range, and maxDet breakdowns,
    while preserving its main mAP interpolation. It is sufficient for model
    selection without adding a platform-sensitive pycocotools dependency.
    """

    def __init__(self, num_classes: int, summary_score_threshold: float = 0.25):
        self.num_classes = num_classes
        self.summary_score_threshold = summary_score_threshold
        self.ground_truth: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
        self.predictions: dict[int, list[tuple[int, float, np.ndarray]]] = defaultdict(list)

    def update(
        self,
        image_id: int,
        detections: Detections,
        true_boxes: np.ndarray,
        true_labels: np.ndarray,
    ) -> None:
        for label in range(self.num_classes):
            self.ground_truth[label][image_id] = true_boxes[true_labels == label].astype(np.float32)
        for box, score, label in zip(
            detections.boxes, detections.scores, detections.labels, strict=True
        ):
            self.predictions[int(label)].append((image_id, float(score), box.astype(np.float32)))

    def _evaluate_class(self, label: int, iou_threshold: float) -> tuple[float, int, int, int]:
        truths = self.ground_truth[label]
        total_truth = sum(len(boxes) for boxes in truths.values())
        if total_truth == 0:
            return float("nan"), 0, 0, 0
        predictions = sorted(self.predictions[label], key=lambda item: item[1], reverse=True)
        matched = {image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in truths.items()}
        true_positive = np.zeros(len(predictions), dtype=np.float32)
        false_positive = np.zeros(len(predictions), dtype=np.float32)
        summary_tp = 0
        summary_fp = 0

        for index, (image_id, score, box) in enumerate(predictions):
            image_truth = truths.get(image_id, np.empty((0, 4), np.float32))
            available = np.flatnonzero(~matched.get(image_id, np.empty((0,), bool)))
            is_match = False
            if available.size:
                ious = _numpy_iou(box, image_truth[available])
                best = int(np.argmax(ious))
                if float(ious[best]) >= iou_threshold:
                    matched[image_id][available[best]] = True
                    true_positive[index] = 1.0
                    is_match = True
            if not is_match:
                false_positive[index] = 1.0
            if score >= self.summary_score_threshold:
                summary_tp += int(is_match)
                summary_fp += int(not is_match)

        cumulative_tp = np.cumsum(true_positive)
        cumulative_fp = np.cumsum(false_positive)
        recall = cumulative_tp / max(total_truth, 1)
        precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
        recall_points = np.linspace(0.0, 1.0, 101)
        interpolated = [np.max(precision[recall >= point], initial=0.0) for point in recall_points]
        summary_fn = total_truth - summary_tp
        return float(np.mean(interpolated)), summary_tp, summary_fp, summary_fn

    def result(self) -> dict[str, float]:
        thresholds = np.arange(0.50, 0.951, 0.05)
        average_precisions: list[float] = []
        ap50: list[float] = []
        ap75: list[float] = []
        total_tp = total_fp = total_fn = 0
        for threshold in thresholds:
            for label in range(self.num_classes):
                ap, tp, fp, fn = self._evaluate_class(label, float(threshold))
                if not np.isnan(ap):
                    average_precisions.append(ap)
                    if np.isclose(threshold, 0.50):
                        ap50.append(ap)
                        total_tp += tp
                        total_fp += fp
                        total_fn += fn
                    if np.isclose(threshold, 0.75):
                        ap75.append(ap)
        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        return {
            "map": float(np.mean(average_precisions)) if average_precisions else 0.0,
            "ap50": float(np.mean(ap50)) if ap50 else 0.0,
            "ap75": float(np.mean(ap75)) if ap75 else 0.0,
            "precision50": precision,
            "recall50": recall,
            "f1_50": f1,
        }
