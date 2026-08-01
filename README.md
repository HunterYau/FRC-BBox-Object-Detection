# FRC TensorFlow Bounding-Box Detector

This project trains a multi-scale, anchor-free object detector from COCO JSON and exports it to TensorFlow Lite. The network uses an ImageNet-pretrained MobileNetV2 backbone, a P3-P6 feature pyramid, and FCOS-style class, box, and centerness heads.

The input is genuinely dynamic: images keep their aspect ratio, a different short-side resolution is sampled during training, and each batch is padded only to its largest image size (rounded to 32). The model and non-INT8 TFLite exports accept different heights and widths. A batch still needs a rectangular tensor, so padding is unavoidable; no image is stretched into a fixed square.

## Project layout

- `frc_detector/config.py` — the easy-to-edit training settings, including paths, batch size, workers, resolutions, model, optimizer, loss, and validation settings.
- `frc_detector/coco.py` and `augment.py` — validated COCO loading and box-aware input transforms.
- `frc_detector/model.py` and `losses.py` — MobileNetV2-FPN-FCOS architecture and training targets/losses.
- `frc_detector/trainer.py` and `metrics.py` — AdamW training, warmup/cosine schedule, resume checkpoints, early stopping, and COCO-style mAP.
- `frc_detector/inference.py` — decoding, NMS, TFLite runtime wrapper, truth matching, and plots.
- `train.py` — an IDE-friendly training entry point; `predict.py` and `export_tflite.py` are utility scripts.

## Dataset layout

The defaults expect:

```text
data/
  train/
    _annotations.coco.json
    image_001.jpg
    ...
  valid/
    _annotations.coco.json
    image_101.jpg
    ...
```

COCO boxes must use `[x, y, width, height]`. Category IDs may be sparse (for example 1, 3, 17); the loader maps them to contiguous model labels and saves the mapping in `artifacts/frc_detector/metadata.json`. Training and validation must describe the same category IDs.

## PyCharm setup and environment

The project interpreter is `.venv/Scripts/python.exe`. In PyCharm, open **Settings > Project > Python Interpreter** and select that path. Install the pinned packages:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On native Windows, current TensorFlow training is CPU-only. An NVIDIA GPU requires WSL2; keep the project in the Linux filesystem for good I/O performance and install the appropriate TensorFlow CUDA extras there. Apple and Linux TensorFlow installations follow their platform-specific package instructions.

## Train and resume in PyCharm

1. Open `frc_detector/config.py`.
2. Change the values in the `EDIT THESE SETTINGS` section at the top. This is
   where `BATCH_SIZE`, `NUM_WORKERS`, `EPOCHS`, dataset paths, output folder,
   resume behavior, pretrained weights, and the main image sizes live.
3. Open `train.py`, right-click in the editor, and select **Run 'train'**. You
   can also use the green Run button after the run configuration is created.
   Leave **Script parameters** empty; training does not read command-line
   arguments.

Training automatically resumes from the configured output folder when
`AUTO_RESUME = True`. A checkpoint stores the model, AdamW state,
learning-rate step, completed epoch, and best mAP. To intentionally start a
fresh run, set `AUTO_RESUME = False` and use a new `OUTPUT_DIR`. If ImageNet
weights cannot be downloaded, set `BACKBONE_WEIGHTS = None` to use random
backbone initialization (usually slower and less accurate).

Outputs:

- `checkpoints/` — exact state used to continue training.
- `best.weights.h5` — weights from the highest validation mAP.
- `trained_detector.keras` — best reloadable model for inference, fine-tuning, or TFLite export.
- `metadata.json` — classes, original COCO IDs, strides, preprocessing, and full configuration.
- `history.json` — epoch losses, AP, precision, recall, F1, learning rate, and checkpoint paths.

Validation uses 101-point interpolated mAP over IoU 0.50–0.95, plus AP50/AP75 and thresholded precision/recall/F1. It does not implement COCO crowd/area/maxDet breakdowns.

## Inspect a custom image, truth boxes, and accuracy

For an image present in a COCO file:

```powershell
.\.venv\Scripts\python.exe predict.py data\valid\image_101.jpg --annotations data\valid\_annotations.coco.json --image-dir data\valid
```

The plot shows green dashed ground-truth boxes and solid predictions. Detection has no useful single classification-style accuracy, so the function reports true/false positives, false negatives, precision, recall, F1, and mean IoU of correct matches at IoU 0.50.

The reusable function can also be added directly to any `main()`:

```python
import tensorflow as tf
from frc_detector.coco import load_coco_index
from frc_detector.inference import find_ground_truth, visualize_prediction

index = load_coco_index("data/valid/_annotations.coco.json", "data/valid")
model = tf.keras.models.load_model("artifacts/frc_detector/trained_detector.keras", compile=False)
image_path = "data/valid/image_101.jpg"
true_boxes, true_labels = find_ground_truth(index, image_path)
visualize_prediction(
    model,
    image_path,
    index.class_names,
    strides=(8, 16, 32, 64),
    true_boxes=true_boxes,
    true_labels=true_labels,
)
```

## Export TensorFlow Lite

Float16 is a good default for mobile size/speed while keeping dynamic input dimensions:

```powershell
.\.venv\Scripts\python.exe export_tflite.py --quantization float16
```

Other options are `float32` (largest, closest numerically), `dynamic` (dynamic-range weight quantization), and calibrated INT8:

```powershell
.\.venv\Scripts\python.exe export_tflite.py --quantization int8 --representative-json data\train\_annotations.coco.json --representative-images data\train
```

INT8 calibration uses a fixed `640x640` input by default because embedded accelerators commonly require fixed shapes. Change it with `--int8-input-size`. The exporter writes `detector_*.tflite` and adjacent `.tflite.json` metadata. `frc_detector.inference.TFLiteDetector` handles dynamic tensor resizing, dequantization, FCOS decoding, and class-aware NMS.

## Augmentations

Enabled transforms include multi-scale aspect-preserving resize, horizontal flip, random crop, zoom-out/canvas translation, color/brightness/contrast/saturation/hue jitter, gamma, grayscale, JPEG degradation, Gaussian blur, sensor noise, and cutout. Right-angle rotation is implemented but disabled by default because a fixed robot camera normally does not rotate; enable it only if deployment images can rotate. Every geometric transform updates and clips boxes.

## Practical tuning

- Start with `batch_size=2` on CPU or limited memory. Reduce the maximum short side if out-of-memory errors occur.
- Keep ImageNet initialization unless the input is unlike natural RGB imagery.
- Small objects benefit from larger input sizes; large inputs cost roughly quadratically more compute.
- Examine AP per dataset revision, not just training loss. Poor labels or inconsistent category IDs cannot be fixed by architecture changes.
- Run the smoke tests after changing model or pipeline code:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
