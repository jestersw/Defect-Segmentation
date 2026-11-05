# Arbitrary-image segmentation demo

Run from the repository root with Python 3.11 and the project dependencies:

```sh
python -m pip install -r requirements.txt
```

The Stage 6 demo uses **E: `cpu_light_dice_focal_512`**, the final model named in
[issue #67](https://github.com/jestersw/Defect-Segmentation/issues/67).
Its saved settings are in
[`results/runs/cpu_light_dice_focal_512/config.yaml`](../results/runs/cpu_light_dice_focal_512/config.yaml).
Use E's matching weights, either its trained `best.pth` or the model-only state
dict published by #67. Place the file at
`checkpoints/cpu_light_dice_focal_512/best.pth` for the commands below; the filename
does not affect checkpoint loading. Weights are not committed to this repository.
The planned [v1.0 release](https://github.com/jestersw/Defect-Segmentation/releases)
was not yet available when these instructions were prepared on 5 November 2026.
After publication, verify the downloaded file against the release's SHA-256.
The presentation dry run with release weights is tracked in #69.

## Five examples for the presentation

Download/extract MVTec AD using `python scripts/download_data.py`, as described in
the README, so the paths below exist under `data/mvtec_ad/`. All five paths belong
to the project's fixed **test** split in [`splits/test.txt`](../splits/test.txt).
They demonstrate a good image, clear defects, a small defect and a failure.

| Purpose | Path relative to `data/mvtec_ad/` | Recorded result for E at threshold 0.5 |
| --- | --- | --- |
| Good image | `capsule/test/good/001.png` | No predicted pixels; no defect flag. |
| Clear texture defect | `tile/test/glue_strip/017.png` | Dice 0.989; GT area 41,189 px. |
| Clear object defect | `hazelnut/test/hole/000.png` | Dice 0.974; GT area 3,594 px. |
| Small defect detected | `capsule/test/poke/011.png` | Dice 0.918; one 708 px component, detected. |
| Failure: complete miss | `capsule/test/squeeze/002.png` | Dice approximately 0; all 543 GT pixels missed, no predicted pixels. |

These observations come from the committed
[per-image test metrics](../results/runs/cpu_light_dice_focal_512/evaluation/per_image_test.csv).
The small-component cutoff is 859.28 px at 512 x 512; detection in the evaluation
requires coverage of at least 25% of the component. The final row deliberately
shows a limitation: a "No defect" prediction does not establish that an image is
defect-free. The demo itself uses no ground truth and does not calculate Dice.
This selection illustrates existing results; it does not change the model,
threshold, checkpoint or reported test metrics.

Exact command for all five images (one line; works in PowerShell and Bash):

```sh
python scripts/demo.py --config results/runs/cpu_light_dice_focal_512/config.yaml --checkpoint checkpoints/cpu_light_dice_focal_512/best.pth --images data/mvtec_ad/capsule/test/good/001.png data/mvtec_ad/tile/test/glue_strip/017.png data/mvtec_ad/hazelnut/test/hole/000.png data/mvtec_ad/capsule/test/poke/011.png data/mvtec_ad/capsule/test/squeeze/002.png --output-dir results/demo/E
```

Open `results/demo/E/001_001.png` through `005_002.png` in order. Each has a matching
JSON file. No additional full-test evaluation is needed to run this illustration.

## Your own files

`--images` accepts one or more file paths, including absolute paths and paths with
spaces, in the order supplied. Relative paths resolve from the current working
directory, **not** from `data.root`. The flag may be repeated. No manifest, category
directory layout, mask or dataset download is needed for your own images.
For example, with a local file named `my image.jpg`:

```sh
python scripts/demo.py --config results/runs/cpu_light_dice_focal_512/config.yaml --checkpoint checkpoints/cpu_light_dice_focal_512/best.pth --images "my image.jpg" --output-dir results/demo/custom
```

OpenCV-decodable formats such as PNG, JPEG and BMP are supported. Images are read
as RGB (grayscale becomes three channels; alpha is discarded), resized directly
to the square `train.input_size` with bilinear interpolation, and normalized with
ImageNet mean/std. Rectangular images are stretched, matching evaluation rather
than preserving aspect ratio. There are no random transforms. Model weights load
offline, without downloading an ImageNet encoder.

Logits are resized to `train.eval_size` using bilinear interpolation with
`align_corners=False` **before sigmoid**. E uses 512 input and evaluation sizes;
to demo B, supply B's saved config and weights together (256 input, 512 evaluation).
The input panel is resized directly from the original to evaluation resolution.
The heatmap always uses a fixed 0–1 probability scale. Orange marks pixels whose
probability is at least the threshold.

`--threshold` optionally overrides `eval.threshold` for this invocation only.
For an exploratory run on your own file, append `--threshold 0.7` and use a distinct
`--output-dir results/demo/custom_t07`. It does not edit the saved config or metrics;
keep the official threshold 0.5 for the selected test examples.

## Outputs and timing

For each input the script writes `<index>_<stem>.png` and `<index>_<stem>.json`.
Indices start at `001`, so identical basenames from different directories remain
distinct within a run. Parent directories are created. Repeating a run with the
same output paths replaces earlier demo artifacts; use separate directories for
different checkpoints or thresholds. Input images are protected from overwrite.

The JSON records the source image and checkpoint paths, original height/width,
input/evaluation sizes, threshold, minimum component size, CPU threads and:

- `predicted_defect_area_px`: all thresholded pixels at **evaluation resolution**,
  including small isolated regions; not area in the original image.
- `max_probability`: maximum pixel probability before thresholding.
- `has_defect`: true if any **8-connected** predicted component contains at least
  `eval.good_fp_min_component_px` pixels (20 for E). Several smaller regions do
  not trigger the flag merely because their total area reaches 20. All thresholded
  regions remain visible in the overlay, even when the flag is false.
- `inference_time_ms`: one synchronous FP32 CPU forward pass, logit resizing and
  sigmoid, batch size 1. Threads come from `train.cpu_threads` (default 8).

Timing excludes model loading, image reading/resizing/normalization, thresholding,
component analysis, plotting and file writes. There is no warm-up; the first image
includes first-forward overhead. `timing_scope` in every JSON states this protocol.
These per-image demo times differ from `scripts/benchmark_inference.py`, which uses
warm-ups, many repetitions and excludes sigmoid. Use that script for reported
benchmark comparisons.

Synthetic tests run without trained weights or a dataset:

```sh
python -m pytest -q tests/test_demo.py
```
