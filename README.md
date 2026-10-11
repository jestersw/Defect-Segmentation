
![CI](https://github.com/jestersw/Defect-Segmentation/actions/workflows/ci.yml/badge.svg)

# Defect Segmentation

Binary defect segmentation on five MVTec AD categories: tile, hazelnut, wood,
metal_nut, and capsule. Shared settings are in `configs/base.yaml`.

## Dataset preparation (issue #6)

Use Python 3.11 (the CI version). Install project dependencies and the optional
Kaggle downloader:

```sh
python -m pip install -r requirements.txt
python -m pip install kaggle
```

Configure Kaggle authentication locally, for example with your account's
`~/.kaggle/kaggle.json`. The downloader also accepts `.kaggle/kaggle.json` in the
repository (git-ignored); an explicit `KAGGLE_CONFIG_DIR` takes precedence.
Never commit credentials. The source is the
[ipythonx/mvtec-ad mirror](https://www.kaggle.com/datasets/ipythonx/mvtec-ad)
of MVTec AD; see its dataset page for attribution and license terms.

Run from the repository root (also works in PowerShell):

```sh
python scripts/download_data.py
python scripts/dataset_stats.py
```

On Linux/macOS/Git Bash, `bash scripts/download_data.sh` wraps the same download.
The downloader runs `python -m kaggle datasets download -d ipythonx/mvtec-ad -p data/`,
retains `data/mvtec-ad.zip`, and extracts only the configured categories into
`data/mvtec_ad/<category>/{train,test,ground_truth}/`. Optional archive wrappers are
removed. The ZIP contains all categories; allow space for the archive and the five
extracted categories. An existing complete ZIP is reused. An incomplete download
is passed back to the Kaggle client, which can resume it when the remote server
supports resuming; do not run two downloaders against the same archive:

```sh
python scripts/download_data.py --archive /path/to/mvtec-ad.zip
python scripts/dataset_stats.py --root /path/to/mvtec_ad --output results
```

Data and the downloaded archive are git-ignored. The analysis checks image
readability, image/mask dimensions, missing/orphan masks, and empty defect masks.
It compares counts against the expected inventory:

| Category | train/good | test/good | test defective | Masks |
| --- | ---: | ---: | ---: | ---: |
| tile | 230 | 33 | 84 | 84 |
| hazelnut | 391 | 40 | 70 | 70 |
| wood | 247 | 19 | 60 | 60 |
| metal_nut | 220 | 22 | 93 | 93 |
| capsule | 219 | 23 | 109 | 109 |

Count differences are printed and saved, rather than silently ignored. Broken
image/mask pairs stop the analysis. Outputs:

- `results/dataset_counts.csv`: observed and expected counts per category.
- `results/dataset_stats.csv`: image/mask/component counts by category, official
  source split, and defect type; minimum, median, and maximum component areas in
  pixels and as percentages of the resized image.
- `results/defect_area_hist.png`: component-area distributions by category with a
  shared logarithmic x-axis.
- `results/dataset_verification.json`: count differences, preprocessing settings,
  and the number of masks that lose all foreground after resizing.

Masks use OpenCV `INTER_NEAREST`, then `> 0` binarization and
`skimage.measure.label(connectivity=2)` (8-connected components). All component
areas are measured after resizing, without augmentation. Empty good-image masks
have zero components and blank area statistics. The histogram and CSV describe
all available masks; they do not set the evaluation threshold. The configured
`data.exclude_defect_types: [metal_nut/flip]` excludes 23 images from the experiment
before balancing and split generation. Inventory checks, the statistics CSV, and
the histogram still cover all 416 original masks; the threshold uses the filtered
training split only.

## Fixed train/validation/test split (issue #7)

The shared manifests are `splits/train.txt`, `splits/val.txt`, and `splits/test.txt`.
Use these files for every experiment. This custom split defines the supervised
evaluation protocol for the project.

The pool contains the 393 retained defective images from official `test`, all 137
official `test/good` images, and 256 sampled official `train/good` images. Each
category has equal good and defective counts in the pool (786 images overall).
The 23 `metal_nut/flip` images are excluded as global orientation anomalies whose
masks cover roughly 48% of an image. Balancing is performed after this exclusion.

```sh
python scripts/make_split.py
```

The script reads seed 42 and ratios `[0.70, 0.15, 0.15]` from `configs/base.yaml`.
It sorts categories and image paths before sampling good images without
replacement using NumPy `default_rng`. Two scikit-learn `train_test_split` calls
use `stratify=(category, defect_type)` and `random_state=42`: first 70/30, then a
50/50 split of the held-out 30%. Good images form one stratum per category.
Rounding within strata means individual split/category counts can differ slightly
between good and defective images.

| Split | Defective | Good | Total |
| --- | ---: | ---: | ---: |
| Train | 275 | 275 | 550 |
| Validation | 61 | 57 | 118 |
| Test | 57 | 61 | 118 |

Before writing outputs, the script checks that every pooled path appears exactly
once and that no byte-identical images (MD5) occur in different splits. Strata
with fewer than seven images produce a warning; an impossible stratification
fails without silently dropping or merging groups. All actual strata have at
least eight images. Two runs on the real dataset produced byte-identical manifests
and reports, and both leakage checks passed.

[`results/split_summary.csv`](results/split_summary.csv) contains counts by category
and split. [`results/split_metadata.json`](results/split_metadata.json) records the
pool composition, seed, ratios, library versions, and SHA-256 manifest hashes.
Manifests use UTF-8 with LF line endings on all platforms. scikit-learn is pinned
to 1.9.0; this run used NumPy 2.1.3 and Python 3.13 (CI uses Python 3.11).

Identical reruns are allowed. A different split is rejected unless `--overwrite`
is supplied deliberately; changing a shared split requires recalculating the
train-only threshold and rerunning the experiments. To generate an experimental
split separately, use `--config`, `--splits-dir`, and `--output` with separate paths.

## Train-only small-defect threshold

The fixed custom split is tracked in [issue #7](https://github.com/jestersw/Defect-Segmentation/issues/7).
The official training set contains only good images. Do not use it or the entire
dataset to estimate the threshold. To reproduce the threshold for the shared split, run:

```sh
python scripts/dataset_stats.py --splits-dir splits
```

Each manifest (`train.txt`, `val.txt`, `test.txt`) must contain one POSIX image path
per line relative to `data/mvtec_ad`, e.g. `tile/test/crack/000.png`. Before computing
the percentile, the script rejects missing images, duplicate paths, overlapping
splits, and byte-identical images across splits. It requires nonempty manifests
and defect components in the training split.

`results/small_defect_threshold.json` records the 33rd percentile of **train
components only**, NumPy's linear interpolation method, and SHA-256 hashes of all
three manifests. Per the shared protocol, this step requires 512 × 512 masks.
A small defect has `area < small_defect_area_px` (strict inequality, no rounding).
The shared train split contains 449 defect components. Its 33rd percentile is
approximately **859.28 pixels** (**0.3278%** of a 512 × 512 image), recorded at
full precision in the JSON and `eval.small_defect_area_px`.
This supersedes the earlier 1,102.11-pixel threshold and 832-image split that
included `metal_nut/flip`. After the approved exclusion, the manifests, summary,
hashes, and threshold were regenerated. Existing clones
with the earlier split must deliberately run `python scripts/make_split.py --overwrite`
followed by the statistics command above before using the new protocol.
Without `--splits-dir`, no threshold is estimated;
any threshold file from an earlier run is not refreshed.

## Verified dataset results

The Kaggle archive was processed on 2026-10-03 with the default configuration.
All five categories match the expected counts above: 1,860 images (1,444 good
and 416 defective) and 416 corresponding masks. All images and masks were
readable, dimensions matched, and there were no missing or orphan masks.

At 512 × 512, the masks contain 638 components with areas from 1 to 127,709 pixels.
No defect mask became entirely empty after resizing. See
[`results/dataset_stats.csv`](results/dataset_stats.csv),
[`results/dataset_counts.csv`](results/dataset_counts.csv),
[`results/dataset_verification.json`](results/dataset_verification.json), and
[`results/defect_area_hist.png`](results/defect_area_hist.png).

These are descriptive results over all available images. The evaluation
threshold comes exclusively from the shared training split; see
[`results/small_defect_threshold.json`](results/small_defect_threshold.json).

## Dataset Preparation report section (issue #8)

Section 2 of [`docs/report_1.tex`](docs/report_1.tex) contains the verified
dataset preparation, actual split counts, and train-only threshold after excluding
`metal_nut/flip`. Its numbers come from the committed CSV/JSON results. It
distinguishes the statistics and leakage checks completed in Stage 1 from the
training transforms implemented in Stage 2. The histogram is available as a
separate artifact to keep the report compact.

With an existing LaTeX installation, build the full report from the repository root:

```sh
python -c "from pathlib import Path; Path('.tools').mkdir(exist_ok=True)"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=.tools docs/report_1.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=.tools docs/report_1.tex
```

Alternatively use `tectonic --outdir .tools docs/report_1.tex`.

## CPU environment and timing (issues #9 and #10)

The task owner selected local CPU compute. See [the reproducible protocol](docs/compute.md)
for Python 3.11 setup and a full batch-8 U-Net benchmark at 512 and 256 pixels.
`scripts/smoke_test.py` records step times, peak process RAM and projections for
both the issue's 582-image reference and the current 550-image train split.
The standalone smoke test verifies the model independently of the training pipeline.

## Stage 2 dataset and loaders (issue #24)

`src.data.MVTecSegDataset(split, cfg, train)` reads the fixed manifests in
`data.splits_dir`, preserving their order in `dataset.records` (`path`, `category`,
`defect_type`). Relative config paths resolve from the repository root; absolute
paths are supported. Images are read as RGB, defect masks come from
`<category>/ground_truth/<defect_type>/<stem>_mask.png`, and good images have zero masks.

Each sample contains contiguous float32 image/mask tensors. Image size comes from
`train.input_size` (256 by default); mask size is `input_size` in train mode and
`train.eval_size` (512) in evaluation mode. Images use bilinear resizing and
ImageNet normalisation. Masks are resized directly from their original resolution
with nearest-neighbour interpolation and binarised at `> 0`; validation masks are
never reduced to the model input size first.

Training applies Albumentations horizontal/vertical flips and 90-degree rotations
jointly to images and masks, plus image-only brightness/contrast limits of 0.1.
Each transform has probability 0.5. Evaluation has no random transforms.
`make_loaders(cfg)` returns `(train_loader, val_loader, test_loader)` with the
configured batch size and worker count. Only train shuffles; the last partial
batch is retained. Separate seeded generators and worker-specific augmentation
seeds reproduce complete runs with the same seed, library versions, worker count,
batch size, and iteration schedule. Augmentations continue to vary across epochs;
changing the worker count can change their sequence.

On Windows, create and iterate loaders inside an `if __name__ == "__main__":`
guard when using workers. Synthetic tests exercise the `spawn` worker mode.
Stage 2 uses train/validation data only; keep the real test loader unconsumed
until the final comparison.

Generate the report illustration (eight seeded training examples after augmentation):

```sh
python scripts/show_batch.py --config configs/base.yaml
```

The output is [`results/s2/data_samples.png`](results/s2/data_samples.png), with
ground-truth masks in orange. The script reverses ImageNet normalisation for display.
Data tests run without MVTec files or a GPU:

```sh
python -m pytest -q tests/test_data.py
```

All 26 data tests passed locally. A real-data check consumed all 550 training
images (69 batches) and 118 validation images (15 batches) with four workers,
checking shapes, finite float32 tensors and binary masks. The real test loader
was not consumed. [The recorded check](results/s2/data_loading_check.json)
includes split hashes, settings, versions and diagnostic timings (worker startup
and tensor checks included, no model). It used Python 3.13 / PyTorch 2.10 locally;
CI validates the project-pinned Python 3.11 / PyTorch 2.6 environment.

## Checks

```sh
python -m pip install -r requirements-dev.txt
ruff check .
python -m pytest -q
```

Statistics and split-generator tests use synthetic masks and temporary data;
the full dataset is not needed in CI. Manifest tests also verify the committed
split files. Real-image MD5 checks run when generating the split and the threshold.
