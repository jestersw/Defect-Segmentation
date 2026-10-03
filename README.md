
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
all available masks; they do not set the evaluation threshold.

## Train-only small-defect threshold

The fixed custom split is tracked in [issue #7](https://github.com/jestersw/Defect-Segmentation/issues/7).
The official training set contains only good images. Do not use it or the entire
dataset to estimate the threshold. Once all three agreed manifests exist, run:

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
Record the resulting value in issues #6 and #1 and in `eval.small_defect_area_px`
after the split is finalized. Without `--splits-dir`, no threshold is estimated;
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

These are descriptive results over all available images. The train-only 33rd
percentile remains pending the fixed manifests from issue #7;
`eval.small_defect_area_px` is still unset.

## Checks

```sh
python -m pip install pytest ruff
ruff check .
python -m pytest -q
```

Statistics tests use synthetic masks and temporary data; the full dataset is not
needed in CI. The existing split tests skip until the project manifests exist.
