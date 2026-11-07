![CI](https://github.com/jestersw/Defect-Segmentation/actions/workflows/ci.yml/badge.svg)

# Industrial Defect Segmentation on MVTec AD

Pixel-level binary defect segmentation on five MVTec AD categories (tile, wood, hazelnut, metal_nut, capsule) with a U-Net and a ResNet18 encoder, trained on CPU. The project compares the loss function (BCE vs Dice + Focal), the augmentation policy (default vs category-aware) and the input resolution (256 vs 512), with a focus on small defects.

Final model weights: [release v1.0](https://github.com/jestersw/Defect-Segmentation/releases/tag/v1.0).
Technical summary: [`docs/technical_summary.tex`](docs/technical_summary.tex).

## Results

Test split, 118 images (61 good, 20 small defect components), threshold 0.5. The final model E was selected by validation Dice before test evaluation.

| | Approach | Dice | IoU | Pixel P | Pixel R | Small-defect R | Good FPR | CPU inference |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | BCE, default aug., 256 | 0.716 | 0.613 | 0.877 | 0.839 | 7/20 | 0.000 | 48 ms |
| B | Dice + Focal, default aug., 256 | 0.757 | 0.663 | 0.899 | 0.860 | 8/20 | 0.000 | 47 ms |
| C | BCE, category-aware aug., 256 | 0.716 | 0.611 | 0.898 | 0.831 | 8/20 | 0.000 | 48 ms |
| D | Dice + Focal, category-aware aug., 256 | 0.756 | 0.653 | 0.897 | 0.842 | 9/20 | 0.000 | 53 ms |
| **E** | **Dice + Focal, default aug., 512 (final)** | **0.809** | **0.719** | **0.913** | **0.876** | **13/20** | **0.000** | **168 ms** |

Dice and IoU are averaged over defective images; pixel precision and recall are pooled over all pixels. A small defect is a connected component below 859.28 px at 512 x 512 and counts as detected when at least 25% of it is predicted. Good FPR is the share of good images with a predicted component of at least 20 px. Inference is the mean CPU time per image with batch 1 (AMD Ryzen AI 9 HX 370, 8 threads, FP32). All metrics, per-category and per-image results are in [`results/runs/`](results/runs).

Main findings: Dice + Focal improves test Dice by about 0.04 over BCE; training at 512 adds another 0.05 and raises small-defect recall from 8 to 13 of 20, at 3.5 times the inference time; the category-aware policy produces only very small differences. Remaining errors are mostly small, low-contrast capsule defects. Each configuration was trained with a single seed.

## Setup

Python 3.11 is required (the CI version). From the repository root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. `requirements.txt` pins the runtime packages (PyTorch 2.6.0 CPU, segmentation-models-pytorch 0.4.0, albumentations 2.0.8, NumPy 2.1.3); `requirements-dev.txt` adds pytest and ruff.

## Data

The dataset comes from the [ipythonx/mvtec-ad](https://www.kaggle.com/datasets/ipythonx/mvtec-ad) Kaggle mirror of MVTec AD; see the dataset page for attribution and license terms. Configure Kaggle credentials (`~/.kaggle/kaggle.json`, never committed), then:

```sh
python -m pip install kaggle
python scripts/download_data.py
```

This extracts the five categories into `data/mvtec_ad/` (git-ignored). The fixed split manifests are committed in [`splits/`](splits) and must not be regenerated. Dataset checks, the split protocol and the small-defect threshold are described in [`docs/project_log.md`](docs/project_log.md).

## Final model weights

```sh
gh release download v1.0 -R jestersw/Defect-Segmentation -p "cpu_light_dice_focal_512.pth" -D checkpoints/cpu_light_dice_focal_512
mv checkpoints/cpu_light_dice_focal_512/cpu_light_dice_focal_512.pth checkpoints/cpu_light_dice_focal_512/best.pth
sha256sum checkpoints/cpu_light_dice_focal_512/best.pth
```

The hash must be `c30e3daaa57e80e5a8c6243ec6ae6e059f7d78f196d6cad6dfb520bb3859b5e6`. Without `gh`, download the file from the release page in a browser. Release v1.0 also contains model B (`cpu_light_dice_focal_256.pth`) and both saved configs.

## Evaluate the final model

```sh
python -m src.evaluate --config results/runs/cpu_light_dice_focal_512/config.yaml --checkpoint checkpoints/cpu_light_dice_focal_512/best.pth --split val --output-dir results/check/E
```

The printed metrics must match [`results/runs/cpu_light_dice_focal_512/evaluation/metrics_val.json`](results/runs/cpu_light_dice_focal_512/evaluation/metrics_val.json) (Dice 0.8104). The test split is reserved: the reported test metrics were generated once and should not be used to tune anything.

## Demo

Run the final model on any image files; each input produces a figure (input, probability heatmap, overlay) and a JSON with predicted area, maximum probability, defect flag and inference time:

```sh
python scripts/demo.py --config results/runs/cpu_light_dice_focal_512/config.yaml --checkpoint checkpoints/cpu_light_dice_focal_512/best.pth --images data/mvtec_ad/capsule/test/good/001.png data/mvtec_ad/tile/test/glue_strip/017.png data/mvtec_ad/hazelnut/test/hole/000.png data/mvtec_ad/capsule/test/poke/011.png data/mvtec_ad/capsule/test/squeeze/002.png --output-dir results/demo/E
```

The five examples (good image, clear texture and object defects, a small defect and a missed defect), output format and timing scope are explained in [`docs/demo.md`](docs/demo.md).

## Reproduce the experiments

Training refuses to overwrite an existing run, so use new run names. Commands for the five approaches:

```sh
python -m src.train --config configs/cpu_light.yaml --run-name repro_A_bce_256
python -m src.train --config configs/cpu_light.yaml --run-name repro_B_dice_focal_256 --loss dice_focal
python -m src.train --config configs/cpu_light_cataug.yaml --run-name repro_C_bce_256_cataug
python -m src.train --config configs/cpu_light_cataug.yaml --run-name repro_D_dice_focal_256_cataug --loss dice_focal
python -m src.train --config configs/cpu_light.yaml --run-name repro_E_dice_focal_512 --loss dice_focal --input-size 512
```

Each run writes metrics, curves and the resolved config to `results/runs/<run>/` and checkpoints to `checkpoints/<run>/`. On the reference laptop CPU an epoch takes about 70 s at 256 and 290 s at 512 (about 1 h and 5 h per run). Then, for each run:

```sh
python -m src.evaluate --config results/runs/<run>/config.yaml --checkpoint checkpoints/<run>/best.pth --split val --output-dir results/runs/<run>/evaluation
python scripts/per_image_metrics.py --config results/runs/<run>/config.yaml --checkpoint checkpoints/<run>/best.pth --split val --output results/runs/<run>/evaluation/per_image_val.csv
python scripts/benchmark_inference.py --config results/runs/<run>/config.yaml --checkpoint checkpoints/<run>/best.pth --output results/runs/<run>/inference_cpu.json
```

`python scripts/summarize_results.py` prints the report tables from `results/runs/`, and `scripts/plot_cases.py` draws selected images for several checkpoints (see [`docs/cpu_training.md`](docs/cpu_training.md)). Runs are deterministic on the same machine with seed 42; results on other hardware may differ slightly. Run E was interrupted and resumed once, so it is not bit-reproducible.

## Tests

```sh
ruff check .
python -m pytest -q
```

The tests use synthetic data and need neither the dataset nor trained weights. GitHub Actions runs the same checks on every pull request.

## Repository structure
```
configs/         base.yaml, cpu_light.yaml (A, B, E), cpu_light_cataug.yaml (C, D)
splits/          fixed train.txt / val.txt / test.txt manifests
src/             data.py, model.py, losses.py, train.py, evaluate.py, metrics.py
scripts/         download_data.py, make_split.py, demo.py, benchmark_inference.py,
                 per_image_metrics.py, plot_cases.py, plot_predictions.py, summarize_results.py
tests/           unit tests on synthetic data (pytest)
results/         dataset statistics and split metadata
  runs/<run>/    config, history, curves, summary, inference_cpu.json for A-E
    evaluation/  val/test metrics, per-category, per-image and threshold sweeps
  s5/            failure-case figures
docs/            technical_summary.tex, report_1-5.tex, demo.md, cpu_training.md,
                 compute.md, project_log.md
data/            MVTec AD after download (not in git)
checkpoints/     trained or downloaded weights (not in git)
```
