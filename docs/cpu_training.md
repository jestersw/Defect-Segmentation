# CPU training on the lightweight candidate

Work branch: `s2/cpu-lightweight-training`, rebased on main `6139798` (PR #33).
The agreed baseline in issue #1 is `configs/cpu_light.yaml`: ResNet18,
ImageNet weights, decoder `[128,64,32,16,8]`, 256px inputs, BCE. Stage 3 condition
B uses exactly the same configuration with `--loss dice_focal`.

The loader, loader tests, sample script and sample image are inherited from
moonshine1l's PR #33. Only an optional pre-augmentation image/mask cache is added
to the loader; cache equivalence is covered separately in `tests/test_data_cache.py`.
Pre-rebase run artifacts are excluded from the current results and kept in a
local ignored archive. The exploratory 384px Dice run is not Stage 2 evidence.

Both encoders and decoders are fully fine-tuned. The CPU candidate uses FP32,
8 computation threads, 1 interop thread, batch 8, 256px inputs, AdamW (lr=1e-4,
weight decay=1e-4), cosine schedule over at most 60 epochs, and seed 42. Checkpoint
selection maximises validation Dice, stopping after 10 epochs without improvement.
No training images are omitted in the full run. Early stopping is a normal
completion condition, so a full run can finish before epoch 60.

Measured on this laptop on 2026-10-10 with synthetic batches of identical shape (8 x 3 x
256 x 256), FP32 BCE+AdamW, 3 warm-ups and 20 measured steps:

| Model | Parameters | Mean step seconds |
| --- | ---: | ---: |
| Original ResNet34 U-Net | 24,436,369 | 1.9195 |
| ResNet18 + narrow decoder | 12,458,025 | 0.9364 |

This is a 2.05x synthetic training-step speedup, not evidence of equal quality or
a guarantee of wall-clock epoch speed. Raw step durations are in
`results/s2/cpu_model_comparison.json`; reproduce with
`python scripts/compare_cpu_models.py`. Both encoder weight files are required
and are downloaded on the first run. At 69 steps/epoch the candidate's training
computation alone projects to about 64.6 seconds/epoch; real validation and I/O
add overhead. The 2026-10-10 comparison ran while the dataset downloaded, so
background system activity was not completely eliminated.

## Data and evaluation contract

The existing manifests remain 550 train / 118 validation / 118 test. Training masks
are 256px. Validation/test masks remain 512px; evaluation upsamples **logits before
sigmoid**, preserving the shared 859.28px small-defect threshold and 20px false
positive component threshold. Test images are not evaluated during training.
Validation loss and metrics use a single inference pass each epoch.

The lightweight config caches resized uint8 training/validation images and masks
before augmentation. It uses zero DataLoader worker processes to avoid copying
the cache or respawning workers on Windows. Spatial augmentation is joint for
image and mask; brightness/contrast changes only the image. Both are seeded.
Image decoding supports Unicode Windows paths.

The Kaggle dataset is not stored in Git. `data/`, downloaded model weights,
checkpoints, tools and background logs are ignored. The original downloader's
entry point was corrected for Kaggle versions without `kaggle.__main__`.

## Run

Use the existing Python 3.11 environment with `requirements-dev.txt` installed.
From the repository root:

```bash
# Required two-epoch integration trial on all real train/validation samples.
python -m src.train --config configs/cpu_light.yaml --run-name cpu_light_trial_2epochs --max-epochs 2

# Full lightweight run; no --limit-batches.
python -m src.train --config configs/cpu_light.yaml --run-name cpu_light_bce_256

# Stage 3 condition B: same architecture/resolution/data, change only loss.
python -m src.train --config configs/cpu_light.yaml --run-name cpu_light_dice_focal_256 --loss dice_focal
```

Run names must be unique: existing results/checkpoints are never overwritten.
`--loss`, `--input-size`, `--max-epochs`, and `--limit-batches` overrides are written
into the saved config. `--limit-batches` limits both training and validation and
marks the summary as a limited diagnostic run; it is not a final experiment.
CI uses synthetic images, random encoder weights and a one-batch training run.

To launch a hidden background process on Windows PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File scripts/start_training.ps1 -Config configs/cpu_light.yaml -RunName cpu_light_bce_256
```

`RemoteSigned` applies only to this PowerShell process and allows the local launcher
on Windows installations with the default restricted script policy; it does not
change the saved execution policy. The launcher uses `.venv/Scripts/python.exe`.
`--keep-awake` prevents automatic
system idle sleep while training on Windows and releases the request when the
process exits; it does not change saved system power settings. Keep the laptop
plugged in and its lid open. Manual shutdown/sleep still interrupts training.

## Monitor

```powershell
Get-Content .tools/training/cpu_light_bce_256/stdout.log -Tail 20 -Wait
Get-Content results/runs/cpu_light_bce_256/summary.json
Get-Content .tools/training/cpu_light_bce_256/stderr.log -Tail 20
```

The process PID and start time are in `.tools/training/<run>/process.json`.
`summary.json` distinguishes `initializing`, `running`, `completed`, `failed` and
`interrupted`; `running` must not be reported as a completed experiment. On an OS
kill or shutdown, the last saved status can be stale; also verify the process PID.

Each run writes `results/runs/<run>/config.yaml`, `history.csv`, `curves.png` and
`summary.json`. Summary includes software/source fingerprints, split hashes,
parameter count, best epoch/Dice and elapsed time. Actual epoch times include
training, validation, checkpoint writes and plotting; initial cache/weight loading
is included in total elapsed time. CSV loss averages weight each batch by sample
count, including the last partial batch. Curves and JSON are replaced atomically.

Checkpoints are `checkpoints/<run>/best.pth` and `last.pth`. They include the model,
optimizer, scheduler, epoch and configuration and are compatible with
`src.evaluate.load_model`; they are not claimed to provide exact resumable RNG state.
The best checkpoint can be evaluated with:

```bash
python -m src.evaluate --config results/runs/cpu_light_bce_256/config.yaml --checkpoint checkpoints/cpu_light_bce_256/best.pth --split val
python -m scripts.plot_predictions --config results/runs/cpu_light_bce_256/config.yaml --checkpoint checkpoints/cpu_light_bce_256/best.pth --split val --output results/runs/cpu_light_bce_256/examples.png
```

Commit completed run artifacts, not checkpoints or in-progress logs. Share the
best checkpoint separately after training completes. Human review and GitHub CI
remain separate from local tests. No checkpoint upload or issue posting is done
by the training process.

## Finalize the rebased baseline

After a completed baseline, evaluate only validation and render its examples:

```bash
python -m src.evaluate --config results/runs/cpu_light_bce_256/config.yaml --checkpoint checkpoints/cpu_light_bce_256/best.pth --split val --output-dir results/runs/cpu_light_bce_256/evaluation
python -m scripts.plot_predictions --config results/runs/cpu_light_bce_256/config.yaml --checkpoint checkpoints/cpu_light_bce_256/best.pth --split val --count 12 --output results/runs/cpu_light_bce_256/evaluation/predictions_val.png
python -m scripts.finalize_baseline
```

The finalizer checks completed-run provenance and fills only Sections 3 and 4 of
the existing team report. Validation small-defect recall has only 18 components
in its denominator; report that limitation. It never evaluates test images.
