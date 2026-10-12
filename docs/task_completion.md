# Stage 2 handoff: issues #26, #27, #28

Branch: `s2/cpu-lightweight-training`. Prepared 12 October 2026.
No push, checkpoint upload, issue comment or reviewer approval has been performed by this handoff.

Local verification on 12 October: **73 tests passed**, Ruff passed and
`git diff --check` passed. Full pytest used a fresh ASCII user-temp directory
after the initial temp location was blocked by permissions. Both saved best
checkpoints were reloaded and evaluated on validation; metrics exactly match
their saved best-epoch summaries. New Dice-run prediction panels were generated.

## #26 Model builder — ready for CI/review

Paste into the issue:

> Implemented `build_model(cfg)` with SMP U-Net, configurable encoder and decoder,
> one output logit channel, and `none`/null encoder weights mapped to `None`.
> Tests check `[2,3,64,64] -> [2,1,64,64]`, backward gradients and the ResNet34
> default without downloading weights. Parameter count is printed. Completed
> CPU experiments use separate ResNet18 configs. GitHub CI and AigulFar review
> remain pending after push.

## #27 Training loop — ready for CI/review

Paste into the issue:

> Implemented config-driven CPU training, saved CLI overrides, seeded deterministic
> execution, AdamW, cosine scheduling, validation Dice checkpoint selection,
> patience-10 early stopping, best/last checkpoints, per-epoch CSV and curves,
> resolved config and summary JSON. Synthetic tests cover BCE and Dice training,
> checkpoint reload, early stopping, time-budget stopping and overwrite protection.
> Real-data validation exceeds the requested two-epoch trial: runs completed for
> 60 and 48 epochs on all 550 training images (69 batches/epoch), with 118 validation
> images and no batch limit. Test data were not evaluated. GitHub CI and reviewer
> sign-off are still pending.

## #28 BCE run — CPU scope deviation must be accepted

Paste into the issue:

> Completed CPU BCE reference: `results/runs/cpu_light_bce_256/`.
> This uses ResNet18 with decoder [128,64,32,16,8] at 256 square, rather than
> the ResNet34 `configs/base.yaml` run named `baseline_bce_256` in the original
> issue. Please accept this CPU scope change before closing #28; it is not a
> completed ResNet34 experiment.
>
> BCE: 60 epochs, best epoch 58, total 74.83 minutes, best validation Dice
> 0.708525, IoU 0.596578, small recall 0.555556, good-image FPR 0.
>
> Additional joint experiment: `results/runs/cpu_dice_384/`, ResNet18 with
> decoder [256,128,64,32,16], 384-square inputs and pure Dice loss. Completed
> 48 epochs; early stopping after ten epochs without improvement, best epoch 38.
> Total 228.98 minutes including a sleep interruption. Best validation Dice
> 0.770742, IoU 0.684881, small recall 0.722222, good-image FPR 0.
>
> Observations: larger defects are easier; small capsule cracks and thin branches
> of tile cracks can be missed. Epoch 12 of the Dice run took 2047.7 seconds during
> laptop sleep; ordinary epochs took about four minutes. Multiple factors changed
> together, so this is not a controlled loss ablation. Only one seed was used.
>
> Artifacts include config.yaml, history.csv, curves.png, summary.json and validation
> evaluation tables/prediction panels. Best checkpoints must be uploaded separately;
> add their shared-drive download links here before closing the issue.

## Files to commit

The earlier implementation is already committed locally in this branch. Commit
the additional Dice code, tests, config, documentation and both completed runs:

```bash
git branch --show-current
git add src/losses.py src/train.py tests/test_losses.py tests/test_train.py
git add configs/cpu_dice_384.yaml docs/cpu_training.md docs/dice_384_experiment.md
git add docs/report_2.tex docs/task_completion.md
git add results/s2/data_samples.png
git add results/runs/cpu_light_bce_256 results/runs/cpu_dice_384
git diff --cached --stat
git commit -m "Report completed CPU BCE and Dice segmentation experiments"
git push -u origin s2/cpu-lightweight-training
```

Do not force-add ignored files. Dataset, archives, environment, checkpoints and
background logs are ignored. Upload `checkpoints/cpu_light_bce_256/best.pth`
and `checkpoints/cpu_dice_384/best.pth` to the team's shared drive separately.
Use each checkpoint with its matching run config. No download link has yet been
provided, and no remote CI result has been verified.

## Reports

`docs/report_2.tex` is the Stage 2 report with actual configurations, timing,
metrics, curves and limitations. Validation CSVs and JSONs are the numerical
evidence. `docs/dice_384_experiment.md` records the experiment and launch commands.
This handoff contains ready-to-paste issue comments. A TeX compiler is not available
in the current shell, so PDF compilation is not verified; compile report_2.tex
from the repository root or docs directory with its referenced image files.

The experiment changed architecture, resolution and loss together. Preserve
threshold 0.5 for this recorded comparison. Select any future threshold using
validation only, before evaluating the held-out test split.
