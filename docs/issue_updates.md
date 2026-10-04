# Prepared GitHub updates

These texts have not been posted. Push `s1/compute-implementation-plan`, open a PR
against `main`, and copy the relevant update into each issue. Request review from
AigulFar. Remote CI and human review are still pending.

## Issue #9: CPU scope and measured results

The task owner changed the compute target to the local laptop **CPU** on
2026-10-04. The revised scope replaces GPU model/VRAM/CUDA memory with CPU model,
system RAM and peak process RAM; Kaggle attachment is not applicable.

Completed a clean Python 3.11.12 environment with pinned dependencies and a full
`smp.Unet("resnet34", encoder_weights="imagenet", classes=1)` benchmark.
Hardware: AMD Ryzen AI 9 HX 370 (12 physical cores / 24 logical processors),
31.12 GiB usable RAM, Windows; 8 PyTorch computation threads, 1 interop thread.
Local sessions have no provider quota; uptime, sleep and available RAM are limits.

Protocol: batch 8, FP32, BCEWithLogitsLoss, AdamW (lr=1e-4, wd=1e-4), 3 warm-up
steps and 20 measured steps per resolution. A step includes zero_grad, forward,
backward and optimizer update. Seeds are 42 and deterministic algorithms are on.
Peak RAM is the OS-reported worker peak working set, including setup/warm-up;
each resolution uses a fresh process.

| Input | Output | Mean s/step | Peak RAM GiB | Train N | Steps/epoch | Min/epoch | Hours/60 epochs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 x 512 | [8, 1, 512, 512] | 7.5996 | 4.806 | 582 | 73 | 9.246 | 9.246 |
| 256 x 256 | [8, 1, 256, 256] | 1.9071 | 2.248 | 582 | 73 | 2.320 | 2.320 |
| 512 x 512 | [8, 1, 512, 512] | 7.5996 | 4.806 | 550 | 69 | 8.739 | 8.739 |
| 256 x 256 | [8, 1, 256, 256] | 1.9071 | 2.248 | 550 | 69 | 2.193 | 2.193 |

The original 582-image reference is retained above as requested. The committed
split has 550 training images after excluding `metal_nut/flip`. Estimates use
`mean_step_time * ceil(N / 8)`. The minutes/epoch and hours/60 epochs happen to
have the same numerical value because there are 60 epochs and 60 minutes/hour.
These are synthetic training-only projections, excluding data loading,
augmentation, validation and checkpoint I/O. **No full 60-epoch run was performed.**
Dice + Focal runtime has not been measured.

Evidence: `results/smoke_test_cpu.json` (all step timings and environment),
`results/smoke_test_cpu.md`, `results/smoke_test_cpu_environment.txt`.
Reproduction: `docs/compute.md`. Script: `scripts/smoke_test.py`.

Revised definition of done:

- [x] Actual output is [8, 1, 512, 512], and [8, 1, 256, 256] for the ablation.
- [x] CPU step time, peak process RAM and both epoch/run estimates measured.
- [x] Requirements pinned; clean Python 3.11 install and local checks pass.
- [ ] Results posted to this issue (check after posting this update).
- [ ] GitHub Actions green on the pushed branch/PR.
- [ ] Reviewed by AigulFar.

## Issue #10: Section 4 ready for review

Updated `docs/plan_report.tex`, Section 4, using the CPU scope selected by the
task owner and the real measurements from `results/smoke_test_cpu.json`.

- Pipeline and repository table match the actual files. Empty training, loader,
  model, loss, evaluation and profiling modules are explicitly described as
  planned, while implemented preparation, metrics and smoke test are identified.
- Library versions match the pinned requirements, including torchvision,
  scikit-learn, tqdm and psutil for RAM measurement.
- Compute includes CPU/RAM/session limits, both output shapes, step times,
  peak process RAM and estimates for both the original 582-image reference and
  current 550-image split. No `[TBD]` remains in Section 4.
- Training contract specifies AdamW, lr=1e-4, wd=1e-4, cosine, batch 8, max 60
  epochs, early stopping on validation Dice with patience 10, best checkpoint,
  per-epoch CSV and loss/Dice curves.
- Reproducibility covers seeds, deterministic algorithms, fixed threads,
  deterministic cuDNN settings (inactive on CPU), seeded workers and config saved
  per run. Smoke configuration and environment are already recorded in JSON.
- CI includes pinned CPU dependencies, consistency check, Ruff, tests and a small
  offline CPU model check. Full timings come from the laptop benchmark.

Definition of done:

- [x] No `[TBD]` in Section 4.
- [x] Numbers come from the full CPU smoke test.
- [ ] Reviewed by AigulFar.

## Suggested PR text

Title: `Measure CPU baseline compute and finalise Implementation Plan`

The selected compute is now the local CPU. Add a reproducible batch-8 U-Net
benchmark at 512/256 with 3 warm-up and 20 measured steps, record peak process RAM,
and replace Section 4's compute placeholders with real measurements. Pin Python
and direct dependencies, extend CPU CI, and distinguish the implemented scripts
from planned training modules. Retain timing estimates for both the issue's 582
images and the actual 550-image train split.

Local validation: clean Python 3.11 installation, compatible dependencies, Ruff,
46 passing tests, full ImageNet-pretrained CPU smoke test, and successful PDF
compilation with Tectonic 0.17.0. Remote CI and AigulFar's
review are required before merge. No full training run was started.

Closes #9. Closes #10.
