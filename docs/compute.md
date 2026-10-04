# CPU compute protocol (issues #9 and #10)

The task owner changed the compute target to the local laptop CPU on 2026-10-04.
The revised acceptance criterion is measured CPU step time and peak process RAM
at both resolutions. CUDA/GPU memory and Kaggle dataset attachment no longer apply.
The synthetic benchmark does not need MVTec images.

The full CPU run completed on 2026-10-04. Raw evidence is in
[`results/smoke_test_cpu.json`](../results/smoke_test_cpu.json), with the ready-to-copy
[measurement table](../results/smoke_test_cpu.md) and
[installed package snapshot](../results/smoke_test_cpu_environment.txt).
At 512 pixels the mean is 7.5996 s/step and peak RAM is 4.806 GiB;
at 256 pixels it is 1.9071 s/step and 2.248 GiB. No full 60-epoch run was started.
Local verification passed: dependency compatibility, Ruff, 46 tests, and PDF
compilation with Tectonic 0.17.0. GitHub Actions is pending the branch push.

## Clean setup

Use Python 3.11 (`.python-version`). Create a fresh virtual environment and install:

```sh
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pip check
ruff check .
python -m pytest -q
```

On Windows, the existing OpenCV test fixtures require an ASCII temporary path.
If the checkout path contains non-ASCII characters, use a fresh directory, e.g.
`python -m pytest -q --basetemp C:/Users/yourname/AppData/Local/Temp/defect-tests`.
Do not point `--basetemp` at a directory containing other files: pytest clears it.

The torch/torchvision pairing follows the
[official PyTorch version matrix](https://docs.pytorch.org/get-started/previous-versions/).
Direct runtime and test dependencies are pinned; this is not a complete transitive
lockfile. The benchmark records runtime package versions.

## Full measurement

Keep the laptop plugged in, disable sleep for the session, and avoid competing
CPU-heavy workloads. The first run downloads ImageNet encoder weights; download
time is excluded. Do not substitute random weights when reporting the full test.

```sh
python scripts/smoke_test.py --device cpu --threads 8 --platform local-laptop --session-limits "No provider quota; limited by uptime, sleep and available RAM" --output results/smoke_test_cpu.json
```

Defaults: `smp.Unet("resnet34", encoder_weights="imagenet", classes=1)`, batch 8,
512 then 256 pixels, FP32, seed 42, 3 warm-up steps, 20 timed steps. Each step is
`zero_grad -> forward -> BCEWithLogitsLoss -> backward -> AdamW.step`, with lr
1e-4 and weight decay 1e-4. Inputs and binary targets are generated once on CPU.
Output must be `[8, 1, 512, 512]` and `[8, 1, 256, 256]` respectively; loss must
stay finite. CPU operations are synchronous, timed with `time.perf_counter`.

Each resolution runs in a fresh process. Windows reports its OS peak working
set; other systems sample process RSS every 10 ms. This is whole-process RAM,
including imports, model, activations, optimizer and warm-up, not tensor-only
memory or total laptop RAM. The separate coordinator process is not included.
CPU threads are fixed and saved; deterministic algorithms are enabled.

The script writes JSON (including every step duration) and a Markdown table for
the issue. It counts the committed train manifest and records its SHA-256 hash.
For mean step time `t` seconds:

| Basis | Images | Steps/epoch (batch 8) | Minutes/epoch | Hours/60 epochs |
| --- | ---: | ---: | ---: | ---: |
| Original issue reference | 582 | 73 | `73*t/60` | `73*t/60` |
| Current split, excluding flip | 550 | 69 | `69*t/60` | `69*t/60` |

These estimates exclude loading, augmentation, validation, CSV/curve generation
and checkpoint I/O; the final partial batch is conservatively counted as full.
They do not measure an actual epoch or establish the speed of Dice + Focal.
Early stopping can reduce a real run's duration.

CI runs a smaller offline check (batch 2, 64 pixels, random weights, 1 warm-up
and 1 timed step). Its results do not replace the full laptop measurement.

## Review

Section 4 describes the actual repository and separates implemented components
from planned training modules. AigulFar's review must be obtained on the PR;
passing tests does not constitute that human review.
