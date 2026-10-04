# CPU compute smoke test

- Recorded (UTC): 2026-10-04T16:14:02.751192+00:00
- Platform: local-laptop; device: AMD Ryzen AI 9 HX 370 w/ Radeon 890M
- System RAM: 31.12 GiB; limits: No provider quota; limited by uptime, sleep and available RAM
- Python: 3.11.12; torch: 2.6.0
- CPU task protocol satisfied: True
- CPU threads: 8; interop threads: 1.
- Seed: 42; FP32; deterministic algorithms enabled.
- Step includes zero_grad, forward, BCEWithLogitsLoss, backward and AdamW update.
- Inputs stay on device; excludes loading, augmentation, validation and checkpoint I/O.
- Estimates are synthetic training-only projections, not measured epoch durations.

| Size | Output | Mean s/step | Peak process RAM GiB | Train N | Steps/epoch | min/epoch | h/60 epochs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | [8, 1, 512, 512] | 7.5996 | 4.806 | 582 | 73 | 9.246 | 9.246 |
| 512 | [8, 1, 512, 512] | 7.5996 | 4.806 | 550 | 69 | 8.739 | 8.739 |
| 256 | [8, 1, 256, 256] | 1.9071 | 2.248 | 582 | 73 | 2.320 | 2.320 |
| 256 | [8, 1, 256, 256] | 1.9071 | 2.248 | 550 | 69 | 2.193 | 2.193 |

582 images is the issue reference; the current committed split is counted independently. RAM is the whole worker process peak, including model setup and warm-up, measured in a fresh process per resolution.
