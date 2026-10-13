"""Fill only Stage 2 report Sections 3 and 4 from a completed baseline run."""

import csv
import hashlib
import json
import math
import re
import statistics

import numpy as np
import yaml

from src.data import ROOT, MVTecSegDataset
from src.metrics import small_defect_hits

RUN = ROOT / "results/runs/cpu_light_bce_256"
KEYS = ("dice", "iou", "precision", "recall", "small_recall", "good_fpr")


def number(value):
    return "--" if value in (None, "") else f"{float(value):.4f}"


def fill_report(text, summary, history, metrics, categories, sweep, small_count):
    start = text.index(r"\section{Baseline Model and Training Setup}")
    middle = text.index(r"\section{Initial Results (Validation Split)}")
    end = text.index(r"\section{Problems and Observations}")
    setup, results = text[start:middle], text[middle:end]
    seconds = statistics.median(float(row["epoch_seconds"]) for row in history)
    timing = (f"Training time & Median epoch {seconds / 60:.2f} min; "
              f"total {summary['total_training_time_seconds'] / 60:.2f} min; "
              f"{summary['epochs_completed']} epochs; best epoch {summary['best_epoch']} "
              + r"\\")
    setup = re.sub(r"^Training time &.*$", lambda _: timing, setup, flags=re.MULTILINE)
    by_category = {row["category"].replace("_", r"\_"): row for row in categories}
    rows = {"All": metrics, **by_category}
    for label, row in rows.items():
        replacement = label + " & " + " & ".join(number(row[key]) for key in KEYS) + r" \\"
        results, count = re.subn(r"^" + re.escape(label) + r"\s*&.*$",
                                 lambda _: replacement, results, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"Expected one table row for {label}")
    results = results.replace(r"cpu\_light\_bce\_256/metrics\_val.json",
                              r"cpu\_light\_bce\_256/evaluation/metrics\_val.json")
    thresholds = {float(row["threshold"]): row for row in sweep}
    sensitivity = (r"\textbf{Threshold sensitivity.} At thresholds 0.3 / 0.5 / 0.7 "
                   + "the pixel recall is "
                   + " / ".join(number(thresholds[t]["recall"]) for t in (0.3, 0.5, 0.7))
                   + " and the pixel precision "
                   + " / ".join(number(thresholds[t]["precision"]) for t in (0.3, 0.5, 0.7))
                   + ". The reported comparison retains threshold 0.5.")
    results = re.sub(r"\\textbf\{Threshold sensitivity\.\}[^\n]*",
                     lambda _: sensitivity, results)
    ordered = sorted(categories, key=lambda row: float(row["dice"]))
    weakest = ordered[0]["category"].replace("_", r"\_")
    strongest = ordered[-1]["category"].replace("_", r"\_")
    interpretation = (
        r"\textbf{Interpretation.} "
        f"{strongest} has the highest category Dice and {weakest} the lowest. "
        f"Small-defect recall is {number(metrics['small_recall'])}; it is based on only "
        f"{small_count} validation components, so a single detection changes it by "
        f"{100 / small_count:.2f} percentage points. Good-image FPR is "
        f"{number(metrics['good_fpr'])}. Dice is averaged over defective images; "
        "small recall requires at least 25 percent coverage of a component smaller "
        "than 859.28 pixels at 512 resolution. Results are validation-selected and "
        "single-seed; the held-out test split remains unused."
    )
    results = re.sub(r"\\textbf\{Interpretation\.\}[^\n]*", lambda _: interpretation, results)
    results = results.replace("results/s2/baseline_examples.png",
                              "results/runs/cpu_light_bce_256/evaluation/predictions_val.png")
    if r"\tbd{" in setup + results:
        raise ValueError("Unfilled placeholder in Sections 3 or 4")
    return text[:start] + setup + results + text[end:]


def main():
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((RUN / "config.yaml").read_text(encoding="utf-8"))
    if summary["status"] != "completed" or summary["limited_run"]:
        raise ValueError("A completed, unlimited run is required")
    if (cfg["loss"] != "bce" or cfg["model"]["encoder"] != "resnet18"
            or cfg["model"]["decoder_channels"] != [128, 64, 32, 16, 8]
            or cfg["train"]["input_size"] != 256):
        raise ValueError("Not the agreed CPU BCE baseline")
    for name, expected in summary["environment"]["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed since training: {name}")
    for name, expected in summary["split_sha256"].items():
        manifest = ROOT / cfg["data"]["splits_dir"] / f"{name}.txt"
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Split changed since training: {name}")
    evaluation = RUN / "evaluation"
    metrics = json.loads((evaluation / "metrics_val.json").read_text())
    for key in KEYS:
        if not math.isclose(metrics[key], summary["best_validation_metrics"][key], abs_tol=1e-8):
            raise ValueError(f"Reloaded best checkpoint disagrees on {key}")
    cfg["data"]["cache_images"] = False
    small_count = sum(small_defect_hits(np.zeros(mask.shape[1:], dtype=bool), mask[0].numpy(),
                                      max_area=cfg["eval"]["small_defect_area_px"])[1]
                      for _, mask in MVTecSegDataset("val", cfg, train=False))
    if small_count != 18:
        raise ValueError(f"Expected 18 small validation components, found {small_count}")
    if not (evaluation / "predictions_val.png").is_file():
        raise FileNotFoundError("Render validation predictions before finalizing")

    def read_csv(path):
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    report = ROOT / "docs/report_2.tex"
    updated = fill_report(report.read_text(encoding="utf-8"), summary,
                          read_csv(RUN / "history.csv"), metrics,
                          read_csv(evaluation / "metrics_val_per_category.csv"),
                          read_csv(evaluation / "threshold_sweep_val.csv"), small_count)
    report.write_text(updated, encoding="utf-8", newline="\n")
    print("Updated only report Sections 3/4; provenance and 18-component denominator verified.")


if __name__ == "__main__":
    main()
