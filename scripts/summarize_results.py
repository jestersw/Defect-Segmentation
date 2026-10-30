import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    ("A", "cpu_light_bce_256"),
    ("B", "cpu_light_dice_focal_256"),
    ("C", "cpu_light_bce_256_cataug"),
    ("D", "cpu_light_dice_focal_256_cataug"),
    ("E", "cpu_light_dice_focal_512"),
)
METRICS = ("dice", "iou", "precision", "recall", "small_recall", "good_fpr")
CATEGORIES = ("capsule", "hazelnut", "metal_nut", "tile", "wood")
MISSING = r"\tbd"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def read_csv(path):
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value, digits=3):
    if value is None or value == "":
        return "--"
    return f"{float(value):.{digits}f}"


def metric_row(label, run_dir, split):
    metrics = read_json(run_dir / "evaluation" / f"metrics_{split}.json")
    if metrics is None:
        return f"{label} & " + " & ".join([MISSING] * len(METRICS)) + r" \\"
    return f"{label} & " + " & ".join(number(metrics.get(key)) for key in METRICS) + r" \\"


def category_rows(runs, results, split):
    tables = {}
    for label, name in runs:
        rows = read_csv(results / name / "evaluation" / f"metrics_{split}_per_category.csv")
        tables[label] = {row["category"]: row for row in rows} if rows else None
    lines = []
    for category in CATEGORIES:
        cells = []
        for label, _ in runs:
            table = tables[label]
            cells.append(number(table[category]["dice"]) if table and category in table
                         else MISSING)
        lines.append(category.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    return lines


def runtime_row(label, run_dir):
    summary = read_json(run_dir / "summary.json")
    history = read_csv(run_dir / "history.csv")
    inference = read_json(run_dir / "inference_cpu.json")
    if summary is None or not history:
        training = [MISSING] * 3
    else:
        median = statistics.median(float(row["epoch_seconds"]) for row in history)
        training = [
            f"{summary['epochs_completed']} ({summary['best_epoch']})",
            number(median, 1),
            number(summary["total_training_time_seconds"] / 60, 1),
        ]
    timing = ([number(inference["mean_ms"], 1), number(inference["p95_ms"], 1)]
              if inference else [MISSING, MISSING])
    return f"{label} & " + " & ".join(training + timing) + r" \\"


def build_report(results, runs=RUNS):
    results = Path(results)
    sections = {
        "test": [metric_row(label, results / name, "test") for label, name in runs],
        "val": [metric_row(label, results / name, "val") for label, name in runs],
        "test_per_category": category_rows(runs, results, "test"),
        "val_per_category": category_rows(runs, results, "val"),
        "runtime": [runtime_row(label, results / name) for label, name in runs],
    }
    return "\n\n".join(f"% {title}\n" + "\n".join(lines) for title, lines in sections.items())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Print LaTeX table rows for the stage 5 report")
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "runs")
    args = parser.parse_args(argv)
    print(build_report(args.results))


if __name__ == "__main__":
    main()
