import json

from scripts.summarize_results import build_report

RUNS = (("A", "run_a"), ("B", "run_b"))


def write_run(root, name, with_test):
    run = root / name
    (run / "evaluation").mkdir(parents=True)
    metrics = {"dice": 0.7085, "iou": 0.5966, "precision": 0.8862, "recall": 0.8574,
               "small_recall": None, "good_fpr": 0.0}
    splits = ("val", "test") if with_test else ("val",)
    for split in splits:
        (run / "evaluation" / f"metrics_{split}.json").write_text(json.dumps(metrics))
        (run / "evaluation" / f"metrics_{split}_per_category.csv").write_text(
            "category,images,dice\ncapsule,34,0.4561\ntile,25,0.8447\n")
    (run / "summary.json").write_text(json.dumps(
        {"epochs_completed": 60, "best_epoch": 58, "total_training_time_seconds": 4264.5}))
    (run / "history.csv").write_text("epoch,epoch_seconds\n1,69.0\n2,70.0\n3,71.0\n")
    if with_test:
        (run / "inference_cpu.json").write_text(json.dumps({"mean_ms": 41.26, "p95_ms": 45.04}))


def test_rows_from_complete_and_partial_runs(tmp_path):
    write_run(tmp_path, "run_a", with_test=True)
    write_run(tmp_path, "run_b", with_test=False)
    report = build_report(tmp_path, RUNS)
    sections = dict(block.split("\n", 1) for block in report.split("\n\n"))
    test_a, test_b = sections["% test"].splitlines()
    assert test_a == r"A & 0.709 & 0.597 & 0.886 & 0.857 & -- & 0.000 \\"
    assert test_b == r"B & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\"
    assert sections["% val"].splitlines()[1].startswith("B & 0.709")
    capsule = sections["% test_per_category"].splitlines()[0]
    assert capsule == r"capsule & 0.456 & \tbd \\"
    assert r"metal\_nut & \tbd & \tbd \\" in sections["% test_per_category"]
    assert r"metal\_nut & \tbd & \tbd \\" in sections["% val_per_category"]
    runtime_a, runtime_b = sections["% runtime"].splitlines()
    assert runtime_a == r"A & 60 (58) & 70.0 & 71.1 & 41.3 & 45.0 \\"
    assert runtime_b == r"B & 60 (58) & 70.0 & 71.1 & \tbd & \tbd \\"


def test_missing_run_directory_is_reported_as_tbd(tmp_path):
    report = build_report(tmp_path, (("E", "absent"),))
    rows = [line for line in report.splitlines() if line.startswith("E &")]
    assert rows[0] == "E & " + " & ".join([r"\tbd"] * 6) + r" \\"
    assert rows[-1] == "E & " + " & ".join([r"\tbd"] * 5) + r" \\"
