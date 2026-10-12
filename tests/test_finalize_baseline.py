from pathlib import Path

from scripts.finalize_baseline import KEYS, fill_report


def test_report_fills_only_owned_sections_and_is_idempotent():
    report = Path(__file__).resolve().parents[1] / "docs/report_2.tex"
    original = report.read_text(encoding="utf-8")
    metrics = dict.fromkeys(KEYS, 0.5)
    categories = [dict(category=name, **metrics)
                  for name in ("capsule", "hazelnut", "metal_nut", "tile", "wood")]
    categories[2]["small_recall"] = ""
    summary = dict(total_training_time_seconds=120, epochs_completed=2, best_epoch=1)
    history = [dict(epoch_seconds=60)]
    sweep = [dict(threshold=t, **metrics) for t in (0.3, 0.5, 0.7)]
    updated = fill_report(original, summary, history, metrics, categories, sweep, 18)
    start = r"\section{Baseline Model and Training Setup}"
    end = r"\section{Problems and Observations}"
    assert original.split(start)[0] == updated.split(start)[0]
    assert original.split(end)[1] == updated.split(end)[1]
    assert "18 validation components" in updated
    assert r"\tbd{" not in updated.split(start)[1].split(end)[0]
    assert fill_report(updated, summary, history, metrics, categories, sweep, 18) == updated
