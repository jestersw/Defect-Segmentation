"""Build the shared, stratified MVTec AD split with byte-level leakage checks."""

import argparse
import csv
import hashlib
import io
import json
import warnings
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import sklearn
import yaml
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
NAMES = ("train", "val", "test")


def stratum(path):
    parts = PurePosixPath(path).parts
    return (parts[0], parts[2])


def build_pool(root, categories, seed, exclude=()):
    """Include all official test images, then sample train/good to balance categories."""
    root = Path(root)
    if not categories or len(categories) != len(set(categories)):
        raise ValueError("Categories must be nonempty and unique")
    exclude = set(exclude)
    unknown = {item for item in exclude if item.split("/")[0] not in set(categories)}
    if unknown:
        raise ValueError(f"Excluded defect types refer to unknown categories: {sorted(unknown)}")
    rng = np.random.default_rng(seed)
    pool, composition = [], []
    for category in sorted(categories):
        base = root / category
        candidates = sorted(p for p in base.glob("test/*/*.png") if p.parent.name != "good")
        defective = [p for p in candidates if f"{category}/{p.parent.name}" not in exclude]
        excluded = len(candidates) - len(defective)
        test_good = sorted(base.glob("test/good/*.png"))
        train_good = sorted(base.glob("train/good/*.png"))
        if not defective:
            raise ValueError(f"No defective images found in {base / 'test'}")
        needed = len(defective) - len(test_good)
        if not 0 <= needed <= len(train_good):
            raise ValueError(f"Cannot balance {category}: need {needed} train/good images, "
                             f"found {len(train_good)}")
        for path in defective:
            mask = base / "ground_truth" / path.parent.name / f"{path.stem}_mask.png"
            if not mask.is_file():
                raise ValueError(f"Missing defect mask: {mask}")
        indices = rng.choice(len(train_good), size=needed, replace=False)
        selected_good = [train_good[int(i)] for i in indices]
        pool.extend(p.relative_to(root).as_posix() for p in defective + test_good + selected_good)
        composition.append({
            "category": category, "defective": len(defective), "good": len(defective),
            "official_test_good": len(test_good), "sampled_train_good": needed,
            "excluded_defective": excluded,
        })
    return sorted(pool), composition


def split_pool(pool, seed, ratios=(0.70, 0.15, 0.15)):
    ratios = np.asarray(ratios, dtype=float)
    if (ratios.shape != (3,) or not np.isfinite(ratios).all()
            or (ratios <= 0).any() or not np.isclose(ratios.sum(), 1.0)):
        raise ValueError("Split ratios must be three positive finite values summing to one")
    pool = sorted(pool)
    counts = Counter(stratum(p) for p in pool)
    for group, count in sorted(counts.items()):
        if count < 7:
            warnings.warn(f"Small stratum {group}: {count} images (< 7)", stacklevel=2)
    try:
        train, held_out = train_test_split(
            pool, test_size=float(ratios[1] + ratios[2]), random_state=seed,
            stratify=[stratum(p) for p in pool],
        )
        val, test = train_test_split(
            held_out, test_size=float(ratios[2] / (ratios[1] + ratios[2])), random_state=seed,
            stratify=[stratum(p) for p in held_out],
        )
    except ValueError as error:
        raise ValueError(
            f"Cannot stratify the pool; review stratum sizes and ratios: {error}"
        ) from error
    return {name: sorted(paths) for name, paths in zip(NAMES, (train, val, test))}


def validate_splits(splits, pool, root):
    """Fail before writing anything if paths or byte-identical images leak across splits."""
    root = Path(root).resolve()
    if set(splits) != set(NAMES) or any(not paths for paths in splits.values()):
        raise ValueError("Exactly three nonempty splits are required")
    paths = [p for name in NAMES for p in splits[name]]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate paths inside or across splits")
    if len(pool) != len(set(pool)) or set(paths) != set(pool):
        raise ValueError("Splits must partition the entire pool exactly once")
    hashes = {}
    for name in NAMES:
        for relative in splits[name]:
            parts = PurePosixPath(relative)
            if (parts.is_absolute() or ".." in parts.parts or "\\" in relative
                    or ":" in relative or parts.as_posix() != relative):
                raise ValueError(f"Invalid relative POSIX path: {relative}")
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Image path escapes the dataset root: {relative}")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "md5").hexdigest()
            previous = hashes.get(digest)
            if previous is not None and previous[0] != name:
                raise ValueError(f"Identical image bytes across splits: {previous[1]} "
                                 f"({previous[0]}) and {relative} ({name})")
            hashes[digest] = (name, relative)


def summarize_splits(splits, categories):
    rows = []
    for category in sorted(categories):
        row = {"category": category}
        for name in NAMES:
            selected = [stratum(p)[1] for p in splits[name] if stratum(p)[0] == category]
            row[f"{name}_defective"] = sum(kind != "good" for kind in selected)
            row[f"{name}_good"] = sum(kind == "good" for kind in selected)
        rows.append(row)
    return rows


def save_outputs(splits, rows, composition, splits_dir, results_dir, seed, ratios, overwrite=False,
                 exclude=()):
    manifests = {name: "\n".join(splits[name]) + "\n" for name in NAMES}
    for name, content in manifests.items():
        target = splits_dir / f"{name}.txt"
        if target.exists() and target.read_text(encoding="utf-8") != content and not overwrite:
            raise ValueError(
                f"Refusing to change fixed split {target}; use --overwrite deliberately"
            )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    metadata = {
        "seed": seed, "ratios": list(ratios), "pool_images": sum(map(len, splits.values())),
        "split_sizes": {name: len(splits[name]) for name in NAMES},
        "pool_composition": composition,
        "excluded_defect_types": sorted(exclude),
        "sampling": "NumPy default_rng; sorted categories and paths; without replacement",
        "splitting": "Two sklearn train_test_split calls, stratify=(category, defect_type)",
        "leakage_checks": {"paths_disjoint": True, "cross_split_md5_disjoint": True},
        "versions": {"numpy": np.__version__, "scikit-learn": sklearn.__version__},
        "split_sha256": {
            name: hashlib.sha256(content.encode("utf-8")).hexdigest()
            for name, content in manifests.items()
        },
    }
    outputs = {splits_dir / f"{name}.txt": content for name, content in manifests.items()}
    outputs[results_dir / "split_summary.csv"] = buffer.getvalue()
    outputs[results_dir / "split_metadata.json"] = json.dumps(metadata, indent=2) + "\n"
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/base.yaml")
    parser.add_argument("--root", type=Path, help="Override data.root")
    parser.add_argument("--splits-dir", type=Path, help="Override data.splits_dir")
    parser.add_argument("--output", type=Path, help="Override output.results_dir")
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow replacing a different split"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = args.root or REPO / config["data"]["root"]
    splits_dir = args.splits_dir or REPO / config["data"]["splits_dir"]
    output = args.output or REPO / config["output"]["results_dir"]
    categories, seed = config["data"]["categories"], config["seed"]
    ratios = config["data"]["split_ratios"]
    exclude = config["data"].get("exclude_defect_types", [])
    pool, composition = build_pool(root, categories, seed, exclude)
    splits = split_pool(pool, seed, ratios)
    print(f"Pool: {len(pool)} images. Checking paths and MD5 hashes...", flush=True)
    validate_splits(splits, pool, root)
    rows = summarize_splits(splits, categories)
    save_outputs(
        splits, rows, composition, splits_dir, output, seed, ratios, args.overwrite, exclude
    )
    for name in NAMES:
        print(f"{name}: {len(splits[name])}")
    print(f"Leakage checks passed. Saved manifests to {splits_dir} and summary to {output}.")


if __name__ == "__main__":
    main()
