"""Verify MVTec AD files and measure defect components after nearest-neighbour resize."""

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

import cv2
import matplotlib
import numpy as np
import yaml
from skimage.measure import label

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EXPECTED_COUNTS = {
    "tile": (230, 33, 84, 84),
    "hazelnut": (391, 40, 70, 70),
    "wood": (247, 19, 60, 60),
    "metal_nut": (220, 22, 93, 93),
    "capsule": (219, 23, 109, 109),
}
COUNT_NAMES = ("train_good", "test_good", "test_defective", "masks")


def read_image(path, grayscale=False):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def component_areas(mask, image_size=512):
    resized = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    components = label(resized > 0, connectivity=2)
    return np.bincount(components.ravel())[1:]


def inspect_dataset(root, categories, image_size):
    """Return a checked inventory, per-image component areas, and count differences."""
    root = Path(root)
    records, counts, differences = [], [], []
    areas_by_image = {}
    for category in categories:
        base = root / category
        images = sorted(base.glob("train/*/*.png")) + sorted(base.glob("test/*/*.png"))
        masks = set(base.glob("ground_truth/*/*.png"))
        if not images:
            raise ValueError(f"No images found in {base}")
        used_masks = set()
        actual = dict.fromkeys(COUNT_NAMES, 0)
        actual["masks"] = len(masks)
        for path in images:
            source_split, defect_type = path.parts[-3:-1]
            good = defect_type == "good"
            if source_split == "train" and not good:
                raise ValueError(f"Unexpected defective image in official train: {path}")
            actual[f"{source_split}_{'good' if good else 'defective'}"] += 1
            image = read_image(path)
            areas = np.array([], dtype=np.int64)
            if not good:
                mask_path = base / "ground_truth" / defect_type / f"{path.stem}_mask.png"
                if mask_path not in masks:
                    raise ValueError(f"Missing mask for {path}: {mask_path}")
                mask = read_image(mask_path, grayscale=True)
                if image.shape[:2] != mask.shape:
                    raise ValueError(f"Image/mask shape mismatch: {path}")
                if not np.any(mask):
                    raise ValueError(f"Empty defect mask: {mask_path}")
                areas = component_areas(mask, image_size)
                used_masks.add(mask_path)
            relative = path.relative_to(root).as_posix()
            areas_by_image[relative] = areas
            records.append({
                "path": relative, "category": category,
                "source_split": source_split, "defect_type": defect_type,
                "components": len(areas),
                "vanished_after_resize": not good and len(areas) == 0,
            })
        if masks != used_masks:
            raise ValueError(f"Orphan masks in {category}: {sorted(masks - used_masks)}")
        expected = EXPECTED_COUNTS[category]
        row = {"category": category, **actual}
        for name, value in zip(COUNT_NAMES, expected):
            row[f"expected_{name}"] = value
            if actual[name] != value:
                differences.append(f"{category}/{name}: expected {value}, found {actual[name]}")
        row["matches_expected"] = all(actual[n] == v for n, v in zip(COUNT_NAMES, expected))
        counts.append(row)
    return records, areas_by_image, counts, differences


def summarize(records, areas_by_image, image_size):
    groups = {}
    for record in records:
        key = (record["category"], record["source_split"], record["defect_type"])
        groups.setdefault(key, []).append(record)
    rows = []
    for (category, source_split, defect_type), group in sorted(groups.items()):
        areas = np.concatenate([areas_by_image[r["path"]] for r in group])
        row = {
            "category": category, "source_split": source_split, "defect_type": defect_type,
            "image_count": len(group), "mask_count": len(group) if defect_type != "good" else 0,
            "component_count": len(areas),
            "masks_vanished_after_resize": sum(r["vanished_after_resize"] for r in group),
            "image_size": image_size,
        }
        for name, function in (("min", np.min), ("median", np.median), ("max", np.max)):
            value = float(function(areas)) if len(areas) else None
            row[f"area_{name}_px"] = value
            row[f"area_{name}_pct"] = 100 * value / image_size**2 if value is not None else None
        rows.append(row)
    return rows


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_histogram(path, records, areas_by_image, image_size):
    all_areas = np.concatenate(list(areas_by_image.values()))
    if not len(all_areas):
        raise ValueError("No components remain after resize; cannot plot log-area histogram")
    bins = np.geomspace(max(0.5, all_areas.min() * 0.8), all_areas.max() * 1.2, 41)
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for category in sorted({r["category"] for r in records}):
        areas = np.concatenate([
            areas_by_image[r["path"]] for r in records if r["category"] == category
        ])
        ax.hist(areas, bins=bins, histtype="step", linewidth=1.5, label=category)
    ax.set_xscale("log")
    ax.set_xlabel(f"Component area in pixels ({image_size} × {image_size} masks)")
    ax.set_ylabel("Component count")
    ax.set_title("MVTec AD: all available defect masks (descriptive statistics)")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def read_splits(directory, known_paths, root):
    """Validate all three manifests, including byte-identical images across splits."""
    splits, owners, hashes = {}, {}, {}
    root = Path(root).resolve()
    for name in ("train", "val", "test"):
        manifest = Path(directory) / f"{name}.txt"
        paths = manifest.read_text(encoding="utf-8").splitlines()
        paths = [p.strip() for p in paths if p.strip()]
        if not paths:
            raise ValueError(f"Empty split: {manifest}")
        for relative in paths:
            path = PurePosixPath(relative)
            if (path.is_absolute() or ".." in path.parts or "\\" in relative
                    or ":" in relative or path.as_posix() != relative):
                raise ValueError(f"Split paths must be canonical relative POSIX paths: {relative}")
            if relative not in known_paths:
                raise ValueError(f"Split image not in the checked inventory: {relative}")
            if relative in owners:
                raise ValueError(f"Duplicate split image: {relative} ({owners[relative]}, {name})")
            owners[relative] = name
            full_path = (root / relative).resolve()
            if not full_path.is_relative_to(root):
                raise ValueError(f"Split image escapes data root: {relative}")
            with full_path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "md5").hexdigest()
            if digest in hashes and hashes[digest] != name:
                raise ValueError(f"Identical image bytes across splits: {relative}")
            hashes[digest] = name
        splits[name] = paths
    return splits


def train_percentile(areas_by_image, train_paths, percentile):
    if not 0 < percentile < 100:
        raise ValueError("Percentile must be between 0 and 100")
    areas = np.concatenate([areas_by_image[p] for p in train_paths])
    if not len(areas):
        raise ValueError("Train has no defect components; the official train/good is insufficient")
    return float(np.percentile(areas, percentile, method="linear")), len(areas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/base.yaml")
    parser.add_argument("--root", type=Path, help="Override data.root")
    parser.add_argument("--output", type=Path, help="Override output.results_dir")
    parser.add_argument(
        "--splits-dir", type=Path, help="Validate splits and compute train percentile"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = args.root or REPO / config["data"]["root"]
    output = args.output or REPO / config["output"]["results_dir"]
    size = int(config["data"]["image_size"])
    if size <= 0:
        parser.error("data.image_size must be positive")
    if args.splits_dir and size != 512:
        parser.error("The agreed small-defect threshold must be computed at 512 x 512")
    records, areas, counts, differences = inspect_dataset(root, config["data"]["categories"], size)
    threshold = None
    if args.splits_dir:
        splits = read_splits(args.splits_dir, areas, root)
        percentile = config["eval"]["small_defect_percentile"]
        value, count = train_percentile(areas, splits["train"], percentile)
        threshold = {
            "small_defect_area_px": value, "percentile": percentile,
            "percentile_method": "linear", "comparison": "area < small_defect_area_px",
            "image_size": size, "connectivity": 2, "resize": "cv2.INTER_NEAREST",
            "train_images": len(splits["train"]), "train_components": count,
            "split_sha256": {
                name: hashlib.sha256((args.splits_dir / f"{name}.txt").read_bytes()).hexdigest()
                for name in splits
            },
        }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "dataset_counts.csv", counts)
    write_csv(output / "dataset_stats.csv", summarize(records, areas, size))
    save_histogram(output / "defect_area_hist.png", records, areas, size)
    if threshold is not None:
        (output / "small_defect_threshold.json").write_text(
            json.dumps(threshold, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Train-only percentile: {threshold['small_defect_area_px']:g} pixels")
    else:
        print("Train threshold not computed. After issue #7, rerun with --splits-dir splits.")
    report = {
        "dataset": "ipythonx/mvtec-ad", "image_size": size,
        "resize": "cv2.INTER_NEAREST", "connectivity": 2,
        "images": len(records), "components": sum(len(a) for a in areas.values()),
        "count_differences": differences,
        "masks_vanished_after_resize": sum(r["vanished_after_resize"] for r in records),
        "threshold_computed_in_this_run": threshold is not None,
        "versions": {"numpy": np.__version__, "opencv": cv2.__version__},
    }
    (output / "dataset_verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    for row in counts:
        print(row["category"], "/".join(str(row[n]) for n in COUNT_NAMES),
              "OK" if row["matches_expected"] else "DIFFERS")
    for difference in differences:
        print(f"WARNING: {difference}")
    print(f"Saved statistics to {output}")


if __name__ == "__main__":
    main()
