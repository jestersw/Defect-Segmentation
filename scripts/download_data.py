"""Download the Kaggle mirror and extract only the configured MVTec AD categories."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, is_zipfile

import yaml

REPO = Path(__file__).resolve().parents[1]
DATASET = "ipythonx/mvtec-ad"


def archive_ready(archive):
    partial = Path(str(archive) + ".kaggle-partial")
    return archive.is_file() and not partial.exists() and is_zipfile(archive)


def selected_path(name, categories):
    """Strip optional archive wrappers, rejecting unsafe paths before extraction."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    for index, part in enumerate(path.parts[:-1]):
        tail = path.parts[index:]
        if part in categories and len(tail) == 4:
            if tail[1] in {"train", "test", "ground_truth"} and path.suffix.lower() == ".png":
                return Path(*tail)
    return None


def extract_categories(archive, root, categories):
    root = Path(root).resolve()
    with ZipFile(archive) as source:
        selected = {}
        for info in source.infolist():
            if info.is_dir():
                continue
            relative = selected_path(info.filename, categories)
            if relative is not None:
                if relative in selected:
                    raise ValueError(f"Duplicate destination in archive: {relative}")
                selected[relative] = info
        for category in categories:
            for split in ("train", "test", "ground_truth"):
                if not any(p.parts[:2] == (category, split) for p in selected):
                    raise ValueError(f"Archive does not contain {category}/{split}")
        for relative, info in sorted(selected.items()):
            target = (root / relative).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Destination escapes data root: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".png.part")
            try:
                with source.open(info) as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/base.yaml")
    parser.add_argument("--archive", type=Path, default=REPO / "data/mvtec-ad.zip")
    parser.add_argument("--root", type=Path, help="Override data.root from the config")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = args.root or REPO / config["data"]["root"]
    if not archive_ready(args.archive):
        if args.archive.name != "mvtec-ad.zip":
            parser.error("--archive must point to a complete ZIP or be named mvtec-ad.zip")
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", DATASET, "-p", str(args.archive.parent),
        ]
        print("Downloading ipythonx/mvtec-ad (all categories in the source ZIP).", flush=True)
        environment = os.environ.copy()
        local_credentials = REPO / ".kaggle" / "kaggle.json"
        if not environment.get("KAGGLE_CONFIG_DIR") and local_credentials.is_file():
            environment["KAGGLE_CONFIG_DIR"] = str(local_credentials.parent)
        subprocess.run(command, check=True, env=environment)
    count = extract_categories(args.archive, root, config["data"]["categories"])
    print(f"Extracted {count} PNG files to {root}")
    print("Archive retained for reuse; run scripts/dataset_stats.py to verify the data.")


if __name__ == "__main__":
    main()
