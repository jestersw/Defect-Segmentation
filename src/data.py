"""Fixed MVTec AD manifests, paired training transforms and deterministic evaluation."""

import random
from pathlib import Path, PurePosixPath

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

ROOT = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_image(path, grayscale=False):
    """Read Unicode paths on Windows too; OpenCV's colour output is BGR."""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image if grayscale else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class MVTecSegDataset(Dataset):
    def __init__(self, split: str, cfg: dict, train: bool):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        if train and split != "train":
            raise ValueError("Training augmentation is only allowed for the train split")
        self.root = (ROOT / cfg["data"]["root"]).resolve()
        self.input_size = int(cfg["train"]["input_size"])
        self.mask_size = self.input_size if train else int(cfg["train"]["eval_size"])
        if min(self.input_size, self.mask_size) <= 0:
            raise ValueError("input_size and eval_size must be positive")
        manifest = ROOT / cfg["data"]["splits_dir"] / f"{split}.txt"
        self.records = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            relative = line.strip()
            if not relative:
                continue
            path = PurePosixPath(relative)
            if (len(path.parts) != 4 or path.is_absolute() or ".." in path.parts
                    or "\\" in relative or ":" in relative or path.as_posix() != relative
                    or path.parts[1] not in ("train", "test") or path.suffix != ".png"):
                raise ValueError(f"Invalid MVTec image path in {manifest}: {relative}")
            if not (self.root / relative).resolve().is_relative_to(self.root):
                raise ValueError(f"Image path escapes dataset root: {relative}")
            self.records.append({"path": relative, "category": path.parts[0],
                                 "defect_type": path.parts[2]})
        if not self.records:
            raise ValueError(f"Empty split: {manifest}")
        if len({r["path"] for r in self.records}) != len(self.records):
            raise ValueError(f"Duplicate image paths in {manifest}")
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        ], seed=int(cfg["seed"])) if train else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = self.root / record["path"]
        image = read_image(image_path)
        if record["defect_type"] == "good":
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask_path = (self.root / record["category"] / "ground_truth"
                         / record["defect_type"] / f"{image_path.stem}_mask.png")
            mask = read_image(mask_path, grayscale=True)
            if mask.shape != image.shape[:2]:
                raise ValueError(f"Image/mask shape mismatch: {image_path} and {mask_path}")
        image = cv2.resize(image, (self.input_size, self.input_size),
                           interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.mask_size, self.mask_size),
                          interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        image = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return (
            torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            torch.from_numpy(np.ascontiguousarray(mask[None], dtype=np.float32)),
        )


def seed_worker(worker_id):
    """Separate reproducible random streams; top-level function supports Windows spawn."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    dataset = get_worker_info().dataset
    if dataset.transform is not None:
        dataset.transform.set_random_seed(seed)


def make_loaders(cfg):
    loaders = []
    for offset, split in enumerate(("train", "val", "test")):
        # Independent generators keep validation iteration from changing training order.
        generator = torch.Generator().manual_seed(int(cfg["seed"]) + offset)
        loaders.append(DataLoader(
            MVTecSegDataset(split, cfg, train=split == "train"),
            batch_size=cfg["train"]["batch_size"],
            num_workers=cfg["train"]["num_workers"],
            shuffle=split == "train", generator=generator, worker_init_fn=seed_worker,
            drop_last=False,
        ))
    return tuple(loaders)
