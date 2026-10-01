import numpy as np
from skimage.measure import label


def _as_bool(mask):
    return np.asarray(mask).astype(bool)


def dice_score(pred, target, eps=1e-7):
    pred, target = _as_bool(pred), _as_bool(target)
    inter = np.logical_and(pred, target).sum()
    return float((2 * inter + eps) / (pred.sum() + target.sum() + eps))


def iou_score(pred, target, eps=1e-7):
    pred, target = _as_bool(pred), _as_bool(target)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float((inter + eps) / (union + eps))


def pixel_counts(pred, target):
    pred, target = _as_bool(pred), _as_bool(target)
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    return tp, fp, fn


def small_defect_hits(pred, target, max_area, coverage=0.25):
    pred, target = _as_bool(pred), _as_bool(target)
    components = label(target, connectivity=2)
    detected, total = 0, 0
    for idx in range(1, components.max() + 1):
        region = components == idx
        area = int(region.sum())
        if area >= max_area:
            continue
        total += 1
        if np.logical_and(pred, region).sum() >= coverage * area:
            detected += 1
    return detected, total


def is_false_positive(pred, min_component=20):
    components = label(_as_bool(pred), connectivity=2)
    if components.max() == 0:
        return False
    sizes = np.bincount(components.ravel())[1:]
    return bool((sizes >= min_component).any())


def aggregate(preds, targets, max_area, coverage=0.25, min_component=20):
    dices, ious = [], []
    tp = fp = fn = 0
    small_detected = small_total = 0
    good_images = good_fp = 0
    for pred, target in zip(preds, targets):
        if _as_bool(target).any():
            dices.append(dice_score(pred, target))
            ious.append(iou_score(pred, target))
            d, t = small_defect_hits(pred, target, max_area, coverage)
            small_detected += d
            small_total += t
        else:
            good_images += 1
            good_fp += is_false_positive(pred, min_component)
        a, b, c = pixel_counts(pred, target)
        tp, fp, fn = tp + a, fp + b, fn + c
    return {
        "dice": float(np.mean(dices)) if dices else float("nan"),
        "iou": float(np.mean(ious)) if ious else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "small_recall": small_detected / small_total if small_total else float("nan"),
        "good_fpr": good_fp / good_images if good_images else float("nan"),
    }
