import hashlib
import os
import urllib.request
from pathlib import Path

import cv2
import gradio as gr
import matplotlib
import numpy as np
import torch
import yaml

from scripts.demo import predict_image, prediction_summary, prepare_image
from src.evaluate import load_model

try:
    import spaces
except ImportError:
    spaces = None

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "results" / "runs" / "cpu_light_dice_focal_512" / "config.yaml"
CHECKPOINT = ROOT / "checkpoints" / "cpu_light_dice_focal_512" / "best.pth"
WEIGHTS_URL = (
    "https://github.com/jestersw/Defect-Segmentation/releases/download/v1.0/"
    "cpu_light_dice_focal_512.pth"
)
WEIGHTS_SHA256 = "c30e3daaa57e80e5a8c6243ec6ae6e059f7d78f196d6cad6dfb520bb3859b5e6"
EXAMPLES = ROOT / "examples"
OVERLAY_COLOR = np.array([255, 90, 0], dtype=np.float32)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_weights(path=CHECKPOINT):
    if path.is_file() and sha256(path) == WEIGHTS_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    urllib.request.urlretrieve(WEIGHTS_URL, partial)
    if sha256(partial) != WEIGHTS_SHA256:
        partial.unlink()
        raise RuntimeError("Downloaded weights do not match the release SHA-256")
    os.replace(partial, path)
    return path


torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
CFG = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
INPUT_SIZE = CFG["train"]["input_size"]
EVAL_SIZE = CFG["train"]["eval_size"]
MIN_COMPONENT = CFG["eval"].get("good_fp_min_component_px", 20)
MODEL = load_model(CFG, ensure_weights()).eval()
HEATMAP = matplotlib.colormaps["magma"]
with torch.inference_mode():
    MODEL(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE))

if spaces is not None:
    @spaces.GPU(duration=10)
    def gpu_available():
        return torch.cuda.is_available()


def resize_rgb(image, size):
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def overlay(image, mask):
    result = image.astype(np.float32)
    result[mask] = 0.45 * result[mask] + 0.55 * OVERLAY_COLOR
    return result.clip(0, 255).astype(np.uint8)


def segment(path, threshold):
    if path is None:
        raise gr.Error("Upload an image or pick an example first.")
    original, tensor = prepare_image(path, INPUT_SIZE)
    probabilities, elapsed_ms = predict_image(MODEL, tensor, EVAL_SIZE)
    mask, summary = prediction_summary(probabilities, float(threshold), MIN_COMPONENT)
    shown = resize_rgb(original, EVAL_SIZE)
    heatmap = (HEATMAP(probabilities)[..., :3] * 255).astype(np.uint8)
    verdict = "Defect" if summary["has_defect"] else "No defect"
    report = (
        f"### {verdict}\n\n"
        f"- Predicted defect area: **{summary['predicted_defect_area_px']:,} px** "
        f"of {EVAL_SIZE} × {EVAL_SIZE}\n"
        f"- Maximum probability: **{summary['max_probability']:.3f}** "
        f"(threshold {float(threshold):.2f})\n"
        f"- Inference time on CPU: **{elapsed_ms:.0f} ms**\n"
    )
    return overlay(shown, mask), heatmap, report


def auto_segment(path, threshold):
    if path is None:
        return None, None, ""
    return segment(path, threshold)


def example_paths():
    if not EXAMPLES.is_dir():
        return []
    files = sorted(p for p in EXAMPLES.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    return [str(p) for p in files]


DESCRIPTION = """
# Industrial Defect Segmentation on MVTec AD

Final model **E**: U-Net with a ResNet18 encoder, Dice + Focal loss, 512 × 512 input,
trained on five MVTec AD categories (capsule, hazelnut, metal nut, tile, wood).
Test Dice 0.809. Upload an image or click an example; the model marks defect pixels in orange.

Code, results and weights: [github.com/jestersw/Defect-Segmentation](https://github.com/jestersw/Defect-Segmentation)
"""

with gr.Blocks(title="Defect Segmentation Demo") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="filepath", label="Input image", height=360)
            threshold = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Probability threshold")
            run = gr.Button("Segment", variant="primary")
            examples = example_paths()
            if examples:
                gr.Examples(examples, inputs=image, examples_per_page=10,
                            label="Test images from MVTec AD: click one to segment it")
        with gr.Column():
            result = gr.Image(label="Prediction overlay", height=360)
            report = gr.Markdown()
    heatmap = gr.Image(label="Defect probability", height=300)
    gr.Markdown(
        "Example images: MVTec AD (Bergmann et al., CVPR 2019), "
        "[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), "
        "used for non-commercial demonstration."
    )
    run.click(segment, inputs=[image, threshold], outputs=[result, heatmap, report])
    image.change(auto_segment, inputs=[image, threshold], outputs=[result, heatmap, report])
    threshold.release(auto_segment, inputs=[image, threshold], outputs=[result, heatmap, report])

if __name__ == "__main__":
    demo.launch()
