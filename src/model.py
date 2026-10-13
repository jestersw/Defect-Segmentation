"""Configurable U-Net builder shared by training and evaluation."""

import segmentation_models_pytorch as smp


def build_model(cfg):
    settings = cfg["model"]
    if settings.get("arch", "unet").lower() != "unet":
        raise ValueError("Only the unet architecture is supported")
    if settings.get("classes", 1) != 1:
        raise ValueError("Binary segmentation requires classes=1")
    weights = settings.get("encoder_weights", "imagenet")
    if isinstance(weights, str) and weights.lower() == "none":
        weights = None
    options = {}
    if "decoder_channels" in settings:
        channels = tuple(settings["decoder_channels"])
        if len(channels) != 5 or any(c <= 0 for c in channels):
            raise ValueError("decoder_channels must contain five positive integers")
        options["decoder_channels"] = channels
    model = smp.Unet(
        encoder_name=settings["encoder"], encoder_weights=weights,
        in_channels=3, classes=1, activation=None, **options,
    )
    print(f"U-Net/{settings['encoder']}: {sum(p.numel() for p in model.parameters()):,} parameters",
          flush=True)
    return model
