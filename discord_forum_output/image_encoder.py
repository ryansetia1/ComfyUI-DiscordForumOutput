"""Convert Comfy IMAGE tensors into bounded, in-memory image bytes."""
from __future__ import annotations

import io
import json
from dataclasses import dataclass

from PIL import Image
from PIL.PngImagePlugin import PngInfo


class ImageEncodingError(ValueError):
    pass


@dataclass
class EncodedImage:
    filename: str
    content_type: str
    data: bytes
    width: int
    height: int
    alpha_dropped: bool = False


def _metadata_disabled() -> bool:
    try:
        from comfy.cli_args import args
        return bool(getattr(args, "disable_metadata", False))
    except Exception:
        return False


def png_metadata(prompt=None, extra_pnginfo=None) -> PngInfo | None:
    """Same text chunks as ComfyUI's SaveImage, so dragging the PNG back in restores the workflow."""
    if _metadata_disabled() or (prompt is None and not extra_pnginfo):
        return None
    info = PngInfo()
    if prompt is not None:
        info.add_text("prompt", json.dumps(prompt))
    for key, value in (extra_pnginfo or {}).items():
        info.add_text(key, json.dumps(value))
    return info


def encode_image(item, index: int, image_format: str, jpeg_quality: int, pnginfo: PngInfo | None = None) -> EncodedImage:
    if getattr(item, "ndim", None) != 3 or item.shape[0] < 1 or item.shape[1] < 1 or item.shape[2] not in (3, 4):
        raise ImageEncodingError("image must have non-empty [H,W,3] or [H,W,4] dimensions")
    if not hasattr(item, "is_floating_point"):
        raise ImageEncodingError("image pixels must be numeric")
    cpu = item.detach().to("cpu")
    if not bool(cpu.isfinite().all().item()):
        raise ImageEncodingError("image contains non-finite pixels")
    if float(cpu.min().item()) < 0 or float(cpu.max().item()) > 1:
        raise ImageEncodingError("image pixels must be in the [0, 1] range")
    array = cpu.mul(255).round().clamp(0, 255).byte().numpy()
    height, width, channels = array.shape
    alpha_dropped = image_format == "jpeg" and channels == 4
    if image_format == "jpeg":
        image = Image.fromarray(array[:, :, :3], "RGB")
        suffix, content_type = "jpg", "image/jpeg"
    elif image_format == "png":
        image = Image.fromarray(array, "RGBA" if channels == 4 else "RGB")
        suffix, content_type = "png", "image/png"
    else:
        raise ImageEncodingError("image_format must be png or jpeg")
    output = io.BytesIO()
    try:
        if image_format == "jpeg":
            image.save(output, format="JPEG", quality=jpeg_quality, optimize=False)
        else:
            image.save(output, format="PNG", compress_level=6, pnginfo=pnginfo)
        data = output.getvalue()
    finally:
        output.close()
    return EncodedImage(f"comfy-{index:03d}.{suffix}", content_type, data, width, height, alpha_dropped)
