"""Caption formatting that keeps workflow text out of logs and disables mentions."""
from __future__ import annotations

import os
import re


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def format_weight(weight: float | int) -> str:
    rounded = round(float(weight), 3)
    text = f"{rounded:.3f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return text


def format_duration(seconds: float | int | str | None) -> str:
    if seconds is None:
        return ""
    if isinstance(seconds, str):
        cleaned = _clean(seconds)
        return cleaned if cleaned else ""
    try:
        sec = float(seconds)
    except (TypeError, ValueError):
        return ""
    if sec <= 0:
        return ""
    if sec < 60:
        return f"{sec:.1f}s"
    minutes = int(sec // 60)
    rem = int(round(sec % 60))
    if rem >= 60:
        minutes += 1
        rem = 0
    return f"{minutes}m {rem}s"


def format_loras(
    loras: tuple[tuple[str, float], ...] | list[tuple[str, float]] | str | None,
) -> tuple[str, str] | None:
    if not loras:
        return None
    if isinstance(loras, str):
        cleaned = _clean(loras)
        return ("loras", cleaned) if cleaned else None
    valid: list[tuple[str, float]] = []
    for item in loras:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            name = os.path.basename(_clean(item[0]))
            try:
                weight = float(item[1])
            except (TypeError, ValueError):
                continue
            if name:
                valid.append((name, weight))
    if not valid:
        return None
    if len(valid) == 1:
        name, weight = valid[0]
        return ("lora", f"{name} ({format_weight(weight)})")
    formatted = ", ".join(f"{name} ({format_weight(weight)})" for name, weight in valid)
    return ("loras", formatted)


def render_caption(
    *,
    prompt: str,
    seed: int | None,
    model_or_checkpoint: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    maximum: int,
    loras: tuple[tuple[str, float], ...] | list[tuple[str, float]] | str | None = None,
    duration: float | int | str | None = None,
) -> str:
    model = os.path.basename(_clean(model_or_checkpoint))
    loras_entry = format_loras(loras)
    duration_str = format_duration(duration)
    fields = [
        ("seed", str(seed) if seed is not None else ""),
        ("model", model),
        *([loras_entry] if loras_entry else []),
        ("resolution", f"{width}x{height}"),
        ("steps", str(steps) if steps > 0 else ""),
        ("CFG", str(cfg) if cfg > 0 else ""),
        ("sampler", _clean(sampler)),
        ("scheduler", _clean(scheduler)),
        ("duration", duration_str),
    ]
    suffix = " | ".join(f"{name}: {value}" for name, value in fields if value)
    prompt = _clean(prompt)
    if not prompt:
        return suffix[:maximum]
    combined = prompt if not suffix else f"{prompt}\n{suffix}"
    if len(combined) <= maximum:
        return combined
    marker = " … [prompt truncated]"
    available = maximum - len(suffix) - len(marker) - (1 if suffix else 0)
    if available > 0:
        return f"{prompt[:available].rstrip()}…{marker[2:]}\n{suffix}"[:maximum]
    return (suffix[: max(0, maximum - len(" … [metadata truncated]"))] + " … [metadata truncated]")[:maximum]
