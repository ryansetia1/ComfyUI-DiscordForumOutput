"""Caption formatting that keeps workflow text out of logs and disables mentions."""
from __future__ import annotations

import os
import re


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def render_caption(*, prompt: str, seed: int | None, model_or_checkpoint: str, width: int, height: int, steps: int, cfg: float, sampler: str, scheduler: str, maximum: int) -> str:
    model = os.path.basename(_clean(model_or_checkpoint))
    fields = [
        ("seed", str(seed) if seed is not None else ""), ("model", model), ("resolution", f"{width}x{height}"),
        ("steps", str(steps) if steps > 0 else ""), ("CFG", str(cfg) if cfg > 0 else ""),
        ("sampler", _clean(sampler)), ("scheduler", _clean(scheduler)),
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
