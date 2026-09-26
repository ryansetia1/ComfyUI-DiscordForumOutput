"""Conservative metadata extraction from ComfyUI's execution prompt graph.

The prompt object is the public legacy-node ``PROMPT`` input: a mapping of node
IDs to ``class_type`` and ``inputs``.  We start at the output node's ``images``
link and only walk upstream from there, so a separate sampler elsewhere in the
workflow can never supply caption data.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

_EXECUTION_START_TIMES: dict[str, float] = {}


def record_execution_start(prompt_id: Any) -> None:
    if prompt_id is not None:
        _EXECUTION_START_TIMES[str(prompt_id)] = time.perf_counter()


def record_execution_finish(prompt_id: Any) -> None:
    if prompt_id is not None:
        _EXECUTION_START_TIMES.pop(str(prompt_id), None)


def ensure_execution_timer_hook() -> None:
    try:
        from server import PromptServer
        server = getattr(PromptServer, "instance", None)
        if server is None or getattr(server, "_discord_timer_hooked", False):
            return
        original_send_sync = server.send_sync

        def hooked_send_sync(event, data, sid=None):
            if event == "execution_start" and isinstance(data, dict):
                pid = data.get("prompt_id")
                if pid:
                    record_execution_start(pid)
            elif event in ("execution_success", "execution_error"):
                pid = data.get("prompt_id") if isinstance(data, dict) else None
                if pid:
                    record_execution_finish(pid)
            return original_send_sync(event, data, sid)

        server.send_sync = hooked_send_sync
        server._discord_timer_hooked = True
    except Exception:
        pass


def resolve_duration(explicit_duration: float | int | str | None = None) -> float | None:
    if explicit_duration is not None:
        try:
            val = float(explicit_duration)
            if val > 0:
                return val
        except (TypeError, ValueError):
            pass
    ensure_execution_timer_hook()
    try:
        from server import PromptServer
        server = getattr(PromptServer, "instance", None)
        if server is not None:
            last_pid = getattr(server, "last_prompt_id", None)
            if last_pid and str(last_pid) in _EXECUTION_START_TIMES:
                start_time = _EXECUTION_START_TIMES[str(last_pid)]
                return max(0.0, time.perf_counter() - start_time)
            queue = getattr(server, "prompt_queue", None)
            if queue is not None:
                running = getattr(queue, "currently_running", {})
                for item in running.values():
                    if len(item) > 3 and isinstance(item[3], dict):
                        create_time = item[3].get("create_time")
                        if create_time:
                            return max(0.0, (time.time() * 1000 - create_time) / 1000.0)
    except Exception:
        pass
    return None


@dataclass(frozen=True)
class RuntimeMetadata:
    prompt: str | None = None
    seed: int | None = None
    model_or_checkpoint: str | None = None
    steps: int | None = None
    cfg: float | None = None
    sampler: str | None = None
    scheduler: str | None = None
    loras: tuple[tuple[str, float], ...] = ()
    duration: float | None = None


def extract_metadata(prompt: Mapping[Any, Any] | None, output_node_id: Any) -> RuntimeMetadata:
    """Extract known metadata from the sampler that produced ``images``.

    Ambiguous or unsupported graph shapes intentionally return missing fields.
    Values are read from the submitted execution prompt, which contains the
    resolved value of ComfyUI's randomize-seed widget.
    """
    graph = _Graph(prompt)
    output = graph.node(output_node_id)
    image_source = graph.link(_inputs(output).get("images"))
    if image_source is None:
        return RuntimeMetadata(duration=resolve_duration())
    sampler = graph.nearest(image_source, {"KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"})
    if sampler is None:
        return RuntimeMetadata(duration=resolve_duration())
    sampler_node = graph.node(sampler)
    kind = _class_type(sampler_node)
    if kind in {"KSampler", "KSamplerAdvanced"}:
        return _standard_metadata(graph, sampler_node, kind)
    return _custom_metadata(graph, sampler_node)


def _standard_metadata(graph: "_Graph", node: Mapping[str, Any], kind: str) -> RuntimeMetadata:
    inputs = _inputs(node)
    seed_key = "seed" if kind == "KSampler" else "noise_seed"
    model_name, loras = _model_and_loras(graph, _source(graph, node, "model"))
    return RuntimeMetadata(
        prompt=_conditioning_prompt(graph, _source(graph, node, "positive")),
        seed=_integer(inputs.get(seed_key)),
        model_or_checkpoint=model_name,
        steps=_positive_integer(inputs.get("steps")),
        cfg=_positive_float(inputs.get("cfg")),
        sampler=_text(inputs.get("sampler_name")),
        scheduler=_text(inputs.get("scheduler")),
        loras=loras,
        duration=resolve_duration(),
    )


def _custom_metadata(graph: "_Graph", node: Mapping[str, Any]) -> RuntimeMetadata:
    inputs = _inputs(node)
    if _class_type(node) == "SamplerCustom":
        sigmas = _source(graph, node, "sigmas")
        model_name, loras = _model_and_loras(graph, _source(graph, node, "model"))
        return RuntimeMetadata(
            prompt=_conditioning_prompt(graph, _source(graph, node, "positive")),
            seed=_integer(inputs.get("noise_seed")),
            model_or_checkpoint=model_name,
            steps=_scheduler_steps(graph, sigmas),
            cfg=_positive_float(inputs.get("cfg")),
            sampler=_sampler_name(graph, _source(graph, node, "sampler")),
            scheduler=_scheduler_name(graph, sigmas),
            loras=loras,
            duration=resolve_duration(),
        )

    # Flux's common SamplerCustomAdvanced shape routes model and positive
    # conditioning through BasicGuider, seed through RandomNoise, and sampling
    # settings through KSamplerSelect/BasicScheduler.
    guider = _source(graph, node, "guider")
    guider_node = graph.node(guider[0]) if guider is not None else None
    sigmas = _source(graph, node, "sigmas")
    noise = _source(graph, node, "noise")
    model_name, loras = _model_and_loras(graph, _source(graph, guider_node, "model"))
    return RuntimeMetadata(
        prompt=_guider_prompt(graph, guider_node),
        seed=_noise_seed(graph, noise),
        model_or_checkpoint=model_name,
        steps=_scheduler_steps(graph, sigmas),
        sampler=_sampler_name(graph, _source(graph, node, "sampler")),
        scheduler=_scheduler_name(graph, sigmas),
        loras=loras,
        duration=resolve_duration(),
    )


def _guider_prompt(graph: "_Graph", guider: Mapping[str, Any] | None) -> str | None:
    if not guider:
        return None
    inputs = _inputs(guider)
    return _conditioning_prompt(graph, _source(graph, guider, "positive") or _source(graph, guider, "conditioning"))


def _conditioning_prompt(graph: "_Graph", source: Any) -> str | None:
    text_node = graph.nearest(source, {"CLIPTextEncode"})
    if text_node is None:
        return None
    return _text(_inputs(graph.node(text_node)).get("text"))


def _model_and_loras(graph: "_Graph", source: Any) -> tuple[str | None, tuple[tuple[str, float], ...]]:
    if source is None:
        return None, ()
    current = source
    visited: set[str] = set()
    loras: list[tuple[str, float]] = []
    base_model_name: str | None = None

    while current is not None:
        node_id = current[0] if isinstance(current, (list, tuple)) else current
        key = str(node_id)
        if key in visited:
            break
        visited.add(key)

        node = graph.node(node_id)
        if node is None:
            break

        c_type = _class_type(node)
        inputs = _inputs(node)

        if c_type in {"CheckpointLoader", "CheckpointLoaderSimple", "UNETLoader", "UnetLoaderGGUF"}:
            base_model_name = _text(inputs.get("ckpt_name") or inputs.get("unet_name"))
            break

        if c_type in {"LoraLoader", "LoraLoaderModelOnly"}:
            name = _text(inputs.get("lora_name"))
            weight = inputs.get("strength_model")
            try:
                weight_val = float(weight) if weight is not None else 1.0
            except (TypeError, ValueError):
                weight_val = 1.0
            if name and weight_val != 0.0:
                loras.append((name, weight_val))

        current = graph.link(inputs.get("model"))

    if base_model_name is None:
        base_model_name = _model_name(graph, source)

    loras.reverse()
    return base_model_name, tuple(loras)


def _model_name(graph: "_Graph", source: Any) -> str | None:
    loader = graph.nearest(source, {"CheckpointLoader", "CheckpointLoaderSimple", "UNETLoader", "UnetLoaderGGUF"})
    if loader is None:
        return None
    inputs = _inputs(graph.node(loader))
    return _text(inputs.get("ckpt_name") or inputs.get("unet_name"))


def _noise_seed(graph: "_Graph", source: Any) -> int | None:
    noise = graph.nearest(source, {"RandomNoise"})
    return _integer(_inputs(graph.node(noise)).get("noise_seed")) if noise is not None else None


def _sampler_name(graph: "_Graph", source: Any) -> str | None:
    select = graph.nearest(source, {"KSamplerSelect"})
    return _text(_inputs(graph.node(select)).get("sampler_name")) if select is not None else None


def _scheduler_steps(graph: "_Graph", source: Any) -> int | None:
    scheduler = graph.nearest(source, {"BasicScheduler"})
    return _positive_integer(_inputs(graph.node(scheduler)).get("steps")) if scheduler is not None else None


def _scheduler_name(graph: "_Graph", source: Any) -> str | None:
    scheduler = graph.nearest(source, {"BasicScheduler"})
    return _text(_inputs(graph.node(scheduler)).get("scheduler")) if scheduler is not None else None


def _source(graph: "_Graph", node: Mapping[str, Any] | None, name: str) -> Any:
    return graph.link(_inputs(node).get(name)) if node else None


class _Graph:
    def __init__(self, prompt: Mapping[Any, Any] | None):
        self.prompt = prompt if isinstance(prompt, Mapping) else {}

    def node(self, node_id: Any) -> Mapping[str, Any] | None:
        if node_id in self.prompt and isinstance(self.prompt[node_id], Mapping):
            return self.prompt[node_id]
        string_id = str(node_id)
        node = self.prompt.get(string_id)
        return node if isinstance(node, Mapping) else None

    def link(self, value: Any) -> tuple[Any, int] | None:
        if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[1], int):
            return (value[0], value[1])
        if isinstance(value, Mapping):
            node_id = value.get("node_id", value.get("node"))
            output = value.get("output_index", value.get("output"))
            if node_id is not None and isinstance(output, int):
                return (node_id, output)
        return None

    def nearest(self, source: tuple[Any, int] | None, classes: set[str]) -> Any:
        if source is None:
            return None
        queue = deque([(source[0], 0)])
        visited: set[str] = set()
        matches: list[Any] = []
        distance: int | None = None
        while queue:
            node_id, depth = queue.popleft()
            key = str(node_id)
            if key in visited:
                continue
            visited.add(key)
            node = self.node(node_id)
            if node is None:
                continue
            if distance is not None and depth > distance:
                break
            if _class_type(node) in classes:
                matches.append(node_id)
                distance = depth
                continue
            for value in _inputs(node).values():
                link = self.link(value)
                if link is not None:
                    queue.append((link[0], depth + 1))
        return matches[0] if len(matches) == 1 else None


def _inputs(node: Mapping[str, Any] | None) -> Mapping[str, Any]:
    inputs = node.get("inputs") if isinstance(node, Mapping) else None
    return inputs if isinstance(inputs, Mapping) else {}


def _class_type(node: Mapping[str, Any] | None) -> str:
    value = node.get("class_type") if isinstance(node, Mapping) else ""
    return value if isinstance(value, str) else ""


def _text(value: Any) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_integer(value: Any) -> int | None:
    value = _integer(value)
    return value if value is not None and value > 0 else None


def _positive_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None
