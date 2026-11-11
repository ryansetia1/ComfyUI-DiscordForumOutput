"""Best-effort removal of a job's input images and the ComfyUI temp folder."""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

LOGGER = logging.getLogger("ComfyUI-DiscordForumOutput")
# Loader nodes whose "image" widget names a file in the input folder.
LOADERS = {"LoadImage", "LoadImageMask", "AILab_LoadImage"}


def input_files(prompt_graph, input_dir) -> list[Path]:
    """Files under ``input_dir`` that the graph's image loaders read."""
    root = Path(input_dir).resolve()
    found: dict[Path, None] = {}
    for node in (prompt_graph or {}).values():
        if not isinstance(node, dict) or node.get("class_type") not in LOADERS:
            continue
        name = (node.get("inputs") or {}).get("image")
        if not isinstance(name, str) or not name.strip():
            continue
        annotated = re.search(r"\s*\[(input|output|temp)\]$", name)
        if annotated:
            if annotated.group(1) != "input":
                continue
            name = name[: annotated.start()]
        path = (root / name).resolve()
        # resolve() follows symlinks, so links pointing outside are rejected too.
        if path.is_relative_to(root) and path.is_file():
            found[path] = None
    return list(found)


def clear_dir(directory) -> None:
    for child in Path(directory).iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def other_prompts():
    """Graphs of every other running or queued job, or None if that cannot be told.

    The queue holds deep copies, so the current job is matched by prompt_id
    (from ComfyUI's executing-context), never by object identity.
    """
    try:
        from comfy_execution.utils import get_executing_context
        from server import PromptServer
        prompt_id = get_executing_context().prompt_id
        running, pending = PromptServer.instance.prompt_queue.get_current_queue()
        return [item[2] for item in (*running, *pending) if item[1] != prompt_id]
    except Exception:
        return None


def cleanup(prompt_graph) -> None:
    """Delete the job's input images, then temp if idle. Never raises."""
    try:
        import folder_paths
        others = other_prompts()
        if others is None:  # cannot tell what else needs these files: delete nothing
            return
        input_dir = folder_paths.get_input_directory()
        needed = {path for graph in others for path in input_files(graph, input_dir)}
        for path in input_files(prompt_graph, input_dir):
            if path in needed:
                continue
            try:
                path.unlink()
            except OSError as error:
                LOGGER.warning("Input cleanup skipped %s: %s", path.name, error)
        # Previews of this job live in temp and are fetched after the node runs.
        has_preview = any("Preview" in str(n.get("class_type")) for n in prompt_graph.values() if isinstance(n, dict))
        if not others and not has_preview:
            clear_dir(folder_paths.get_temp_directory())
    except Exception as error:
        LOGGER.warning("Discord cleanup failed: %s", error)
