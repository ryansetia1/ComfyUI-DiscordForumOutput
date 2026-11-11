"""Best-effort removal of a job's input images and the ComfyUI temp folder."""
from __future__ import annotations

import logging
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
        if name.endswith(" [input]"):
            name = name[: -len(" [input]")]
        elif name.endswith(("[output]", "[temp]")):
            continue
        path = (root / name).resolve()
        # resolve() follows symlinks, so links pointing outside are rejected too.
        if path != root and path.is_relative_to(root) and path.is_file():
            found[path] = None
    return list(found)


def clear_dir(directory) -> None:
    for child in Path(directory).iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def other_jobs_active() -> bool:
    """True when another job runs or waits; assume busy if the queue is unreadable."""
    try:
        from server import PromptServer
        running, pending = PromptServer.instance.prompt_queue.get_current_queue()
        return len(running) > 1 or len(pending) > 0
    except Exception:
        return True


def cleanup(prompt_graph) -> None:
    """Delete the job's input images, then temp if idle. Never raises."""
    try:
        import folder_paths
        for path in input_files(prompt_graph, folder_paths.get_input_directory()):
            try:
                path.unlink()
            except OSError as error:
                LOGGER.warning("Input cleanup skipped %s: %s", path.name, error)
        if not other_jobs_active():
            clear_dir(folder_paths.get_temp_directory())
    except Exception as error:
        LOGGER.warning("Discord cleanup failed: %s", error)
