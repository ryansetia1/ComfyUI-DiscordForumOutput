"""Best-effort removal of a job's input images and the ComfyUI temp folder.

Runs on a background thread once the job has left ComfyUI's running set, so the
frontend has already fetched previews and the current job needs no special case.
"""
from __future__ import annotations

import logging
import re
import shutil
import threading
import time
from pathlib import Path

LOGGER = logging.getLogger("ComfyUI-DiscordForumOutput")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")
# The mask editor saves four files per edit, but the graph names only one of them.
CLIPSPACE = re.compile(r"^clipspace-(?:mask|paint|painted-masked|painted)-(.+)$")
POLL_SECONDS, GRACE_SECONDS, TIMEOUT_SECONDS = 0.5, 2.0, 600.0


def _clipspace_siblings(path: Path) -> list[Path]:
    match = CLIPSPACE.match(path.name)
    if not match:
        return []
    return [p for p in path.parent.iterdir() if p != path and p.is_file() and (m := CLIPSPACE.match(p.name)) and m.group(1) == match.group(1)]


def input_files(prompt_graph, input_dir) -> list[Path]:
    """Image files under ``input_dir`` that any node input of the graph names."""
    root = Path(input_dir).resolve()
    found: dict[Path, None] = {}
    for node in (prompt_graph or {}).values():
        for name in ((node.get("inputs") or {}).values() if isinstance(node, dict) else ()):
            if not isinstance(name, str):
                continue
            annotated = re.search(r"\s*\[(input|output|temp)\]$", name)
            if annotated:
                if annotated.group(1) != "input":
                    continue
                name = name[: annotated.start()]
            if not name.lower().endswith(IMAGE_EXTS):
                continue
            try:
                path = (root / name).resolve()  # follows symlinks, so links pointing outside are rejected
                if path.is_relative_to(root) and path.is_file():
                    for p in (path, *_clipspace_siblings(path)):
                        found[p] = None
            except (OSError, ValueError):
                continue
    return list(found)


def clear_dir(directory) -> None:
    for child in Path(directory).iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _queue():
    """(running, pending) queue items; raises if ComfyUI's queue is unreachable."""
    from server import PromptServer
    return PromptServer.instance.prompt_queue.get_current_queue()


def current_prompt_id():
    try:
        from comfy_execution.utils import get_executing_context
        return get_executing_context().prompt_id
    except Exception:
        pass
    try:
        from server import PromptServer
        return PromptServer.instance.last_prompt_id
    except Exception:
        return None


def run(prompt_graph, others) -> str:
    """Delete this job's inputs (minus any another job needs), then temp if idle."""
    import folder_paths
    input_dir = folder_paths.get_input_directory()
    needed = {p for graph in others for p in input_files(graph, input_dir)}
    removed = 0
    for path in input_files(prompt_graph, input_dir):
        if path in needed:
            continue
        try:
            path.unlink()
            removed += 1
        except OSError as error:
            LOGGER.warning("Input cleanup skipped %s: %s", path.name, error)
    if others:
        return f"removed {removed} input file(s); temp kept (other jobs queued)"
    clear_dir(folder_paths.get_temp_directory())
    return f"removed {removed} input file(s); temp cleared"


def after_job(prompt_id, prompt_graph) -> None:
    """Wait for the job to finish, then clean. Never raises."""
    try:
        deadline = time.monotonic() + TIMEOUT_SECONDS
        while any(item[1] == prompt_id for item in _queue()[0]):
            if time.monotonic() > deadline:
                LOGGER.warning("Discord cleanup skipped: job did not finish in time")
                return
            time.sleep(POLL_SECONDS)
        time.sleep(GRACE_SECONDS)  # let the frontend finish loading previews
        running, pending = _queue()
        LOGGER.info("Discord cleanup: %s", run(prompt_graph, [item[2] for item in (*running, *pending)]))
    except Exception as error:
        LOGGER.warning("Discord cleanup failed: %s", error)


def schedule(prompt_graph) -> None:
    prompt_id = current_prompt_id()
    if prompt_id is None:
        LOGGER.warning("Discord cleanup skipped: cannot tell which job is running")
        return
    threading.Thread(target=after_job, args=(prompt_id, prompt_graph), daemon=True, name="discord-cleanup").start()
