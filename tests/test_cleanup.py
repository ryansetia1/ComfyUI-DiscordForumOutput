from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from test_components import DiscordForumOutput, Settings, node_result, torch  # noqa: F401  (loads the package)
from discord_forum_node.discord_forum_output import cleanup as mod

CONFIGURED = Settings("https://discord.com/api/webhooks/123456789012345678/SECRET", {"x": "123456789012345678"}, {})


class Client:
    fail = False

    def __init__(self, _settings): pass

    def send(self, *_a, **_k):
        if Client.fail:
            raise RuntimeError("boom")


def graph(*names, cls="LoadImage"):
    return {str(i): {"class_type": cls, "inputs": {"image": n}} for i, n in enumerate(names)}


class CleanupTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.input, self.temp = self.root / "input", self.root / "temp"
        (self.input / "rsg-inpaint").mkdir(parents=True); self.temp.mkdir()
        for name in ("rsg-inpaint/a.png", "keep.png", "orig.png"):
            (self.input / name).write_bytes(b"x")
        (self.root / "outside.png").write_bytes(b"x")
        (self.temp / "preview.png").write_bytes(b"x")
        folder_paths = types.SimpleNamespace(get_input_directory=lambda: str(self.input), get_temp_directory=lambda: str(self.temp))
        patcher = patch.dict(sys.modules, {"folder_paths": folder_paths}); patcher.start(); self.addCleanup(patcher.stop)
        Client.fail = False

    # --- input_files -------------------------------------------------------
    def test_input_files_allows_subfolders_and_rejects_escapes(self):
        g = graph("rsg-inpaint/a.png [input]", "../outside.png", "missing.png", "x.png [output]", "keep.png [temp]", "keep.png[output]", str(self.root / "outside.png"), "notes.txt")
        self.assertEqual(mod.input_files(g, self.input), [self.input / "rsg-inpaint" / "a.png"])

    def test_any_node_input_naming_an_image_counts(self):
        g = {"1": {"class_type": "SomeCustomLoader", "inputs": {"image_path": "keep.png", "image": ["2", 0], "n": 3}}}
        self.assertEqual(mod.input_files(g, self.input), [self.input / "keep.png"])

    def test_symlink_pointing_outside_is_rejected(self):
        (self.input / "link.png").symlink_to(self.root / "outside.png")
        self.assertEqual(mod.input_files(graph("link.png"), self.input), [])

    def test_mask_editor_siblings_are_included_but_unrelated_files_are_not(self):
        for kind in ("mask", "paint", "painted", "painted-masked"):
            (self.input / f"clipspace-{kind}-1762.543.png").write_bytes(b"x")
        (self.input / "clipspace-mask-9999.png").write_bytes(b"x")
        found = {p.name for p in mod.input_files(graph("clipspace-painted-masked-1762.543.png [input]"), self.input)}
        self.assertEqual(found, {f"clipspace-{k}-1762.543.png" for k in ("mask", "paint", "painted", "painted-masked")})

    def test_mask_editor_siblings_in_clipspace_subfolder(self):
        (self.input / "clipspace").mkdir()
        for kind in ("mask", "paint", "painted-masked"):
            (self.input / "clipspace" / f"clipspace-{kind}-1.png").write_bytes(b"x")
        found = mod.input_files(graph("clipspace/clipspace-painted-masked-1.png [input]"), self.input)
        self.assertEqual(len(found), 3)

    # --- run ---------------------------------------------------------------
    def test_run_with_idle_queue_deletes_inputs_and_temp(self):
        message = mod.run(graph("rsg-inpaint/a.png"), [])
        self.assertFalse((self.input / "rsg-inpaint" / "a.png").exists())
        self.assertTrue((self.input / "keep.png").exists())
        self.assertEqual(list(self.temp.iterdir()), [])
        self.assertIn("temp cleared", message)

    def test_run_with_busy_queue_keeps_temp_and_files_other_jobs_use(self):
        mod.run(graph("rsg-inpaint/a.png", "keep.png"), [graph("keep.png [input]")])
        self.assertFalse((self.input / "rsg-inpaint" / "a.png").exists())
        self.assertTrue((self.input / "keep.png").exists())
        self.assertTrue((self.temp / "preview.png").exists())

    def test_preview_nodes_no_longer_block_temp_cleanup(self):
        g = graph("rsg-inpaint/a.png"); g["p"] = {"class_type": "PreviewImage", "inputs": {}}
        mod.run(g, [])
        self.assertEqual(list(self.temp.iterdir()), [])

    # --- purge of stale inputs ---------------------------------------------
    def age(self, rel, seconds_ago):
        import os, time
        path = self.input / rel
        os.utime(path, (time.time() - seconds_ago,) * 2)

    def test_purge_removes_old_images_in_scope_only(self):
        (self.input / "clipspace").mkdir(); (self.input / "pixaroma").mkdir(); (self.input / "3d").mkdir()
        for rel in ("clipspace/c.png", "pixaroma/project.png", "3d/m.png", "notes.txt"):
            (self.input / rel).write_bytes(b"x")
        for rel in ("rsg-inpaint/a.png", "keep.png", "orig.png", "clipspace/c.png", "pixaroma/project.png", "3d/m.png", "notes.txt"):
            self.age(rel, 600)
        import time
        mod.run({}, [], cutoff=time.time() - 60)
        gone = lambda rel: not (self.input / rel).exists()
        self.assertTrue(all(gone(r) for r in ("rsg-inpaint/a.png", "keep.png", "orig.png", "clipspace/c.png")))
        self.assertFalse(any(gone(r) for r in ("pixaroma/project.png", "3d/m.png", "notes.txt")))

    def test_purge_keeps_files_newer_than_the_job_and_files_other_jobs_use(self):
        import time
        for rel in ("rsg-inpaint/a.png", "keep.png"):
            self.age(rel, 600)
        # orig.png was uploaded after the job was queued: possibly for the next job
        mod.run({}, [graph("keep.png")], cutoff=time.time() - 60)
        self.assertTrue((self.input / "keep.png").exists())      # used by a queued job
        self.assertTrue((self.input / "orig.png").exists())      # newer than the cutoff
        self.assertFalse((self.input / "rsg-inpaint" / "a.png").exists())

    def test_without_a_cutoff_nothing_unreferenced_is_purged(self):
        self.age("orig.png", 600)
        mod.run(graph("keep.png"), [])
        self.assertTrue((self.input / "orig.png").exists())
        self.assertFalse((self.input / "keep.png").exists())

    def test_queued_at_reads_create_time_from_the_running_item(self):
        running = [(0, "me", {}, {"create_time": 1_700_000_000_000}, [])]
        server = self.fake_server((running, []))
        with patch.dict(sys.modules, {"server": server}):
            self.assertEqual(mod.queued_at("me"), 1_700_000_000.0)
            self.assertIsNone(mod.queued_at("other"))

    # --- after_job ---------------------------------------------------------
    def fake_server(self, *snapshots):
        """PromptServer whose queue returns each (running, pending) snapshot in turn, then the last."""
        calls = iter(snapshots); last = []
        def get_current_queue():
            nonlocal last
            last = next(calls, last)
            return last
        queue = types.SimpleNamespace(get_current_queue=get_current_queue)
        return types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(prompt_queue=queue, last_prompt_id="me")))

    def test_after_job_waits_for_the_job_to_leave_running_then_cleans(self):
        mine = (0, "me", graph("rsg-inpaint/a.png"), {}, [])
        server = self.fake_server(([mine], []), ([mine], []), ([], []), ([], []))
        with patch.dict(sys.modules, {"server": server}), patch.object(mod, "POLL_SECONDS", 0), patch.object(mod, "GRACE_SECONDS", 0):
            mod.after_job("me", mine[2])
        self.assertFalse((self.input / "rsg-inpaint" / "a.png").exists())
        self.assertEqual(list(self.temp.iterdir()), [])

    def test_after_job_protects_inputs_of_queued_jobs(self):
        queued = (1, "next", graph("rsg-inpaint/a.png"), {}, [])
        server = self.fake_server(([], []), ([], [queued]))
        with patch.dict(sys.modules, {"server": server}), patch.object(mod, "POLL_SECONDS", 0), patch.object(mod, "GRACE_SECONDS", 0):
            mod.after_job("me", graph("rsg-inpaint/a.png"))
        self.assertTrue((self.input / "rsg-inpaint" / "a.png").exists())
        self.assertTrue((self.temp / "preview.png").exists())

    def test_after_job_unreadable_queue_deletes_nothing_and_never_raises(self):
        mod.after_job("me", graph("rsg-inpaint/a.png"))
        self.assertTrue((self.input / "rsg-inpaint" / "a.png").exists())

    def test_after_job_gives_up_if_the_job_never_finishes(self):
        mine = (0, "me", graph("rsg-inpaint/a.png"), {}, [])
        server = self.fake_server(([mine], []))
        with patch.dict(sys.modules, {"server": server}), patch.object(mod, "POLL_SECONDS", 0), patch.object(mod, "TIMEOUT_SECONDS", -1):
            mod.after_job("me", mine[2])
        self.assertTrue((self.input / "rsg-inpaint" / "a.png").exists())

    def test_current_prompt_id_prefers_executing_context_then_last_prompt_id(self):
        server = self.fake_server()
        ctx = types.SimpleNamespace(get_executing_context=lambda: types.SimpleNamespace(prompt_id="ctx"))
        with patch.dict(sys.modules, {"server": server, "comfy_execution": types.ModuleType("comfy_execution"), "comfy_execution.utils": ctx}):
            self.assertEqual(mod.current_prompt_id(), "ctx")
        with patch.dict(sys.modules, {"server": server}):
            self.assertEqual(mod.current_prompt_id(), "me")
        self.assertIsNone(mod.current_prompt_id())

    def test_schedule_starts_a_daemon_thread_for_the_current_job(self):
        thread = MagicMock()
        with patch.object(mod, "current_prompt_id", return_value="me"), patch.object(mod.threading, "Thread", return_value=thread) as make:
            mod.schedule({"g": 1})
        self.assertEqual(make.call_args.kwargs["args"][:2], ("me", {"g": 1})); self.assertTrue(make.call_args.kwargs["daemon"])
        thread.start.assert_called_once()
        with patch.object(mod, "current_prompt_id", return_value=None), patch.object(mod.threading, "Thread") as make:
            mod.schedule({}); make.assert_not_called()

    # --- node wiring -------------------------------------------------------
    def run_node(self, **kwargs):
        with patch("discord_forum_node.nodes.discord_forum_output.load_settings", return_value=CONFIGURED), \
                patch("discord_forum_node.nodes.discord_forum_output.WebhookClient", Client), \
                patch("discord_forum_node.nodes.discord_forum_output.schedule_cleanup") as schedule:
            result = DiscordForumOutput().send(torch.zeros((1, 2, 2, 3)), "x", caption_enabled=False, prompt_graph={"g": 1}, **kwargs)
        return result, schedule

    def test_node_schedules_cleanup_only_after_full_success(self):
        _, schedule = self.run_node(); schedule.assert_called_once_with({"g": 1})
        Client.fail = True
        result, schedule = self.run_node()
        self.assertEqual(node_result(result)[2], 1); schedule.assert_not_called()

    def test_node_toggle_off_schedules_nothing(self):
        _, schedule = self.run_node(cleanup_after_send=False); schedule.assert_not_called()


if __name__ == "__main__":
    unittest.main()
