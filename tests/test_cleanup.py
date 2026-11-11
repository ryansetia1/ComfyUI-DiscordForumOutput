from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from test_components import DiscordForumOutput, Settings, node_result, torch  # noqa: F401  (loads the package)
from discord_forum_node.discord_forum_output import cleanup as mod

CONFIGURED = Settings("https://discord.com/api/webhooks/123456789012345678/SECRET", {"x": "123456789012345678"}, {})


class Client:
    fail = False

    def __init__(self, _settings): pass

    def send(self, *_a, **_k):
        if Client.fail:
            raise RuntimeError("boom")


class CleanupTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.input, self.temp = self.root / "input", self.root / "temp"
        (self.input / "rsg-inpaint").mkdir(parents=True); self.temp.mkdir()
        (self.input / "rsg-inpaint" / "a.png").write_bytes(b"x")
        (self.input / "keep.png").write_bytes(b"x")
        (self.root / "outside.png").write_bytes(b"x")
        (self.temp / "preview.png").write_bytes(b"x")
        folder_paths = types.SimpleNamespace(get_input_directory=lambda: str(self.input), get_temp_directory=lambda: str(self.temp))
        patcher = patch.dict(sys.modules, {"folder_paths": folder_paths}); patcher.start(); self.addCleanup(patcher.stop)
        Client.fail = False

    def graph(self, *names, cls="LoadImage"):
        return {str(i): {"class_type": cls, "inputs": {"image": n}} for i, n in enumerate(names)}

    def test_input_files_allows_subfolders_and_rejects_escapes(self):
        graph = self.graph("rsg-inpaint/a.png [input]", "../outside.png", "missing.png", "x.png [output]")
        graph["9"] = {"class_type": "KSampler", "inputs": {"image": "keep.png"}}
        self.assertEqual(mod.input_files(graph, self.input), [self.input / "rsg-inpaint" / "a.png"])

    def test_absolute_and_temp_annotated_names_are_rejected(self):
        graph = self.graph(str(self.root / "outside.png"), "keep.png [temp]", "keep.png[output]")
        self.assertEqual(mod.input_files(graph, self.input), [])

    def test_symlink_pointing_outside_is_rejected(self):
        (self.input / "link.png").symlink_to(self.root / "outside.png")
        self.assertEqual(mod.input_files(self.graph("link.png"), self.input), [])

    def run_node(self, graph, **kwargs):
        with patch("discord_forum_node.nodes.discord_forum_output.load_settings", return_value=CONFIGURED), \
                patch("discord_forum_node.nodes.discord_forum_output.WebhookClient", Client), \
                patch.object(mod, "other_prompts", return_value=kwargs.pop("others", [])):
            return DiscordForumOutput().send(torch.zeros((1, 2, 2, 3)), "x", caption_enabled=False, prompt_graph=graph, **kwargs)

    def test_success_deletes_inputs_and_temp_only(self):
        self.run_node(self.graph("rsg-inpaint/a.png"))
        self.assertFalse((self.input / "rsg-inpaint" / "a.png").exists())
        self.assertTrue((self.input / "keep.png").exists())
        self.assertEqual(list(self.temp.iterdir()), [])

    def test_failed_delivery_keeps_everything(self):
        Client.fail = True
        result = self.run_node(self.graph("rsg-inpaint/a.png"))
        self.assertEqual(node_result(result)[2], 1)
        self.assertTrue((self.input / "rsg-inpaint" / "a.png").exists())
        self.assertTrue((self.temp / "preview.png").exists())

    def test_busy_queue_keeps_temp_but_deletes_unshared_inputs(self):
        self.run_node(self.graph("rsg-inpaint/a.png"), others=[self.graph("keep.png")])
        self.assertFalse((self.input / "rsg-inpaint" / "a.png").exists())
        self.assertTrue((self.temp / "preview.png").exists())

    def test_input_still_used_by_a_queued_job_is_kept(self):
        self.run_node(self.graph("rsg-inpaint/a.png"), others=[self.graph("rsg-inpaint/a.png [input]")])
        self.assertTrue((self.input / "rsg-inpaint" / "a.png").exists())

    def test_unreadable_queue_deletes_nothing(self):
        self.run_node(self.graph("rsg-inpaint/a.png"), others=None)
        self.assertTrue((self.input / "rsg-inpaint" / "a.png").exists())
        self.assertTrue((self.temp / "preview.png").exists())

    def test_graph_with_preview_node_keeps_temp(self):
        graph = self.graph("rsg-inpaint/a.png")
        graph["p"] = {"class_type": "PreviewImage", "inputs": {}}
        self.run_node(graph)
        self.assertFalse((self.input / "rsg-inpaint" / "a.png").exists())
        self.assertTrue((self.temp / "preview.png").exists())

    def test_toggle_off_keeps_everything(self):
        self.run_node(self.graph("rsg-inpaint/a.png"), cleanup_after_send=False)
        self.assertTrue((self.input / "rsg-inpaint" / "a.png").exists())
        self.assertTrue((self.temp / "preview.png").exists())

    def test_cleanup_errors_never_change_the_result(self):
        with patch.object(mod, "input_files", side_effect=RuntimeError("x")):
            result = self.run_node(self.graph("rsg-inpaint/a.png"))
        self.assertEqual(node_result(result)[1:], (1, 0))

    def test_other_prompts_matches_current_job_by_prompt_id_not_identity(self):
        other, queued = {"o": 1}, {"q": 1}
        # the queue holds copies of the current job, as ComfyUI's PromptQueue does
        for running, pending, expected in (
            ([(0, "me", {"c": 1}, {}, [])], [(1, "b", queued, {}, [])], [queued]),
            ([(0, "me", {"c": 1}, {}, []), (1, "x", other, {}, [])], [], [other]),
            ([(0, "me", {"c": 1}, {}, [])], [], []),
        ):
            queue = types.SimpleNamespace(get_current_queue=lambda r=running, p=pending: (r, p))
            server = types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(prompt_queue=queue)))
            execution = types.SimpleNamespace(get_executing_context=lambda: types.SimpleNamespace(prompt_id="me"))
            with patch.dict(sys.modules, {"server": server, "comfy_execution": types.ModuleType("comfy_execution"), "comfy_execution.utils": execution}):
                self.assertEqual(mod.other_prompts(), expected)

    def test_unreadable_queue_returns_none(self):
        self.assertIsNone(mod.other_prompts())


if __name__ == "__main__":
    unittest.main()
