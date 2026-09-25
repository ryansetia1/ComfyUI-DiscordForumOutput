from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import unittest
from unittest.mock import patch
from urllib.error import URLError
import socket

import torch
from PIL import Image

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("discord_forum_node", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
assert spec.loader
import sys
sys.modules[spec.name] = package
spec.loader.exec_module(package)

from discord_forum_node.discord_forum_output.caption import render_caption
from discord_forum_node.discord_forum_output.config import ConfigError, Settings, load_settings, redact, resolve_destination
from discord_forum_node.discord_forum_output.image_encoder import EncodedImage, ImageEncodingError, encode_image
from discord_forum_node.discord_forum_output.prompt_metadata import extract_metadata
from discord_forum_node.discord_forum_output.webhook_client import DeliveryError, HttpResponse, WebhookClient
from discord_forum_node.nodes.discord_forum_output import DiscordForumOutput


def settings():
    return Settings("https://discord.com/api/webhooks/123456789012345678/SECRET", {"x": "123456789012345678"}, {})


class ComponentsTest(unittest.TestCase):
    def test_web_directory_omitted_for_headless_node(self):
        self.assertFalse(hasattr(package, "WEB_DIRECTORY"))

    def test_registration_and_schema_has_only_delivery_widgets_and_runtime_hidden_inputs(self):
        node = package.NODE_CLASS_MAPPINGS["DiscordForumOutput"]
        schema = node.INPUT_TYPES()["required"]
        self.assertEqual(package.NODE_DISPLAY_NAME_MAPPINGS, {"DiscordForumOutput": "Discord Forum Output"})
        self.assertTrue(node.OUTPUT_NODE); self.assertEqual(node.CATEGORY, "output/discord")
        self.assertEqual(node.RETURN_TYPES, ("STRING", "INT", "INT"))
        self.assertNotIn("webhook", " ".join(schema).lower())
        self.assertEqual(schema["images"][0], "IMAGE")
        self.assertIn("destination_alias", schema)
        self.assertNotIn("existing_thread_id", schema)
        self.assertEqual(node.INPUT_TYPES()["hidden"], {"prompt_graph": "PROMPT", "unique_id": "UNIQUE_ID"})
        self.assertTrue({"prompt", "seed", "model_or_checkpoint", "steps", "cfg", "sampler", "scheduler"}.isdisjoint(schema))

    def test_runtime_metadata_uses_connected_standard_ksampler_and_resolved_seed(self):
        graph = {
            "discord": {"class_type": "DiscordForumOutput", "inputs": {"images": ["decode", 0]}},
            "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0]}},
            "sampler": {"class_type": "KSampler", "inputs": {"model": ["checkpoint", 0], "seed": 847362951, "steps": 28, "cfg": 6.5, "sampler_name": "euler", "scheduler": "normal", "positive": ["positive", 0]}},
            "positive": {"class_type": "CLIPTextEncode", "inputs": {"text": "a red fox"}},
            "checkpoint": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "fox.safetensors"}},
            "unrelated": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 99, "cfg": 99, "sampler_name": "bad", "scheduler": "bad"}},
        }
        metadata = extract_metadata(graph, "discord")
        self.assertEqual(metadata.seed, 847362951)
        self.assertEqual(metadata.prompt, "a red fox")
        self.assertEqual(metadata.model_or_checkpoint, "fox.safetensors")
        self.assertEqual((metadata.steps, metadata.cfg, metadata.sampler, metadata.scheduler), (28, 6.5, "euler", "normal"))
        self.assertEqual(DiscordForumOutput._metadata(graph, "discord", "", 0, "", 0, 0.0, "", ""), metadata)

    def test_runtime_metadata_supports_flux_custom_sampler_lineage(self):
        graph = {
            "discord": {"class_type": "DiscordForumOutput", "inputs": {"images": ["decode", 0]}},
            "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0]}},
            "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["selector", 0], "sigmas": ["schedule", 0]}},
            "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": 294857102}},
            "guider": {"class_type": "BasicGuider", "inputs": {"model": ["unet", 0], "conditioning": ["guidance", 0]}},
            "guidance": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["text", 0], "guidance": 3.5}},
            "text": {"class_type": "CLIPTextEncode", "inputs": {"text": "glowing glass forest"}},
            "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-dev.safetensors"}},
            "selector": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "schedule": {"class_type": "BasicScheduler", "inputs": {"model": ["unet", 0], "scheduler": "simple", "steps": 24}},
        }
        metadata = extract_metadata(graph, "discord")
        self.assertEqual(metadata.seed, 294857102)
        self.assertEqual(metadata.prompt, "glowing glass forest")
        self.assertEqual(metadata.model_or_checkpoint, "flux1-dev.safetensors")
        self.assertEqual((metadata.steps, metadata.sampler, metadata.scheduler), (24, "euler", "simple"))
        self.assertIsNone(metadata.cfg)

    def test_runtime_metadata_supports_ksampler_advanced_noise_seed(self):
        graph = {
            "discord": {"class_type": "DiscordForumOutput", "inputs": {"images": ("decode", 0)}},
            "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0]}},
            "sampler": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["checkpoint", 0], "noise_seed": 444, "steps": 16, "cfg": 4.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "positive": ["positive", 0]}},
            "positive": {"class_type": "CLIPTextEncode", "inputs": {"text": "advanced prompt"}},
            "checkpoint": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "advanced.safetensors"}},
        }
        metadata = extract_metadata(graph, "discord")
        self.assertEqual((metadata.seed, metadata.steps, metadata.cfg, metadata.sampler, metadata.scheduler), (444, 16, 4.0, "dpmpp_2m", "karras"))

    def test_runtime_metadata_omits_unknown_values_and_legacy_overrides_win(self):
        graph = {"discord": {"class_type": "DiscordForumOutput", "inputs": {"images": ["image", 0]}}, "image": {"class_type": "LoadImage", "inputs": {}}}
        self.assertEqual(extract_metadata(graph, "discord").seed, None)
        metadata = DiscordForumOutput._metadata(graph, "discord", "manual prompt", 7, "manual.ckpt", 12, 2.5, "heun", "karras")
        self.assertEqual((metadata.prompt, metadata.seed, metadata.model_or_checkpoint, metadata.steps, metadata.cfg, metadata.sampler, metadata.scheduler), ("manual prompt", 7, "manual.ckpt", 12, 2.5, "heun", "karras"))

    def test_node_legacy_metadata_arguments_override_runtime_metadata(self):
        node, captions = DiscordForumOutput(), []
        graph = {"discord": {"class_type": "DiscordForumOutput", "inputs": {"images": ["image", 0]}}, "image": {"class_type": "LoadImage", "inputs": {}}}
        configured = Settings("https://discord.com/api/webhooks/123456789012345678/SECRET", {"x": "123456789012345678"}, {})
        class Client:
            def __init__(self, _settings): pass
            def send(self, _images, caption, **_kwargs): captions.append(caption)
        with patch("discord_forum_node.nodes.discord_forum_output.load_settings", return_value=configured), patch("discord_forum_node.nodes.discord_forum_output.WebhookClient", Client):
            result = node.send(torch.zeros((1, 2, 2, 3)), "x", caption_enabled=True, prompt="manual prompt", seed=7, model_or_checkpoint="manual.ckpt", steps=12, cfg=2.5, sampler="heun", scheduler="karras", prompt_graph=graph, unique_id="discord")
        self.assertEqual(result[1:], (1, 0))
        self.assertEqual(captions, ["manual prompt\nseed: 7 | model: manual.ckpt | resolution: 2x2 | steps: 12 | CFG: 2.5 | sampler: heun | scheduler: karras"])

    def test_node_new_schema_call_uses_runtime_metadata_without_manual_arguments(self):
        node, captions = DiscordForumOutput(), []
        graph = {
            "discord": {"class_type": "DiscordForumOutput", "inputs": {"images": ["decode", 0]}},
            "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0]}},
            "sampler": {"class_type": "KSampler", "inputs": {"model": ["checkpoint", 0], "seed": 42, "steps": 20, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "positive": ["positive", 0]}},
            "positive": {"class_type": "CLIPTextEncode", "inputs": {"text": "runtime prompt"}},
            "checkpoint": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "runtime.safetensors"}},
        }
        configured = Settings("https://discord.com/api/webhooks/123456789012345678/SECRET", {"x": "123456789012345678"}, {})
        class Client:
            def __init__(self, _settings): pass
            def send(self, _images, caption, **_kwargs): captions.append(caption)
        with patch("discord_forum_node.nodes.discord_forum_output.load_settings", return_value=configured), patch("discord_forum_node.nodes.discord_forum_output.WebhookClient", Client):
            result = node.send(torch.zeros((1, 2, 2, 3)), "x", caption_enabled=True, prompt_graph=graph, unique_id="discord")
        self.assertEqual(result[1:], (1, 0))
        self.assertEqual(captions, ["runtime prompt\nseed: 42 | model: runtime.safetensors | resolution: 2x2 | steps: 20 | CFG: 7.0 | sampler: euler | scheduler: normal"])

    def test_config_env_precedence_validation_and_redaction(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "local.toml"
            path.write_text('[webhook]\nurl_env = "HOOK"\nurl = "https://discord.com/api/webhooks/123456789012345678/LOCAL"\n[destinations]\nx = { thread_id = "123456789012345678" }\n')
            configured = load_settings(path, {"HOOK": "https://discord.com/api/webhooks/123456789012345678/ENV"})
            self.assertTrue(configured.webhook_url.endswith("/ENV"))
            self.assertNotIn("ENV", configured.redact("failed https://discord.com/api/webhooks/123456789012345678/ENV"))
            with self.assertRaises(ConfigError): load_settings(Path(directory) / "missing.toml", {})
            path.write_text('[webhook]\nurl = "https://discord.com/api/webhooks/123456789012345678/LOCAL"\n[destinations]\nx = { thread_id = 123 }\n')
            with self.assertRaises(ConfigError): load_settings(path, {})

    def test_destination_alias_resolution(self):
        configured = Settings("", {"x": "123456789012345678"}, {})
        self.assertEqual(resolve_destination(configured, "x"), "123456789012345678")
        with self.assertRaises(ConfigError):
            resolve_destination(configured, "unknown")

    def test_node_resolves_destination_alias_and_ignores_legacy_existing_thread_id(self):
        node, sent = DiscordForumOutput(), []
        configured = Settings("https://discord.com/api/webhooks/123456789012345678/SECRET", {"x": "123456789012345678"}, {})
        class Client:
            def __init__(self, _settings): pass
            def send(self, _images, _caption, **kwargs): sent.append(kwargs["thread_id"])
        with patch("discord_forum_node.nodes.discord_forum_output.load_settings", return_value=configured), patch("discord_forum_node.nodes.discord_forum_output.WebhookClient", Client):
            result = node.send(torch.zeros((1, 2, 2, 3)), "x", caption_enabled=False, existing_thread_id="testing")
        self.assertEqual(result[1:], (1, 0))
        self.assertEqual(sent, ["123456789012345678"])

    def test_redaction_covers_url_path_and_standalone_token(self):
        secret = "https://discord.com/api/webhooks/123456789012345678/SECRET"
        text = f"{secret} /api/webhooks/123456789012345678/OTHER token=SECRET"
        redacted = redact(text, secret)
        self.assertNotIn("SECRET", redacted)
        self.assertNotIn("OTHER", redacted)
        self.assertNotIn("123456789012345678/", redacted)

    def test_encoder_is_in_memory_and_decodable(self):
        tensor = torch.tensor([[[1.0, 0.0, 0.0, 0.5]]])
        png, jpeg = encode_image(tensor, 1, "png", 95), encode_image(tensor, 2, "jpeg", 95)
        self.assertEqual(png.filename, "comfy-001.png"); self.assertTrue(jpeg.alpha_dropped)
        self.assertEqual(Image.open(io.BytesIO(png.data)).mode, "RGBA")
        self.assertEqual(Image.open(io.BytesIO(jpeg.data)).mode, "RGB")
        with self.assertRaises(ImageEncodingError): encode_image(torch.zeros((1, 1, 2)), 1, "png", 95)
        with self.assertRaises(ImageEncodingError): encode_image(torch.tensor([[[float("nan"), 0, 0]]]), 1, "png", 95)

    def test_caption_order_limit_and_no_mention_payload(self):
        caption = render_caption(prompt="  hello   @everyone ", seed=0, model_or_checkpoint="/models/a.safetensors", width=10, height=20, steps=5, cfg=7.0, sampler="Euler", scheduler="normal", maximum=1800)
        self.assertEqual(caption, "hello @everyone\nseed: 0 | model: a.safetensors | resolution: 10x20 | steps: 5 | CFG: 7.0 | sampler: Euler | scheduler: normal")
        self.assertLessEqual(len(render_caption(prompt="x" * 500, seed=0, model_or_checkpoint="", width=1, height=1, steps=0, cfg=0, sampler="", scheduler="", maximum=50)), 50)

    def test_multipart_existing_thread_and_rate_retry(self):
        image = encode_image(torch.zeros((2, 2, 3)), 1, "png", 95)
        responses = [HttpResponse(429, {"Retry-After": "1.5"}, b'{"retry_after": 1.5}'), HttpResponse(200, {}, b'{"id": "99"}')]
        calls, sleeps = [], []
        def transport(url, body, content_type, timeout): calls.append((url, body, content_type, timeout)); return responses.pop(0)
        WebhookClient(settings(), transport, sleeps.append, lambda: 0).send([image], "caption", thread_id="123456789012345679")
        self.assertEqual(len(calls), 2); self.assertEqual(parse_qs(urlsplit(calls[0][0]).query), {"wait": ["true"], "thread_id": ["123456789012345679"]})
        self.assertIn(b'name="payload_json"', calls[0][1]); self.assertIn(b'name="files[0]"', calls[0][1])
        self.assertIn(b'"allowed_mentions":{"parse":[]}', calls[0][1]); self.assertEqual(sleeps, [1.5])

    def test_rate_limit_headers_are_case_insensitive_and_global_limit_paces_retry(self):
        image = encode_image(torch.zeros((2, 2, 3)), 1, "png", 95)
        calls, sleeps = [], []
        responses = [HttpResponse(200, {"x-rAtElImIt-rEmAiNiNg": "0", "X-rAtElImIt-rEsEt-AfTeR": "1.25"}, b'{"id":"1"}'), HttpResponse(200, {}, b'{"id":"2"}')]
        client = WebhookClient(settings(), lambda *args: (calls.append(1), responses.pop(0))[1], sleeps.append, lambda: 0)
        client.send([image], ""); client.send([image], "")
        self.assertEqual(len(calls), 2); self.assertEqual(sleeps, [1.25])
        responses = [HttpResponse(429, {"rEtRy-AfTeR": "2"}, b'{"global":true}'), HttpResponse(200, {}, b'{"id":"3"}')]
        sleeps = []
        WebhookClient(settings(), lambda *args: responses.pop(0), sleeps.append, lambda: 0).send([image], "")
        self.assertEqual(sleeps, [2.0])

    def test_new_forum_post_uses_title_and_tags_without_thread_id(self):
        image = encode_image(torch.zeros((2, 2, 3)), 1, "png", 95)
        calls = []
        def transport(url, body, content_type, timeout): calls.append((url, body)); return HttpResponse(200, {}, b'{"id":"1"}')
        WebhookClient(settings(), transport, lambda _: None, lambda: 0).send([image], "", thread_name="My post", applied_tags=["123456789012345679"])
        self.assertEqual(parse_qs(urlsplit(calls[0][0]).query), {"wait": ["true"]})
        self.assertIn(b'"thread_name":"My post"', calls[0][1]); self.assertIn(b'"applied_tags":["123456789012345679"]', calls[0][1])

    def test_no_retry_on_ambiguous_timeout_or_400_and_no_secret_in_error(self):
        image = encode_image(torch.zeros((2, 2, 3)), 1, "png", 95)
        client = WebhookClient(settings(), lambda *args: (_ for _ in ()).throw(TimeoutError()), lambda _: None, lambda: 0)
        with self.assertRaisesRegex(DeliveryError, "unknown delivery"): client.send([image], "")
        client = WebhookClient(settings(), lambda *args: HttpResponse(400, {}, b'{"message":"bad"}'), lambda _: None, lambda: 0)
        with self.assertRaises(DeliveryError) as error: client.send([image], "")
        self.assertNotIn("SECRET", str(error.exception))

    def test_retries_5xx_but_not_ambiguous_connection_failure(self):
        image = encode_image(torch.zeros((2, 2, 3)), 1, "png", 95)
        responses, calls = [HttpResponse(502, {}, b""), HttpResponse(200, {}, b'{"id":"1"}')], []
        WebhookClient(settings(), lambda *args: (calls.append(1), responses.pop(0))[1], lambda _: None, lambda: 0).send([image], "")
        self.assertEqual(len(calls), 2)
        client = WebhookClient(settings(), lambda *args: (_ for _ in ()).throw(URLError(OSError("connection reset"))), lambda _: None, lambda: 0)
        with self.assertRaises(DeliveryError) as error:
            client.send([image], "")
        self.assertTrue(error.exception.ambiguous)

    def test_retries_known_pre_send_dns_failure(self):
        image = encode_image(torch.zeros((2, 2, 3)), 1, "png", 95)
        responses, calls = [URLError(socket.gaierror()), HttpResponse(200, {}, b'{"id":"1"}')], []
        def transport(*args):
            calls.append(1)
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        WebhookClient(settings(), transport, lambda _: None, lambda: 0).send([image], "")
        self.assertEqual(len(calls), 2)

    def test_node_preflight_returns_safe_result_without_network(self):
        node = DiscordForumOutput()
        with patch("discord_forum_node.nodes.discord_forum_output.load_settings", side_effect=ConfigError("webhook URL is not configured")):
            result = node.send(torch.zeros((2, 2, 2, 3)), "", "existing_thread", "one_message_per_image", "png", 95, False, "", 0, "", 0, 0.0, "", "", "", "")
        self.assertEqual(result[1:], (0, 2))
        self.assertIn("webhook URL is not configured", result[0])
        with patch("discord_forum_node.nodes.discord_forum_output.load_settings", side_effect=ConfigError("webhook URL is not configured")):
            result = node.send(torch.zeros((2, 2, 2, 3)), "", "existing_thread", "single_message", "png", 95, False, "", 0, "", 0, 0.0, "", "", "", "")
        self.assertEqual(result[1:], (0, 1))

    def test_node_titles_include_original_batch_indexes_and_never_truncate(self):
        node, sent = DiscordForumOutput(), []
        class Client:
            def __init__(self, _settings): pass
            def send(self, _images, _caption, **kwargs): sent.append(kwargs["thread_name"])
        configured = Settings("https://discord.com/api/webhooks/123456789012345678/SECRET", {}, {})
        args = (torch.zeros((2, 2, 2, 3)), "", "new_forum_post", "one_message_per_image", "png", 95, False, "", 0, "", 0, 0.0, "", "")
        with patch("discord_forum_node.nodes.discord_forum_output.load_settings", return_value=configured), patch("discord_forum_node.nodes.discord_forum_output.WebhookClient", Client):
            result = node.send(*args, "x" * 94, "")
        self.assertEqual(result[1:], (2, 0)); self.assertEqual(sent, ["x" * 94 + " (1/2)", "x" * 94 + " (2/2)"])
        with patch("discord_forum_node.nodes.discord_forum_output.load_settings", return_value=configured):
            result = node.send(*args, "x" * 95, "")
        self.assertEqual(result[1:], (0, 2)); self.assertIn("image 1", result[0]); self.assertIn("image 2", result[0])

    def test_node_single_message_counts_messages_and_releases_buffers(self):
        node, sent, made = DiscordForumOutput(), [], []
        configured = Settings("https://discord.com/api/webhooks/123456789012345678/SECRET", {"x": "123456789012345678"}, {}, max_attachments_per_request=1)
        def encoder(_image, index, *_args):
            item = EncodedImage(f"{index}.png", "image/png", b"data", 2, 2)
            made.append(item)
            return item
        class Client:
            def __init__(self, _settings): pass
            def send(self, images, _caption, **_kwargs): sent.append(images)
        args = (torch.zeros((2, 2, 2, 3)), "x", "existing_thread", "single_message", "png", 95, False, "", 0, "", 0, 0.0, "", "", "", "")
        with patch("discord_forum_node.nodes.discord_forum_output.load_settings", return_value=configured), patch("discord_forum_node.nodes.discord_forum_output.encode_image", side_effect=encoder), patch("discord_forum_node.nodes.discord_forum_output.WebhookClient", Client):
            result = node.send(*args)
        self.assertEqual(result[1:], (0, 1)); self.assertFalse(sent); self.assertTrue(all(not item.data for item in made))
        made.clear()
        oversized = Settings("https://discord.com/api/webhooks/123456789012345678/SECRET", {"x": "123456789012345678"}, {}, max_upload_bytes=3)
        with patch("discord_forum_node.nodes.discord_forum_output.load_settings", return_value=oversized), patch("discord_forum_node.nodes.discord_forum_output.encode_image", side_effect=encoder), patch("discord_forum_node.nodes.discord_forum_output.WebhookClient", Client):
            result = node.send(*args)
        self.assertEqual(result[1:], (0, 1)); self.assertFalse(sent); self.assertTrue(all(not item.data for item in made))
        configured = Settings("https://discord.com/api/webhooks/123456789012345678/SECRET", {"x": "123456789012345678"}, {})
        made.clear()
        with patch("discord_forum_node.nodes.discord_forum_output.load_settings", return_value=configured), patch("discord_forum_node.nodes.discord_forum_output.encode_image", side_effect=encoder), patch("discord_forum_node.nodes.discord_forum_output.WebhookClient", Client):
            result = node.send(*args)
        self.assertEqual(result[1:], (1, 0)); self.assertEqual(len(sent), 1); self.assertTrue(all(not item.data for item in made))
        class FailingClient:
            def __init__(self, _settings): pass
            def send(self, *_args, **_kwargs): raise DeliveryError("failed")
        with patch("discord_forum_node.nodes.discord_forum_output.load_settings", return_value=configured), patch("discord_forum_node.nodes.discord_forum_output.encode_image", side_effect=encoder), patch("discord_forum_node.nodes.discord_forum_output.WebhookClient", FailingClient):
            result = node.send(*args)
        self.assertEqual(result[1:], (0, 1)); self.assertIn("single_message: failed", result[0])


if __name__ == "__main__":
    unittest.main()
