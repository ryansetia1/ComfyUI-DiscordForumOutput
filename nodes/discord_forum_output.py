from __future__ import annotations

import logging
import re
import traceback

from ..discord_forum_output.caption import render_caption
from ..discord_forum_output.config import ConfigError, load_settings, resolve_destination, resolve_tags
from ..discord_forum_output.image_encoder import ImageEncodingError, encode_image
from ..discord_forum_output.prompt_metadata import RuntimeMetadata, extract_metadata
from ..discord_forum_output.webhook_client import DeliveryError, WebhookClient

LOGGER = logging.getLogger("ComfyUI-DiscordForumOutput")
TITLE_MAX = 100


class DiscordForumOutput:
    CATEGORY = "output/discord"
    FUNCTION = "send"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("status", "uploaded_count", "failed_count")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",), "destination_alias": ("STRING", {"default": ""}),
            "delivery_mode": (["existing_thread", "new_forum_post"], {"default": "existing_thread"}),
            "batch_mode": (["one_message_per_image", "single_message"], {"default": "one_message_per_image"}),
            "image_format": (["png", "jpeg"], {"default": "png"}), "jpeg_quality": ("INT", {"default": 95, "min": 1, "max": 100}),
            "caption_enabled": ("BOOLEAN", {"default": True}),
            "new_thread_title": ("STRING", {"default": ""}), "forum_tag_aliases": ("STRING", {"default": ""}),
        }, "hidden": {"prompt_graph": "PROMPT", "unique_id": "UNIQUE_ID"}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, delivery_mode="existing_thread", batch_mode="one_message_per_image", image_format="png", jpeg_quality=95, new_thread_title="", **kwargs):
        if delivery_mode not in ("existing_thread", "new_forum_post") or batch_mode not in ("one_message_per_image", "single_message") or image_format not in ("png", "jpeg"):
            return "invalid Discord Forum Output mode"
        if not isinstance(jpeg_quality, int) or not 1 <= jpeg_quality <= 100:
            return "jpeg_quality must be between 1 and 100"
        if delivery_mode == "new_forum_post" and not str(new_thread_title).strip():
            return "new_thread_title is required for new_forum_post"
        return True

    def send(self, images, destination_alias="", delivery_mode="existing_thread", batch_mode="one_message_per_image", image_format="png", jpeg_quality=95, caption_enabled=True, prompt="", seed=0, model_or_checkpoint="", steps=0, cfg=0.0, sampler="", scheduler="", new_thread_title="", forum_tag_aliases="", existing_thread_id="", prompt_graph=None, unique_id=None, **kwargs):
        uploaded = failed = 0
        notes: list[str] = []
        try:
            settings = load_settings()
            if delivery_mode == "existing_thread":
                thread_id, tags = resolve_destination(settings, destination_alias), []
                title = None
            elif delivery_mode == "new_forum_post":
                title = str(new_thread_title).strip()
                if not title or len(title) > TITLE_MAX:
                    raise ConfigError(f"new_thread_title must be 1-{TITLE_MAX} characters")
                thread_id, tags = None, resolve_tags(settings, forum_tag_aliases)
            else:
                raise ConfigError("invalid delivery_mode")
            if batch_mode not in ("one_message_per_image", "single_message"):
                raise ConfigError("invalid batch_mode")
            if getattr(images, "ndim", None) != 4 or images.shape[0] < 1:
                raise ImageEncodingError("images must be a non-empty [B,H,W,C] tensor")
        except (ConfigError, ImageEncodingError, ValueError) as error:
            count = 1 if batch_mode == "single_message" else int(images.shape[0]) if getattr(images, "ndim", None) == 4 else 1
            return self._result(0, count, str(error))
        batch_size = int(images.shape[0])
        metadata = self._metadata(prompt_graph, unique_id, prompt, seed, model_or_checkpoint, steps, cfg, sampler, scheduler)
        if title and batch_mode == "one_message_per_image" and batch_size > 1:
            invalid = [index for index in range(1, batch_size + 1) if len(f"{title} ({index}/{batch_size})") > TITLE_MAX]
            if invalid:
                details = "; ".join(f"image {index}: new_thread_title including batch suffix must be 1-{TITLE_MAX} characters" for index in invalid)
                return self._result(0, batch_size, details)
        encoded: list[tuple[int, object]] = []
        for index in range(images.shape[0]):
            try:
                item = encode_image(images[index], index + 1, image_format, jpeg_quality)
                if len(item.data) > settings.max_upload_bytes:
                    item.data = b""
                    raise ImageEncodingError(f"exceeds {settings.max_upload_bytes // (1024 * 1024)} MiB")
                encoded.append((index + 1, item))
            except (ImageEncodingError, ValueError) as error:
                failed += 1; notes.append(f"image {index + 1}: {error}")
            except Exception:
                LOGGER.error("Discord image encoding failed: %s", settings.redact(traceback.format_exc())[:1000])
                failed += 1; notes.append(f"image {index + 1}: unexpected encoding failure")
        if batch_mode == "single_message" and (failed or len(encoded) > settings.max_attachments_per_request):
            self._release(encoded)
            detail = "; ".join(notes) or f"attachment count exceeds {settings.max_attachments_per_request}"
            return self._result(0, 1, f"single_message preflight failed: {detail}")
        units = [encoded] if batch_mode == "single_message" else [[entry] for entry in encoded]
        if batch_mode == "one_message_per_image":
            encoded = []
        client = WebhookClient(settings)
        for unit in units:
            original_index, image = unit[0]
            caption = render_caption(prompt=metadata.prompt or "", seed=metadata.seed, model_or_checkpoint=metadata.model_or_checkpoint or "", width=image.width, height=image.height, steps=metadata.steps or 0, cfg=metadata.cfg or 0, sampler=metadata.sampler or "", scheduler=metadata.scheduler or "", maximum=settings.caption_max_chars) if caption_enabled else ""
            unit_title = f"{title} ({original_index}/{batch_size})" if title and batch_mode == "one_message_per_image" and batch_size > 1 else title
            try:
                client.send([entry[1] for entry in unit], caption, thread_id=thread_id, thread_name=unit_title, applied_tags=tags)
                uploaded += 1 if batch_mode == "single_message" else len(unit)
            except DeliveryError as error:
                failed += 1 if batch_mode == "single_message" else len(unit)
                label = "single_message" if batch_mode == "single_message" else f"image {original_index}"
                notes.append(f"{label}: {error}")
            except Exception:
                LOGGER.error("Discord upload failed: %s", settings.redact(traceback.format_exc())[:1000])
                failed += 1 if batch_mode == "single_message" else len(unit)
                label = "single_message" if batch_mode == "single_message" else f"image {original_index}"
                notes.append(f"{label}: unexpected upload failure")
            finally:
                self._release(unit)
                unit.clear()
        if image_format == "png":
            notes.append("jpeg_quality ignored for PNG")
        return self._result(uploaded, failed, "; ".join(notes))

    @staticmethod
    def _release(entries: list[tuple[int, object]]) -> None:
        for _, image in entries:
            image.data = b""

    @staticmethod
    def _metadata(prompt_graph, unique_id, prompt, seed, model_or_checkpoint, steps, cfg, sampler, scheduler) -> RuntimeMetadata:
        runtime = extract_metadata(prompt_graph, unique_id)
        return RuntimeMetadata(
            prompt=str(prompt).strip() or runtime.prompt,
            seed=int(seed) if int(seed) != 0 else runtime.seed,
            model_or_checkpoint=str(model_or_checkpoint).strip() or runtime.model_or_checkpoint,
            steps=int(steps) if int(steps) > 0 else runtime.steps,
            cfg=float(cfg) if float(cfg) > 0 else runtime.cfg,
            sampler=str(sampler).strip() or runtime.sampler,
            scheduler=str(scheduler).strip() or runtime.scheduler,
        )

    @staticmethod
    def _result(uploaded: int, failed: int, detail: str):
        compact_detail = re.sub("\\s+", " ", detail)[:500]
        status = f"uploaded={uploaded} failed={failed}" + (f"; {compact_detail}" if detail else "")
        LOGGER.info("%s", status)
        return {
            "ui": {"status": [status]},
            "result": (status, uploaded, failed),
        }
