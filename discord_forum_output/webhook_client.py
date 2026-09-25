"""Small standard-library client for Discord Execute Webhook uploads."""
from __future__ import annotations

import json
import random
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .config import Settings
from .image_encoder import EncodedImage


@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes


class DeliveryError(RuntimeError):
    def __init__(self, message: str, ambiguous: bool = False):
        super().__init__(message)
        self.ambiguous = ambiguous


def multipart(payload: dict, images: list[EncodedImage]) -> tuple[bytes, str]:
    boundary = f"----ComfyDiscord{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    def part(headers: list[str], data: bytes) -> None:
        chunks.extend([f"--{boundary}\r\n".encode(), "\r\n".join(headers).encode(), b"\r\n\r\n", data, b"\r\n"])
    part(["Content-Disposition: form-data; name=\"payload_json\"", "Content-Type: application/json"], json.dumps(payload, separators=(",", ":")).encode())
    for number, image in enumerate(images):
        part([
            f"Content-Disposition: form-data; name=\"files[{number}]\"; filename=\"{image.filename}\"",
            f"Content-Type: {image.content_type}",
        ], image.data)
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _url(base: str, thread_id: str | None) -> str:
    parsed = urlsplit(base)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["wait"] = "true"
    if thread_id:
        query["thread_id"] = thread_id
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), ""))


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def default_transport(url: str, body: bytes, content_type: str, timeout: int) -> HttpResponse:
    request = Request(url, data=body, method="POST", headers={"Content-Type": content_type, "Content-Length": str(len(body)), "User-Agent": "ComfyUI-DiscordForumOutput/1.0"})
    try:
        with build_opener(_NoRedirect).open(request, timeout=timeout) as response:
            return HttpResponse(response.status, dict(response.headers.items()), response.read())
    except HTTPError as error:
        return HttpResponse(error.code, dict(error.headers.items()) if error.headers else {}, error.read())


class WebhookClient:
    def __init__(self, settings: Settings, transport: Callable[[str, bytes, str, int], HttpResponse] = default_transport, sleeper: Callable[[float], None] = time.sleep, jitter: Callable[[], float] = random.random):
        self.settings, self.transport, self.sleeper, self.jitter = settings, transport, sleeper, jitter
        self._pace_seconds = 0.0

    def _error(self, response: HttpResponse) -> str:
        code = ""
        try:
            decoded = json.loads(response.body.decode("utf-8", "replace"))
            if isinstance(decoded, dict):
                code = f" Discord {decoded.get('code', '')} {decoded.get('message', '')}".strip()
        except (ValueError, UnicodeError):
            pass
        return self.settings.redact(f"Discord HTTP {response.status}{(': ' + code) if code else ''}")[:240]

    def _retry_after(self, response: HttpResponse) -> float:
        headers = {name.lower(): value for name, value in response.headers.items()}
        raw = headers.get("retry-after")
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            try:
                body = json.loads(response.body.decode("utf-8", "replace"))
                return max(0.0, float(body.get("retry_after", 0)))
            except (ValueError, TypeError, AttributeError):
                return 0.0

    def _record_pacing(self, response: HttpResponse) -> None:
        headers = {name.lower(): value for name, value in response.headers.items()}
        remaining = headers.get("x-ratelimit-remaining")
        reset = headers.get("x-ratelimit-reset-after")
        if remaining == "0":
            try:
                self._pace_seconds = max(self._pace_seconds, float(reset))
            except (TypeError, ValueError):
                pass

    @staticmethod
    def _known_pre_send_failure(error: OSError | URLError) -> bool:
        reason = error.reason if isinstance(error, URLError) else error
        return isinstance(reason, (socket.gaierror, ConnectionRefusedError))

    def _is_global_rate_limit(self, response: HttpResponse) -> bool:
        try:
            body = json.loads(response.body.decode("utf-8", "replace"))
            return isinstance(body, dict) and body.get("global") is True
        except (ValueError, UnicodeError):
            return False

    def send(self, images: list[EncodedImage], caption: str, *, thread_id: str | None = None, thread_name: str | None = None, applied_tags: list[str] | None = None) -> None:
        payload: dict = {"allowed_mentions": {"parse": []}}
        if caption:
            payload["content"] = caption
        if thread_name:
            payload["thread_name"] = thread_name
            if applied_tags:
                payload["applied_tags"] = applied_tags
        body, content_type = multipart(payload, images)
        endpoint = _url(self.settings.webhook_url, thread_id)
        for attempt in range(self.settings.max_transient_retries + 1):
            if self._pace_seconds:
                delay, self._pace_seconds = self._pace_seconds, 0.0
                self.sleeper(delay)
            try:
                response = self.transport(endpoint, body, content_type, self.settings.request_timeout_seconds)
            except (socket.timeout, TimeoutError):
                raise DeliveryError("unknown delivery; check Discord before rerun", ambiguous=True)
            except URLError as error:
                if not self._known_pre_send_failure(error):
                    raise DeliveryError("unknown delivery; check Discord before rerun", ambiguous=True) from error
                if attempt >= self.settings.max_transient_retries:
                    raise DeliveryError("connection failed before send") from error
                self.sleeper(min(4.0, 0.5 * (2 ** attempt)) + self.jitter() * 0.1)
                continue
            except OSError as error:
                if not self._known_pre_send_failure(error):
                    raise DeliveryError("unknown delivery; check Discord before rerun", ambiguous=True) from error
                if attempt >= self.settings.max_transient_retries:
                    raise DeliveryError("connection failed before send") from error
                self.sleeper(min(4.0, 0.5 * (2 ** attempt)) + self.jitter() * 0.1)
                continue
            self._record_pacing(response)
            if 200 <= response.status < 300:
                try:
                    parsed = json.loads(response.body.decode("utf-8"))
                    if not isinstance(parsed, dict) or not parsed.get("id"):
                        raise ValueError
                except (ValueError, UnicodeError):
                    raise DeliveryError("Discord returned a malformed success response")
                return
            if response.status == 429 and attempt < self.settings.max_transient_retries:
                delay = self._retry_after(response)
                if self._is_global_rate_limit(response):
                    self._pace_seconds = max(self._pace_seconds, delay)
                    continue
                self.sleeper(delay + self.jitter() * 0.1)
                continue
            if 500 <= response.status < 600 and attempt < self.settings.max_transient_retries:
                self.sleeper(min(4.0, 0.5 * (2 ** attempt)) + self.jitter() * 0.1)
                continue
            raise DeliveryError(self._error(response))
