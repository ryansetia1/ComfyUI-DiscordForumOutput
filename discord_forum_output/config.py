"""Secret-safe local configuration loading and validation."""
from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

SNOWFLAKE = re.compile(r"^[0-9]{1,20}$")
ALIAS = re.compile(r"^[a-z0-9_-]+$")
LOCAL_CONFIG = Path(__file__).resolve().parent.parent / "discord_forum_output.local.toml"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    webhook_url: str
    destinations: dict[str, str]
    forum_tags: dict[str, str]
    forum_channels: dict[str, str] = field(default_factory=dict)
    bot_token: str | None = None
    max_upload_bytes: int = 8 * 1024 * 1024
    max_attachments_per_request: int = 10
    caption_max_chars: int = 1800
    request_timeout_seconds: int = 30
    max_transient_retries: int = 2

    def redact(self, value: object) -> str:
        text = str(value)
        return redact(text, self.webhook_url)


def redact(value: str, webhook_url: str = "") -> str:
    token = ""
    if webhook_url:
        value = value.replace(webhook_url, "[discord webhook redacted]")
        path = urlparse(webhook_url).path.rstrip("/").split("/")
        if len(path) >= 2:
            token = path[-1]
    value = re.sub(r"https?://[^\s/]+(?:/api(?:/v\d+)?)?/webhooks/[0-9]{1,20}/[^/?\s#]+(?:\?[^\s#]*)?", "[discord webhook redacted]", value, flags=re.I)
    value = re.sub(r"(?:/api(?:/v\d+)?)?/webhooks/[0-9]{1,20}/[^/?\s#]+", "[discord webhook redacted]", value, flags=re.I)
    if token:
        value = value.replace(token, "[discord webhook token redacted]")
    return value


def _alias_map(value: object, field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be a TOML table")
    result: dict[str, str] = {}
    for alias, item in value.items():
        if not isinstance(alias, str) or not ALIAS.fullmatch(alias):
            raise ConfigError(f"invalid {field} alias")
        ident = item.get("thread_id") if field == "destinations" and isinstance(item, dict) else item
        if not isinstance(ident, str) or not SNOWFLAKE.fullmatch(ident):
            raise ConfigError(f"{field}.{alias} must be a decimal snowflake string")
        result[alias] = ident
    return result


def _destination_maps(value: object) -> tuple[dict[str, str], dict[str, str]]:
    if not isinstance(value, dict):
        raise ConfigError("destinations must be a TOML table")
    destinations: dict[str, str] = {}
    forum_channels: dict[str, str] = {}
    for alias, item in value.items():
        if not isinstance(alias, str) or not ALIAS.fullmatch(alias):
            raise ConfigError("invalid destinations alias")
        if isinstance(item, str):
            thread_id, forum_channel_id = item, None
        elif isinstance(item, dict):
            thread_id, forum_channel_id = item.get("thread_id"), item.get("forum_channel_id")
        else:
            raise ConfigError(f"destinations.{alias} must be a decimal snowflake string or TOML table")
        if not isinstance(thread_id, str) or not SNOWFLAKE.fullmatch(thread_id):
            raise ConfigError(f"destinations.{alias}.thread_id must be a decimal snowflake string")
        if forum_channel_id is not None:
            if not isinstance(forum_channel_id, str) or not SNOWFLAKE.fullmatch(forum_channel_id):
                raise ConfigError(f"destinations.{alias}.forum_channel_id must be a decimal snowflake string")
            forum_channels[alias] = forum_channel_id
        destinations[alias] = thread_id
    return destinations, forum_channels


def _positive(table: dict, name: str, default: int, maximum: int | None = None) -> int:
    value = table.get(name, default)
    if type(value) is not int or value <= 0 or (maximum is not None and value > maximum):
        suffix = f" (maximum {maximum})" if maximum else ""
        raise ConfigError(f"limits.{name} must be a positive integer{suffix}")
    return value


def _validate_url(url: object) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ConfigError("Discord webhook URL is not configured")
    parsed = urlparse(url.strip())
    path = parsed.path.rstrip("/").split("/")
    allowed_hosts = {"discord.com", "canary.discord.com", "ptb.discord.com", "discordapp.com"}
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts or len(path) < 4 or path[-3] != "webhooks" or not SNOWFLAKE.fullmatch(path[-2]) or not path[-1]:
        raise ConfigError("webhook URL must be an HTTPS Discord incoming-webhook URL")
    return url.strip()


def load_settings(config_path: Path = LOCAL_CONFIG, environ: dict[str, str] | None = None, require_webhook: bool = True) -> Settings:
    environ = os.environ if environ is None else environ
    raw: dict = {}
    if config_path.exists():
        try:
            with config_path.open("rb") as handle:
                raw = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise ConfigError("could not read local Discord configuration") from error
    webhook = raw.get("webhook", {})
    if not isinstance(webhook, dict):
        raise ConfigError("webhook must be a TOML table")
    env_name = webhook.get("url_env", "COMFY_DISCORD_FORUM_WEBHOOK_URL")
    if not isinstance(env_name, str) or not env_name:
        raise ConfigError("webhook.url_env must be a non-empty environment variable name")
    url = environ.get(env_name) or webhook.get("url")
    if require_webhook:
        webhook_url = _validate_url(url)
    elif url is None or url == "":
        webhook_url = ""
    else:
        webhook_url = _validate_url(url)
    bot = raw.get("discord_bot", {})
    if not isinstance(bot, dict):
        raise ConfigError("discord_bot must be a TOML table")
    token_env = bot.get("token_env", "COMFY_DISCORD_FORUM_BOT_TOKEN")
    if not isinstance(token_env, str) or not token_env:
        raise ConfigError("discord_bot.token_env must be a non-empty environment variable name")
    token = environ.get(token_env) or bot.get("token")
    if token is not None and (not isinstance(token, str) or not token.strip()):
        raise ConfigError("Discord bot token must be a non-empty string")
    limits = raw.get("limits", {})
    if not isinstance(limits, dict):
        raise ConfigError("limits must be a TOML table")
    destinations, forum_channels = _destination_maps(raw.get("destinations", {}))
    return Settings(
        webhook_url=webhook_url,
        destinations=destinations,
        forum_tags=_alias_map(raw.get("forum_tags", {}), "forum_tags"),
        forum_channels=forum_channels,
        bot_token=token.strip() if isinstance(token, str) else None,
        max_upload_bytes=_positive(limits, "max_upload_bytes", 8 * 1024 * 1024, 20 * 1024 * 1024),
        max_attachments_per_request=_positive(limits, "max_attachments_per_request", 10, 10),
        caption_max_chars=_positive(limits, "caption_max_chars", 1800, 2000),
        request_timeout_seconds=_positive(limits, "request_timeout_seconds", 30, 300),
        max_transient_retries=_positive(limits, "max_transient_retries", 2, 10),
    )


def resolve_destination(settings: Settings, alias: str) -> str:
    if not ALIAS.fullmatch(alias or "") or alias not in settings.destinations:
        available = ", ".join(sorted(settings.destinations)) if settings.destinations else "none"
        raise ConfigError(f"destination_alias '{alias}' is not configured in [destinations] (available: {available})")
    return settings.destinations[alias]


def resolve_forum_channel(settings: Settings, alias: str) -> str:
    if not ALIAS.fullmatch(alias or "") or alias not in settings.forum_channels:
        raise ConfigError("destination_alias has no configured forum_channel_id for thread discovery")
    return settings.forum_channels[alias]


def resolve_thread_target(settings: Settings, alias: str, selected_thread_id: str) -> str:
    selected = str(selected_thread_id or "").strip()
    if selected:
        if not SNOWFLAKE.fullmatch(selected):
            raise ConfigError("existing_thread_id must be a decimal Discord thread ID selected from the dropdown")
        return selected
    return resolve_destination(settings, alias)


def resolve_tags(settings: Settings, aliases: str) -> list[str]:
    names = [item.strip() for item in aliases.split(",") if item.strip()]
    if len(names) != len(set(names)):
        raise ConfigError("forum_tag_aliases contains a duplicate alias")
    if len(names) > 5:
        raise ConfigError("at most five forum tags are allowed")
    missing = [name for name in names if name not in settings.forum_tags]
    if missing:
        raise ConfigError("forum_tag_aliases contains an unconfigured alias")
    return [settings.forum_tags[name] for name in names]
