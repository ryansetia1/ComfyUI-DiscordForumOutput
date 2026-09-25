# ComfyUI Discord Forum Output

ComfyUI custom node that posts generated images directly from memory to Discord forum threads via incoming webhooks.

## Commands

Run the test suite using ComfyUI's virtual environment:
```sh
"/Users/ryan.setiawan/ComfyUI-Installs/ComfyUI (1)/ComfyUI/.venv/bin/python3" -m unittest discover -s tests -v
```

## Architectural Guidelines & Rules

- **Zero Disk Footprint**: Images are encoded in RAM using Pillow (`BytesIO`) and sent directly as multipart requests. Never write temporary or output image files to disk.
- **Strict Secret Safety**:
  - Never read, display, or commit `discord_forum_output.local.toml`.
  - Never log or expose webhook URLs, bot tokens, credentials, or secrets.
  - All errors and logs must use redaction (`settings.redact()`).
- **Standard Library First**: Use Python's standard library (`urllib.request`, `json`, `tomllib`, `dataclasses`) without adding unnecessary external dependencies.
- **Precedence & Destination Routing**:
  - Destination aliases are configured in TOML / environment variables.
  - `destination_alias` maps to a configured thread snowflake ID.
- **Clean Code & Ponytail**:
  - Avoid speculative abstraction and unnecessary complexity.
  - Keep node schema clean and aligned with ComfyUI's native widget system.
