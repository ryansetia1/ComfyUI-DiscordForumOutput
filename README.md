# ComfyUI Discord Forum Output

`Discord Forum Output` is a server-side output node that sends a ComfyUI `IMAGE`
batch to an existing Discord forum thread or creates forum posts through an
incoming webhook. Images are encoded in memory and are never written to ComfyUI's
output directory.

## Install and configure

Place this folder in `ComfyUI/custom_nodes/`, then restart ComfyUI. Copy
`discord_forum_output.example.toml` to `discord_forum_output.local.toml`; the
local file is ignored by Git. Set either the environment variable named by
`[webhook].url_env` (default `COMFY_DISCORD_FORUM_WEBHOOK_URL`) or the local
fallback `webhook.url`. Configure destination thread IDs and forum tag IDs
under `[destinations]` and `[forum_tags]` as decimal strings.

On macOS Desktop, an app launched from Finder may not inherit your shell
environment. The ignored local config is the practical alternative.

## Use

Connect an `IMAGE` batch, choose `existing_thread`, and input your configured
destination alias (e.g. `general`, `testing`, `showcase`). The node looks up the
destination alias in your configuration and delivers directly to that thread via
incoming webhook.

For creating new posts in the forum, choose `delivery_mode = new_forum_post`,
specify `new_thread_title`, and optionally add comma-separated `forum_tag_aliases`.

The default sends one message per image. `single_message` sends one multipart
request only when every image passes the 8 MiB guard and the attachment limit.
PNG and JPEG are encoded with Pillow in memory; JPEG drops alpha.

### Automatic generation metadata

Only the `IMAGE` connection is needed for generation metadata. At execution, the
node receives ComfyUI's runtime `PROMPT` graph and follows the upstream lineage
of that specific image. It uses the sampler that produced that image rather than
searching the whole workflow, so disconnected samplers cannot affect a caption.
The seed is read from the execution prompt after ComfyUI has resolved a
randomized seed; resolution comes from the in-memory generated image itself.

Common supported shapes are `KSampler` and `KSampler Advanced` through a VAE
decode, plus the usual Flux custom-sampler route: `SamplerCustomAdvanced` with
`RandomNoise`, `BasicGuider`, `FluxGuidance`, `KSamplerSelect`, and
`BasicScheduler`. When that lineage is available, the caption includes the
positive `CLIPTextEncode` text, model/checkpoint (checkpoint or UNet loader),
seed, steps, CFG where the workflow has CFG, sampler, scheduler, and actual
image resolution.

Extraction is deliberately conservative. Custom nodes, ambiguous branches,
unsupported loaders, and parameters absent from the image's upstream lineage are
omitted rather than guessed. Flux guidance is not labelled as CFG. No image-file
metadata is read and no output files are written.

Caption metadata is automatic; prompt, seed, model/checkpoint, steps, CFG,
sampler, and scheduler are not node widgets. Existing workflows that retain
those serialized inputs still load and run, but this ComfyUI runtime ignores
inputs absent from a node's current schema. Direct/API calls may still pass any
of those legacy arguments as explicit overrides: non-empty strings and non-zero
numeric values win over extracted metadata. A legacy seed value of `0` is the
automatic default, while an actual runtime seed of `0` is still reported when
the connected sampler resolves to it.

The node always executes when queued, so intentionally requeueing posts again.
It sends the prompt and enabled generation metadata to Discord as message content;
Discord's privacy policies apply. Mentions are disabled. Webhook URLs
are secrets: never put them in a workflow, tracked config, or issue. The node does not
return or log credentials, and follows no redirects for uploads.

Failures are returned as a safe status output. A timeout after sending may be an
unknown delivery and is not retried automatically; check Discord before rerunning.

## Tests

Run the mocked, in-memory test suite with ComfyUI's virtualenv:

```sh
"/Users/ryan.setiawan/ComfyUI-Installs/ComfyUI (1)/ComfyUI/.venv/bin/python3" -m unittest discover -s tests -v
```

System Python does not include ComfyUI's Torch and Pillow dependencies, so use
that virtualenv for tests and import checks.

## Cleanup after send

With `cleanup_after_send` on (default), once every image was delivered the node waits for the job to leave ComfyUI's running set (plus a 2 s grace so the frontend finishes loading previews), then:

- deletes every image file under ComfyUI's `input/` that a node input of the graph names (subfolders allowed; `[output]`/`[temp]` references, absolute paths, `..` and symlinks pointing outside are ignored), together with the four files the mask editor saves per edit (`clipspace-mask|paint|painted|painted-masked-<id>.png`);
- also purges **every other image** in the `input/` root, `input/clipspace/` and `input/rsg-*/` that existed before this job was queued, which removes the original of a mask-edited image. Images uploaded after the job was queued (possibly for the next job) and all other subfolders (e.g. `pixaroma/`, `3d/`) are never touched;
- keeps any file that another running or queued job still references;
- clears `temp/` only when no other job is running or queued.

If the queue cannot be read nothing is deleted. Any delivery failure skips cleanup entirely. Each run logs one `Discord cleanup: ...` line (or the reason it was skipped); cleanup errors never change the Discord result. Turn `cleanup_after_send` off to keep everything.
