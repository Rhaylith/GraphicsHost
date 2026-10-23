# GraphicsHost HTTP API

The server listens on `http://127.0.0.1:1234` by default (settings `server.port`, `server.lan`; or `graphicshost serve
--port N`). The same port serves the browser UI at `/`.

Every error, for every route and status, has one shape (the OpenAI one):

```json
{"error": {"message": "…", "type": "invalid_request_error", "code": "invalid_parameters"}}
```

`code` is a stable machine-readable name (`model_not_found`, `out_of_memory`, `job_not_found`, …). A route that does not
exist answers 404 with `"code": null`.

## Access

* Programs on this computer need no token while `server.requireToken` is off. Everything else needs one:
  `Authorization: Bearer gh_…` (or `x-api-key: gh_…`). Create tokens in Settings, with `graphicshost token create
  <name> --perm read,generate`, or `POST /api/v1/tokens`. A secret is shown once; only its hash is stored.
* Listening for other computers (`server.lan`) is refused until at least one token exists.
* Permissions: `read` (look), `generate` (queue and cancel generations), `manage` (load models, download, presets,
  defaults, delete images), `admin` (settings, tokens, request log; includes the others).
* `GET /api/v1/health` is public. `GET /api/v1/whoami` tells who you are and what you may do.

## OpenAI compatible

| Route | Permission | Notes |
|---|---|---|
| `GET /v1/models` | read | the checkpoints that can run, `loaded` marks the one in memory |
| `POST /v1/images/generations` | generate | waits for the image(s). Hanging up cancels the generation |
| `POST /v1/images/edits` | generate | image to image, see below |

`POST /v1/images/generations` body (all optional except `prompt`):

```json
{"model": "v1-5-pruned-emaonly", "prompt": "a lighthouse", "n": 2, "size": "512x512",
 "response_format": "b64_json", "negative_prompt": "blurry", "steps": 20, "cfg_scale": 7.5, "seed": 42,
 "sampler": "euler_a", "scheduler": "karras", "clip_skip": 2, "preset": "fast", "store": true,
 "load": {"device": "vulkan0", "flashAttention": true}}
```

`model` may be a file name, the name without extension or a unique prefix; left out, the loaded model is used. The native
spellings (`width`, `height`, `batchCount`, `cfgScale`, `negativePrompt`, …) work too. `response_format` is `b64_json`
(default) or `url` (a link to `/outputs/…`; needs `store: true`). The answer:

```json
{"created": 1790000000, "model": "v1-5-pruned-emaonly.safetensors", "mime_type": "image/png",
 "data": [{"b64_json": "…", "seed": 42, "gallery_id": 17}], "notes": ["…"]}
```

Parameters are layered: the model's defaults, then the preset, then the request. An image that cannot fit in memory is
refused with 507 `out_of_memory`; a different model is loaded on demand when no generation is running.

## Image to image

Add a source picture to a generation and it becomes image to image: the result starts from the picture instead of noise.

* `POST /v1/images/edits`: OpenAI's route. `multipart/form-data` with `image` (a PNG, JPEG or WebP file), `prompt`, and
  the same optional fields as above (`model`, `n`, `size`, `strength`, `steps`, …), or a JSON body with `image` as base64.
* `POST /v1/images/generations` and `POST /api/v1/generate` accept `init_image` (base64, a `data:` URL is fine) or
  `init_image_gallery_id` (an image that is already in the gallery) next to the usual fields.
* `strength` (0.05 to 1, default 0.75) says how much of the picture is replaced: low keeps it, 1 ignores it.
* Without `size`/`width`/`height` the result has the picture's size (rounded to a multiple of 8, at most twice the
  model's default size); with a size, the picture is scaled to it.
* Inpainting: add a `mask` (base64, or the `mask` file of `/v1/images/edits`). **White repaints, black keeps** (not
  OpenAI's transparency convention: alpha is ignored). The mask is scaled to the result's size. A mask without a source
  image is refused. The Qt UI has a mask painter ("Edit mask…"); the browser UI does not yet.
* A runtime DLL built before this feature answers 400 (`invalid_parameters`): update the DLLs in `runtimes\`.

## Native API (`/api/v1`)

| Route | Permission | Notes |
|---|---|---|
| `GET status`, `GET hardware`, `GET parameters` | read | state, CPU/GPU/runtimes, the names and limits a request may use |
| `GET settings`, `PUT settings` | read, admin | `PUT` takes a partial document; `restartRequired` is set for port/LAN changes |
| `GET logs?limit=&level=` | read | newest last |
| `GET requests?limit=`, `GET /metrics` | admin, read | request log (no query strings kept); Prometheus text |
| `GET models`, `POST models/load`, `POST models/unload` | read, manage | the folder with cards; load takes `{model, load, preset}` |
| `POST models/get` | manage | `{target: <catalog key or publisher/repo>, file?, companions?, dryRun?}` |
| `GET catalog?text=&family=&kind=`, `GET hub/search?q=`, `GET hub/files?repo=` | read | discovery, with a fit hint for this computer |
| `GET downloads`, `POST downloads/{id}/pause\|resume\|cancel`, `GET updates` | read, manage | |
| `GET/PUT/DELETE defaults/{file}`, `GET defaults` | read, manage | per-model defaults |
| `GET/PUT/DELETE presets/{name}`, `GET presets` | read, manage | |
| `POST generate` | generate | same body as above; answers 202 with the job |
| `GET jobs`, `GET jobs/{id}`, `POST jobs/{id}/cancel` | read, generate | |
| `GET jobs/{id}/events` | read | Server-Sent Events: `progress`, then `done` (the job document) |
| `GET gallery?text=&limit=&offset=`, `GET/DELETE gallery/{id}` | read, manage | |
| `GET gallery/{id}/image`, `GET gallery/{id}/thumbnail?size=` | read | |
| `GET /outputs/<path>` | read | the image files only (never the index) |

A job document: `{"id", "state": "queued|running|succeeded|failed|cancelled", "step", "steps", "request", "images":
[{"seed", "width", "height", "galleryId", "url"}], "seconds", "warnings", "error"}`.
