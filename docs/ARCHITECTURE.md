# Architecture

## Data flow

```
phone video
  -> video.to_sam_mp4        H.264 MP4, fps and size chosen for cost
  -> sam.track  x2           "insect" and "flower" tracks (SAM 3.1, cached)
  -> flowers.group_touching  touching flowers merged into one head (umbels, spikes)
  -> motion.detect           still camera only: small insects SAM missed, found from their movement
  -> visits.find_visits      insect still on a flower for >= 0.3 s
  -> video.crop              best frames of each visiting insect
  -> classify.classify_crops FIT Count group, or "not an insect" (motion candidates need confidence >= 0.3)
  -> export.fit_count        distinct insects per group (FIT Count) and visits, JSON and CSV
```

## Layers

| Layer | Modules | May use | Never |
|---|---|---|---|
| core | `geometry`, `flowers`, `visits`, `export` | standard library, other core modules | network, subprocess, environment variables |
| adapters | `sam`, `classify`, `video` | core, one external service or tool each | business rules (thresholds, counting) |
| orchestration | `pipeline` | core, adapters | thresholds or service details |
| entry points | `cli`, `server`, `web/` | `pipeline` | logic that is not wiring or presentation |
| evaluation | `evaluation` | core, annotation files | anything the product needs at runtime |

Dependencies only point down: entry points -> `pipeline` -> adapters -> core.
The web app draws the tracks over the video in the browser from `Analysis.to_json()`, so no video is re-encoded.

The core stays pure so the counting
logic is tested without any API, video or GPU.

Each adapter takes an optional `client` so tests inject a fake instead of calling the service.

## Rules

- One module, one responsibility. A new external service is a new adapter, not a branch in an existing one.
- Thresholds live in `VisitRule` and CLI flags, never as literals inside functions.
- Every paid API call goes through a cache keyed by the input content.
- Behaviour changes come with a test; a change to the counted result updates the regression fixture on purpose.

## Security

- API keys only come from environment variables (`META_API_KEY`, `BASETEN_API_KEY`), never from files or arguments.
- Videos uploaded to the Meta Model API expire after 24 h (`sam.UPLOAD_TTL_S`).
- External tools run with argument lists, never through a shell.
- The `glimmer` backend expects `llama-server` bound to `127.0.0.1`, never exposed on the network.
- Model replies are parsed as untrusted data: unknown groups fall back to "other insects", confidences are clamped.
- Before every push: secret scan (gitleaks), dependency audit (pip-audit) and security lint rules (ruff `S`).

## Tests

`uv run python scripts/check.py` runs lint, format, unit tests, the end-to-end regression scenario,
the dependency audit and the secret scan. CI runs the same checks on every push.

The regression scenario (`tests/fixtures`) is a fixed SAM output in the exact wire format. To change it on purpose:

```
uv run python tests/fixtures/build_regression.py
POLLICOUNT_UPDATE_GOLDEN=1 uv run pytest tests/test_regression.py
```

then review the diff of `regression_expected.json`.
