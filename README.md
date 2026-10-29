# PolliCount

Zero-shot pollinator counting from an ordinary phone video, for citizen science.

Built during the Meta Global AI Developer Hackathon (Oct 26 - Nov 4, 2026). Everything in this repository was written during the event.

## Idea

A volunteer films a flower patch with a phone for 10 minutes. PolliCount:

1. tracks every insect and every flower with **SAM 3.1** (Meta Model API);
2. classifies each insect track into FIT Count groups with **Muse Spark**, **Muse Spark Open** or a local **Muse Glimmer** (bumblebee, honeybee, solitary bee, hoverfly, other fly, butterfly/moth, wasp, beetle, small insect), and drops tracks that are not insects;
3. counts a **visit** when an insect stays on a flower for at least 0.3 s;
4. exports a FIT Count record (JSON and CSV) the volunteer can check.

## Pipeline

| Module | Role |
|---|---|
| `video.py` | converts any video to the H.264 MP4 SAM accepts, cuts insect crops |
| `sam.py` | one SAM 3.1 request per concept (`insect`, `flower`), raw output cached so a video is never billed twice |
| `visits.py` | visit rule: insect mostly inside a flower box, nearly still, for at least 0.3 s; short tracking dropouts are bridged |
| `classify.py` | sends the best crops of each visiting insect to an OpenAI-compatible vision model |
| `export.py` | distinct insects per FIT Count group, plus every visit |

## Usage

```
uv sync
uv run pollicount run my_video.mp4 --out outputs/my_video --backend spark-open
```

| Backend | Model | Key |
|---|---|---|
| `spark` | `muse-spark-1.3` on the Meta Model API | `META_API_KEY` |
| `spark-open` | `meta-models/muse-spark-open-1026` on Baseten | `BASETEN_API_KEY` |
| `spark-open-fireworks` | `muse-spark-open` on Fireworks, fallback for Baseten | `FIREWORKS_API_KEY` |
| `glimmer` | Muse Glimmer served by `llama-server` on `127.0.0.1:8089` | none |

Web app: `uv run uvicorn --factory pollicount.server:production_app`, then open http://127.0.0.1:8000.
`POLLICOUNT_BACKEND` picks the classifier (default `spark-open`).

SAM always runs on the Meta Model API (`META_API_KEY`). It is billed per frame, so `--fps` (default 15) sets the cost.

Tests use synthetic SAM outputs and never call an API. All checks (lint, tests, regression scenario,
dependency audit, secret scan): `uv run python scripts/check.py`. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Status

Work in progress.

## What counts

FIT Count asks volunteers to count every insect *sitting on or landing on* a target flower, each insect once.
There is no agreed minimum duration in the literature, so PolliCount makes "landing" measurable: an insect
counts when most of it lies on a flower and it stays nearly still for at least 0.3 s (`--min-visit`), which
leaves out insects flying past. The record gives distinct insects per group, as FIT Count expects, and keeps
every visit (insect, flower, start, duration) for finer analyses. Spiders and caterpillars are not counted.

## Evaluation

`pollicount evaluate` scores a run against the annotated insect tracks of the Monash dataset: visit precision
and recall, insects found and classification accuracy. Annotators start a new visit when an insect hops off a
flower for a fraction of a second; the evaluation merges such hops (under 0.2 s) so both sides count the same way.

With the annotated tracks themselves as input, the visit rule reaches F1 0.90 on test video T1: the ceiling
for any tracker with this rule.

## Credits

Uses Meta's Segment Anything Model (SAM 3.1) through the Meta Model API, and `meta-sam-parser` (SAM License).

Evaluation data: Ratnayake, M. N., Amarathunga, D. C., Zaman, A., Dyer, A. G., Dorin, A. (2022).
Spatial Monitoring and Insect Behavioural Analysis Dataset. Monash University. CC BY 4.0.
https://doi.org/10.26180/21533760

## License

MIT
