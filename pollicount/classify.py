"""Insect classification into FIT Count groups with a vision-language model.

Every backend speaks the OpenAI chat API, so Muse Spark (Meta Model API),
Muse Spark Open (Baseten) and a local Muse Glimmer (llama-server) share one code path.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

FIT_GROUPS = (
    "bumblebees",
    "honeybees",
    "solitary bees",
    "wasps",
    "hoverflies",
    "other flies",
    "butterflies & moths",
    "beetles",
    "small insects",
    "other insects",
)
NOT_AN_INSECT = "not an insect"


@dataclass(frozen=True)
class Backend:
    name: str
    base_url: str
    model: str
    api_key_env: str | None


BACKENDS = {
    "spark": Backend("spark", "https://api.meta.ai/v1", "muse-spark-1.3", "META_API_KEY"),
    "spark-open": Backend(
        "spark-open", "https://inference.baseten.co/v1", "meta-models/muse-spark-open-1026", "BASETEN_API_KEY"
    ),
    "glimmer": Backend("glimmer", "http://127.0.0.1:8089/v1", "muse-glimmer", None),
}

PROMPT = f"""These crops show the same insect, tracked in a video of a flower patch.
Classify it into exactly one FIT Count group: {", ".join(FIT_GROUPS)}.
"small insects" means shorter than 3 mm. Answer "{NOT_AN_INSECT}" for debris, a leaf, a shadow,
a spider or a caterpillar: FIT Count does not count them.
Reply with JSON only: {{"group": "...", "confidence": 0.0, "visible_cues": ["..."]}}"""


@dataclass
class Classification:
    group: str
    confidence: float
    cues: list[str] = field(default_factory=list)
    raw: str = ""


def parse_reply(text: str) -> Classification:
    """Read the model's JSON answer, tolerating code fences and stray reasoning."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    group = str(data.get("group", "")).strip().lower()
    if group not in FIT_GROUPS and group != NOT_AN_INSECT:
        return Classification("other insects", 0.0, [], text)
    try:
        confidence = min(1.0, max(0.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    cues = [str(c) for c in data.get("visible_cues", []) if isinstance(c, str)]
    return Classification(group, confidence, cues, text)


def _image_part(path: Path) -> dict:
    encoded = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def classify_crops(crops: list[Path], backend: Backend, client=None) -> Classification:
    if client is None:
        from openai import OpenAI

        key = os.environ[backend.api_key_env] if backend.api_key_env else "none"
        client = OpenAI(base_url=backend.base_url, api_key=key)
    response = client.chat.completions.create(
        model=backend.model,
        messages=[
            {
                "role": "user",
                "content": [*(_image_part(c) for c in crops), {"type": "text", "text": PROMPT}],
            }
        ],
        max_tokens=2000,
    )
    return parse_reply(response.choices[0].message.content or "")
