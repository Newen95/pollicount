"""One natural-history plate per FIT Count group, drawn by Muse Image and reviewed by Muse Spark.

    uv run python scripts/make_plates.py OUT_DIR [group ...]

Each plate is regenerated (up to three times) until Muse Spark, asked to review it as an entomologist,
finds every diagnostic trait of the group. Plates are then checked by eye before they ship.
"""

import base64
import json
import os
import re
import sys
from pathlib import Path

import httpx
from openai import OpenAI

OUT = Path(sys.argv[1])
client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["META_API_KEY"])

STYLE = (
    "Natural history plate, as in a museum entomology collection. Fine black ink line engraving with delicate "
    "cross-hatching and a single soft watercolour wash in the insect's natural colours. Dorsal view of a pinned "
    "specimen, symmetrical, scientifically accurate anatomy. Plain flat very pale cool grey-blue paper "
    "background, colour #EEF3F6. Centred, generous empty margin. No text, no labels, no frame, no pin, no shadow."
)

GROUPS = {
    "bumblebees": (
        "a buff-tailed bumblebee worker (Bombus terrestris), wings spread",
        [
            "very hairy, rounded, stocky body",
            "two pairs of membranous wings, forewings larger",
            "elbowed (geniculate) antennae",
            "black body with yellow bands and a white or buff tail",
            "six legs",
        ],
    ),
    "honeybees": (
        "a western honeybee worker (Apis mellifera), wings spread",
        [
            "elbowed (geniculate) antennae",
            "two pairs of membranous wings, forewings larger",
            "hairy thorax, amber and brown banded abdomen",
            "pollen baskets on the hind legs",
            "six legs",
        ],
    ),
    "solitary-bees": (
        "a female tawny mining bee (Andrena fulva), wings spread; its antennae are short and elbowed "
        "exactly like a honeybee's, bending at a sharp angle near the head",
        [
            "elbowed (geniculate) antennae: a long first segment then a sharp bend",
            "two pairs of membranous wings",
            "dense bright orange-red hair on the thorax and abdomen",
            "dark face and legs",
            "all six legs attached to the thorax",
        ],
    ),
    "wasps": (
        "a common wasp worker (Vespula vulgaris), wings spread",
        [
            "narrow waist between thorax and abdomen",
            "two pairs of membranous wings",
            "elbowed (geniculate) antennae",
            "smooth, almost hairless, bright yellow and black body",
            "six legs",
        ],
    ),
    "hoverflies": (
        "a hoverfly (Syrphus ribesii), wings spread",
        [
            "exactly one pair of wings (flies have two wings, not four)",
            "very large eyes",
            "short antennae with a bristle",
            "no narrow waist",
            "yellow and black banded abdomen",
            "six legs",
        ],
    ),
    "other-flies": (
        "a common greenbottle fly (Lucilia sericata), wings spread",
        [
            "exactly one pair of wings (flies have two wings, not four)",
            "large red-brown eyes",
            "short antennae with a bristle",
            "bristly metallic green body",
            "six legs",
        ],
    ),
    "butterflies-moths": (
        "a small tortoiseshell butterfly (Aglais urticae), wings open flat",
        [
            "two pairs of scaled wings",
            "clubbed antennae",
            "orange wings with black and yellow patches and a row of blue spots on the edges",
            "slender hairy body",
            "six legs or legs hidden under the wings",
        ],
    ),
    "beetles": (
        "a common red soldier beetle (Rhagonycha fulva), wing cases closed",
        [
            "hardened wing cases (elytra) closed over the abdomen, meeting in a straight line down the back",
            "thread-like antennae about half the body length, clearly shorter than the body",
            "orange-red body with dark tips to the wing cases",
            "six legs",
        ],
    ),
    "small-insects": (
        "a pollen beetle (Brassicogethes aeneus), wing cases closed, drawn enlarged",
        [
            "hardened wing cases (elytra) closed, meeting in a straight line",
            "short clubbed antennae",
            "small compact oval body, shiny metallic black-green",
            "six legs",
        ],
    ),
    "other-insects": (
        "a green shieldbug (Palomena prasina), wings folded",
        [
            "shield-shaped body with a large triangular plate (scutellum) on the back",
            "antennae with five segments",
            "green body",
            "six legs",
            "no elbowed antennae",
        ],
    ),
}


def generate(subject: str) -> bytes:
    kwargs = {"model": "muse-image-1.0", "prompt": f"{subject}. {STYLE}", "n": 1, "size": "1024x1024"}
    try:
        r = client.images.generate(**kwargs)
    except Exception:
        kwargs.pop("size")
        r = client.images.generate(**kwargs)
    d = r.data[0]
    if getattr(d, "b64_json", None):
        return base64.b64decode(d.b64_json)
    if not d.url.startswith("https://"):
        raise ValueError(f"unexpected image URL: {d.url}")
    return httpx.get(d.url, timeout=60).content


def check(image: bytes, traits: list[str]) -> dict:
    url = "data:image/png;base64," + base64.b64encode(image).decode()
    question = (
        "You are an entomologist reviewing a scientific illustration. For each trait below, say whether the "
        "drawing shows it correctly. Also list any anatomical error an entomologist would notice "
        "(wrong number of wings or legs, wrong antennae, missing waist, extra body parts).\n"
        + "\n".join(f"- {t}" for t in traits)
        + '\nReply with JSON only: {"traits": [{"trait": "...", "ok": true, "note": "..."}], "errors": ["..."]}'
    )
    r = client.responses.create(
        model="muse-spark-1.3",
        input=[
            {
                "role": "user",
                "content": [{"type": "input_image", "image_url": url}, {"type": "input_text", "text": question}],
            }
        ],
    )
    m = re.search(r"\{.*\}", r.output_text, re.S)
    return json.loads(m.group(0)) if m else {"traits": [], "errors": ["unreadable review"]}


for name in sys.argv[2:] or GROUPS:
    subject, traits = GROUPS[name]
    for attempt in range(1, 4):
        image = generate(f"{subject}, with {', '.join(traits)}")
        path = OUT / f"{name}_{attempt}.png"
        path.write_bytes(image)
        review = check(image, traits)
        failed = [t["trait"] for t in review.get("traits", []) if not t.get("ok")]
        ok = not failed and not review.get("errors")
        print(
            json.dumps(
                {
                    "group": name,
                    "attempt": attempt,
                    "file": path.name,
                    "ok": ok,
                    "failed": failed,
                    "errors": review.get("errors", []),
                }
            ),
            flush=True,
        )
        if ok:
            break
