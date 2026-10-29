"""FIT Count record from detected visits, ready for the volunteer to check.

FIT Count asks to count each insect sitting on or landing on a target flower once, so ``counts`` holds
distinct insects per group. Visits stay in the record for finer analyses.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from .classify import FIT_GROUPS, NOT_AN_INSECT, Classification
from .visits import Visit


def fit_count(visits: list[Visit], groups: dict[str, Classification], duration_s: float, flowers: int) -> dict:
    """Distinct insects and visits per FIT Count group. Tracks classified as non-insects are dropped."""
    unknown = Classification("other insects", 0.0)
    label = {v.insect_id: groups.get(v.insect_id, unknown) for v in visits}
    counted = [v for v in visits if label[v.insect_id].group != NOT_AN_INSECT]
    insects = {v.insect_id: label[v.insect_id].group for v in counted}
    insects_per_group = Counter(insects.values())
    visits_per_group = Counter(label[v.insect_id].group for v in counted)
    return {
        "duration_min": round(duration_s / 60, 2),
        "flowers_tracked": flowers,
        "counts": {g: insects_per_group.get(g, 0) for g in FIT_GROUPS},
        "insects": len(insects),
        "visit_counts": {g: visits_per_group.get(g, 0) for g in FIT_GROUPS},
        "total_visits": len(counted),
        "visits": [
            {
                "insect": v.insect_id,
                "flower": v.flower_id,
                "group": label[v.insect_id].group,
                "confidence": label[v.insect_id].confidence,
                "start_s": round(v.start_s, 2),
                "duration_s": round(v.duration_s, 2),
            }
            for v in counted
        ],
    }


def write(record: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fit_count.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    with (out_dir / "fit_count.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["duration_min", "flowers_tracked", *FIT_GROUPS, "total"])
        writer.writerow(
            [record["duration_min"], record["flowers_tracked"], *record["counts"].values(), record["insects"]]
        )
