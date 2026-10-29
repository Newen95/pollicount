"""End-to-end non-regression on a fixed SAM output.

If a change alters the count on purpose, review the diff of the expected file after:
    POLLICOUNT_UPDATE_GOLDEN=1 uv run pytest tests/test_regression.py
"""

import json
import os
from pathlib import Path

from pollicount import export, sam
from pollicount.classify import Classification
from pollicount.visits import find_visits

FIXTURES = Path(__file__).parent / "fixtures"
FPS, DURATION_S = 15, 20


def run_scenario() -> dict:
    insects = sam.parse_tracks((FIXTURES / "regression_insects.txt").read_text(encoding="utf-8"), "insect")
    flowers = sam.parse_tracks((FIXTURES / "regression_flowers.txt").read_text(encoding="utf-8"), "flower")
    groups = {
        k: Classification(v["group"], v["confidence"])
        for k, v in json.loads((FIXTURES / "regression_groups.json").read_text(encoding="utf-8")).items()
    }
    return export.fit_count(find_visits(insects, flowers, FPS), groups, DURATION_S, len(flowers))


def test_count_matches_golden_record():
    expected_path = FIXTURES / "regression_expected.json"
    record = run_scenario()
    if os.environ.get("POLLICOUNT_UPDATE_GOLDEN") == "1":
        expected_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    assert record == json.loads(expected_path.read_text(encoding="utf-8"))


def test_golden_record_tells_the_expected_story():
    record = run_scenario()
    visits = [(v["insect"], v["flower"]) for v in record["visits"]]

    assert visits == [("1", "10"), ("1", "11"), ("2", "12")]
    assert record["counts"]["honeybees"] == 2
    assert record["counts"]["hoverflies"] == 1
    assert record["total_visits"] == 3
