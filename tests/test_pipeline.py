import shutil
import subprocess
from pathlib import Path

import pytest

from pollicount import pipeline
from pollicount.classify import Classification
from pollicount.geometry import Box, Track

from samples import track

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "source.webm"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=2", str(path)],
        check=True,
    )
    return path


def fake_tracker(clip: Path, concept: str, on_frame=None):
    if on_frame:
        on_frame(29)
    if concept == "flower":
        return [track("10", "flower", {f: (100, 80, 200, 180) for f in range(30)})]
    return [
        track("1", "insect", {f: (140, 120, 160, 140) for f in range(5, 20)}),
        track("2", "insect", {f: (10 * f, 10, 10 * f + 15, 25) for f in range(10)}),
    ]


def test_analyse_counts_visits_and_classifies_only_visiting_insects(source: Path, tmp_path: Path):
    seen_crops: list[list[Path]] = []
    stages: list[tuple[str, float]] = []

    def classifier(crops):
        seen_crops.append(crops)
        return Classification("honeybees", 0.9)

    analysis = pipeline.analyse(
        source,
        tmp_path / "work",
        fake_tracker,
        classifier,
        pipeline.Settings(fps=15),
        progress=lambda s, v: stages.append((s, v)),
    )

    assert analysis.info.fps == 15
    assert analysis.info.width == 320
    assert [(v.insect_id, v.flower_id) for v in analysis.visits] == [("1", "10")]
    assert analysis.record["counts"]["honeybees"] == 1
    assert len(seen_crops) == 1
    assert all(p.exists() for p in seen_crops[0])
    assert (tmp_path / "work" / "fit_count.json").exists()
    values = [v for _, v in stages]
    assert values == sorted(values)
    assert stages[0][1] == 0.0 and stages[-1] == ("done", 1.0)


def test_analyse_without_classifier_keeps_unknown_group(source: Path, tmp_path: Path):
    analysis = pipeline.analyse(source, tmp_path / "work", fake_tracker, None)
    assert analysis.record["counts"]["other insects"] == 1


def test_result_lists_the_crops_shown_to_the_classifier(source: Path, tmp_path: Path):
    analysis = pipeline.analyse(source, tmp_path / "work", fake_tracker, lambda crops: Classification("wasps", 0.8))
    insects = {i["id"]: i for i in analysis.to_json()["insects"]}

    assert len(insects["1"]["crops"]) == 3
    assert all((tmp_path / "work" / "crops" / name).is_file() for name in insects["1"]["crops"])
    assert insects["2"]["crops"] == []
    assert isinstance(analysis.to_json()["camera_moves"], list)


def test_motion_candidates_already_tracked_by_sam_are_left_out():
    sam_track = Track("0", "insect", {f: Box(100, 100, 140, 140) for f in range(10)})
    same = Track("m1", "insect", {f: Box(110, 110, 125, 125) for f in range(5, 15)})
    other = Track("m2", "insect", {f: Box(400, 300, 410, 310) for f in range(5, 15)})

    assert pipeline._seen_by(same, [sam_track])
    assert not pipeline._seen_by(other, [sam_track])


def test_flowers_missed_as_flower_are_tracked_by_the_plant_name(source: Path, tmp_path: Path):
    asked: list[str] = []

    def tracker(clip, concept, on_frame=None):
        asked.append(concept)
        return fake_tracker(clip, "flower" if concept == "lavender" else "insect") if concept != "flower" else []

    analysis = pipeline.analyse(
        source, tmp_path / "work", tracker, lambda crops: Classification("bumblebees", 0.9), namer=lambda f: "lavender"
    )

    assert asked == ["insect", "flower", "lavender"]
    assert [(v.insect_id, v.flower_id) for v in analysis.visits] == [("1", "10")]
