import shutil
import subprocess
from pathlib import Path

import pytest

from pollicount import pipeline
from pollicount.classify import Classification

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
