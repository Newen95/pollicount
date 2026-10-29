import csv
from pathlib import Path

import pytest

from pollicount import evaluation as ev
from pollicount.geometry import Box, Track
from pollicount.visits import Visit, find_visits

DAY_FRAME_OF_START = 162_000
VIDEO = "cam_4_N_video_20210309_132604.h264.avi"


def write_track(root: Path, track_id: str, start_nframe_abs: int, rows: list[tuple]):
    """rows: (x, y, flower, visit_num), one per frame; None x means no position for that frame."""
    path = root / "Extracted_Tracks" / "Insect_Tracks_CSV" / f"{track_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["nframe", "absframe", "x0", "y0", "flower", "visit_num"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", *(f"{c}_{track_id}" for c in cols)])
        for i, (x, y, flower, visit) in enumerate(rows):
            w.writerow(
                [i, i, start_nframe_abs + i, x if x is not None else "", y if y is not None else "", flower, visit]
            )


@pytest.fixture
def trajectories(tmp_path: Path) -> Path:
    flowers = tmp_path / "Extracted_Tracks" / "Flower_Positions_CSV"
    flowers.mkdir(parents=True)
    (flowers / "flowers_409.csv").write_text(",flower_num,x0,y0,radius\n0,0,300,300,40\n1,1,900,500,40\n")
    # Honeybee starting 13:26:14, i.e. 10 s into the video: lands on flower 0 for 1 s,
    # hops off for 0.1 s, lands again for 0.5 s.
    landing = [(300, 300, "0.0", "1.0")] * 30 + [(380, 300, "", "")] * 3 + [(300, 300, "0.0", "2.0")] * 15
    write_track(tmp_path, "40913261400", DAY_FRAME_OF_START + 300, [(200, 300, "", "")] * 10 + landing)
    # Hoverfly at 13:27:04 that only touches flower 1 for 0.1 s.
    write_track(tmp_path, "40913270401", DAY_FRAME_OF_START + 1800, [(900, 500, "1.0", "1.0")] * 3)
    # Another collection point, never loaded.
    write_track(tmp_path, "50913261400", 0, [(1, 1, "", "")])
    return tmp_path


def test_recording_from_video_name():
    rec = ev.Recording.from_video_name(VIDEO, 18002)
    assert (rec.point, rec.day, rec.start_s, rec.frames) == ("4", "09", 13 * 3600 + 26 * 60 + 4, 18002)
    with pytest.raises(ValueError):
        ev.Recording.from_video_name("garden.mp4", 10)


def test_load_ground_truth_aligns_frames_and_splits_visits(trajectories: Path):
    truth = ev.load_ground_truth(trajectories, ev.Recording.from_video_name(VIDEO, 18002))

    assert sorted((i.id, i.group) for i in truth.insects) == [
        ("40913261400", "honeybees"),
        ("40913270401", "hoverflies"),
    ]
    bee = next(i for i in truth.insects if i.group == "honeybees")
    assert min(bee.positions) == 300
    assert [(v.flower, v.start, v.end) for v in truth.visits if v.insect == bee.id] == [
        ("0", 310, 339),
        ("0", 343, 357),
    ]
    assert truth.flowers["1"] == (900.0, 500.0, 40.0)


def test_harmonise_merges_short_hops_and_drops_touches(trajectories: Path):
    truth = ev.load_ground_truth(trajectories, ev.Recording.from_video_name(VIDEO, 18002))
    merged = ev.harmonise(truth.visits, merge_gap_s=0.2, min_visit_s=0.3)
    assert [(v.flower, v.start, v.end) for v in merged] == [("0", 310, 357)]


def perfect_prediction(truth: ev.GroundTruth, frames: int):
    insects = [
        Track(i.id, "insect", {f: Box(x - 15, y - 15, x + 15, y + 15) for f, (x, y) in i.positions.items()})
        for i in truth.insects
    ]
    flowers = [
        Track(k, "flower", {f: Box(x - r, y - r, x + r, y + r) for f in range(frames)})
        for k, (x, y, r) in truth.flowers.items()
    ]
    return insects, flowers


def test_perfect_tracks_give_full_scores(trajectories: Path):
    truth = ev.load_ground_truth(trajectories, ev.Recording.from_video_name(VIDEO, 18002))
    insects, flowers = perfect_prediction(truth, 2000)
    visits = find_visits(insects, flowers, ev.SOURCE_FPS)
    report = ev.evaluate(truth, insects, flowers, visits, {i.id: i.group for i in truth.insects}, ev.Scale(30, 1.0))

    assert (report["visits_expected"], report["visits_found"], report["visits_matched"]) == (1, 1, 1)
    assert report["visit_f1"] == 1.0
    assert report["insects_found"] == 2
    assert report["classification_accuracy"] == 1.0


def test_visit_on_the_wrong_flower_or_time_is_a_false_positive(trajectories: Path):
    truth = ev.load_ground_truth(trajectories, ev.Recording.from_video_name(VIDEO, 18002))
    insects, flowers = perfect_prediction(truth, 2000)
    wrong = [Visit("x", "1", 310, 357, 30), Visit("y", "0", 1500, 1600, 30)]
    report = ev.evaluate(truth, insects, flowers, wrong, {}, ev.Scale(30, 1.0))

    assert (report["visits_matched"], report["visit_precision"], report["visit_recall"]) == (0, 0.0, 0.0)


def test_scale_maps_source_to_clip(trajectories: Path):
    truth = ev.load_ground_truth(trajectories, ev.Recording.from_video_name(VIDEO, 18002))
    scale = ev.Scale(fps=15, factor=0.5)
    insects = [
        Track(
            i.id,
            "insect",
            {scale.frame(f): Box(x / 2 - 8, y / 2 - 8, x / 2 + 8, y / 2 + 8) for f, (x, y) in i.positions.items()},
        )
        for i in truth.insects
    ]
    flowers = [
        Track(k, "flower", {f: Box((x - r) / 2, (y - r) / 2, (x + r) / 2, (y + r) / 2) for f in range(1000)})
        for k, (x, y, r) in truth.flowers.items()
    ]
    report = ev.evaluate(truth, insects, flowers, find_visits(insects, flowers, 15), {}, scale)
    assert report["visit_f1"] == 1.0
