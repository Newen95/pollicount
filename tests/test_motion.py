import numpy as np

from pollicount import motion
from pollicount.geometry import Box, Track

S = motion.MotionSettings(min_detections=3, max_jump=30, max_missing_s=0.2)


def at(x, y, size=10):
    return Box(x, y, x + size, y + size)


def test_link_follows_two_insects_separately():
    detections = {f: [at(10 + 5 * f, 10), at(200, 100 + 4 * f)] for f in range(6)}
    tracks = motion.link(detections, fps=10, s=S)

    assert sorted(len(t.boxes) for t in tracks) == [6, 6]
    assert all(abs(t.boxes[5].left - t.boxes[0].left) in (0, 25) for t in tracks)


def test_link_bridges_short_gaps_and_splits_long_ones():
    short_gap = {f: [at(50, 50)] for f in (0, 1, 2, 4, 5)}
    long_gap = {f: [at(50, 50)] for f in (0, 1, 2, 9, 10, 11)}

    assert [len(t.boxes) for t in motion.link(short_gap, 10, S)] == [5]
    assert [len(t.boxes) for t in motion.link(long_gap, 10, S)] == [3, 3]


def test_link_drops_jumps_and_short_tracks():
    detections = {0: [at(0, 0)], 1: [at(5, 0)], 2: [at(300, 300)], 3: [at(305, 300)]}
    assert motion.link(detections, 10, S) == []


def piece(object_id, frames, x, y):
    return Track(object_id, "insect", {f: at(x, y) for f in frames})


def test_merge_joins_a_resting_insect_and_carries_its_box():
    first, again = piece("a", range(0, 10), 50, 50), piece("b", range(60, 70), 55, 52)
    merged = motion.merge_tracklets([first, again], fps=10, s=S)

    assert len(merged) == 1
    assert merged[0].frames == list(range(0, 70))
    assert merged[0].boxes[30] == at(50, 50)
    assert merged[0].confidences[30] == 0.0 and 5 not in merged[0].confidences


def test_merge_joins_duplicates_and_short_flights_but_not_far_strangers():
    side_by_side = [piece("a", range(0, 10), 50, 50), piece("b", range(3, 12), 52, 50)]
    flight = [piece("a", range(0, 10), 50, 50), piece("b", range(15, 25), 150, 50)]
    stranger = [piece("a", range(0, 10), 50, 50), piece("b", range(100, 110), 400, 300)]

    assert len(motion.merge_tracklets(side_by_side, 10, S)) == 1
    assert [len(t.boxes) for t in motion.merge_tracklets(flight, 10, S)] == [20]
    assert len(motion.merge_tracklets(stranger, 10, S)) == 2


def test_blobs_keep_insect_sized_shapes_outside_foliage():
    mask = np.zeros((100, 200), np.uint8)
    mask[10:20, 10:20] = 255  # insect
    mask[50:52, 10:90] = 255  # thin stem
    mask[30:40, 150:160] = 255  # inside foliage
    foliage = np.zeros_like(mask, dtype=bool)
    foliage[25:45, 140:170] = True
    found = motion.blobs(mask, foliage, motion.MotionSettings(min_area=50))

    assert [(b.left, b.top, b.right, b.bottom) for b in found] == [(10, 10, 20, 20)]


def _clip(tmp_path, source: str):
    import shutil
    import subprocess

    import pytest

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    path = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", source, str(path)], check=True)
    return path


def test_still_camera_has_no_moves(tmp_path):
    clip = _clip(tmp_path, "smptebars=size=640x360:rate=15:duration=3")
    assert motion.camera_moves(motion.camera_offsets(clip, 640, 360), 15, 640) == []


def test_panning_camera_is_detected(tmp_path):
    clip = _clip(
        tmp_path,
        "smptebars=size=960x360:rate=15:duration=3,crop=480:360:'if(between(t,1,2),(t-1)*400,if(gt(t,2),400,0))':0",
    )
    offsets = motion.camera_offsets(clip, 480, 360)
    moves = motion.camera_moves(offsets, 15, 480)

    assert len(moves) == 1
    start, end = moves[0]
    assert 0.8 <= start <= 1.2 and 1.7 <= end <= 2.2
    assert abs(offsets[-1][0] + 400) < 20  # the scene slid 400 px to the left
