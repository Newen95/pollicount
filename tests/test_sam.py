from pathlib import Path

from pollicount import sam
from samples import sam_frame, sam_record


def test_parse_tracks_groups_records_by_object():
    raw = (
        sam_frame(0, sam_record(3, 10, 20, 29, 39), sam_record(7, 100, 100, 149, 149, c=0.5))
        + sam_frame(1, sam_record(3, 12, 20, 31, 39))
        + sam_frame(4, sam_record(7, 101, 100, 150, 149))
    )
    tracks = {t.object_id: t for t in sam.parse_tracks(raw, "insect")}

    assert set(tracks) == {"3", "7"}
    assert tracks["3"].frames == [0, 1]
    assert tracks["7"].frames == [0, 4]
    box = tracks["3"].boxes[0]
    assert (box.left, box.top, box.right, box.bottom) == (10, 20, 30, 40)
    assert tracks["7"].confidences[0] == 0.5


def test_parse_tracks_empty_output_means_no_match():
    assert sam.parse_tracks("", "insect") == []


class _FailingClient:
    def __getattr__(self, name):
        raise AssertionError("cached video must not call the API")


def test_track_reads_cache_without_calling_api(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake video")
    cache = tmp_path / "cache"
    cached = sam.cache_path(clip, "insect", cache)
    cached.parent.mkdir()
    cached.write_text(sam_frame(0, sam_record(1, 0, 0, 9, 9)), encoding="utf-8")

    tracks = sam.track(clip, "insect", cache, client=_FailingClient())

    assert [t.object_id for t in tracks] == ["1"]
