from pathlib import Path

import pytest

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


class _FakeStream:
    def __init__(self, deltas):
        self._events = [type("E", (), {"type": "response.output_text.delta", "delta": d}) for d in deltas]

    def __enter__(self):
        return iter(self._events)

    def __exit__(self, *exc):
        return False


class _FakeClient:
    def __init__(self, deltas):
        self.uploads, self.requests = [], []
        self.files = type("Files", (), {"create": self._upload})()
        self.responses = type("Responses", (), {"stream": self._stream})()
        self._deltas = deltas

    def _upload(self, **kwargs):
        self.uploads.append(kwargs)
        return type("F", (), {"id": "file-1"})()

    def _stream(self, **kwargs):
        self.requests.append(kwargs)
        return _FakeStream(self._deltas)


def test_request_uploads_with_expiry_and_joins_stream(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake video")
    client = _FakeClient(["<0f>", "1<|box"])

    raw = sam.request_tracking(clip, "insect", client)

    assert raw == "<0f>1<|box"
    assert client.uploads[0]["purpose"] == "user_data"
    assert client.uploads[0]["expires_after"] == {"anchor": "created_at", "seconds": sam.UPLOAD_TTL_S}
    content = client.requests[0]["input"][0]["content"]
    assert content == [{"type": "input_text", "text": "insect"}, {"type": "input_video", "file_id": "file-1"}]
    assert client.requests[0]["model"] == "sam-3.1"


def _as_dict(tracks):
    return {t.object_id: (t.boxes, t.confidences) for t in tracks}


def test_fast_parser_matches_meta_parser():
    fixtures = Path(__file__).parent / "fixtures"
    raw = (fixtures / "regression_insects.txt").read_text(encoding="utf-8")
    raw += "".join((fixtures / "regression_flowers.txt").read_text(encoding="utf-8").splitlines(True)[:5])

    assert _as_dict(sam.parse_tracks(raw, "x")) == _as_dict(sam.parse_tracks_strict(raw, "x"))


def test_cost_estimate_follows_meta_pricing():
    assert sam.estimate_cost(9000, 2) == pytest.approx(3.6)
    assert sam.estimate_cost(10, 1) == pytest.approx(0.01)
