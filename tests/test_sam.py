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


def _event(type_, **fields):
    return type(type_, (), {"type": type_, **fields})()


class _FakeStream:
    def __init__(self, events, drop_after=None):
        self._events, self._drop_after = events, drop_after

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        import httpx
        from openai import APIConnectionError

        for i, event in enumerate(self._events):
            if i == self._drop_after:
                raise APIConnectionError(request=httpx.Request("POST", "https://api.meta.ai/v1/responses"))
            yield event


class _FakeClient:
    def __init__(self, deltas, drop_after=None, stored=None):
        self.uploads, self.requests, self.retrieved = [], [], []
        events = [_event("response.created", response=type("R", (), {"id": "resp-1"})())]
        events += [_event("response.output_text.delta", delta=d) for d in deltas]
        self._stream_obj = _FakeStream(events, drop_after)
        self._stored = stored
        self.files = type("Files", (), {"create": self._upload})()
        self.responses = type("Responses", (), {"stream": self._stream, "retrieve": self._retrieve})()

    def _upload(self, **kwargs):
        self.uploads.append(kwargs)
        return type("F", (), {"id": f"file-{len(self.uploads)}"})()

    def _stream(self, **kwargs):
        self.requests.append(kwargs)
        return self._stream_obj

    def _retrieve(self, response_id):
        self.retrieved.append(response_id)
        return type("R", (), {"status": "completed", "output_text": self._stored})()


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"fake video")
    return path


def test_request_uploads_with_expiry_streams_and_reports_frames(clip: Path, tmp_path: Path):
    client = _FakeClient(["<0f>1<|box", "|>\n<7f>"])
    frames = []

    raw = sam.request_tracking(clip, "insect", tmp_path / "cache", client, on_frame=frames.append)

    assert raw == "<0f>1<|box|>\n<7f>"
    assert frames == [0, 7]
    assert client.uploads[0]["purpose"] == "user_data"
    assert client.uploads[0]["expires_after"] == {"anchor": "created_at", "seconds": sam.UPLOAD_TTL_S}
    content = client.requests[0]["input"][0]["content"]
    assert content == [{"type": "input_text", "text": "insect"}, {"type": "input_video", "file_id": "file-1"}]
    assert client.requests[0]["model"] == "sam-3.1"


def test_dropped_stream_is_recovered_from_the_stored_response(clip: Path, tmp_path: Path):
    client = _FakeClient(["<0f>", "partial"], drop_after=2, stored="<0f>full output")

    raw = sam.request_tracking(clip, "insect", tmp_path / "cache", client, poll_s=0)

    assert raw == "<0f>full output"
    assert client.retrieved == ["resp-1"]


def test_interrupted_run_resumes_without_paying_again(clip: Path, tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    sam.cache_path(clip, "insect", cache).with_suffix(".response").write_text("resp-9", encoding="utf-8")
    client = _FakeClient([], stored="<0f>stored")

    assert sam.request_tracking(clip, "insect", cache, client, poll_s=0) == "<0f>stored"
    assert client.requests == [] and client.uploads == []


def test_upload_is_reused_for_the_second_concept(clip: Path, tmp_path: Path):
    client = _FakeClient(["<0f>"])
    sam.request_tracking(clip, "insect", tmp_path / "cache", client)
    sam.request_tracking(clip, "flower", tmp_path / "cache", client)

    assert len(client.uploads) == 1
    assert [r["input"][0]["content"][1]["file_id"] for r in client.requests] == ["file-1", "file-1"]


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


def test_fast_parser_matches_meta_parser_on_real_sam_output():
    raw = (Path(__file__).parent / "fixtures" / "sam31_real_insect.txt").read_text(encoding="utf-8")
    tracks = sam.parse_tracks(raw, "insect")

    assert len(tracks) == 1 and len(tracks[0].boxes) == 2
    assert _as_dict(tracks) == _as_dict(sam.parse_tracks_strict(raw, "insect"))
