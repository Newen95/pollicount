import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pollicount import server
from pollicount.classify import Classification

from test_pipeline import fake_tracker

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = server.create_app(tmp_path / "data", lambda cache: fake_tracker, lambda crops: Classification("wasps", 0.8))
    return TestClient(app)


@pytest.fixture
def video_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "clip.webm"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=2", str(path)],
        check=True,
    )
    return path.read_bytes()


def test_upload_runs_analysis_and_serves_result_and_video(client: TestClient, video_bytes: bytes):
    created = client.post("/api/analyses", files={"file": ("garden.webm", video_bytes, "video/webm")})
    assert created.status_code == 202
    job_id = created.json()["id"]

    status = client.get(f"/api/analyses/{job_id}").json()
    assert (status["status"], status["progress"]) == ("done", 1.0)

    result = client.get(f"/api/analyses/{job_id}/result").json()
    assert result["video"]["fps"] == 15
    assert result["record"]["counts"]["wasps"] == 1
    visiting = [i for i in result["insects"] if i["visiting"]]
    assert [(i["id"], i["group"]) for i in visiting] == [("1", "wasps")]
    assert result["insects"][0]["boxes"][0] == [5, 140.0, 120.0, 160.0, 140.0]

    video = client.get(f"/api/analyses/{job_id}/video")
    assert video.headers["content-type"] == "video/mp4"
    assert len(video.content) > 0


def test_not_a_video_fails_with_a_readable_message(client: TestClient):
    job_id = client.post("/api/analyses", files={"file": ("notes.txt", b"hello", "text/plain")}).json()["id"]
    status = client.get(f"/api/analyses/{job_id}").json()

    assert status["status"] == "failed"
    assert status["error"] == "This file could not be read as a video."
    assert client.get(f"/api/analyses/{job_id}/result").status_code == 409


def test_oversized_upload_is_rejected_and_cleaned(client: TestClient, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 10)
    response = client.post("/api/analyses", files={"file": ("big.mp4", b"x" * 100, "video/mp4")})

    assert response.status_code == 413
    assert [p.name for p in (tmp_path / "data").iterdir()] == []


def test_unknown_analysis_is_404(client: TestClient):
    assert client.get("/api/analyses/../../etc").status_code == 404
    assert client.get("/api/analyses/deadbeef/result").status_code == 404


def test_web_app_is_served(client: TestClient):
    page = client.get("/")
    assert page.status_code == 200
    assert "Count the pollinators" in page.text
    assert client.get("/app.js").status_code == 200


def test_too_long_video_fails_before_any_api_call(tmp_path: Path, video_bytes: bytes):
    def tracker_must_not_run(cache):
        raise AssertionError("SAM must not be called")

    app = server.create_app(tmp_path / "data", tracker_must_not_run, None, server.Limits(max_video_s=1))
    client = TestClient(app)
    job_id = client.post("/api/analyses", files={"file": ("long.webm", video_bytes, "video/webm")}).json()["id"]
    status = client.get(f"/api/analyses/{job_id}").json()

    assert status["status"] == "failed"
    assert "limited to" in status["error"]


def test_hourly_limit_refuses_new_uploads(tmp_path: Path, video_bytes: bytes):
    app = server.create_app(tmp_path / "data", lambda cache: fake_tracker, None, server.Limits(max_per_hour=1))
    client = TestClient(app)
    assert client.post("/api/analyses", files={"file": ("a.webm", video_bytes, "video/webm")}).status_code == 202
    refused = client.post("/api/analyses", files={"file": ("b.webm", video_bytes, "video/webm")})

    assert refused.status_code == 429
    assert "hourly limit" in refused.json()["detail"]


def test_examples_are_listed_served_and_never_purged(tmp_path: Path, video_bytes: bytes, monkeypatch):
    folder = tmp_path / "data" / "examples" / "hoverfly"
    folder.mkdir(parents=True)
    (folder / "clip.mp4").write_bytes(b"mp4")
    (folder / "result.json").write_text('{"record": {"insects": 1}}', encoding="utf-8")
    meta = {"title": "Hoverfly on wild carrot", "credit": "A. Author", "license": "CC BY 4.0", "source": "https://x"}
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "data" / "examples" / "Bad Name").mkdir()
    monkeypatch.setattr(server, "JOB_TTL_S", -1)
    client = TestClient(
        server.create_app(tmp_path / "data", lambda cache: fake_tracker, None, server.Limits(max_per_hour=1))
    )

    assert client.get("/api/examples").json() == [{"id": "hoverfly", **meta}]
    assert client.post("/api/analyses", files={"file": ("a.webm", video_bytes, "video/webm")}).status_code == 202
    assert client.get("/api/analyses/hoverfly").json()["example"]["title"] == "Hoverfly on wild carrot"
    assert client.get("/api/analyses/hoverfly/result").json() == {"record": {"insects": 1}}
    assert client.get("/api/analyses/hoverfly/video").content == b"mp4"
