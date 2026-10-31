"""Web API: upload a video, follow the analysis, fetch the result.

uv run uvicorn --factory pollicount.server:production_app
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import pipeline, video

logger = logging.getLogger("pollicount")

MAX_UPLOAD_BYTES = 500 * 1024 * 1024
JOB_TTL_S = 24 * 3600
WEB_DIR = Path(__file__).parent / "web"


@dataclass(frozen=True)
class Limits:
    """Keeps a public deployment within its API budget."""

    max_video_s: float = 120
    max_per_hour: int = 20


class VideoTooLong(Exception):
    pass


def user_message(error: Exception) -> str:
    """What the volunteer sees when an analysis fails. Internals stay in the server logs."""
    if isinstance(error, VideoTooLong):
        return str(error)
    if isinstance(error, subprocess.CalledProcessError):
        return "This file could not be read as a video."
    if isinstance(error, KeyError):
        return "The server is not configured for analyses yet."
    return "The analysis service is unavailable. Please try again in a moment."


@dataclass
class Job:
    id: str
    dir: Path
    created: float = field(default_factory=time.time)
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    error: str | None = None
    result: dict | None = None
    example: dict | None = None
    """Title and credits of a pre-analysed example; examples are never purged or rate limited."""

    def public(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 3),
            "error": self.error,
            "example": self.example,
        }


_EXAMPLE_ID = re.compile(r"[a-z0-9-]{1,40}")
_JOB_ID = re.compile(r"[0-9a-f]{32}")
_CROP_NAME = re.compile(r"[A-Za-z0-9-]{1,40}_\d{1,7}\.png")
RESTARTED = "The server restarted during this analysis. Upload the video again."


def save(job: Job) -> None:
    """Analyses live on disk too, so a server restart does not lose them."""
    state = {"created": job.created, "status": job.status, "error": job.error}
    if job.result is not None:
        (job.dir / "result.json").write_text(json.dumps(job.result), encoding="utf-8")
    (job.dir / "job.json").write_text(json.dumps(state), encoding="utf-8")


def load_jobs(data_dir: Path) -> dict[str, Job]:
    jobs = {}
    for state_file in data_dir.glob("*/job.json"):
        folder = state_file.parent
        if not _JOB_ID.fullmatch(folder.name):
            continue
        state = json.loads(state_file.read_text(encoding="utf-8"))
        job = Job(folder.name, folder, state["created"], state["status"], state["status"], 1.0, state["error"])
        if job.status == "done" and (folder / "result.json").is_file():
            job.result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        elif job.status != "failed":
            job.status, job.stage, job.error = "failed", "failed", RESTARTED
        jobs[job.id] = job
    return jobs


def load_examples(examples_dir: Path) -> dict[str, Job]:
    """Each example folder holds clip.mp4, result.json and meta.json (title, credit, licence, source)."""
    examples = {}
    for folder in sorted(examples_dir.glob("*")):
        files = [folder / name for name in ("clip.mp4", "result.json", "meta.json")]
        if not _EXAMPLE_ID.fullmatch(folder.name) or not all(f.is_file() for f in files):
            continue
        result = json.loads(files[1].read_text(encoding="utf-8"))
        meta = json.loads(files[2].read_text(encoding="utf-8"))
        examples[folder.name] = Job(folder.name, folder, 0.0, "done", "done", 1.0, None, result, meta)
    return examples


def create_app(
    data_dir: Path,
    tracker: Callable[[Path], pipeline.Tracker],
    classifier: pipeline.Classifier | None,
    limits: Limits | None = None,
) -> FastAPI:
    """``tracker`` builds a tracker from a cache directory, so tests can inject a fake one."""
    limits = limits or Limits()
    app = FastAPI(title="PolliCount")
    data_dir.mkdir(parents=True, exist_ok=True)
    jobs: dict[str, Job] = {**load_examples(data_dir / "examples"), **load_jobs(data_dir)}
    lock = threading.Lock()

    def purge_old_jobs() -> None:
        now = time.time()
        with lock:
            expired = [
                j
                for j in jobs.values()
                if not j.example and now - j.created > JOB_TTL_S and j.status in ("done", "failed")
            ]
            for job in expired:
                del jobs[job.id]
        for job in expired:
            shutil.rmtree(job.dir, ignore_errors=True)

    def run(job: Job, source: Path) -> None:
        def progress(stage: str, value: float) -> None:
            job.stage, job.progress = stage, value

        job.status = "running"
        save(job)
        try:
            duration = video.probe(source).duration_s
            if duration > limits.max_video_s:
                raise VideoTooLong(
                    f"This video lasts {duration / 60:.1f} min. Videos are limited to "
                    f"{limits.max_video_s / 60:g} min here: trim it and upload it again."
                )
            analysis = pipeline.analyse(source, job.dir, tracker(data_dir / "sam-cache"), classifier, progress=progress)
            job.result = analysis.to_json()
            job.status = "done"
        except Exception as error:  # the job must never hang in "running"
            logger.exception("analysis %s failed", job.id)
            job.status, job.error = "failed", user_message(error)
        finally:
            source.unlink(missing_ok=True)
            save(job)

    def get_job(job_id: str) -> Job:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "analysis not found")
        return job

    @app.post("/api/analyses", status_code=202)
    async def create_analysis(file: UploadFile, background: BackgroundTasks) -> dict:
        purge_old_jobs()
        hour_ago = time.time() - 3600
        recent = sorted(j.created for j in jobs.values() if not j.example and j.created > hour_ago)
        if len(recent) >= limits.max_per_hour:
            wait_min = max(1, round((recent[0] - hour_ago) / 60))
            raise HTTPException(429, f"PolliCount has reached its hourly limit. Try again in {wait_min} min.")
        job_id = uuid.uuid4().hex
        job = Job(job_id, data_dir / job_id)
        job.dir.mkdir()
        source = job.dir / "source"
        size = 0
        with source.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    out.close()
                    shutil.rmtree(job.dir, ignore_errors=True)
                    raise HTTPException(413, f"video larger than {MAX_UPLOAD_BYTES // 2**20} MB")
                out.write(chunk)
        with lock:
            jobs[job_id] = job
        background.add_task(run, job, source)
        return job.public()

    @app.get("/api/examples")
    def list_examples() -> list[dict]:
        return [{"id": j.id, **j.example} for j in jobs.values() if j.example]

    @app.get("/api/examples/{example_id}/poster")
    def read_poster(example_id: str) -> FileResponse:
        job = jobs.get(example_id)
        poster = job.dir / "poster.jpg" if job and job.example else None
        if poster is None or not poster.is_file():
            raise HTTPException(404, "poster not found")
        return FileResponse(poster, media_type="image/jpeg")

    @app.get("/api/analyses/{job_id}")
    def read_analysis(job_id: str) -> dict:
        return get_job(job_id).public()

    @app.get("/api/analyses/{job_id}/result")
    def read_result(job_id: str) -> dict:
        job = get_job(job_id)
        if job.result is None:
            raise HTTPException(409, f"analysis is {job.status}")
        return job.result

    @app.get("/api/analyses/{job_id}/video")
    def read_video(job_id: str) -> FileResponse:
        clip = get_job(job_id).dir / "clip.mp4"
        if not clip.exists():
            raise HTTPException(409, "video not ready")
        return FileResponse(clip, media_type="video/mp4")

    @app.get("/api/analyses/{job_id}/crops/{name}")
    def read_crop(job_id: str, name: str) -> FileResponse:
        crop = get_job(job_id).dir / "crops" / name
        if not _CROP_NAME.fullmatch(name) or not crop.is_file():
            raise HTTPException(404, "crop not found")
        return FileResponse(crop, media_type="image/png")

    @app.middleware("http")
    async def revalidate_web_files(request, call_next):
        """The page, its script and its styles are revalidated on every visit, so updates reach phones at once."""
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def production_app() -> FastAPI:
    backend = os.environ.get("POLLICOUNT_BACKEND", "spark-open")
    return create_app(
        Path(os.environ.get("POLLICOUNT_DATA", "outputs/server")),
        pipeline.sam_tracker,
        None if backend == "none" else pipeline.model_classifier(backend),
        Limits(
            max_video_s=float(os.environ.get("POLLICOUNT_MAX_VIDEO_S", 120)),
            max_per_hour=int(os.environ.get("POLLICOUNT_MAX_PER_HOUR", 20)),
        ),
    )
