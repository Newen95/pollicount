"""Web API: upload a video, follow the analysis, fetch the result.

uv run uvicorn --factory pollicount.server:production_app
"""

from __future__ import annotations

import logging
import os
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

from . import pipeline

logger = logging.getLogger("pollicount")

MAX_UPLOAD_BYTES = 500 * 1024 * 1024
JOB_TTL_S = 24 * 3600
WEB_DIR = Path(__file__).parent / "web"


def user_message(error: Exception) -> str:
    """What the volunteer sees when an analysis fails. Internals stay in the server logs."""
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

    def public(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 3),
            "error": self.error,
        }


def create_app(
    data_dir: Path,
    tracker: Callable[[Path], pipeline.Tracker],
    classifier: pipeline.Classifier | None,
) -> FastAPI:
    """``tracker`` builds a tracker from a cache directory, so tests can inject a fake one."""
    app = FastAPI(title="PolliCount")
    jobs: dict[str, Job] = {}
    lock = threading.Lock()
    data_dir.mkdir(parents=True, exist_ok=True)

    def purge_old_jobs() -> None:
        now = time.time()
        with lock:
            expired = [j for j in jobs.values() if now - j.created > JOB_TTL_S and j.status in ("done", "failed")]
            for job in expired:
                del jobs[job.id]
        for job in expired:
            shutil.rmtree(job.dir, ignore_errors=True)

    def run(job: Job, source: Path) -> None:
        def progress(stage: str, value: float) -> None:
            job.stage, job.progress = stage, value

        job.status = "running"
        try:
            analysis = pipeline.analyse(source, job.dir, tracker(data_dir / "sam-cache"), classifier, progress=progress)
            job.result = analysis.to_json()
            job.status = "done"
        except Exception as error:  # the job must never hang in "running"
            logger.exception("analysis %s failed", job.id)
            job.status, job.error = "failed", user_message(error)
        finally:
            source.unlink(missing_ok=True)

    def get_job(job_id: str) -> Job:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "analysis not found")
        return job

    @app.post("/api/analyses", status_code=202)
    async def create_analysis(file: UploadFile, background: BackgroundTasks) -> dict:
        purge_old_jobs()
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

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def production_app() -> FastAPI:
    backend = os.environ.get("POLLICOUNT_BACKEND", "spark-open")
    return create_app(
        Path(os.environ.get("POLLICOUNT_DATA", "outputs/server")),
        pipeline.sam_tracker,
        None if backend == "none" else pipeline.model_classifier(backend),
    )
