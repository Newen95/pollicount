"""SAM 3.1 video tracking through the Meta Model API.

SAM takes one concept per request, so insects and flowers are two calls.
Raw outputs are cached on disk: a video is never billed twice for the same concept.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

from meta_sam_parser import CompletedOutcome, SegmentationBoxRecord, video_segmentation_format

from .geometry import Box, Track

META_BASE_URL = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1"
API_KEY_ENV = "META_API_KEY"
UPLOAD_TTL_S = 24 * 3600
"""Volunteer videos are deleted from Meta's storage after a day."""
PRICE_PER_FRAME_USD = 0.20 / 1000
MIN_BILLED_FRAMES = 50


def estimate_cost(frames: int, concepts: int) -> float:
    return concepts * max(frames, MIN_BILLED_FRAMES) * PRICE_PER_FRAME_USD


_FRAME = re.compile(r"<(\d+)f>(.*)")
_FRAME_MARK = re.compile(r"<(\d+)f>")
# Mask payloads never contain "|", so a box token cannot appear inside one.
_BOX = re.compile(r"(?:^|,)([^,<>|]+)<\|box;([^|]*)\|>")


def parse_tracks(raw: str, concept: str) -> list[Track]:
    """Turn SAM's raw video output into one track per object id, reading boxes only.

    Masks are skipped: decoding them costs seconds per frame and visits only need boxes.
    """
    tracks: dict[str, Track] = {}
    for line in raw.splitlines():
        header = _FRAME.match(line.strip())
        if header is None:
            continue
        frame = int(header.group(1))
        for object_id, body in _BOX.findall(header.group(2)):
            fields = dict(item.split("=", 1) for item in body.split(";") if "=" in item)
            track = tracks.setdefault(object_id, Track(object_id, concept))
            track.boxes[frame] = Box(
                float(fields["x1"]), float(fields["y1"]), float(fields["x2"]) + 1, float(fields["y2"]) + 1
            )
            if "c" in fields:
                track.confidences[frame] = float(fields["c"])
    return list(tracks.values())


def parse_tracks_strict(raw: str, concept: str) -> list[Track]:
    """Same as ``parse_tracks`` through Meta's validating parser, which also decodes every mask."""
    parser = video_segmentation_format().create_parser()
    parser.push(raw)
    result = parser.finish(CompletedOutcome()).result
    tracks: dict[str, Track] = {}
    for record in result.records:
        if not isinstance(record, SegmentationBoxRecord) or record.frame is None:
            continue
        track = tracks.setdefault(record.object_id, Track(record.object_id, concept))
        frame = record.frame.frame_index
        track.boxes[frame] = Box(record.left, record.top, record.right, record.bottom)
        if record.confidence is not None:
            track.confidences[frame] = record.confidence
    return list(tracks.values())


def _digest(video: Path) -> str:
    with video.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()[:16]


def cache_path(video: Path, concept: str, cache_dir: Path) -> Path:
    slug = "".join(c if c.isalnum() else "-" for c in concept.lower())
    return cache_dir / f"{_digest(video)}.{SAM_MODEL}.{slug}.txt"


def _client():
    from openai import OpenAI

    return OpenAI(base_url=META_BASE_URL, api_key=os.environ[API_KEY_ENV])


def _upload(video: Path, client, cache_dir: Path) -> str:
    """Upload once per video; reuse the file while Meta still stores it."""
    record = cache_dir / f"{_digest(video)}.upload.json"
    if record.exists():
        saved = json.loads(record.read_text(encoding="utf-8"))
        if time.time() - saved["at"] < UPLOAD_TTL_S - 3600:
            return saved["id"]
    with video.open("rb") as f:
        uploaded = client.files.create(
            file=f, purpose="user_data", expires_after={"anchor": "created_at", "seconds": UPLOAD_TTL_S}
        )
    record.write_text(json.dumps({"id": uploaded.id, "at": time.time()}), encoding="utf-8")
    return uploaded.id


def _wait_for(client, response_id: str, poll_s: float, timeout_s: float = 3600) -> str:
    """Meta keeps processing a streamed response after a disconnect and stores the result."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        response = client.responses.retrieve(response_id)
        if response.status == "completed":
            return response.output_text
        if response.status in ("failed", "cancelled", "incomplete"):
            raise RuntimeError(f"SAM response {response_id} ended as {response.status}")
        time.sleep(poll_s)
    raise TimeoutError(f"SAM response {response_id} still running after {timeout_s:.0f} s")


def request_tracking(
    video: Path,
    concept: str,
    cache_dir: Path,
    client=None,
    on_frame: Callable[[int], None] | None = None,
    poll_s: float = 10,
) -> str:
    """Run SAM on one concept and return the raw text, surviving a dropped stream."""
    from openai import APIConnectionError

    client = client or _client()
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending = cache_path(video, concept, cache_dir).with_suffix(".response")
    if pending.exists():
        return _wait_for(client, pending.read_text(encoding="utf-8"), poll_s)

    file_id = _upload(video, client, cache_dir)
    parts: list[str] = []
    response_id = None
    try:
        with client.responses.stream(
            model=SAM_MODEL,
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": concept},
                        {"type": "input_video", "file_id": file_id},
                    ],
                }
            ],
            metadata={"include_confidence": "true", "mask_encoding": "one_bit"},
        ) as stream:
            for event in stream:
                if event.type == "response.created":
                    response_id = event.response.id
                    pending.write_text(response_id, encoding="utf-8")
                elif event.type == "response.output_text.delta":
                    parts.append(event.delta)
                    if on_frame and (frames := _FRAME_MARK.findall(event.delta)):
                        on_frame(int(frames[-1]))
    except APIConnectionError:
        if response_id is None:
            raise
        return _wait_for(client, response_id, poll_s)
    return "".join(parts)


def track(
    video: Path, concept: str, cache_dir: Path, client=None, on_frame: Callable[[int], None] | None = None
) -> list[Track]:
    """Tracks for ``concept`` in ``video``, from the cache when available."""
    cached = cache_path(video, concept, cache_dir)
    if not cached.exists():
        raw = request_tracking(video, concept, cache_dir, client, on_frame)
        cached.write_text(raw, encoding="utf-8")
        cached.with_suffix(".response").unlink(missing_ok=True)
    return parse_tracks(cached.read_text(encoding="utf-8"), concept)
