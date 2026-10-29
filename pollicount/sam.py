"""SAM 3.1 video tracking through the Meta Model API.

SAM takes one concept per request, so insects and flowers are two calls.
Raw outputs are cached on disk: a video is never billed twice for the same concept.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from meta_sam_parser import CompletedOutcome, SegmentationBoxRecord, video_segmentation_format

from .geometry import Box, Track

META_BASE_URL = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1"
API_KEY_ENV = "MODEL_API_KEY"


def parse_tracks(raw: str, concept: str) -> list[Track]:
    """Turn SAM's raw video output into one track per object id."""
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


def cache_path(video: Path, concept: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(video.read_bytes()).hexdigest()[:16]
    slug = "".join(c if c.isalnum() else "-" for c in concept.lower())
    return cache_dir / f"{digest}.{SAM_MODEL}.{slug}.txt"


def request_tracking(video: Path, concept: str, client=None) -> str:
    """Upload the video, run SAM on one concept and return the raw streamed text."""
    if client is None:
        from openai import OpenAI

        client = OpenAI(base_url=META_BASE_URL, api_key=os.environ[API_KEY_ENV])
    with video.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="user_data")
    parts: list[str] = []
    with client.responses.stream(
        model=SAM_MODEL,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": concept},
                    {"type": "input_video", "file_id": uploaded.id},
                ],
            }
        ],
        metadata={"include_confidence": "true", "mask_encoding": "one_bit"},
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                parts.append(event.delta)
    return "".join(parts)


def track(video: Path, concept: str, cache_dir: Path, client=None) -> list[Track]:
    """Tracks for ``concept`` in ``video``, from the cache when available."""
    cached = cache_path(video, concept, cache_dir)
    if cached.exists():
        raw = cached.read_text(encoding="utf-8")
    else:
        raw = request_tracking(video, concept, client)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached.write_text(raw, encoding="utf-8")
    return parse_tracks(raw, concept)
