"""ffmpeg helpers: probe, convert for SAM, cut insect crops."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .geometry import Box


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} not found on PATH")
    return path


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps


def probe(path: Path) -> VideoInfo:
    out = subprocess.run(
        [_tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_read_packets", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    return VideoInfo(
        stream["width"], stream["height"], float(Fraction(stream["avg_frame_rate"])), int(stream["nb_read_packets"])
    )


def to_sam_mp4(src: Path, dst: Path, fps: float | None = None, max_side: int | None = 1280) -> Path:
    """H.264 MP4 with faststart, the format SAM accepts (no WebM, no VP9)."""
    filters = []
    if fps:
        filters.append(f"fps={fps}")
    if max_side:
        filters.append(f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [_tool("ffmpeg"), "-y", "-loglevel", "error", "-i", str(src)]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-an", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)
    return dst


def crop(video: Path, frame: int, fps: float, box: Box, dst: Path, pad: float = 0.3, size: int = 384) -> Path:
    """Square crop around ``box`` at ``frame``, padded so the flower context stays visible."""
    side = max(box.width, box.height) * (1 + 2 * pad)
    cx, cy = box.center
    x, y = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [_tool("ffmpeg"), "-y", "-loglevel", "error", "-ss", f"{frame / fps:.4f}", "-i", str(video),
         "-frames:v", "1", "-vf", f"crop='min({int(side)},iw-{x})':'min({int(side)},ih-{y})':{x}:{y},scale={size}:-2",
         str(dst)],
        check=True,
    )
    return dst
