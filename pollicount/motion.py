"""Insect candidates from motion, for static cameras where insects are too small for SAM.

A background model marks what moves; foliage that moves all the time, oversized blobs and frames
shaken by wind are discarded; the remaining blobs are linked into tracks by nearest neighbour.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .geometry import Box, Track
from .video import _tool


@dataclass(frozen=True)
class MotionSettings:
    min_area: int = 60
    """Smallest blob kept, in pixels of the analysed clip."""
    max_area: int = 6000
    max_aspect: float = 4.0
    learning_rate: float = 0.001
    """Slow, so an insect resting on a flower is not absorbed into the background too quickly."""
    warmup_s: float = 2.0
    busy_fraction: float = 0.04
    """Frames where more than this share of pixels moves (wind, shake) are skipped."""
    foliage_rate: float = 0.25
    """Pixels moving more often than this are foliage."""
    max_jump: float = 90.0
    """Largest move between two detections of one track, in pixels."""
    max_missing_s: float = 1.0
    min_detections: int = 5


def frames(video: Path, width: int, height: int):
    """Grayscale frames streamed from ffmpeg at ``width`` x ``height``, without temporary files."""
    proc = subprocess.Popen(  # noqa: S603
        [
            _tool("ffmpeg"),
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"scale={width}:{height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    size = width * height
    try:
        while chunk := proc.stdout.read(size):
            if len(chunk) < size:
                break
            yield np.frombuffer(chunk, np.uint8).reshape(height, width)
    finally:
        proc.stdout.close()
        proc.wait()


def blobs(mask: np.ndarray, foliage: np.ndarray, s: MotionSettings) -> list[Box]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found = []
    for x, y, w, h, area in stats[1:count]:
        if not s.min_area <= area <= s.max_area or max(w, h) > s.max_aspect * min(w, h):
            continue
        if foliage[y : y + h, x : x + w].mean() > 0.5:
            continue
        found.append(Box(float(x), float(y), float(x + w), float(y + h)))
    return found


@dataclass
class _Open:
    track: Track
    last_frame: int
    center: tuple[float, float]


def link(detections: dict[int, list[Box]], fps: float, s: MotionSettings) -> list[Track]:
    """Greedy nearest-neighbour linking, closest pairs first."""
    max_missing = round(s.max_missing_s * fps)
    open_tracks: list[_Open] = []
    done: list[Track] = []
    next_id = 0
    for frame in sorted(detections):
        still_open = []
        for t in open_tracks:
            if frame - t.last_frame > max_missing:
                done.append(t.track)
            else:
                still_open.append(t)
        open_tracks = still_open
        pairs = sorted(
            (np.hypot(b.center[0] - t.center[0], b.center[1] - t.center[1]), i, j)
            for i, t in enumerate(open_tracks)
            for j, b in enumerate(detections[frame])
        )
        used_t, used_b = set(), set()
        for dist, i, j in pairs:
            if dist > s.max_jump or i in used_t or j in used_b:
                continue
            used_t.add(i)
            used_b.add(j)
            box = detections[frame][j]
            t = open_tracks[i]
            t.track.boxes[frame], t.last_frame, t.center = box, frame, box.center
        for j, box in enumerate(detections[frame]):
            if j not in used_b:
                open_tracks.append(_Open(Track(f"m{next_id}", "insect", {frame: box}), frame, box.center))
                next_id += 1
    done += [t.track for t in open_tracks]
    return [t for t in done if len(t.boxes) >= s.min_detections]


def detect(video: Path, width: int, height: int, fps: float, settings: MotionSettings | None = None) -> list[Track]:
    s = settings or MotionSettings()
    model = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    moving = np.zeros((height, width), np.float32)
    detections: dict[int, list[Box]] = {}
    warmup = round(s.warmup_s * fps)
    for i, frame in enumerate(frames(video, width, height)):
        mask = model.apply(frame, learningRate=-1 if i < warmup else s.learning_rate)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        hit = mask > 0
        cv2.accumulateWeighted(hit.astype(np.float32), moving, 0.02)
        if i < warmup or hit.mean() > s.busy_fraction:
            continue
        found = blobs(mask, moving > s.foliage_rate, s)
        if found:
            detections[i] = found
    return link(detections, fps, s)


def camera_offsets(
    video: Path, width: int, height: int, min_step: float = 0.004, min_response: float = 0.1
) -> list[tuple[float, float]]:
    """Cumulative shift of the whole picture at each frame, in pixels of the video.

    Consecutive frames are compared by phase correlation on a 160 px wide copy. A scene point seen
    at ``p`` on frame 0 is seen near ``p + offsets[f]`` on frame ``f``. Steps under ``min_step`` of
    the width, or measured with a weak correlation peak, are noise (leaves, flicker): summed over
    thousands of frames they would make a still camera drift, so they are dropped.
    """
    small_w = 160
    small_h = max(2, round(height * small_w / width / 2) * 2)
    scale = width / small_w
    window = cv2.createHanningWindow((small_w, small_h), cv2.CV_32F)
    offsets = [(0.0, 0.0)]
    previous = None
    for frame in frames(video, small_w, small_h):
        current = frame.astype(np.float32)
        if previous is not None:
            (dx, dy), response = cv2.phaseCorrelate(previous, current, window)
            if np.hypot(dx, dy) < min_step * small_w or response < min_response:
                dx = dy = 0.0
            x, y = offsets[-1]
            offsets.append((x + dx * scale, y + dy * scale))
        previous = current
    return offsets


def camera_moves(
    offsets: list[tuple[float, float]], fps: float, width: int, threshold: float = 0.01, min_s: float = 0.3
) -> list[tuple[float, float]]:
    """Time ranges where the camera moved: a shift above ``threshold`` of the width per frame, held for ``min_s``."""
    moving = [
        i
        for i in range(1, len(offsets))
        if np.hypot(offsets[i][0] - offsets[i - 1][0], offsets[i][1] - offsets[i - 1][1]) / width > threshold
    ]
    ranges: list[tuple[float, float]] = []
    for i in moving:
        if ranges and i - round(ranges[-1][1] * fps) <= 1:
            ranges[-1] = (ranges[-1][0], i / fps)
        else:
            ranges.append((i / fps, i / fps))
    return [(round(a, 2), round(b, 2)) for a, b in ranges if b - a >= min_s]
