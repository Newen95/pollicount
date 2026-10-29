"""Visit detection: an insect that stays on the same flower long enough.

A frame counts as "on a flower" when most of the insect box lies inside the
flower box and the insect is nearly still. Short dropouts (SAM losing the
insect for a few frames) are bridged so one landing is not split in two.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from .geometry import Track


@dataclass(frozen=True)
class VisitRule:
    min_inside: float = 0.5
    """Share of the insect box that must lie inside the flower box."""
    max_speed: float = 2.0
    """Insect speed limit, in insect diagonals per second."""
    min_duration_s: float = 0.3
    max_gap_s: float = 0.2


@dataclass(frozen=True)
class Visit:
    insect_id: str
    flower_id: str
    start_frame: int
    end_frame: int
    fps: float

    @property
    def start_s(self) -> float:
        return self.start_frame / self.fps

    @property
    def duration_s(self) -> float:
        return (self.end_frame - self.start_frame + 1) / self.fps


def _speeds(insect: Track, fps: float) -> dict[int, float]:
    """Center speed at each frame, in insect diagonals per second."""
    frames = insect.frames
    speeds: dict[int, float] = {}
    for i, frame in enumerate(frames):
        j = i - 1 if i > 0 else i + 1
        if j >= len(frames):
            speeds[frame] = 0.0
            continue
        a, b = insect.boxes[frame], insect.boxes[frames[j]]
        dt = abs(frame - frames[j]) / fps
        size = max(a.diagonal, 1.0)
        speeds[frame] = hypot(a.center[0] - b.center[0], a.center[1] - b.center[1]) / size / dt
    return speeds


def _host(insect: Track, frame: int, flowers: list[Track], rule: VisitRule) -> str | None:
    box = insect.boxes[frame]
    best, best_inside = None, rule.min_inside
    for flower in flowers:
        flower_box = flower.boxes.get(frame)
        if flower_box is None:
            continue
        inside = box.inside_fraction(flower_box)
        if inside >= best_inside:
            best, best_inside = flower.object_id, inside
    return best


def find_visits(insects: list[Track], flowers: list[Track], fps: float, rule: VisitRule | None = None) -> list[Visit]:
    rule = rule or VisitRule()
    max_gap = round(rule.max_gap_s * fps)
    visits: list[Visit] = []
    for insect in insects:
        speeds = _speeds(insect, fps)
        runs: list[list] = []  # [flower_id, start, end]
        for frame in insect.frames:
            host = _host(insect, frame, flowers, rule) if speeds[frame] <= rule.max_speed else None
            if host is None:
                continue
            last = runs[-1] if runs else None
            if last and last[0] == host and frame - last[2] - 1 <= max_gap:
                last[2] = frame
            else:
                runs.append([host, frame, frame])
        for flower_id, start, end in runs:
            visit = Visit(insect.object_id, flower_id, start, end, fps)
            if visit.duration_s >= rule.min_duration_s:
                visits.append(visit)
    return sorted(visits, key=lambda v: (v.start_frame, v.insect_id))
