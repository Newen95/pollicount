"""Visit detection: an insect that stays on the same flower long enough.

A frame counts as "on a flower" when the insect is nearly still and its centre, a third of its box,
half of the bottom third of its box (where the legs are in a side view), or the middle of its bottom edge
(its feet, under wings folded upright) lies inside the flower box.
Close-ups need the last two: open wings and the body stand above the flower the legs are touching.
Short breaks (the tracker losing the insect, or the insect walking fast across its flower for up to a second)
are bridged so one landing is not split in two.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from .geometry import Box, Track


@dataclass(frozen=True)
class VisitRule:
    min_inside: float = 0.3
    """Share of the insect box inside the flower box that counts as landed, even off-centre."""
    min_legs_inside: float = 0.5
    """Share of the bottom third of the insect box inside the flower box that counts as landed."""
    feet_on_top: bool = True
    """Also landed when the middle of the box's bottom edge, where the feet are, lies in the flower box:
    a butterfly with its wings folded upright stands on a flower with nothing else inside it."""
    max_speed: float = 2.0
    """Insect speed limit, in insect diagonals per second."""
    min_duration_s: float = 0.3
    max_gap_s: float = 1.0


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


def _nearest_box(track: Track, frame: int, max_gap: int):
    """The box at ``frame``, or the closest one within ``max_gap`` frames: flowers barely move."""
    for d in range(max_gap + 1):
        box = track.boxes.get(frame - d) or track.boxes.get(frame + d)
        if box:
            return box
    return None


def _host(insect: Track, frame: int, flowers: list[Track], rule: VisitRule, max_gap: int) -> str | None:
    """The flower the insect sits on: one holding its centre first, then the one holding most of it,
    then the smallest (the most precise)."""
    box = insect.boxes[frame]
    cx, cy = box.center
    legs = Box(box.left, box.bottom - box.height / 3, box.right, box.bottom)
    best, best_key = None, None
    for flower in flowers:
        flower_box = _nearest_box(flower, frame, max_gap)
        if flower_box is None:
            continue
        centred = flower_box.left <= cx < flower_box.right and flower_box.top <= cy < flower_box.bottom
        inside = box.inside_fraction(flower_box)
        feet = (
            rule.feet_on_top
            and flower_box.left <= cx < flower_box.right
            and flower_box.top <= box.bottom < flower_box.bottom
        )
        landed = (
            centred or feet or inside >= rule.min_inside or legs.inside_fraction(flower_box) >= rule.min_legs_inside
        )
        key = (centred, inside, -flower_box.area)
        if landed and (best_key is None or key > best_key):
            best, best_key = flower.object_id, key
    return best


def find_visits(insects: list[Track], flowers: list[Track], fps: float, rule: VisitRule | None = None) -> list[Visit]:
    rule = rule or VisitRule()
    max_gap = round(rule.max_gap_s * fps)
    visits: list[Visit] = []
    for insect in insects:
        speeds = _speeds(insect, fps)
        runs: list[list] = []  # [flower_id, start, end]
        for frame in insect.frames:
            host = _host(insect, frame, flowers, rule, max_gap) if speeds[frame] <= rule.max_speed else None
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
