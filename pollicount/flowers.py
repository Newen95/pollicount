"""Flower units as a volunteer counts them.

SAM often splits a compound bloom (an umbel, a spike, a flower head) into many small flowers, and
finds them one by one as the bloom comes into view. FIT Count counts the natural unit of the plant,
so touching flowers are merged into one head, and a head keeps its full size on frames where some
of its flowers are not detected yet. Positions are compared in scene coordinates, so a moving
camera does not merge or split heads.
"""

from __future__ import annotations

from .geometry import Box, Track

Offsets = list[tuple[float, float]]
"""Cumulative camera shift per frame, from ``motion.camera_offsets``."""


def _shift(box: Box, dx: float, dy: float) -> Box:
    return Box(box.left + dx, box.top + dy, box.right + dx, box.bottom + dy)


def _offset(offsets: Offsets | None, frame: int) -> tuple[float, float]:
    if not offsets:
        return 0.0, 0.0
    return offsets[min(frame, len(offsets) - 1)]


def _in_scene(track: Track, offsets: Offsets | None) -> Box:
    frame = track.frames[len(track.frames) // 2]
    dx, dy = _offset(offsets, frame)
    return _shift(track.boxes[frame], -dx, -dy)


def _touch(a: Box, b: Box, gap_ratio: float) -> bool:
    gap = gap_ratio * min(max(a.width, a.height), max(b.width, b.height))
    return a.left - gap <= b.right and b.left - gap <= a.right and a.top - gap <= b.bottom and b.top - gap <= a.bottom


def _box_at(track: Track, frame: int, offsets: Offsets | None, max_fill: int, size: tuple[int, int] | None):
    """The flower's box at ``frame``, or its nearest box within ``max_fill`` frames moved with the camera."""
    if frame in track.boxes:
        return track.boxes[frame]
    for d in range(1, max_fill + 1):
        for source in (frame - d, frame + d):
            if source in track.boxes:
                (fx, fy), (sx, sy) = _offset(offsets, frame), _offset(offsets, source)
                box = _shift(track.boxes[source], fx - sx, fy - sy)
                if size:
                    w, h = size
                    box = Box(max(box.left, 0), max(box.top, 0), min(box.right, w), min(box.bottom, h))
                return box if box.area > 0 else None
    return None


def group_touching(
    flowers: list[Track],
    gap_ratio: float = 0.25,
    offsets: Offsets | None = None,
    size: tuple[int, int] | None = None,
    max_fill: int = 0,
) -> list[Track]:
    """Merge flowers whose boxes touch, allowing a gap of ``gap_ratio`` times the smaller flower.

    ``max_fill`` frames bridge a flower that is missing from its head for a moment, using
    ``offsets`` to follow the camera and ``size`` (width, height) to clip at the frame edges.
    """
    flowers = [t for t in flowers if t.boxes]
    parent = {t.object_id: t.object_id for t in flowers}

    def root(key: str) -> str:
        while parent[key] != key:
            key = parent[key]
        return key

    scene = {t.object_id: _in_scene(t, offsets) for t in flowers}
    ids = list(scene)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if _touch(scene[a], scene[b], gap_ratio):
                parent[root(a)] = root(b)

    members: dict[str, list[Track]] = {}
    for track in flowers:
        members.setdefault(root(track.object_id), []).append(track)

    heads = []
    for group in members.values():
        frames = range(min(min(t.boxes) for t in group), max(max(t.boxes) for t in group) + 1)
        boxes: dict[int, Box] = {}
        for frame in frames:
            parts = [b for t in group if (b := _box_at(t, frame, offsets, max_fill, size))]
            if parts:
                boxes[frame] = Box(
                    min(b.left for b in parts),
                    min(b.top for b in parts),
                    max(b.right for b in parts),
                    max(b.bottom for b in parts),
                )
        head_id = min((t.object_id for t in group), key=lambda k: (len(k), k))
        heads.append(Track(head_id, group[0].concept, boxes))
    return heads
