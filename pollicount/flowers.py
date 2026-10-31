"""Flower units as a volunteer counts them.

SAM often splits a compound bloom (an umbel, a spike, a flower head) into many small flowers.
FIT Count counts the natural unit of the plant, so touching flowers are merged into one head.
"""

from __future__ import annotations

from .geometry import Box, Track


def _reference(track: Track) -> Box:
    return track.boxes[track.frames[len(track.frames) // 2]]


def _touch(a: Box, b: Box, gap_ratio: float) -> bool:
    gap = gap_ratio * min(max(a.width, a.height), max(b.width, b.height))
    return a.left - gap <= b.right and b.left - gap <= a.right and a.top - gap <= b.bottom and b.top - gap <= a.bottom


def group_touching(flowers: list[Track], gap_ratio: float = 0.25) -> list[Track]:
    """Merge flowers whose boxes touch, allowing a gap of ``gap_ratio`` times the smaller flower."""
    parent = {t.object_id: t.object_id for t in flowers}

    def root(key: str) -> str:
        while parent[key] != key:
            key = parent[key]
        return key

    ref = {t.object_id: _reference(t) for t in flowers if t.boxes}
    ids = list(ref)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if _touch(ref[a], ref[b], gap_ratio):
                parent[root(a)] = root(b)

    members: dict[str, list[Track]] = {}
    for track in flowers:
        if track.boxes:
            members.setdefault(root(track.object_id), []).append(track)

    heads = []
    for group in members.values():
        if len(group) == 1:
            heads.append(group[0])
            continue
        boxes: dict[int, Box] = {}
        for track in group:
            for frame, b in track.boxes.items():
                u = boxes.get(frame, b)
                boxes[frame] = Box(
                    min(u.left, b.left), min(u.top, b.top), max(u.right, b.right), max(u.bottom, b.bottom)
                )
        head_id = min((t.object_id for t in group), key=lambda k: (len(k), k))
        heads.append(Track(head_id, group[0].concept, boxes))
    return heads
