"""Boxes and tracks shared by every pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot


@dataclass(frozen=True)
class Box:
    """Half-open box in source-video pixels."""

    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    @property
    def diagonal(self) -> float:
        return hypot(self.width, self.height)

    def intersection(self, other: Box) -> float:
        w = min(self.right, other.right) - max(self.left, other.left)
        h = min(self.bottom, other.bottom) - max(self.top, other.top)
        return w * h if w > 0 and h > 0 else 0.0

    def inside_fraction(self, other: Box) -> float:
        """Share of this box that lies inside ``other``."""
        return self.intersection(other) / self.area if self.area else 0.0


@dataclass
class Track:
    """One object followed across frames by SAM."""

    object_id: str
    concept: str
    boxes: dict[int, Box] = field(default_factory=dict)
    confidences: dict[int, float] = field(default_factory=dict)

    @property
    def frames(self) -> list[int]:
        return sorted(self.boxes)

    def best_frames(self, n: int) -> list[int]:
        """Frames where the object is most visible: large and confident, spread over the track."""
        ranked = sorted(
            self.boxes,
            key=lambda f: self.boxes[f].area * self.confidences.get(f, 1.0),
            reverse=True,
        )
        picked: list[int] = []
        min_spacing = max(1, len(self.boxes) // (2 * n))
        for frame in ranked:
            if all(abs(frame - p) >= min_spacing for p in picked):
                picked.append(frame)
            if len(picked) == n:
                break
        return sorted(picked)
