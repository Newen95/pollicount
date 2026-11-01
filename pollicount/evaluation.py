"""Accuracy against the Monash insect trajectory dataset (Ratnayake et al. 2022, CC BY 4.0).

Each test video `cam_<point>_<N|S>_video_<YYYYMMDD>_<HHMMSS>.h264.avi` matches the annotated
tracks of the same collection point and day. Track files are named `<point><day><HHMMSS><type>.csv`
and give one centre position per frame, plus the flower visited and the visit number.
"""

from __future__ import annotations

import csv
import re
import statistics
from dataclasses import dataclass, field
from math import hypot
from pathlib import Path

from .geometry import Track
from .visits import Visit

SOURCE_FPS = 30
GROUPS = {"0": "honeybees", "1": "hoverflies", "2": "butterflies & moths", "3": "wasps"}
_VIDEO_NAME = re.compile(r"cam_(\d)_[NS]_video_\d{6}(\d{2})_(\d{2})(\d{2})(\d{2})")


@dataclass(frozen=True)
class Recording:
    point: str
    day: str
    start_s: int
    frames: int

    @classmethod
    def from_video_name(cls, name: str, frames: int) -> Recording:
        match = _VIDEO_NAME.search(name)
        if match is None:
            raise ValueError(f"not a Monash test video name: {name}")
        point, day, h, m, s = match.groups()
        return cls(point, day, int(h) * 3600 + int(m) * 60 + int(s), frames)


@dataclass
class TruthInsect:
    id: str
    group: str
    positions: dict[int, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class TruthVisit:
    insect: str
    flower: str
    start: int
    end: int


@dataclass
class GroundTruth:
    """Annotations in source-video frames and pixels."""

    insects: list[TruthInsect]
    visits: list[TruthVisit]
    flowers: dict[str, tuple[float, float, float]]
    """Flower id -> (x, y, radius)."""


def _track_start_s(track_id: str) -> int:
    h, m, s = int(track_id[3:5]), int(track_id[5:7]), int(track_id[7:9])
    return h * 3600 + m * 60 + s


def load_ground_truth(trajectories: Path, rec: Recording) -> GroundTruth:
    tracks_dir = trajectories / "Extracted_Tracks" / "Insect_Tracks_CSV"
    rows_by_track: dict[str, list[dict]] = {}
    for path in sorted(tracks_dir.glob(f"{rec.point}{rec.day}*.csv")):
        track_id = path.stem
        if rec.start_s - 3600 <= _track_start_s(track_id) < rec.start_s + rec.frames / SOURCE_FPS:
            with path.open(newline="") as f:
                rows_by_track[track_id] = list(csv.DictReader(f))

    # Track files count frames from the start of the day; their names give the start second.
    # The median over all tracks cancels the one-second rounding of file names.
    offsets = [
        int(rows[0][f"absframe_{tid}"])
        - int(rows[0][f"nframe_{tid}"])
        - (_track_start_s(tid) - rec.start_s) * SOURCE_FPS
        for tid, rows in rows_by_track.items()
        if rows
    ]
    day_frame_of_start = round(statistics.median(offsets)) if offsets else 0

    insects, visits = [], []
    for tid, rows in rows_by_track.items():
        insect = TruthInsect(tid, GROUPS.get(tid[-1], "other insects"))
        current: tuple[str, str] | None = None
        start = end = 0
        for row in rows:
            frame = int(row[f"absframe_{tid}"]) - day_frame_of_start
            if not 0 <= frame < rec.frames:
                continue
            if row[f"x0_{tid}"]:
                insect.positions[frame] = (float(row[f"x0_{tid}"]), float(row[f"y0_{tid}"]))
            key = (row[f"flower_{tid}"], row[f"visit_num_{tid}"]) if row[f"flower_{tid}"] else None
            if key != current:
                if current:
                    visits.append(TruthVisit(tid, current[0].split(".")[0], start, end))
                current, start = key, frame
            end = frame
        if current:
            visits.append(TruthVisit(tid, current[0].split(".")[0], start, end))
        if insect.positions:
            insects.append(insect)

    flowers: dict[str, tuple[float, float, float]] = {}
    with (trajectories / "Extracted_Tracks" / "Flower_Positions_CSV" / f"flowers_{rec.point}{rec.day}.csv").open(
        newline=""
    ) as f:
        for row in csv.DictReader(f):
            if row["x0"] and row["y0"] and row["radius"]:
                flowers[row["flower_num"]] = (float(row["x0"]), float(row["y0"]), float(row["radius"]))
    return GroundTruth(insects, visits, flowers)


@dataclass(frozen=True)
class Scale:
    """Maps source-video frames and pixels to the analysed clip."""

    fps: float
    factor: float

    def frame(self, source_frame: int) -> int:
        return round(source_frame * self.fps / SOURCE_FPS)


def _match_flowers(flowers: list[Track], truth: GroundTruth, scale: Scale) -> dict[str, str]:
    """Predicted flower id -> annotated flower id, when its centre lies within the annotated circle."""
    matched = {}
    for flower in flowers:
        box = flower.boxes[flower.frames[len(flower.frames) // 2]]
        cx, cy = box.center
        best = min(
            truth.flowers.items(),
            key=lambda item: hypot(item[1][0] * scale.factor - cx, item[1][1] * scale.factor - cy),
            default=None,
        )
        if (
            best
            and hypot(best[1][0] * scale.factor - cx, best[1][1] * scale.factor - cy) <= best[1][2] * scale.factor * 1.5
        ):
            matched[flower.object_id] = best[0]
    return matched


def _match_insects(insects: list[Track], truth: GroundTruth, scale: Scale, max_dist_px: float) -> dict[str, str]:
    """Predicted insect id -> annotated insect id, by mean centre distance over shared frames."""
    matched = {}
    for track in insects:
        best, best_dist = None, max_dist_px * scale.factor
        for gt in truth.insects:
            shared = [
                (track.boxes[scale.frame(f)].center, p)
                for f, p in gt.positions.items()
                if scale.frame(f) in track.boxes
            ]
            if len(shared) < 3:
                continue
            dist = statistics.mean(hypot(c[0] - p[0] * scale.factor, c[1] - p[1] * scale.factor) for c, p in shared)
            if dist < best_dist:
                best, best_dist = gt.id, dist
        if best:
            matched[track.object_id] = best
    return matched


def _median_s(frame_errors) -> float | None:
    values = list(frame_errors)
    return round(statistics.median(values) / SOURCE_FPS, 2) if values else None


def harmonise(visits: list[TruthVisit], merge_gap_s: float, min_visit_s: float) -> list[TruthVisit]:
    """Annotators start a new visit when an insect hops off for a fraction of a second; PolliCount
    bridges such gaps. Merge them, then drop visits shorter than the counting threshold."""
    merged: list[TruthVisit] = []
    for v in sorted(visits, key=lambda v: (v.insect, v.flower, v.start)):
        last = merged[-1] if merged else None
        if (
            last
            and (last.insect, last.flower) == (v.insect, v.flower)
            and v.start - last.end <= merge_gap_s * SOURCE_FPS
        ):
            merged[-1] = TruthVisit(v.insect, v.flower, last.start, max(last.end, v.end))
        else:
            merged.append(v)
    return [v for v in merged if (v.end - v.start + 1) / SOURCE_FPS >= min_visit_s]


def evaluate(
    truth: GroundTruth,
    insects: list[Track],
    flowers: list[Track],
    visits: list[Visit],
    groups: dict[str, str],
    scale: Scale,
    min_visit_s: float = 0.3,
    merge_gap_s: float = 0.2,
    tolerance_s: float = 0.5,
    max_dist_px: float = 60,
) -> dict:
    """Visit precision and recall, insect recall and classification accuracy, with both sides
    following the same visit definition (see ``harmonise``)."""
    flower_map = _match_flowers(flowers, truth, scale)
    insect_map = _match_insects(insects, truth, scale, max_dist_px)
    tol = tolerance_s * SOURCE_FPS
    expected = harmonise(truth.visits, merge_gap_s, min_visit_s)

    unmatched = list(expected)
    true_positives = 0
    pairs: list[tuple[float, float, TruthVisit]] = []
    for visit in visits:
        start, end = visit.start_frame * SOURCE_FPS / scale.fps, visit.end_frame * SOURCE_FPS / scale.fps
        hit = next(
            (
                gt
                for gt in unmatched
                if flower_map.get(visit.flower_id) == gt.flower
                and start <= gt.end + tol
                and gt.start <= end + tol
                and insect_map.get(visit.insect_id, gt.insect) == gt.insect
            ),
            None,
        )
        if hit:
            unmatched.remove(hit)
            true_positives += 1
            pairs.append((start, end, hit))

    truth_groups = {gt.id: gt.group for gt in truth.insects}
    classified = [(groups[p], truth_groups[g]) for p, g in insect_map.items() if p in groups]
    precision = true_positives / len(visits) if visits else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    return {
        "visits_expected": len(expected),
        "visits_found": len(visits),
        "visits_matched": true_positives,
        "visit_precision": round(precision, 3),
        "visit_recall": round(recall, 3),
        "visit_f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0,
        "visitors_expected": len({v.insect for v in expected}),
        "visitors_found": len({v.insect_id for v in visits}),
        "visit_seconds_expected": round(sum(v.end - v.start + 1 for v in expected) / SOURCE_FPS, 1),
        "visit_seconds_found": round(sum(v.duration_s for v in visits), 1),
        "start_error_s": _median_s(abs(start - gt.start) for start, _, gt in pairs),
        "duration_error_s": _median_s(abs((end - start) - (gt.end - gt.start)) for start, end, gt in pairs),
        "insects_annotated": len(truth.insects),
        "insects_found": len(set(insect_map.values())),
        "flowers_annotated": len(truth.flowers),
        "flowers_found": len(flowers),
        "flowers_matched": len(set(flower_map.values())),
        "classification_accuracy": round(sum(p == t for p, t in classified) / len(classified), 3)
        if classified
        else None,
        "classified": len(classified),
    }
