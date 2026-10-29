"""The full analysis, shared by the CLI and the web server."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import classify, export, sam, video
from .geometry import Track
from .visits import Visit, VisitRule, find_visits

Tracker = Callable[[Path, str], list[Track]]
Classifier = Callable[[list[Path]], classify.Classification]
Progress = Callable[[str, float], None]
"""Receives a stage name and the overall progress, from 0 to 1."""


@dataclass
class Settings:
    fps: float = 15
    max_side: int = 1280
    insect_prompt: str = "insect"
    flower_prompt: str = "flower"
    rule: VisitRule = field(default_factory=VisitRule)
    crops_per_insect: int = 3


@dataclass
class Analysis:
    clip: Path
    info: video.VideoInfo
    insects: list[Track]
    flowers: list[Track]
    visits: list[Visit]
    groups: dict[str, classify.Classification]
    record: dict

    def to_json(self) -> dict:
        """Everything the web app needs to draw tracks over the video and show the record."""

        def boxes(track: Track) -> list[list[float]]:
            return [
                [f, *(round(v, 1) for v in (b.left, b.top, b.right, b.bottom))] for f, b in sorted(track.boxes.items())
            ]

        def insect(track: Track) -> dict:
            group = self.groups.get(track.object_id)
            return {
                "id": track.object_id,
                "visiting": track.object_id in visiting,
                "group": group.group if group else None,
                "confidence": group.confidence if group else None,
                "cues": group.cues if group else [],
                "boxes": boxes(track),
            }

        visiting = {v.insect_id for v in self.visits}
        return {
            "video": {
                "width": self.info.width,
                "height": self.info.height,
                "fps": self.info.fps,
                "frames": self.info.frames,
            },
            "flowers": [{"id": t.object_id, "boxes": boxes(t)} for t in self.flowers],
            "insects": [insect(t) for t in self.insects],
            "record": self.record,
        }


def sam_tracker(cache_dir: Path) -> Tracker:
    return lambda clip, concept: sam.track(clip, concept, cache_dir)


def model_classifier(backend_name: str) -> Classifier:
    backend = classify.BACKENDS[backend_name]
    return lambda crops: classify.classify_crops(crops, backend)


def analyse(
    source: Path,
    work_dir: Path,
    tracker: Tracker,
    classifier: Classifier | None,
    settings: Settings | None = None,
    progress: Progress | None = None,
) -> Analysis:
    settings = settings or Settings()

    def report(stage: str, value: float) -> None:
        if progress:
            progress(stage, value)

    report("preparing", 0.0)
    clip = video.to_sam_mp4(source, work_dir / "clip.mp4", fps=settings.fps, max_side=settings.max_side)
    info = video.probe(clip)

    report("tracking insects", 0.1)
    insects = tracker(clip, settings.insect_prompt)
    report("tracking flowers", 0.4)
    flowers = tracker(clip, settings.flower_prompt)

    report("finding visits", 0.7)
    visits = find_visits(insects, flowers, info.fps, settings.rule)

    groups: dict[str, classify.Classification] = {}
    visiting = [t for t in insects if any(v.insect_id == t.object_id for v in visits)]
    if classifier:
        for i, insect in enumerate(visiting):
            report("identifying insects", 0.75 + 0.2 * i / len(visiting))
            crops = [
                video.crop(clip, f, info.fps, insect.boxes[f], work_dir / "crops" / f"{insect.object_id}_{f}.png")
                for f in insect.best_frames(settings.crops_per_insect)
            ]
            groups[insect.object_id] = classifier(crops)

    record = export.fit_count(visits, groups, info.duration_s, len(flowers))
    export.write(record, work_dir)
    report("done", 1.0)
    return Analysis(clip, info, insects, flowers, visits, groups, record)
