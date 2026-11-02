"""The full analysis, shared by the CLI and the web server."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import classify, export, motion, sam, video
from . import flowers as flower_units
from .geometry import Box, Track
from .visits import Visit, VisitRule, find_visits

Tracker = Callable[[Path, str, Callable[[int], None]], list[Track]]
"""Tracks one concept in a clip, reporting the last frame it has processed."""
Classifier = Callable[[list[Path]], classify.Classification]
Namer = Callable[[Path], str | None]
"""Common name of the plant in a frame, or None."""
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
    group_flowers: bool = True
    """Merge touching flowers into one head (umbels, spikes), the unit FIT Count counts."""
    motion_candidates: bool = True
    """On a still camera where SAM finds no visit, add the insects found from their movement (too small for SAM)."""
    motion_min_confidence: float = 0.3
    """Motion candidates are kept only when the classifier calls them an insect with this confidence."""


@dataclass
class Analysis:
    clip: Path
    info: video.VideoInfo
    insects: list[Track]
    flowers: list[Track]
    visits: list[Visit]
    groups: dict[str, classify.Classification]
    record: dict
    crops: dict[str, list[str]] = field(default_factory=dict)
    """Crop file names (in ``work_dir/crops``) shown to the classifier, per insect."""
    camera_moves: list[tuple[float, float]] = field(default_factory=list)
    """Time ranges where the camera moved: flower identities are unreliable there."""
    from_motion: set[str] = field(default_factory=set)
    """Insects too small for SAM, found from their movement."""

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
                "from_motion": track.object_id in self.from_motion,
                "group": group.group if group else None,
                "confidence": group.confidence if group else None,
                "cues": group.cues if group else [],
                "crops": self.crops.get(track.object_id, []),
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
            "camera_moves": [list(m) for m in self.camera_moves],
        }


def _alone(candidate: Track, tracks: list[Track]) -> Track:
    """The candidate on the frames where its crop (see ``video.crop``) holds no part of the other tracks."""
    kept = {}
    for f, box in candidate.boxes.items():
        half = max(max(box.width, box.height) * 1.6, 96) / 2
        cx, cy = box.center
        window = Box(cx - half, cy - half, cx + half, cy + half)
        if not any(f in t.boxes and t.boxes[f].inside_fraction(window) > 0 for t in tracks):
            kept[f] = box
    return Track(candidate.object_id, candidate.concept, kept, {f: candidate.confidences.get(f, 1.0) for f in kept})


def _seen_by(candidate: Track, tracks: list[Track]) -> bool:
    """True when the candidate's centre lies in one of the tracks' boxes on most of their shared frames."""
    for track in tracks:
        shared = [f for f in candidate.boxes if f in track.boxes]
        inside = sum(
            track.boxes[f].left <= candidate.boxes[f].center[0] <= track.boxes[f].right
            and track.boxes[f].top <= candidate.boxes[f].center[1] <= track.boxes[f].bottom
            for f in shared
        )
        if shared and inside >= len(shared) / 2:
            return True
    return False


def sam_tracker(cache_dir: Path) -> Tracker:
    return lambda clip, concept, on_frame: sam.track(clip, concept, cache_dir, on_frame=on_frame)


def model_classifier(backend_name: str) -> Classifier:
    backend = classify.BACKENDS[backend_name]
    return lambda crops: classify.classify_crops(crops, backend)


def model_namer(backend_name: str) -> Namer:
    backend = classify.BACKENDS[backend_name]
    return lambda frame: classify.name_plant(frame, backend)


def analyse(
    source: Path,
    work_dir: Path,
    tracker: Tracker,
    classifier: Classifier | None,
    settings: Settings | None = None,
    progress: Progress | None = None,
    namer: Namer | None = None,
) -> Analysis:
    settings = settings or Settings()

    def report(stage: str, value: float) -> None:
        if progress:
            progress(stage, value)

    report("preparing", 0.0)
    clip = video.to_sam_mp4(source, work_dir / "clip.mp4", fps=settings.fps, max_side=settings.max_side)
    info = video.probe(clip)
    offsets = motion.camera_offsets(clip, info.width, info.height)
    camera_moves = motion.camera_moves(offsets, info.fps, info.width)

    def frames_done(stage: str, start: float) -> Callable[[int], None]:
        furthest = [0]

        def on_frame(frame: int) -> None:
            furthest[0] = max(furthest[0], frame)
            report(stage, start + 0.3 * min(1.0, furthest[0] / max(info.frames, 1)))

        return on_frame

    report("tracking insects", 0.1)
    insects = tracker(clip, settings.insect_prompt, frames_done("tracking insects", 0.1))
    report("tracking flowers", 0.4)
    flowers = tracker(clip, settings.flower_prompt, frames_done("tracking flowers", 0.4))
    if not flowers and namer:
        # Some flowers (lavender spikes, for one) are not seen as "flower" but are found by the plant's name.
        plant = namer(video.still(clip, info.duration_s / 2, work_dir / "plant.jpg", width=info.width))
        if plant:
            flowers = tracker(clip, plant, frames_done("tracking flowers", 0.4))
    if settings.group_flowers:
        # Camera offsets are left out: measured on blurred pans they split heads more than they help.
        flowers = flower_units.group_touching(
            flowers,
            size=(info.width, info.height),
            max_fill=round(2 * info.fps),
            camera_moves=[(round(a * info.fps), round(b * info.fps)) for a, b in camera_moves],
        )

    report("finding visits", 0.7)
    small: set[str] = set()
    # Motion is for videos whose insects SAM cannot see. When SAM already follows insects on the flowers,
    # moving blobs are mostly petals and leaves stirred by those insects or the wind.
    sam_visits = find_visits(insects, flowers, info.fps, settings.rule)
    if settings.motion_candidates and classifier and not camera_moves and not sam_visits:
        extra = [
            t for t in motion.detect(clip, info.width, info.height, info.fps, known=insects) if not _seen_by(t, insects)
        ]
        small = {t.object_id for t in extra}
        insects = insects + extra
    visits = find_visits(insects, flowers, info.fps, settings.rule)

    groups: dict[str, classify.Classification] = {}
    crops: dict[str, list[str]] = {}
    visiting = [t for t in insects if any(v.insect_id == t.object_id for v in visits)]
    if classifier:
        for i, insect in enumerate(visiting):
            report("identifying insects", 0.75 + 0.2 * i / len(visiting))
            # A motion candidate is judged only on crops without a tracked insect in view: a crop of moving
            # petals next to a real bee would otherwise show the bee and be confirmed.
            shown = (
                _alone(insect, [t for t in insects if t.object_id not in small])
                if insect.object_id in small
                else insect
            )
            if not shown.boxes:
                continue
            paths = [
                video.crop(clip, f, info.fps, insect.boxes[f], work_dir / "crops" / f"{insect.object_id}_{f}.png")
                for f in shown.best_frames(settings.crops_per_insect)
            ]
            groups[insect.object_id] = classifier(paths)
            crops[insect.object_id] = [p.name for p in paths]
    # Motion also picks up leaves and shadows: only those the classifier confirms are counted.
    unconfirmed = {i for i in small if i not in groups or groups[i].confidence < settings.motion_min_confidence}
    visits = [v for v in visits if v.insect_id not in unconfirmed]

    record = export.fit_count(visits, groups, info.duration_s, len(flowers))
    export.write(record, work_dir)
    report("done", 1.0)
    return Analysis(clip, info, insects, flowers, visits, groups, record, crops, camera_moves, small)
