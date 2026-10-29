"""pollicount run VIDEO: video in, FIT Count record out."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import classify, export, sam, video
from .visits import VisitRule, find_visits


def run(args: argparse.Namespace) -> dict:
    out: Path = args.out
    clip = video.to_sam_mp4(args.video, out / "clip.mp4", fps=args.fps, max_side=args.max_side)
    info = video.probe(clip)
    insects = sam.track(clip, args.insect_prompt, args.cache)
    flowers = sam.track(clip, args.flower_prompt, args.cache)
    visits = find_visits(insects, flowers, info.fps, VisitRule(min_duration_s=args.min_visit))
    print(f"{len(insects)} insect tracks, {len(flowers)} flowers, {len(visits)} visits")

    groups: dict[str, classify.Classification] = {}
    if args.backend != "none":
        backend = classify.BACKENDS[args.backend]
        visiting = {t.object_id: t for t in insects if any(v.insect_id == t.object_id for v in visits)}
        for object_id, track in visiting.items():
            crops = [
                video.crop(clip, f, info.fps, track.boxes[f], out / "crops" / f"insect{object_id}_f{f}.png")
                for f in track.best_frames(args.crops)
            ]
            groups[object_id] = classify.classify_crops(crops, backend)
            print(f"insect {object_id}: {groups[object_id].group} ({groups[object_id].confidence:.2f})")

    record = export.fit_count(visits, groups, info.duration_s, len(flowers))
    export.write(record, out)
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pollicount")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="count pollinator visits in a video")
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, default=Path("outputs/run"))
    p.add_argument("--cache", type=Path, default=Path(".cache/sam"))
    p.add_argument("--fps", type=float, default=15, help="frame rate sent to SAM (billed per frame)")
    p.add_argument("--max-side", type=int, default=1280)
    p.add_argument("--insect-prompt", default="insect")
    p.add_argument("--flower-prompt", default="flower")
    p.add_argument("--min-visit", type=float, default=0.3, help="seconds on a flower to count a visit")
    p.add_argument("--backend", choices=[*classify.BACKENDS, "none"], default="spark")
    p.add_argument("--crops", type=int, default=3, help="crops per insect sent to the classifier")
    args = parser.parse_args(argv)
    record = run(args)
    print(json.dumps(record["counts"], indent=2))
