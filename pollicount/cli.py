"""pollicount run VIDEO: video in, FIT Count record out."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import classify, pipeline
from .visits import VisitRule


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
    p.add_argument("--backend", choices=[*classify.BACKENDS, "none"], default="spark-open")
    p.add_argument("--crops", type=int, default=3, help="crops per insect sent to the classifier")
    args = parser.parse_args(argv)

    settings = pipeline.Settings(
        fps=args.fps,
        max_side=args.max_side,
        insect_prompt=args.insect_prompt,
        flower_prompt=args.flower_prompt,
        rule=VisitRule(min_duration_s=args.min_visit),
        crops_per_insect=args.crops,
    )
    analysis = pipeline.analyse(
        args.video,
        args.out,
        pipeline.sam_tracker(args.cache),
        None if args.backend == "none" else pipeline.model_classifier(args.backend),
        settings,
        progress=lambda stage, value: print(f"{value:4.0%} {stage}", flush=True),
    )
    print(f"{len(analysis.insects)} insect tracks, {len(analysis.flowers)} flowers, {len(analysis.visits)} visits")
    print(json.dumps(analysis.record["counts"], indent=2))
