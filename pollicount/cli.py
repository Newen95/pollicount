"""pollicount run VIDEO: video in, FIT Count record out.
pollicount evaluate VIDEO: the same on a Monash test video, scored against its annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import classify, evaluation, pipeline, sam, video
from .visits import VisitRule


def _add_analysis_args(p: argparse.ArgumentParser) -> None:
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
    p.add_argument("--yes", action="store_true", help="run without asking to confirm the SAM cost")


def _printer() -> pipeline.Progress:
    last = {"stage": None, "value": -1.0}

    def show(stage: str, value: float) -> None:
        if stage != last["stage"] or value - last["value"] >= 0.05:
            print(f"{value:4.0%} {stage}", flush=True)
            last.update(stage=stage, value=value)

    return show


def _analyse(args: argparse.Namespace) -> pipeline.Analysis:
    source = video.probe(args.video)
    frames = round(source.duration_s * args.fps)
    print(f"SAM 3.1: 2 concepts x {frames} frames, about ${sam.estimate_cost(frames, 2):.2f} (cached videos are free)")
    if not args.yes and input("Run? [y/N] ").strip().lower() != "y":
        raise SystemExit("cancelled")
    settings = pipeline.Settings(
        fps=args.fps,
        max_side=args.max_side,
        insect_prompt=args.insect_prompt,
        flower_prompt=args.flower_prompt,
        rule=VisitRule(min_duration_s=args.min_visit),
        crops_per_insect=args.crops,
    )
    return pipeline.analyse(
        args.video,
        args.out,
        pipeline.sam_tracker(args.cache),
        None if args.backend == "none" else pipeline.model_classifier(args.backend),
        settings,
        progress=_printer(),
        namer=None if args.backend == "none" else pipeline.model_namer(args.backend),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pollicount")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_analysis_args(sub.add_parser("run", help="count pollinator visits in a video"))
    p = sub.add_parser("evaluate", help="score a Monash test video against its annotations")
    _add_analysis_args(p)
    p.add_argument("--trajectories", type=Path, required=True, help="extracted Insect_Trajectory_Dataset folder")
    args = parser.parse_args(argv)

    analysis = _analyse(args)
    print(f"{len(analysis.insects)} insect tracks, {len(analysis.flowers)} flowers, {len(analysis.visits)} visits")
    if args.command == "run":
        print(json.dumps(analysis.record["counts"], indent=2))
        return

    source = video.probe(args.video)
    truth = evaluation.load_ground_truth(
        args.trajectories, evaluation.Recording.from_video_name(args.video.name, source.frames)
    )
    report = evaluation.evaluate(
        truth,
        analysis.insects,
        analysis.flowers,
        analysis.visits,
        {k: c.group for k, c in analysis.groups.items()},
        evaluation.Scale(analysis.info.fps, analysis.info.width / source.width),
        min_visit_s=args.min_visit,
    )
    report["settings"] = {"fps": args.fps, "max_side": args.max_side, "backend": args.backend}
    (args.out / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
