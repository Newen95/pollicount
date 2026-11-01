"""Pre-analyse a video as an example shown in the web app.

    uv run python scripts/make_example.py VIDEO ID --title ... --credit ... --license ... --source URL

Writes outputs/examples/ID/{clip.mp4, result.json, meta.json}. Upload it to the Modal volume with:
    modal volume put pollicount-data outputs/examples/ID /examples/ID
"""

import argparse
import json
import shutil
from pathlib import Path

from pollicount import pipeline, video


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("id")
    p.add_argument("--title", required=True)
    p.add_argument("--credit", required=True)
    p.add_argument("--license", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--backend", default="spark-open")
    p.add_argument("--out", type=Path, default=Path("outputs/examples"))
    args = p.parse_args()

    work = Path("outputs/work") / args.id
    analysis = pipeline.analyse(
        args.video,
        work,
        pipeline.sam_tracker(Path(".cache/sam")),
        pipeline.model_classifier(args.backend),
        namer=pipeline.model_namer(args.backend),
    )
    folder = args.out / args.id
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(analysis.clip, folder / "clip.mp4")
    shutil.rmtree(folder / "crops", ignore_errors=True)
    if (work / "crops").exists():
        shutil.copytree(work / "crops", folder / "crops")
    (folder / "result.json").write_text(json.dumps(analysis.to_json()), encoding="utf-8")
    first = min((v.start_s for v in analysis.visits), default=0.0)
    video.still(folder / "clip.mp4", first + 0.5, folder / "poster.jpg")
    meta = {"title": args.title, "credit": args.credit, "license": args.license, "source": args.source}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{folder}: {analysis.record['insects']} insects, {analysis.record['total_visits']} visits")


if __name__ == "__main__":
    main()
