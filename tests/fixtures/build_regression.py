"""Builds the regression scenario: 20 s at 15 fps, 3 flowers, 5 tracked objects.

    uv run python tests/fixtures/build_regression.py

Expected behaviour, checked by test_regression.py:
- insect 1 (honeybee) rests on flower 10 for 1 s with a short tracking dropout, then on flower 11 for 0.8 s
- insect 2 (hoverfly) hovers fast around flower 12, then lands for 0.4 s
- insect 3 touches flower 10 for 0.2 s only
- object 4 is debris lying on flower 10, rejected by the classifier
- insect 5 flies across the patch without landing
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from samples import sam_frame, sam_record  # noqa: E402

FPS, FRAMES, W, H = 15, 300, 640, 360
FLOWERS = {10: (100, 100, 199, 199), 11: (300, 120, 399, 219), 12: (480, 80, 579, 179)}


def on(flower: int, size: int = 16, dx: int = 0) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = FLOWERS[flower]
    cx, cy = (x1 + x2) // 2 + dx, (y1 + y2) // 2
    return (cx - size // 2, cy - size // 2, cx + size // 2 - 1, cy + size // 2 - 1)


def insects() -> dict[int, dict[int, tuple[int, int, int, int]]]:
    tracks: dict[int, dict] = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
    for f in [*range(30, 40), *range(42, 45)]:
        tracks[1][f] = on(10)
    for f in range(45, 60):
        tracks[1][f] = on(10)
    for f in range(80, 92):
        tracks[1][f] = on(11)
    for i, f in enumerate(range(120, 150)):
        tracks[2][f] = on(12, dx=(-30 if i % 2 else 30))
    for f in range(150, 156):
        tracks[2][f] = on(12)
    for f in range(200, 203):
        tracks[3][f] = on(10, dx=20)
    for f in range(0, FRAMES, 3):
        tracks[4][f] = on(10, size=10, dx=-30)
    for i, f in enumerate(range(240, 270)):
        x = 20 * i
        tracks[5][f] = (x, 150, x + 15, 165)
    return tracks


def main() -> None:
    by_frame: dict[int, list[str]] = {}
    for oid, boxes in insects().items():
        for f, box in boxes.items():
            by_frame.setdefault(f, []).append(sam_record(oid, *box, w=W, h=H))
    (HERE / "regression_insects.txt").write_text(
        "".join(sam_frame(f, *by_frame[f]) for f in sorted(by_frame)), encoding="utf-8"
    )
    (HERE / "regression_flowers.txt").write_text(
        "".join(
            sam_frame(f, *(sam_record(oid, *box, w=W, h=H) for oid, box in FLOWERS.items())) for f in range(0, FRAMES)
        ),
        encoding="utf-8",
    )
    groups = {
        "1": {"group": "honeybees", "confidence": 0.9},
        "2": {"group": "hoverflies", "confidence": 0.7},
        "3": {"group": "solitary bees", "confidence": 0.5},
        "4": {"group": "not an insect", "confidence": 0.8},
        "5": {"group": "butterflies & moths", "confidence": 0.6},
    }
    (HERE / "regression_groups.json").write_text(json.dumps(groups, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
