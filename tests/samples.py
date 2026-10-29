"""Synthetic SAM outputs and tracks for tests, in the exact wire format."""

from meta_sam_parser._mask_codec import _encode_raster

from pollicount.geometry import Box, Track


def sam_record(object_id: int, x1: int, y1: int, x2: int, y2: int, w: int = 640, h: int = 360, c: float = 0.9) -> str:
    mh, mw = y2 - y1 + 1, x2 - x1 + 1
    payload = _encode_raster([1] * (mw * mh), mw, mh)
    return (
        f"{object_id}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={w};h={h};c={c}|>"
        f"<|mask;x=0;y=0;c={c};data={mh},{mw},{payload}|>"
    )


def sam_frame(frame: int, *records: str) -> str:
    return f"<{frame}f>" + ",".join(records) + "\n"


def track(object_id: str, concept: str, boxes: dict[int, tuple[float, float, float, float]]) -> Track:
    return Track(object_id, concept, {f: Box(*b) for f, b in boxes.items()})
