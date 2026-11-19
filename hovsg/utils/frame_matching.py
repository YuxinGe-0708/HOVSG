"""Pair RGB/depth/pose streams by source frame ID, never sorted position."""

from pathlib import Path


def match_frame_paths(*streams, prefixes=None):
    prefixes = prefixes or [""] * len(streams)
    if len(prefixes) != len(streams):
        raise ValueError("Each stream must have a filename prefix")
    indexed = []
    for paths, prefix in zip(streams, prefixes):
        frames = {}
        for path in paths:
            stem = Path(path).stem
            if not stem.startswith(prefix):
                raise ValueError(f"Unexpected frame filename: {path}")
            frame_id = stem[len(prefix):]
            if not frame_id or frame_id in frames:
                raise ValueError(f"Duplicate or empty frame ID: {path}")
            frames[frame_id] = path
        indexed.append(frames)
    if not indexed or not indexed[0]:
        raise ValueError("No RGB-D frames found")
    ids = set(indexed[0])
    if any(set(frames) != ids for frames in indexed[1:]):
        raise ValueError("RGB/depth/pose frame IDs do not match")
    ordered = sorted(ids, key=lambda value: (0, int(value)) if value.isdigit() else (1, value))
    return [tuple(frames[frame_id] for frames in indexed) for frame_id in ordered]
