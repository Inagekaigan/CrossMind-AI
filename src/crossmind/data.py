"""Validated portable track input and group-disjoint split manifests."""
import csv
import json
from collections import defaultdict
import numpy as np

UNITS = {"pixels", "normalized_image", "meters"}


def load_tracks(path):
    groups = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"scene_id", "track_id", "timestamp", "x", "y", "units"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain {sorted(required)}")
        for row in reader:
            if not row["scene_id"] or not row["track_id"] or row["units"] not in UNITS:
                raise ValueError("Nonempty identifiers and recognized units required")
            values = [float(row[k]) for k in ("timestamp", "x", "y")]
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite CSV values")
            groups[(row["scene_id"], row["track_id"], row["units"])].append(values)
    if not groups:
        raise ValueError("No tracks found")
    if len({key[2] for key in groups}) != 1:
        raise ValueError("Do not aggregate incompatible coordinate units")
    output = {}
    for key, rows in groups.items():
        array = np.asarray(sorted(rows), float)
        if np.any(np.diff(array[:, 0]) <= 0):
            raise ValueError(f"Duplicate timestamp in {key}")
        output[key] = array
    return output


def load_splits(path, scenes):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if set(manifest) != {"train", "validation", "test"}:
        raise ValueError("Manifest requires train, validation, test scene lists")
    if any(not isinstance(v, list) or any(not isinstance(s, str) for s in v) for v in manifest.values()):
        raise ValueError("Split entries must be lists of scene IDs")
    flat = [s for values in manifest.values() for s in values]
    if len(flat) != len(set(flat)):
        raise ValueError("Scene leakage or duplicate scene in split manifest")
    if not set(scenes).issubset(flat):
        raise ValueError("Input contains scenes missing from split manifest")
    return manifest
