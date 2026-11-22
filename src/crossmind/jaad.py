"""JAAD 2.0 native XML adapter. No image downloads or identity inference."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .labels import crossing_initiation_label
from .data import load_splits


def read_annotation(path):
    root = ET.parse(path).getroot()
    size = root.find("./meta/task/original_size")
    width, height = float(size.findtext("width")), float(size.findtext("height"))
    if min(width, height) <= 0:
        raise ValueError("Invalid frame dimensions")
    tracks = []
    for track in root.findall("track"):
        if track.get("label") not in {"pedestrian", "ped"}:
            continue
        records = []
        identity = None
        for box in track.findall("box"):
            attrs = {a.get("name"): a.text for a in box.findall("attribute")}
            if box.get("outside", "0") == "1":
                continue
            if identity is None:
                identity = attrs.get("id")
            if not identity or attrs.get("id") != identity:
                raise ValueError("Missing or changing track ID")
            x1, y1, x2, y2 = [float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")]
            if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
                raise ValueError("Invalid box")
            state = {"not-crossing": 0., "crossing": 1.}.get(attrs.get("cross"), float("nan"))
            # Occluded boxes are retained for timeline auditing but cannot be model inputs/targets.
            visible = attrs.get("occlusion", "none") != "full"
            records.append([int(box.get("frame")), (x1+x2)/2/width, y2/height, state, float(visible)])
        if records:
            array = np.asarray(sorted(records), float)
            if np.any(np.diff(array[:, 0]) <= 0):
                raise ValueError("Duplicate frame in annotation")
            tracks.append({"id": identity, "values": array})
    return tracks


def features(positions, times, neighbors=None, groups=("position", "velocity", "interaction")):
    """[absolute xy, displacement xy, velocity xy, mean neighbor offset xy, count]."""
    positions = np.asarray(positions, np.float32)
    times = np.asarray(times, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2 or times.shape != (len(positions),):
        raise ValueError("Positions and timestamps must have matching shapes")
    if not np.isfinite(positions).all() or not np.isfinite(times).all():
        raise ValueError("Finite observations required")
    if neighbors is not None:
        neighbors = np.asarray(neighbors, dtype=np.float32)
        if neighbors.shape != (len(positions), 3) or not np.isfinite(neighbors).all():
            raise ValueError("Finite neighbor features with shape (samples, 3) required")
    if set(groups) - {"position", "velocity", "interaction"}:
        raise ValueError("Unknown feature group")
    dt = np.diff(times)
    if len(positions) < 2 or np.any(dt <= 0):
        raise ValueError("Increasing sample timestamps required")
    velocity = np.zeros_like(positions)
    velocity[1:] = np.diff(positions, axis=0) / dt[:, None]
    out = np.zeros((len(positions), 9), np.float32)
    if "position" in groups:
        out[:, :2] = positions
        out[:, 2:4] = positions - positions[-1]
    if "velocity" in groups:
        out[:, 4:6] = velocity
    if "interaction" in groups and neighbors is not None:
        out[:, 6:] = neighbors
    return out


def prepare(raw, splits_path, output, config):
    splits = load_splits(Path(splits_path), set())
    fps, step = config["source_fps"], config["sample_step"]
    if fps <= 0 or not isinstance(step, int) or step < 1:
        raise ValueError("Invalid sampling configuration")
    hz = fps / step
    durations = [config[k]*hz for k in ("observation_seconds", "prediction_seconds")]
    if any(v < 1 or abs(v-round(v)) > 1e-8 for v in durations):
        raise ValueError("Duration must align with sampling")
    obs, future = int(round(durations[0]))+1, int(round(durations[1]))
    summary = {"config": config, "splits_sha256": hashlib.sha256(Path(splits_path).read_bytes()).hexdigest(),
               "coordinate_system": "normalized_image", "sources": {}, "counts": {}}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for split, videos in splits.items():
        xs, ys, labels, positions, keys = [], [], [], [], []
        rejected = 0
        for video in videos:
            path = Path(raw) / "annotations" / f"{video}.xml"
            summary["sources"][video] = hashlib.sha256(path.read_bytes()).hexdigest()
            tracks = read_annotation(path)
            by_frame = {}
            for track in tracks:
                for row in track["values"]:
                    if row[4]:
                        by_frame.setdefault(int(row[0]), []).append((track["id"], row[1:3]))
            for track in tracks:
                full = track["values"]
                sampled = full[(full[:, 0].astype(int) % step) == 0]
                lookup = {int(r[0]): i for i, r in enumerate(full)}
                for start in range(0, len(sampled)-obs-future+1, config["stride_samples"]):
                    window = sampled[start:start+obs+future]
                    if np.any(np.diff(window[:, 0]) != step) or not window[:, 4].all():
                        rejected += 1
                        continue
                    anchor_frame = int(window[obs-1, 0])
                    target = crossing_initiation_label(full[:, 0]/fps, full[:, 3], lookup[anchor_frame],
                                                       config["intention_seconds"], 1.01/fps)
                    # Matched at-risk cohort across all three learned model variants.
                    if target is None:
                        rejected += 1
                        continue
                    observed = window[:obs, 1:3]
                    social = np.zeros((obs, 3), np.float32)
                    for j, row in enumerate(window[:obs]):
                        nearby = [p-row[1:3] for tid, p in by_frame.get(int(row[0]), [])
                                  if tid != track["id"] and np.linalg.norm(p-row[1:3]) <= config["neighbor_radius"]]
                        if nearby:
                            social[j, :2] = np.mean(nearby, axis=0)
                            social[j, 2] = len(nearby)
                    xs.append(features(observed, window[:obs, 0]/fps, social, config["features"]))
                    ys.append(window[obs:, 1:3]-observed[-1])
                    positions.append(observed)
                    labels.append(target)
                    keys.append(f"{video}/{track['id']}/{anchor_frame}")
        if not xs:
            raise ValueError(f"No eligible samples in {split}")
        np.savez_compressed(output/f"{split}.npz", x=np.asarray(xs, np.float32),
                            y=np.asarray(ys, np.float32), label=np.asarray(labels, np.float32),
                            positions=np.asarray(positions, np.float32), keys=np.asarray(keys))
        summary["counts"][split] = {"windows": len(xs), "positive": int(sum(labels)),
                                    "negative": len(labels)-int(sum(labels)), "excluded_windows": rejected,
                                    "clips": len(videos), "tracks": len({k.rsplit('/', 1)[0] for k in keys})}
    (output/"manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.raw, args.splits, args.output,
                            json.loads(args.config.read_text()))["counts"], indent=2))
