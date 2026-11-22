"""Evaluate baselines on explicit, complete, regularly sampled CSV windows."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
from .data import load_tracks, load_splits
from .metrics import trajectory_metrics
from .trajectory import predict


def evaluate(csv_path, split_path, config):
    tracks = load_tracks(csv_path)
    splits = load_splits(split_path, {k[0] for k in tracks})
    hz = config["sampling_hz"]
    if not np.isfinite(hz) or hz <= 0:
        raise ValueError("Positive finite sampling rate required")
    counts = [config[k] * hz for k in ("observation_seconds", "prediction_seconds")]
    if any(not np.isfinite(v) or v < 1 or abs(v - round(v)) > 1e-8 for v in counts):
        raise ValueError("Horizons must be positive integer multiples of sample interval")
    # Observation duration includes both endpoints: 3 seconds at 10 Hz = 31 samples.
    obs, future = int(round(counts[0])) + 1, int(round(counts[1]))
    stride = config["stride_samples"]
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride_samples must be a positive integer")
    if config["split"] not in splits:
        raise ValueError("Unknown split")
    if config["provenance"] not in {"synthetic_fixture", "dataset"}:
        raise ValueError("Explicit provenance required")
    rows, skipped = [], 0
    for (scene, track, units), data in sorted(tracks.items()):
        if scene not in splits[config["split"]]:
            continue
        if units != config["units"]:
            raise ValueError("Configuration units do not match data")
        for start in range(0, len(data) - obs - future + 1, stride):
            window = data[start:start + obs + future]
            if not np.allclose(np.diff(window[:, 0]), 1 / hz, atol=1e-6, rtol=0):
                skipped += 1
                continue
            prediction = predict(window[:obs, 1:], window[:obs, 0], window[obs:, 0],
                                 config["method"], config["max_gap_seconds"])
            metrics = trajectory_metrics(prediction, window[obs:, 1:], config["miss_threshold"])
            rows.append({"scene": scene, "track": track, "anchor": float(window[obs-1, 0]), **metrics})
    if not rows:
        raise ValueError("No eligible windows; check lengths, sampling and split")
    return {"status": "synthetic verification only" if config["provenance"] == "synthetic_fixture" else "dataset evaluation",
            "config": config, "window_count": len(rows), "skipped_gap_windows": skipped,
            "metrics": {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in {"scene", "track", "anchor"}},
            "per_window": rows, "python": platform.python_version(), "numpy": np.__version__,
            "input_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
            "split_sha256": hashlib.sha256(Path(split_path).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = evaluate(args.data, args.splits, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "window_count", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
