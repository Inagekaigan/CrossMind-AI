import json
from pathlib import Path
import numpy as np
import pytest
from crossmind.trajectory import predict
from crossmind.labels import crossing_initiation_label as label
from crossmind.metrics import trajectory_metrics
from crossmind.data import load_tracks, load_splits
from crossmind.cli import evaluate

ROOT = Path(__file__).resolve().parents[1]


def test_irregular_timestamp_velocity():
    t = np.array([0, .2, .5])
    p = np.column_stack([2*t, -3*t])
    result = predict(p, t, [.7, 1.0])
    np.testing.assert_allclose(result[0], [[1.4, -2.1], [2, -3]])


def test_stationary():
    np.testing.assert_allclose(predict([[0, 0], [1, 2]], [0, .5], [1], "stationary"), [[[1, 2]]])


@pytest.mark.parametrize("positions,times,future", [
    ([[0, 0]], [0], [1]),
    ([[0, 0], [1, 1]], [0, 0], [1]),
    ([[0, 0], [1, 1]], [0, 1], [2]),
    ([[0, 0], [np.nan, 1]], [0, .5], [1]),
    ([[0, 0], [1, 1]], [0, .5], [.5]),
    ([[0, 0], [1, 1]], [0, .5], [1, .8]),
])
def test_invalid_history(positions, times, future):
    with pytest.raises(ValueError):
        predict(positions, times, future)


def test_hand_computed_metrics():
    metrics = trajectory_metrics([[[3, 4], [0, 2]], [[0, 0], [0, 3]]], [[0, 0], [0, 0]], 2)
    assert metrics == {"ADE_mode0": 3.5, "FDE_mode0": 2., "minADE": 1.5,
                       "minFDE": 2., "miss": 0., "K": 2}


def test_invalid_metric_shapes():
    with pytest.raises(ValueError):
        trajectory_metrics(np.zeros((2, 3, 2)), np.zeros((2, 2)), 2)


@pytest.mark.parametrize("states,anchor,horizon,expected", [
    ([0, 0, 1, 1], 0, 1, 1),
    ([0, 0, 0, 0], 0, 1, 0),
    ([0, 1, 1, 1], 1, 1, None),
    ([0, 0, 0, 1], 2, 1, None),
    ([0, np.nan, 1, 1], 0, 1, None),
    ([0, 0, 1, 1], 0, .75, 0),
])
def test_onset_labels(states, anchor, horizon, expected):
    assert label([0, .5, 1, 1.5], states, anchor, horizon, .5) == expected


def test_label_gap():
    assert label([0, .5, 1.5], [0, 0, 1], 0, 1, .5) is None


def test_split_leakage(tmp_path):
    p = tmp_path / "splits.json"
    p.write_text(json.dumps({"train": ["a"], "validation": [], "test": ["a"]}))
    with pytest.raises(ValueError, match="leakage"):
        load_splits(p, {"a"})


def test_csv_duplicate(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("scene_id,track_id,timestamp,x,y,units\na,1,0,1,2,meters\na,1,0,2,3,meters\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_tracks(p)


def test_end_to_end_fixture():
    config = json.loads((ROOT / "configs/baseline_fixture.json").read_text())
    result = evaluate(ROOT / "tests/fixtures/tracks.csv", ROOT / "tests/fixtures/splits.json", config)
    assert result["status"] == "synthetic verification only"
    assert result["window_count"] == 2
    assert result["metrics"]["ADE_mode0"] == .625
    assert result["metrics"]["FDE_mode0"] == 1
    config["method"] = "stationary"
    second = evaluate(ROOT / "tests/fixtures/tracks.csv", ROOT / "tests/fixtures/splits.json", config)
    assert second["per_window"][0]["FDE_mode0"] == 2
    assert second["per_window"][1]["FDE_mode0"] == 0


def test_windows_never_bridge_missing_sample(tmp_path):
    source = (ROOT / "tests/fixtures/tracks.csv").read_text()
    p = tmp_path / "missing.csv"
    p.write_text(source.replace("synthetic_linear,1,1.5,1.5,0,meters", "synthetic_linear,1,1.6,1.5,0,meters"))
    config = json.loads((ROOT / "configs/baseline_fixture.json").read_text())
    result = evaluate(p, ROOT / "tests/fixtures/splits.json", config)
    assert result["window_count"] == 1
    assert result["skipped_gap_windows"] == 1
