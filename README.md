# CrossMind AI

Pedestrian crossing-initiation and multimodal trajectory prediction research.
**Current milestone: a working research pilot with trained models and video analysis.**
Three GRU variants were trained on JAAD with three seeds each, plus six feature
ablation runs. YOLO pose, ByteTrack and learned forecasts were verified on an
attributed JAAD clip. This is not a finished research validation.

## Open the app

**Verified 2026-11-22:** 38 tests pass locally, including trained-model UI integration. All 15 saved checkpoints reproduce their recorded validation/test scores. See [audit report](reports/AUDIT_REPORT.md).

Visit http://127.0.0.1:8501 while the local server is running. To restart it:

```powershell
.\scripts\launch_app.ps1
```

Use **Track explorer** for held-out annotated tracks, **Video analysis** for video
upload, and **Research comparison** for measured experiments. Select a trained
model in the sidebar; Constant velocity has no intention scores. Onset scores are
uncalibrated and conditional on the pedestrian not currently crossing. Current
crossing state is explicitly unavailable.

## Reproduce the learned pilot

The verified interpreter is `.venv/Scripts/python.exe` (Python 3.13.9, CPU PyTorch).
For a fresh environment, create a Python 3.13 virtual environment without system
site packages. The lock file records the original Windows CPU environment;
for a portable installation use the project extras:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[research,app,video,test]"
```

The public repository includes source, reports and measured JSON results.
Dataset annotations, processed arrays, trained checkpoints and video are generated
locally by the commands below and are not bundled. The app can display research
results before these downloads; learned predictions require training first.

```powershell
$env:PYTHONPATH = "$PWD\src"
$env:TEMP = "$PWD\.tmp"
$env:TMP = $env:TEMP
$env:MPLCONFIGDIR = "$PWD\.cache\matplotlib"
$env:YOLO_CONFIG_DIR = "$PWD\.cache\ultralytics"
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
& .\.venv\Scripts\python.exe scripts/fetch_jaad.py
& .\.venv\Scripts\python.exe -m crossmind.jaad --raw data/raw/jaad --splits data/splits/jaad_default.json --config configs/jaad_pilot.json --output data/processed/jaad_default
& .\.venv\Scripts\python.exe scripts/run_pilot.py
& .\.venv\Scripts\python.exe scripts/run_ablations.py
& .\.venv\Scripts\python.exe -m pytest -q --basetemp "$PWD\.test-tmp-local"
```

All 38 tests passed in the local runner; the interface test requires local
sockets. Upstream Streamlit/Pandas deprecation warnings remain non-fatal.
Training uses 703 / 149 / 562 train/validation/test windows. See
`reports/DATA_CARD.md` for selection, exclusions and semantics.
Artifacts are in `experiments/jaad_default` and `experiments/ablations`.
Resume via `python -m crossmind.train --data <processed-dir> --output <run-dir>
--task joint --seed 17 --epochs 30 --resume <run-dir>/last.pt`.
An exact resumed-versus-uninterrupted training test passes.

## Reproduce the real-video demo

```powershell
& .\.venv\Scripts\python.exe scripts/fetch_demo.py
& .\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('assets/yolo11n-pose.pt')"
& .\.venv\Scripts\python.exe -m crossmind.video --source data/raw/demo/video_0005.mp4 --output runs/joint_demo --weights assets/yolo11n-pose.pt --checkpoint experiments/jaad_default/joint_17/best.pt --max-frames 150
```

Output: `runs/joint_demo/annotated.mp4` and `predictions.json`. The measured run
processed 150 frames, tracked three IDs and emitted 173 predictions with three
modes each at about 8.23 processing FPS on CPU. This measures one-clip throughput,
not tracking accuracy. Small/occluded pedestrians can be missed. Constant-frame-rate
video is required. The demo is attributed to Rasouli, Kotseruba and Tsotsos, JAAD,
ICCVW 2017; its source and license are recorded in `data/raw/demo/source.json`.

The pretrained YOLO11n-pose weights came from the official Ultralytics assets
v8.4.0 release; ByteTrack uses `bytetrack.yaml`. Pose is displayed but not consumed
by prediction heads. Ultralytics dependencies retain their AGPL-3.0/enterprise
terms. Dataset/media/weights retain their own licenses; no blanket project license
is granted over third-party assets.

## Research status

See `reports/PILOT_REPORT.md` and `reports/PROGRESS.md`. Joint learning has mixed
results; this pilot establishes neither novelty nor statistical significance.
PIE integration, longer histories/horizons, pose/scene ablations, geographic
generalization, occlusion stress tests, calibration, cluster-bootstrap uncertainty
and GPU validation remain outstanding.

## Run the verified baseline

Use Python 3.10+ with NumPy and pytest. From this directory:

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
crossmind-baseline --data tests/fixtures/tracks.csv --splits tests/fixtures/splits.json --config configs/baseline_fixture.json --output experiments/baseline_fixture.json
```

If the environment cannot create temporary test files, append
`--basetemp "$PWD\.test-tmp-local"` to pytest. Reserve that directory for pytest;
pytest clears its contents on subsequent runs.

The fixture is explicitly synthetic: one linear track and one abruptly stopping
track. It tests arithmetic and plumbing; it is not evidence of predictive quality.
The JSON output includes configuration, per-window metrics, input/split hashes,
package versions and skipped-window counts. Stationary and least-squares constant
velocity are supported. They return one deterministic path without probabilities.

## Evaluate permitted tracks

Convert annotations into CSV columns `scene_id,track_id,timestamp,x,y,units`.
Timestamps are seconds, units are `pixels`, `normalized_image`, or `meters`.
Record annotation version, conversion, coordinate reference and permission in a
dataset card before calling a result a benchmark. JAAD native conversion is
implemented in `crossmind.jaad`. No camera-coordinate conversion is implicit.
For image tracks, use bounding-box bottom center consistently. For metric tracks,
use the dataset's documented ground-plane coordinates. Do not combine the two.

Supply a JSON manifest with disjoint `train`, `validation`, `test` scene lists.
Use stable scene IDs shared across clips of the same scene. The validator cannot
detect renamed scenes; audit grouping against source recordings. The evaluator
uses complete regular windows only, skipping gaps without interpolation. Copy
the fixture config, set actual units, sampling, horizons, threshold and
`provenance: "dataset"`; this flag is a declaration, not automatic authenticity verification.

Observation duration spans first to last sample (31 observations for 3 s at
10 Hz); prediction starts one step after the last observation (50 predictions
for 5 s). Nonintegral horizon/sample counts are rejected. Aggregate errors are
window-weighted. Mode-0 ADE/FDE and best-of-K metrics are distinct; K=1 here.
Miss rate means minimum endpoint distance strictly exceeds the configured
threshold. NLL and coverage are unavailable for these deterministic baselines.

## Project map

- `src/crossmind`: validated tracks, labels, baselines, metrics, command line.
- `tests`: core and end-to-end correctness tests; explicit synthetic fixtures.
- `configs`: reproducible run settings.
- `experiments`: measured verification outputs.
- `reports/RESEARCH_PLAN.md`: literature, data, protocol, phased acceptance gates.
- `reports/PROGRESS.md`: verified work, blockers, next steps.

No facial identification is performed. Predictions are statistical estimates;
the eventual application is research software, not validated driving software.
Dataset and pretrained-weight licenses remain separate from project code.
