# JAAD onset-cohort pilot

Source: https://github.com/ykotseruba/JAAD
Pinned revision: `d2feeac998366e47ecc54e5533c8a3fa61141dda`.
Acquisition is reproducible with `scripts/fetch_jaad.py`. SHA256 for each local
annotation is in `data/raw/jaad/acquisition.json`; processed inputs and checkpoints
carry additional hashes. Source content transferred through the GitHub connector
may have its final newline normalized; local hashes describe exact used bytes.

The official default lists contain 177 training, 29 validation and 117 test clips.
These are clip-disjoint, not a verified source-route or geographic scene split.
Do not call this a scene-generalization experiment. The first small acquisition
was expanded before any model evaluation because it had no positive test onset
windows. No clips were selected or removed based on model scores.

JAAD documentation states 30 FPS:
https://data.nvision2.eecs.yorku.ca/JAAD_dataset/
Annotations are MIT licensed; a copy remains with raw data. The current JAAD 2.0
README identifies video clips as CC BY 4.0; the older dataset webpage refers to MIT.
The demo carries attribution to Rasouli, Kotseruba and Tsotsos, JAAD, ICCVW 2017.
No media or weights should be committed to the source repository.

## Conversion and eligibility

- Read individual `pedestrian` and `ped` tracks; exclude group boxes (`people`).
- Position = bounding-box bottom-center, divided by image width/height.
  Units are normalized image coordinates. Pixel aspect scaling and ego-motion
  remain; distances are not meters and are not calibrated ground-plane motion.
- `cross=not-crossing` maps to 0; `cross=crossing` maps to 1; absent, irrelevant
  and other values are unavailable. Never use track-level crossing attributes
  as future initiation labels.
- Retain 30 FPS states for onset labeling. Sample model coordinates every third
  frame (10 Hz). Use 1 s observed span (11 samples), 2 s forecast (20 samples)
  and a separate 2 s onset horizon. Anchor stride is five samples (0.5 s).
- Require complete sampled coordinate history/future and no full occlusion at
  sampled points. Partial occlusion remains. Require complete dense crossing
  states from the anchor through the onset horizon, and a non-crossing anchor.
- All learned models and deterministic comparisons use the same eligible cohort.
  This favors long visible non-crossing prefixes; it is not representative of all
  pedestrians and not the published JAAD action-classification protocol.
- Interactions use observed neighbors within 0.2 normalized-image Euclidean units:
  mean relative offset and count. Missing neighbors produce zeros. There is no
  pose input, vehicle model, scene segmentation or future neighbor information.

Prepared counts: train 703 windows (295 positive), validation 149 (47 positive),
test 562 (201 positive). Windows overlap and are not independent people.
There are respectively 154, 24 and 135 eligible track IDs. All source lists and
exclusion counts are retained in processed `manifest.json`.

## Interpretation

Training observes annotation-quality tracks; video inference observes noisy model
tracks. That distribution shift has not been quantified. Onset probabilities are
conditional on a non-crossing current state, which the video pipeline does not
classify. The UI explicitly reports current crossing state as unavailable.
Mixture weights and classification outputs are uncalibrated model scores; no
reliability or safety claim is made. Models currently use CPU training and inference.
