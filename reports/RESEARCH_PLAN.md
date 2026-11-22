# Research protocol — 2026-10-31

## Question and minimum viable research system

Does a shared temporal encoder improve crossing-initiation classification or
trajectory prediction over equally budgeted independent encoders? Positive,
negative and inconclusive findings are all valid. Shared heads are not a novelty
claim. Minimum research system: one audited PIE subset, disjoint source groups,
stationary/CV and temporal GRU baselines, independent and shared models, identical
features/horizons, three seeds, calibration and trajectory evaluation. A working
video interface alone does not satisfy this milestone.

## Initial literature review

1. Rasouli et al., *PIE: A Large-Scale Dataset and Models for Pedestrian Intention
   Estimation and Trajectory Prediction*, ICCV 2019.
   https://openaccess.thecvf.com/content_ICCV_2019/papers/Rasouli_PIE_A_Large-Scale_Dataset_and_Models_for_Pedestrian_Intention_Estimation_ICCV_2019_paper.pdf
   Motivates behavior and trajectory evaluation on driving footage. Its annotated
   intention score must not be substituted for future crossing onset.
2. Rasouli et al., *Bifold and Semantic Reasoning for Pedestrian Behavior
   Prediction*, ICCV 2021 (BiPed).
   https://arxiv.org/abs/2012.03298
   Direct prior work on joint action/trajectory prediction; compare input access,
   ego-motion and task definitions before interpreting any score differences.
3. Alahi et al., *Social LSTM: Human Trajectory Prediction in Crowded Spaces*,
   CVPR 2016.
   https://openaccess.thecvf.com/content_cvpr_2016/html/Alahi_Social_LSTM_Human_CVPR_2016_paper.html
   Interaction-aware temporal prediction is established prior work. Social pooling
   is an appropriate later baseline, not a new contribution.
4. Neumann and Vedaldi, *Pedestrian and Ego-Vehicle Trajectory Prediction From
   Monocular Camera*, CVPR 2021.
   https://openaccess.thecvf.com/content/CVPR2021/html/Neumann_Pedestrian_and_Ego-Vehicle_Trajectory_Prediction_From_Monocular_Camera_CVPR_2021_paper.html
   Ego-motion confounds image-space motion; metric conclusions need calibration.

This is an initial targeted review, not a completed systematic review or a
reproduction of these methods. No published score is presented as ours.

## Data audit and acquisition

| Dataset | Intended role | Verified access information | Remaining audit |
|---|---|---|---|
| PIE | Primary joint experiments | Official code/annotations repository and video download links; repository MIT; video collection about 74 GB | Inspect exact per-frame crossing semantics, split version, video terms and camera conventions before ingestion |
| JAAD 2.0 | Crossing generalization | Official XML annotations, predefined split IDs; behavior labels only for a subset; repository MIT and videos CC BY 4.0; about 3.1 GB | Restrict onset labels to annotated tracks; inspect state values and scene grouping |
| ETH/UCY | Separate trajectory benchmark | Available loader documentation describes common trajectory files | Verify original distribution terms, source provenance, sampling and world coordinates before downloading; no crossing labels assumed |
| SDD | Separate interaction benchmark | Official site: eight scenes, multiple agent classes, about 69 GB, CC BY-NC-SA 3.0 | Filter pedestrians and lost annotations; preserve image units unless calibrated; no crossing labels assumed |

Sources checked 2026-10-31:
https://github.com/aras62/PIE
https://github.com/aras62/PIE/blob/master/annotations/README.md
https://github.com/ykotseruba/JAAD
https://github.com/NVlabs/trajdata/blob/main/DATASETS.md
https://cvgl.stanford.edu/projects/uav_data/

No datasets were downloaded. Access links do not establish that every download
currently succeeds. Log terms, attribution, source revision, file SHA256, FPS,
coordinate origin and annotation exclusions in each local dataset card. Keep
videos/checkpoints outside version control. Missing pose/scene inputs require
explicit availability masks; never synthesize ground truth.

## Task formulation

At last observation time t, for a person annotated non-crossing, label y=1 when
the first subsequent annotated crossing transition falls in (t,t+2 s]. Label
y=0 only with full annotated follow-up through t+2 s and no such transition.
Already-crossing persons form a separate descriptive state, not positive onset
examples. Censored/missing/gapped follow-up is excluded from both classes.
Onset precision is bounded by annotation sampling. An onset between frames is
not recoverable exactly. Track-level intention scores are not onset labels.
The implemented helper operates on mapped binary crossing states; native mapping
still needs dataset verification. Model inputs use only observations through t.

Start with 3 s observation and 5 s prediction, configurable at dataset-specific
rates. Report actual elapsed observation span and future sample timestamps.
Coordinates must retain declared units. Estimate velocities with real timestamps.
Never interpret uncalibrated image displacement as ground-plane speed.

## Planned learned methods

A: temporal GRU with intention head and binary cross entropy.
B: same encoder budget with K-mode trajectory head.
C: shared GRU with both heads; optionally condition trajectory head on predicted
intention (never the ground-truth future class). Use Gaussian mixture trajectory
likelihood with log-sum-exp and bounded positive scales; record whether modes
are complete-trajectory components or per-step mixtures. Select loss weighting
using validation data. Initially K=5; report K identically for comparisons.
Fit normalization only on training observations. Pose, scene and relative-neighbor
features enter with masks. Neighbor construction uses only current/past frames.

## Evaluation protocol

- Fix train/validation/test by source recording before window generation. Respect
  official splits and separately audit shared scenes; publish deviations. Test
  scene generalization with held-out scenes and no tuning on their labels.
- Seeds 17, 29, 43; record environment, data hashes, split hashes, checkpoints,
  code version, configuration and all exclusions. Resume optimizer, scheduler,
  scaler and RNG states, not merely weights. Early stop on validation loss.
- Report class counts and prior/majority intention baselines. Tune class weights
  on train, threshold and calibration on validation, then freeze for test.
- Intention: accuracy, precision, recall, F1, ROC-AUC, average precision (explicit
  PR summary convention), confusion matrix, Brier, 10-bin ECE. Undefined
  single-class metrics are null with a reason, never silently zero.
- Trajectory: ADE/FDE for designated mode, minADE/minFDE at fixed K, miss rate
  with explicit threshold and units. NLL/coverage only for genuine distributions.
  minADE and minFDE may select different modes and do not measure calibration.
- Compare observations 1/2/3 s; lead times 0.5/1/2 s using complete at-risk
  examples; pose, velocity, scene, interactions and all-features ablations;
  controlled 10/30/50% observation removal with a fixed corruption seed.
- Report all seed results plus mean/std and recording-level bootstrap intervals.
  Overlapping windows are correlated; never bootstrap them as independent people.
  Hold dataset, sample eligibility and input availability constant for comparisons.

## Hardware and dependencies

Observed: Windows, Anaconda Python 3.13.9, NumPy/pytest/scikit-learn available;
NVIDIA RTX 3070-class GPU with 8 GB VRAM. PyTorch and OpenCV are absent in the
inspected interpreter; CUDA execution has not been validated. CPU baseline needs
only NumPy. Future training should use an isolated environment and a compatible
official PyTorch build. A small GRU is the initial budget; memory and runtime
must be measured, not promised. Cache features instead of repeatedly decoding
all videos; storage requirements depend on extraction policy.

Planned pretrained dependencies: a compact Ultralytics YOLO person detector with
ByteTrack, and a YOLO pose model. Verify current API/weight license before
installation, pin package/weight revisions and hashes, and document training
sources. No weights are installed. Scene annotations are preferred initially;
semantic segmentation is optional, not silently assumed available.

## Phases and acceptance gates

1. Protocol + baseline: validated timestamp/coordinate contracts, scene manifest,
   onset labels, hand-computed metric tests and reproducible fixture run.
2. Data + video: audited native annotations, permitted video, detector/tracker,
   pose alignment, timing and interruption handling; visually inspect real output.
3. Intention: training prior/logistic/GRU baselines, saved/resumable checkpoints,
   held-out classification and calibration report.
4. Trajectory: CV vs trained GRU mixture, genuine K paths, evaluation in one
   coordinate protocol, observed-only interaction features.
5. Joint: same-data independent/shared models, multiple seeds, ablations and
   documented null or negative outcomes.
6. Application: Streamlit upload, model selector, track inspection, dashed
   forecasts, latency; explicitly unavailable probabilities without a checkpoint.
7. Report + demo: paper sections, actual tables/figures, attributable permitted
   demo footage and fresh-environment reproduction. Final completion requires
   every deliverable in the project brief; phase 1 alone is not completion.
