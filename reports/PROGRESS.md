# Verified project status

Updated 2026-11-22. See AUDIT_REPORT.md for the release checks and PILOT_REPORT.md for the scientific results.

Completed: JAAD preparation with official clip splits; stationary and constant-velocity baselines; nine main GRU runs and six feature ablations; CPU checkpoint inference; YOLO pose and ByteTrack video integration; Streamlit workbench; 38 passing tests. All 15 saved models were independently reloaded and their scores and input hashes verified.

A native-library loading restriction observed on 2026-11-02 no longer reproduces in the current environment. No system security policy was changed. Saved-results fallback remains available if dependencies cannot load.

Outstanding scientific work: cross-dataset testing, geographic grouping, calibration, cluster-bootstrap intervals, stronger scene and interaction features, pose inputs, current crossing-state classification, tracking accuracy, occlusion robustness, and GPU validation. The pilot is not a deployment validation or a new state-of-the-art claim.
