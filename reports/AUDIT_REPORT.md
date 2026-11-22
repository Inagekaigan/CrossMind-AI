# CrossMind AI release audit

Date: 2026-11-22. Verdict: functioning research pilot with reproducible saved results; predictive quality remains moderate and uncalibrated.

## Verification performed

- Full local suite: 38 passed in 10.62 seconds, including exact resume behavior, gradients, metrics, split validation, timestamp gaps, native annotation conversion, trained-model UI inference and recorded-demo loading. Upstream Streamlit/Pandas emits 38 non-fatal deprecation warnings.
- Reloaded all nine main and six ablation checkpoints. Recomputed both validation and test scores and all baselines; every value matches the saved results within relative tolerance 1e-5 and absolute tolerance 1e-6.
- Verified checkpoint SHA-256, train/validation/test archive hashes, and model normalization against training features only. Machine-readable evidence: model_audit.json. Historical source hashes describe training-time code; input-validation fixes have since changed source files without changing valid-input model scores.
- Fixed rejection of nonfinite/misshaped observations and neighbor inputs, unknown feature groups, and zero-epoch training. Video inference now reads the checkpoint's neighbor radius instead of hard-coding it.
- Earlier video evidence: 150 real JAAD frames, three track IDs, 173 forecasts, 8.23 processing FPS on CPU. This audit exercises the recorded-demo UI; it does not remeasure detection recall or tracking accuracy.

## What the model quality supports

Joint crossing-initiation F1 is 0.5758, ROC-AUC 0.7028 and Brier score 0.2183 (three-seed means). These are useful pilot signals, not strong assurance of real-world intention recognition. Joint highest-scored trajectory ADE is 0.04129 against constant velocity 0.04817 in normalized-image units, approximately 14.3% lower error on this cohort. Best-of-three scores must not be compared directly with deterministic forecasts.

Joint training does not consistently improve both tasks. Position-plus-velocity ablation gives higher intention ROC-AUC (0.7460) than adding the current neighbor features. No external validation, significance claim, calibrated risk estimate, world-coordinate accuracy or driving-safety claim is supported. See PILOT_REPORT.md for the complete tables and limitations.

## Public package and privacy

Publication uses an explicit file allowlist. Personal filesystem paths and environment-specific account names were removed from documentation. Raw data, video frames, uploaded videos, trained binary weights, cache/log files, unrelated workspace documents and prior local history are excluded. Commit metadata uses a generic contributor identity. The hosting account's public GitHub username remains visible as repository owner.

The package contains code, synthetic test fixtures, official split identifiers, measured experiment JSON, a research figure, research plan, data card, pilot report, audit evidence and reproduction scripts. Users obtain third-party assets from their documented sources under their respective terms.

## Reproduce checks

After preparing data and training as described in README.md:

```text
python -m pytest -q
python scripts/audit_models.py
```

A source-only checkout skips the local trained-demo integration test until its checkpoint is generated. The checkpoint audit intentionally requires all 15 runs and fails rather than inventing missing results.
