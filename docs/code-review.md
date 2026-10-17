# Code review and validation scope

Prediction, training-label mapping and random-selection retry handling match the corrections described in `docs/code-review.md`. Strict-accuracy selection remains the active policy; ties do not introduce a new diversity policy. Empty PSO masks have a deterministic fallback and its scoring/return threshold is consistent. The legacy NCA path used held-out test loss to choose a threshold; it is now disabled with a clear error. Keep `params.NCA = false` until a training/validation-only replacement is implemented. Local `normalize.m` still shadows MATLAB's built-in name. Third-party optimizer examples and degenerate diversity formulas remain experimental.

## Changed behavior and acceptance

The fixes preserve original source attribution and the documented experiment entry points. They change previously erroneous edge cases and may change reported results. Stored outputs were not rewritten or relabelled as new results.

Run in a clean MATLAB session with only this repository on the path:

```matlab
results = runtests('tests');
assertSuccess(results);
```

The test fixtures are synthetic. `FixedClassifier`, where supplied, implements `predict` without fitting a model and is on the path only for the test session. Majority ties choose the smallest numeric label. Empty or invalid votes fail explicitly. ANN vocabularies are learned from training labels, not held-out labels. PSO thresholds retain the repository's setting, with one shared nonempty mask policy for objective scoring and final return. Random selection, where used, has at most pool-size times elimination-budget attempts.

## Executed MATLAB validation

Actual SVM/tree and ANN training, particleswarm, ga, gamultiobj, all eight binary PSO variants, feature selection with fitted decision trees, and core utility regressions.

GA uses the same >0.5 nonempty mask during scoring and return. Multi-objective GA now measures pairwise double fault on selected models, returns one score-aligned ensemble, and exposes the full Pareto masks/scores separately. BPSO uses a vector for personal-best scores. Feature PSO uses current solver option names, a nonempty feature mask, reproducible learner initialization and validation-only scoring. Original third-party author notices remain.

See [validation.md](validation.md) for the exact run, release and boundaries.

## Remaining validation

- Compatibility beyond the tested MATLAB R2024b/toolbox release.
- Full training runs with the documented datasets, preserving training/validation/test separation.
- Numerical quality, random seeds, paper-method agreement, runtime and memory consumption.
- Optional historical optimizer/diversity/alternate-driver files are not a validated library.

The regression/smoke suite passed in MATLAB R2024b using MathWorks GitHub Actions. Synthetic fixtures check implementation behavior; full research experiments, historical accuracy and performance benchmarks have not been reproduced.
