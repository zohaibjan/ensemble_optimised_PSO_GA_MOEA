# Ensemble Optimisation Research Experiments

MATLAB research experiments combining classifier ensembles, optimisation and diversity measures.

## Research reference and attribution

Some optimisation files retain original third-party author and paper notices. `Main.m`, `BPSO.m`, `MyCost.m` and `DrawConvergenceCurves.m` credit Seyedali Mirjalili and the S-shaped/V-shaped binary PSO work. Preserve those notices and review the applicable terms before reuse.

Please cite the relevant publication and retain source author notices. Existing dataset and third-party terms still apply; this cleanup does not grant a new licence.

## Data and environment

Supply `P-Data/<name>.mat` for names in `dataSetNames.m`. Each MAT file needs numeric `X` (samples × features) and column labels `y`. No datasets are bundled in this checkout.

**MATLAB dependencies:** Statistics and Machine Learning Toolbox; Deep Learning Toolbox; Global Optimization Toolbox for optional PSO/GA experiments.

No exact original MATLAB/toolbox version is recorded. Use MATLAB rather than assuming GNU Octave compatibility for the learner APIs.

## Start here

1. Open this repository as the MATLAB current folder. Use a separate session/path for each research repository: helper filenames overlap across projects.
2. Supply or extract the data described above. Check feature/label row counts and class labels.
3. Edit `Problem` in `mainProgram.m` (or `dataSetNames.m` where used), and start with one dataset and one repeat.
4. Run the synthetic utility checks, then inspect `mainProgram.m` before starting the full experiment:

```matlab
results = runtests('tests');
assertSuccess(results);
rng(42, 'twister');
mainProgram;
```

The seed is a starting point for recording runs; nested seed resets, parallel workers and toolbox versions can still affect reproducibility. `runMultiple.m` performs 50 full driver repetitions and is not the quick-start command.

## Workflow and output

`mainProgram.m` drives ten-fold experiments; `runTraining.m` builds a candidate pool, selects via `rBCS.m` and compares selected versus full-pool accuracy. The separate GA/PSO files now have bounded synthetic runtime tests; they remain research helpers rather than a unified optimiser API.

`saveResults.m` appends rows to `results.csv` in the current folder and writes the header only when creating the file. Driver-level CSV writes are sequential to avoid concurrent append races. Archive old output before a new configuration; rows do not contain a complete environment/configuration record. Committed CSV files are historical outputs, not freshly validated benchmarks.

## Review notes and reproducibility limits

Prediction, training-label mapping and random-selection retry handling match the corrections described in `docs/code-review.md`. Strict-accuracy selection remains the active policy; ties do not introduce a new diversity policy. Empty PSO masks have a deterministic fallback and its scoring/return threshold is consistent. The legacy NCA path used held-out test loss to choose a threshold; it is now disabled with a clear error. Keep `params.NCA = false` until a training/validation-only replacement is implemented. Local `normalize.m` still shadows MATLAB's built-in name. Third-party optimizer examples and degenerate diversity formulas remain experimental.

## Runtime follow-up — October 2026

GA uses the same >0.5 nonempty mask during scoring and return. Multi-objective GA now measures pairwise double fault on selected models, returns one score-aligned ensemble, and exposes the full Pareto masks/scores separately. BPSO uses a vector for personal-best scores. Feature PSO uses current solver option names, a nonempty feature mask, reproducible learner initialization and validation-only scoring. Original third-party author notices remain.

## Correctness review — October 2026

This pass fixes implementation defects; it does not validate the associated research results. Changes can alter model pools, predictions and measured scores. Re-run the experiments before comparing corrected code with historical CSVs or published results.

- Label conversion uses numeric category names, not categorical indices.
- Voting rejects empty/nonfinite inputs; a prediction failure is reported rather than counted as a fabricated class-one vote.
- Neural models store their training labels in `net.userdata.classLabels`. Retrain old models or restore that exact training vocabulary before prediction; inferring it from a validation subset is unsafe.
- Synthetic checks cover normalization, CSV append behavior and the new prediction/selection boundaries. Additional cluster checks run where applicable.

See [the code review](docs/code-review.md) for repository-specific changes, tests and remaining limitations.

**Validation:** 27 regression/smoke tests passed in MATLAB R2024b on Linux through [GitHub Actions](https://github.com/zohaibjan/ensemble_optimised_PSO_GA_MOEA/actions/workflows/matlab-tests.yml). Tests include real toolbox execution on synthetic data. Full research datasets and published accuracy results have not been reproduced. See [validation evidence and limits](docs/validation.md).

[Profile and project directory](https://github.com/zohaibjan)
