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

`mainProgram.m` drives ten-fold experiments; `runTraining.m` builds a candidate pool, selects via `rBCS.m` and compares selected versus full-pool accuracy. The separate GA/PSO files are experimental helpers, not a verified unified optimiser API.

`saveResults.m` appends rows to `results.csv` in the current folder and writes the header only when creating the file. Driver-level CSV writes are sequential to avoid concurrent append races. Archive old output before a new configuration; rows do not contain a complete environment/configuration record. Committed CSV files are historical outputs, not freshly validated benchmarks.

## Review notes and reproducibility limits

Duplicate nested retraining was removed and normalisation of constant/zero-mean features was corrected; new results may differ. The optional NCA path uses the held-out test set to choose a threshold; keep `params.NCA = false` until selection is moved to training/validation data. The legacy local `normalize.m` shadows MATLAB’s built-in function. Review degenerate diversity formulas and random-selection termination before large runs. `Main.m` requires adapting its calls to the modified `BPSO`/`MyCost` signatures; `clusteringPSO.m` references an unprovided `fusionPSO`.

The cleanup aligns function/file names, restores visible warnings, makes CSV handles close on errors, and tests constant-feature normalisation plus CSV append behaviour. MATLAB experiments and the new utility tests have **not** been executed in the cleanup environment, which has no MATLAB installation. Static parsing is not a reproduction of the paper.

[Profile and project directory](https://github.com/zohaibjan)
