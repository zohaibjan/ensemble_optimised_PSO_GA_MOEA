# Ensemble Optimisation Research Experiments

MATLAB experiments combining classifier ensembles with optimisation and diversity measures.

## Code map

| Files | Purpose |
|---|---|
| `mainProgram.m`, `runTraining.m` | Ensemble experiment workflow |
| `runGA.m`, `runMultGA.m`, `BPSO.m` | Optimisation experiments |
| `Correlation.m`, `Disagreement.m`, `Double_Fault.m`, `Q_Test.m` | Diversity measures |
| `ensembleSelection.m`, `fusion.m` | Selection and prediction combination |
| `saveResults.m` | Result export |

Review dataset paths, learner settings and required MATLAB toolboxes before running any experiment. This repository does not provide a complete pinned environment or a verified one-command reproduction.

## Attribution

Some optimisation files retain original third-party author and paper notices. For example, `Main.m` credits Seyedali Mirjalili and the S-shaped/V-shaped binary PSO work. Preserve these notices and check the applicable terms before reuse.

## Status

Historical experimental code. The repository name describes research areas; it is not a claim that every algorithm is packaged, benchmarked or production-ready.

[Profile and project directory](https://github.com/zohaibjan)
