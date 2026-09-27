# MachSense research log

## Dataset design

Paderborn is the primary condition-aware bearing benchmark. FEMTO/PRONOSTIA is the independent raw bearing benchmark. NASA C-MAPSS FD001 is an optional small cross-domain RUL benchmark.

The datasets are trained separately. They are not pooled into one feature matrix.

## Experiments

1. Bearing-wise holdout rather than random-row splitting.
2. Condition-aware temporal model vs condition-agnostic baseline.
3. Single-task RUL vs multi-output MachSense.
4. Health/RUL consistency ablation.
5. Reconstruction anomaly evidence vs Isolation Forest baseline.
6. Paderborn vs raw FEMTO benchmark behavior.
7. Optional cross-domain C-MAPSS FD001 RUL check.

## Important limitation

The MachSense Health Index is currently supervised with a run-to-failure-position proxy (`normalized RUL`). It is a useful engineering health-state representation, not a physical health sensor ground truth. The thesis should describe this honestly and include an ablation/consistency analysis.
