# Experiment Artifacts

This directory contains the CIFAR-10 artifacts used by the paper: NAS search
CSVs, retraining histories, threshold sweeps, checkpoints, exported models, and
generated plots.

## Main Directories

| Directory | Contents |
| --- | --- |
| `cifar10_cosearch_nas/` | Joint NSGA-II co-search results. |
| `cifar10_independent_nas/` | Independent little/big NAS baseline and combined cascade results. |
| `cifar10_A_retrain_ce_only/` | C1 retraining: cross-entropy. |
| `cifar10_A_retrain_ce_smooth/` | C2 retraining: cross-entropy with label smoothing. |
| `cifar10_A_retrain_cascade_only/` | C3 retraining: cascade-aware loss. |
| `cifar10_A_retrain_cascade_smooth/` | C4 retraining: cascade-aware loss with label smoothing. |
| `plots/search/` | NAS search-progress and selected-architecture figures. |
| `plots/rq1/` | Joint-vs-independent comparison figures. |
| `plots/ablations/` | C1-C4 ablation accuracy and accuracy/FLOPs figures. |
| `plots/routing/` | Threshold, ECE, confidence, and routing diagnostics. |
| `plots/flops/` | Little, big, and expected cascade FLOPs breakdowns. |
| `plots/training/` | Training-curve overlays and per-architecture curves. |

The CIFAR-10 dataset itself is not stored here. It is downloaded automatically
by the data loader when running the pipeline.
Confidence-accuracy plotting may create a local `plot_data/` cache; this cache
is ignored by Git.

The plotting scripts write regenerated figures to `plots/`. The report figures
are stored separately in `../../paper/figures/` so the paper remains
self-contained.
