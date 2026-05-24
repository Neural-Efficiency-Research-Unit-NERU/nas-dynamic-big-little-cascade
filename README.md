# Neural Architecture Search for Big/Little Dynamic Inference

This repository contains the implementation, saved experiment results, plots,
model checkpoints, and paper source for a project on neural architecture search
for input-adaptive little/big CNN cascades.

The project studies memory-constrained dynamic inference for embedded targets.
Each input is first evaluated by a little model. If the little model's softmax
confidence is at least a fixed threshold, the cascade exits early. Otherwise the
input is deferred to a larger big model.

## What Is Included

```text
.
  README.md
  REPRODUCING_RESULTS.md
  paper/
    main.tex
    references.bib
    main.pdf
  development/
    environment.yml
    configs/
    scripts/
    src/
    tests/
    experiments/
```

The `development/` directory contains the runnable framework:

- `src/`: dataset loading, CNN search space, NAS utilities, training, metrics,
  and plotting helpers.
- `configs/`: CIFAR-10 search and retraining configurations.
- `scripts/`: entry points for joint search, independent search, retraining,
  analysis, and plot generation.
- `experiments/`: saved CSVs, checkpoints, and figures used in the paper.

The `paper/` directory contains the LaTeX source, compiled PDF, and the figures
used in the report. Regenerated experiment plots are written separately under
`development/experiments/plots/`.

## Main Experiment

The project compares two ways to build a little/big cascade:

- **Joint NAS co-search:** NSGA-II searches complete cascade pairs directly
  using cascade accuracy and expected cascade FLOPs.
- **Independent NAS baseline:** little and big models are searched separately
  and then combined into cascades.

The selected joint architectures are retrained with four objectives:

| Variant | Config | Objective |
| --- | --- | --- |
| C1 | `configs/cifar10_retrain_ce_only.yaml` | Cross-entropy |
| C2 | `configs/cifar10_retrain_ce_smooth.yaml` | Cross-entropy with label smoothing |
| C3 | `configs/cifar10_retrain_cascade_only.yaml` | Cascade-aware loss |
| C4 | `configs/cifar10_retrain_cascade_smooth.yaml` | Cascade-aware loss with label smoothing |

The reported results are saved under:

```text
development/experiments/cifar10_cosearch_nas/
development/experiments/cifar10_independent_nas/
development/experiments/cifar10_A_retrain_ce_only/
development/experiments/cifar10_A_retrain_ce_smooth/
development/experiments/cifar10_A_retrain_cascade_only/
development/experiments/cifar10_A_retrain_cascade_smooth/
development/experiments/plots/
paper/figures/
```

## Quick Start

Create the Python environment:

```bash
cd development
conda env create -f environment.yml
conda activate nas-dev
```

Or install the required packages manually:

```bash
python -m pip install torch torchvision numpy pyyaml matplotlib pandas tqdm pytest "pymoo>=0.6" "ptflops>=0.7"
```

Run a quick sanity check:

```bash
python -m pytest tests -q
```

Regenerate plots from the saved experiment artifacts:

```bash
python scripts/generate_plots.py --dataset cifar10 --mode all
python scripts/generate_extra_result_plots.py --experiments-root experiments --output-dir experiments/plots
python scripts/generate_confidence_accuracy_plots.py --experiments-root experiments --output-dir experiments/plots --dataset cifar10 --data-root experiments/plot_data --top-k 3 --num-workers 0
```

Plot outputs are organized by topic:

```text
development/experiments/plots/search/
development/experiments/plots/rq1/
development/experiments/plots/ablations/
development/experiments/plots/routing/
development/experiments/plots/flops/
development/experiments/plots/training/
```

The plotting scripts write regenerated figures to `development/experiments/plots/`.
The report figures are stored separately in `paper/figures/` so the paper
remains self-contained.

For full reproduction from scratch, see
[`REPRODUCING_RESULTS.md`](REPRODUCING_RESULTS.md).

## Paper

The report is available at:

```text
paper/main.pdf
```

To rebuild it:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
