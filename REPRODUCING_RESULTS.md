# Reproducing Results

This document describes how to reproduce the saved CIFAR-10 NAS, retraining,
analysis, and plot results.

## Environment

From the repository root:

```bash
cd development
conda env create -f environment.yml
conda activate nas-dev
```

Manual installation is also possible:

```bash
python -m pip install torch torchvision numpy pyyaml matplotlib pandas tqdm pytest "pymoo>=0.6" "ptflops>=0.7"
```

If running on CUDA, install a CUDA-enabled PyTorch build that matches your
driver. The pipeline also runs on CPU, but the full NAS and retraining run can
take a long time.

## Saved Results

The repository already includes the CSVs, checkpoints, and figures used in the
paper. To regenerate plots from those saved outputs without rerunning NAS or
training:

```bash
cd development
python scripts/generate_plots.py --dataset cifar10 --mode all
python scripts/generate_extra_result_plots.py --experiments-root experiments --output-dir experiments/plots
python scripts/generate_confidence_accuracy_plots.py --experiments-root experiments --output-dir experiments/plots --dataset cifar10 --data-root experiments/plot_data --top-k 3 --num-workers 0
```

Plots are written to:

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

## Full Pipeline From Scratch

The main end-to-end script is:

```bash
cd development
bash scripts/run_full_evaluation.sh
```

The script name is historical. It is a normal Bash pipeline and can run on a
local Linux shell, Git Bash, WSL, or a compute node.

For CPU runs, set the number of worker threads:

```bash
CPU_THREADS=64 bash scripts/run_full_evaluation.sh
```

For a detached Linux or cluster run:

```bash
nohup env CPU_THREADS=64 bash scripts/run_full_evaluation.sh > /work/run.out 2>&1 &
tail -f /work/run.out
```

On Windows PowerShell with Git Bash:

```powershell
& "C:\Program Files\Git\bin\bash.exe" scripts/run_full_evaluation.sh
```

The full pipeline performs:

1. joint NSGA-II cascade co-search,
2. independent little/big NAS baseline,
3. final retraining of the selected joint architectures with C1-C4 objectives,
4. plot generation.

Outputs are written under:

```text
development/experiments/
development/logs/
```

The script is resumable. Completed steps create marker files under
`development/experiments/` named `.step*_done`. Delete the relevant marker if a
step should be rerun.

## Individual Commands

Joint co-search:

```bash
python scripts/run_nas_search.py --config configs/cifar10_cosearch.yaml
```

Independent NAS baseline:

```bash
python scripts/run_independent_nas.py --config configs/cifar10_independent_search.yaml
```

Retrain the selected joint architectures:

```bash
python scripts/retrain_best.py --search-dir experiments/cifar10_cosearch_nas --config configs/cifar10_retrain_ce_only.yaml --selection knee --top-k 3
python scripts/retrain_best.py --search-dir experiments/cifar10_cosearch_nas --config configs/cifar10_retrain_ce_smooth.yaml --selection knee --top-k 3
python scripts/retrain_best.py --search-dir experiments/cifar10_cosearch_nas --config configs/cifar10_retrain_cascade_only.yaml --selection knee --top-k 3
python scripts/retrain_best.py --search-dir experiments/cifar10_cosearch_nas --config configs/cifar10_retrain_cascade_smooth.yaml --selection knee --top-k 3
```

If the joint search has already completed and only C1-C4 retraining should be
rerun:

```bash
bash scripts/run_cosearch_retrain_only.sh
```

## Main Settings

Current CIFAR-10 settings:

- joint NAS: population 20, generations 20;
- independent NAS: population 20, generations 10 per role;
- proxy training: 5 epochs;
- final retraining: 50 epochs;
- fixed search threshold: 0.70;
- threshold sweep: 0.50, 0.60, 0.70, 0.80, 0.90, 0.95;
- deployment-memory cap: 450 KiB;
- combined parameter window: 50K-100K;
- little parameter range: 15K-30K;
- big parameter range: 40K-85K;
- little channels: 8, 12, 16, 24, 32;
- big channels: 16, 24, 32, 48, 64;
- CIFAR-10 split: 45K train, 5K validation, 10K official test.

The pipeline uses seed 42 for Python, NumPy, PyTorch, DataLoader shuffling,
NSGA-II, feasible sampling, and repair where applicable. Exact bitwise results
can still vary slightly across hardware, PyTorch versions, CUDA versions, and
threading backends.

## Checkpoints

The retrained `.pt` checkpoints are stored in the corresponding experiment
folders under `development/experiments/`.
