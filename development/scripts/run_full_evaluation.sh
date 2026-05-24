#!/usr/bin/env bash
# =============================================================================
# Full PyTorch evaluation pipeline (CIFAR-10, n_gen=20, knee selection)
# =============================================================================
#
# The pipeline supports GPU if available, but also runs on CPU-only machines.
# Python dependencies are installed at job start when missing. Torch and
# torchvision are not installed automatically, because CUDA-enabled builds must
# match the local driver and platform.
#
# Usage:
#   cd development
#   bash scripts/run_full_evaluation.sh
#
# Detached Linux or cluster run:
#   nohup env CPU_THREADS=64 bash scripts/run_full_evaluation.sh > /work/run.out 2>&1 &
#   tail -f /work/run.out
#
# Resume behavior:
#   Each step writes a marker file under experiments/.step*_done when it
#   completes successfully. Re-running the script skips completed steps.
#   Delete the relevant marker to force a step to run again.
#
# What it runs:
#   Step 1: Joint NAS co-search
#   Step 2: Independent NAS baseline
#   Step 3: C1 retrain - cross-entropy
#   Step 4: C2 retrain - cross-entropy with label smoothing
#   Step 5: C3 retrain - cascade-aware loss
#   Step 6: C4 retrain - cascade-aware loss with label smoothing
#   Step 7: Standard plots
#   Step 8: Result and diagnostic plots
#   Step 9: Confidence-vs-accuracy plots
#
# Steps 1-6 are hard-fail. Steps 7-9 are soft-fail, because a plotting issue
# should not invalidate a completed NAS and retraining run.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$DEV_DIR")"

EXPERIMENTS_DIR="$DEV_DIR/experiments"
LOGS_DIR="$DEV_DIR/logs"
mkdir -p "$EXPERIMENTS_DIR" "$LOGS_DIR"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOGS_DIR/run_${RUN_TS}.log"
touch "$RUN_LOG"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$RUN_LOG"
}

trap 'log "interrupt or error caught - aborting (exit $?)"; exit 130' INT TERM

log "============================================================"
log "Evaluation pipeline starting"
log "  repo root  : $REPO_ROOT"
log "  dev dir    : $DEV_DIR"
log "  experiments: $EXPERIMENTS_DIR"
log "  log file   : $RUN_LOG"
log "============================================================"

if [ -d "/work" ]; then
    export TMPDIR="/work/tmp"
    mkdir -p "$TMPDIR"
    log "TMPDIR redirected to $TMPDIR"
fi

# Single-process DataLoader is the safest default for restricted containers.
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
log "DATALOADER_NUM_WORKERS=$DATALOADER_NUM_WORKERS"

if command -v nproc >/dev/null 2>&1; then
    DETECTED_CPU_THREADS="$(nproc)"
else
    DETECTED_CPU_THREADS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
fi
export CPU_THREADS="${CPU_THREADS:-$DETECTED_CPU_THREADS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_THREADS}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-$CPU_THREADS}"
log "CPU_THREADS=$CPU_THREADS (OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS)"

if [ -z "${PYTHON:-}" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON="python3"
    else
        PYTHON="python"
    fi
fi
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

log "--- Pre-flight ---"
log "Python: $($PYTHON -c 'import sys; print(sys.executable, sys.version.split()[0])')"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi 2>&1 | tee -a "$RUN_LOG"
else
    log "WARNING: nvidia-smi not found; PyTorch will fall back to CPU if CUDA is unavailable."
fi

if ! $PYTHON -c "import pymoo, ptflops, yaml, numpy, matplotlib, pandas, tqdm" >/dev/null 2>&1; then
    log "Installing missing Python dependencies..."
    $PYTHON -m pip install --quiet \
        "pymoo>=0.6" "ptflops>=0.7" \
        "pyyaml" "numpy" "matplotlib" "pandas" "tqdm" "pytest" \
        2>&1 | tee -a "$RUN_LOG"
fi

log "Verifying imports..."
$PYTHON - <<'PY' 2>&1 | tee -a "$RUN_LOG"
import torch, torchvision, pymoo, ptflops, yaml, numpy, matplotlib, pandas, tqdm
print(f"  torch       : {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  torchvision : {torchvision.__version__}")
if torch.cuda.is_available():
    print(f"  device      : {torch.cuda.get_device_name(0)}")
    print(f"  capability  : {torch.cuda.get_device_capability(0)}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  gpu memory  : {mem_gb:.2f} GB")
print(f"  pymoo       : {pymoo.__version__}")
print(f"  numpy       : {numpy.__version__}")
PY

if ! $PYTHON -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    log "CUDA is not available; continuing in CPU mode with CPU_THREADS=$CPU_THREADS."
fi

REQUIRED_CONFIGS=(
    "$DEV_DIR/configs/cifar10_cosearch.yaml"
    "$DEV_DIR/configs/cifar10_independent_search.yaml"
    "$DEV_DIR/configs/cifar10_retrain_ce_only.yaml"
    "$DEV_DIR/configs/cifar10_retrain_ce_smooth.yaml"
    "$DEV_DIR/configs/cifar10_retrain_cascade_only.yaml"
    "$DEV_DIR/configs/cifar10_retrain_cascade_smooth.yaml"
)
for cfg in "${REQUIRED_CONFIGS[@]}"; do
    if [ ! -f "$cfg" ]; then
        log "FATAL: missing config $cfg"
        exit 1
    fi
done
log "All ${#REQUIRED_CONFIGS[@]} configs present."

mkdir -p "$DEV_DIR/data"
log "Data dir: $DEV_DIR/data (CIFAR-10 will be downloaded on first use if absent)"

if command -v df >/dev/null 2>&1; then
    df -h "$DEV_DIR" 2>&1 | tail -1 | tee -a "$RUN_LOG" || true
fi

mark_done() { touch "$EXPERIMENTS_DIR/.${1}_done"; }
is_done() { [ -f "$EXPERIMENTS_DIR/.${1}_done" ]; }

run_hard_step() {
    local marker="$1"; shift
    local label="$1"; shift
    if is_done "$marker"; then
        log "[$marker] already complete - skipping ($label)"
        return 0
    fi
    log "[$marker] BEGIN - $label"
    local t0=$SECONDS
    set +e
    "$@" 2>&1 | tee -a "$RUN_LOG"
    local rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -ne 0 ]; then
        log "[$marker] FAIL exit=$rc after $((SECONDS - t0))s - $label"
        exit "$rc"
    fi
    log "[$marker] DONE in $((SECONDS - t0))s - $label"
    mark_done "$marker"
}

run_soft_step() {
    local marker="$1"; shift
    local label="$1"; shift
    if is_done "$marker"; then
        log "[$marker] already complete - skipping ($label)"
        return 0
    fi
    log "[$marker] BEGIN - $label"
    local t0=$SECONDS
    set +e
    "$@" 2>&1 | tee -a "$RUN_LOG"
    local rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -ne 0 ]; then
        log "[$marker] WARN exit=$rc after $((SECONDS - t0))s - continuing"
    else
        log "[$marker] DONE in $((SECONDS - t0))s - $label"
        mark_done "$marker"
    fi
}

cd "$DEV_DIR"
log "cwd: $(pwd)"

run_hard_step step1_A_nas "Joint NAS co-search - CIFAR-10, FS proxy, pop=20, gen=20" \
    "$PYTHON" scripts/run_nas_search.py \
        --config configs/cifar10_cosearch.yaml
[ -f "$EXPERIMENTS_DIR/cifar10_cosearch_nas/pareto_front.csv" ] \
    || { log "FATAL: cifar10_cosearch_nas/pareto_front.csv not produced"; exit 1; }

run_hard_step step2_Bnas "Independent NAS - CIFAR-10, FS-aligned, pop=20, gen=10 per role" \
    "$PYTHON" scripts/run_independent_nas.py \
        --config configs/cifar10_independent_search.yaml
[ -f "$EXPERIMENTS_DIR/cifar10_independent_nas/combined_cascade_results.csv" ] \
    || { log "FATAL: cifar10_independent_nas/combined_cascade_results.csv not produced"; exit 1; }

SEARCH_DIR="$EXPERIMENTS_DIR/cifar10_cosearch_nas"

run_hard_step step3_C1 "C1 retrain - CE only" \
    "$PYTHON" scripts/retrain_best.py \
        --search-dir "$SEARCH_DIR" \
        --config configs/cifar10_retrain_ce_only.yaml \
        --selection knee \
        --top-k 3

run_hard_step step4_C2 "C2 retrain - CE + smoothing" \
    "$PYTHON" scripts/retrain_best.py \
        --search-dir "$SEARCH_DIR" \
        --config configs/cifar10_retrain_ce_smooth.yaml \
        --selection knee \
        --top-k 3

run_hard_step step5_C3 "C3 retrain - cascade-aware loss" \
    "$PYTHON" scripts/retrain_best.py \
        --search-dir "$SEARCH_DIR" \
        --config configs/cifar10_retrain_cascade_only.yaml \
        --selection knee \
        --top-k 3

run_hard_step step6_C4 "C4 retrain - cascade-aware loss + smoothing" \
    "$PYTHON" scripts/retrain_best.py \
        --search-dir "$SEARCH_DIR" \
        --config configs/cifar10_retrain_cascade_smooth.yaml \
        --selection knee \
        --top-k 3

run_soft_step step7_standard_plots "Generate standard plots" \
    "$PYTHON" scripts/generate_plots.py \
        --dataset cifar10 \
        --mode all

run_soft_step step8_result_plots "Generate result plots" \
    "$PYTHON" scripts/generate_extra_result_plots.py \
        --experiments-root experiments \
        --output-dir experiments/plots

run_soft_step step9_confidence_plots "Generate confidence-accuracy plots" \
    "$PYTHON" scripts/generate_confidence_accuracy_plots.py \
        --experiments-root experiments \
        --output-dir experiments/plots \
        --dataset cifar10 \
        --data-root experiments/plot_data \
        --top-k 3 \
        --num-workers 0

log "============================================================"
log "PIPELINE COMPLETE"
log "============================================================"
log "Outputs under $EXPERIMENTS_DIR/"
log "  cifar10_cosearch_nas/             pareto_front.csv, search_log.csv"
log "  cifar10_independent_nas/          combined_cascade_results.csv"
log "  cifar10_A_retrain_ce_only/        C1: 3 retrained checkpoints + sweeps"
log "  cifar10_A_retrain_ce_smooth/      C2"
log "  cifar10_A_retrain_cascade_only/   C3"
log "  cifar10_A_retrain_cascade_smooth/ C4"
log "  plots/                            figures (PNG + PDF)"
log ""
log "Run log: $RUN_LOG"

echo "PIPELINE_COMPLETE $(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    > "$EXPERIMENTS_DIR/PIPELINE_DONE.marker"
log "Completion marker written."
