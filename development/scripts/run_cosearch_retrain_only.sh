#!/usr/bin/env bash
# =============================================================================
# Resume script: retrain joint co-search architectures only
# =============================================================================
#
# Use this when Joint NAS has already produced:
#   experiments/cifar10_cosearch_nas/pareto_front.csv
#
# The script skips both NAS searches and reruns the normal C1--C4 retraining
# ablations for the co-searched architectures, using the same retrain_best.py
# calls as run_full_evaluation.sh.
#
# Usage:
#   cd development
#   nohup env CPU_THREADS=64 bash scripts/run_cosearch_retrain_only.sh > /work/retrain.out 2>&1 &
#   tail -f /work/retrain.out
#
# Resume behavior:
#   Uses the same step marker names as the full pipeline:
#     experiments/.step3_C1_done ... experiments/.step6_C4_done
#   Delete a marker to rerun that condition.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$DEV_DIR")"

EXPERIMENTS_DIR="$DEV_DIR/experiments"
LOGS_DIR="$DEV_DIR/logs"
mkdir -p "$EXPERIMENTS_DIR" "$LOGS_DIR"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOGS_DIR/cosearch_retrain_${RUN_TS}.log"
touch "$RUN_LOG"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$RUN_LOG"
}

trap 'log "interrupt or error caught - aborting (exit $?)"; exit 130' INT TERM

log "============================================================"
log "Co-search retrain-only pipeline starting"
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

# Avoid PyTorch DataLoader multiprocessing socket issues in restricted containers.
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

PYTHON="${PYTHON:-python3}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

log "--- Pre-flight ---"
log "Python: $($PYTHON -c 'import sys; print(sys.executable, sys.version.split()[0])')"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi 2>&1 | tee -a "$RUN_LOG"
else
    log "WARNING: nvidia-smi not found; PyTorch will fall back to CPU."
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
import torch, pymoo, ptflops, yaml, numpy, matplotlib, pandas, tqdm
print(f"  torch       : {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device      : {torch.cuda.get_device_name(0)}")
print(f"  pymoo       : {pymoo.__version__}")
print(f"  numpy       : {numpy.__version__}")
PY

REQUIRED_CONFIGS=(
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
log "All ${#REQUIRED_CONFIGS[@]} retrain configs present."

mkdir -p "$DEV_DIR/data"
log "Data dir: $DEV_DIR/data (CIFAR-10 will be downloaded on first use if absent)"

if command -v df >/dev/null 2>&1; then
    df -h "$DEV_DIR" 2>&1 | tail -1 | tee -a "$RUN_LOG" || true
fi

mark_done() { touch "$EXPERIMENTS_DIR/.${1}_done"; }
is_done()   { [ -f "$EXPERIMENTS_DIR/.${1}_done" ]; }

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

cd "$DEV_DIR"
log "cwd: $(pwd)"

SEARCH_DIR="$EXPERIMENTS_DIR/cifar10_cosearch_nas"
if [ ! -f "$SEARCH_DIR/pareto_front.csv" ]; then
    log "FATAL: missing $SEARCH_DIR/pareto_front.csv"
    log "Run scripts/run_nas_search.py first, or copy the co-search results into $SEARCH_DIR."
    exit 1
fi

if [ ! -f "$SEARCH_DIR/search_log.csv" ]; then
    log "WARNING: $SEARCH_DIR/search_log.csv not found."
    log "retrain_best.py will fall back to pareto_front.csv for knee selection."
fi

log "Using co-search results from $SEARCH_DIR"
log "Architecture selection: duplicate-safe knee on the co-search Pareto frontier"

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

run_hard_step step5_C3 "C3 retrain - Cascade loss" \
    "$PYTHON" scripts/retrain_best.py \
        --search-dir "$SEARCH_DIR" \
        --config configs/cifar10_retrain_cascade_only.yaml \
        --selection knee \
        --top-k 3

run_hard_step step6_C4 "C4 retrain - Cascade loss + smoothing" \
    "$PYTHON" scripts/retrain_best.py \
        --search-dir "$SEARCH_DIR" \
        --config configs/cifar10_retrain_cascade_smooth.yaml \
        --selection knee \
        --top-k 3

log "============================================================"
log "CO-SEARCH RETRAIN-ONLY PIPELINE COMPLETE"
log "============================================================"
log "Outputs under $EXPERIMENTS_DIR/"
log "  cifar10_A_retrain_ce_only/        C1"
log "  cifar10_A_retrain_ce_smooth/      C2"
log "  cifar10_A_retrain_cascade_only/   C3"
log "  cifar10_A_retrain_cascade_smooth/ C4"
