#!/usr/bin/env bash
# Benchmarks: schedulers × threads for Mandelbrot + Gaussian blur.
# Writes results/timings.csv (appends — delete the file to start fresh).
#
# Usage:
#   bash scripts/benchmark.sh
#
# Requires: binary ./optimized already compiled.

set -euo pipefail

BINARY="./optimized"
OUT="results/timings.csv"
WARMUP_THREADS=1

mkdir -p results

if [[ ! -f "$OUT" ]]; then
    echo "threads,task,scheduler,chunk,time_s" > "$OUT"
fi

THREAD_COUNTS=(1 2 3 4 5 6 7 8 9 10 11 12)

# scheduler:chunk pairs to test
declare -a SCHEDULERS=(
    "static:"
    "dynamic:1"
    "dynamic:4"
    "dynamic:16"
    "dynamic:64"
    "guided:1"
    "guided:4"
)

compile_optimized() {
    echo "==> Compiling optimized..."
    g++ -O3 -march=native -fopenmp -o optimized src/optimized.cpp
}

run_one() {
    local threads=$1
    local sched_label=$2   # e.g. "dynamic:16"
    local chunk="${sched_label#*:}"
    local sched="${sched_label%%:*}"
    local omp_sched
    if [[ -z "$chunk" ]]; then
        omp_sched="$sched"
    else
        omp_sched="${sched},${chunk}"
    fi

    # Run binary, capture output lines like "Mandelbrot ...: 12.3456 s"
    local output
    output=$(OMP_NUM_THREADS="$threads" OMP_SCHEDULE="$omp_sched" "$BINARY" 2>&1)

    local t_mb t_blur
    t_mb=$(echo "$output"   | grep -i "mandelbrot"    | grep -oP '[\d]+\.[\d]+(?= s)' | head -1)
    t_blur=$(echo "$output" | grep -i "gaussian blur" | grep -oP '[\d]+\.[\d]+(?= s)' | head -1)

    [[ -n "$t_mb"   ]] && echo "${threads},mandelbrot,${sched},${chunk:-auto},${t_mb}" >> "$OUT"
    [[ -n "$t_blur" ]] && echo "${threads},blur,${sched},${chunk:-auto},${t_blur}"     >> "$OUT"
}

# Warmup run (discard)
echo "==> Warmup run..."
OMP_NUM_THREADS=$WARMUP_THREADS OMP_SCHEDULE="static" "$BINARY" > /dev/null 2>&1 || true

total=$((${#THREAD_COUNTS[@]} * ${#SCHEDULERS[@]}))
done_count=0

for threads in "${THREAD_COUNTS[@]}"; do
    for sched in "${SCHEDULERS[@]}"; do
        done_count=$((done_count + 1))
        echo "[${done_count}/${total}] threads=${threads} schedule=${sched}"
        run_one "$threads" "$sched"
    done
done

echo ""
echo "==> Done. Results written to $OUT"
echo "    Rows: $(wc -l < "$OUT") (including header)"
