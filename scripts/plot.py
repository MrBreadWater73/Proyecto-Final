#!/usr/bin/env python3
"""
Genera dos graficas a partir de results/timings.csv:
  1. results/exec_time_vs_threads.png  — Tiempo de ejecucion vs hilos
  2. results/speedup_vs_threads.png    — Speedup vs hilos + curva de Amdahl

Uso:
  python3 scripts/plot.py
"""
import csv
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "results/timings.csv"
OUT_DIR  = "results"

# ─── load data ────────────────────────────────────────────────────────────────

def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "threads":   int(r["threads"]),
                "task":      r["task"],
                "scheduler": r["scheduler"],
                "chunk":     r["chunk"],
                "time_s":    float(r["time_s"]),
            })
    return rows

# ─── helpers ──────────────────────────────────────────────────────────────────

def label(row):
    c = row["chunk"]
    return row["scheduler"] if c in ("", "auto") else f"{row['scheduler']}:{c}"

def group_by(rows, key):
    d = defaultdict(list)
    for r in rows:
        d[key(r)].append(r)
    return d

def best_time_per_threads(rows):
    """For each thread count return the minimum time (best scheduler run)."""
    by_t = group_by(rows, lambda r: r["threads"])
    return {t: min(rr, key=lambda r: r["time_s"]) for t, rr in by_t.items()}

# ─── plot 1: execution time vs threads ────────────────────────────────────────

def plot_exec_time(rows, task, ax):
    by_label = group_by(rows, label)
    thread_counts = sorted({r["threads"] for r in rows})

    for lbl, rr in sorted(by_label.items()):
        by_t = {r["threads"]: r["time_s"] for r in rr}
        ys = [by_t.get(t) for t in thread_counts]
        valid = [(t, y) for t, y in zip(thread_counts, ys) if y is not None]
        if not valid:
            continue
        xs, ys = zip(*valid)
        ax.plot(xs, ys, marker="o", label=lbl)

    ax.set_title(f"Tiempo de ejecución vs hilos — {task.capitalize()}")
    ax.set_xlabel("Número de hilos")
    ax.set_ylabel("Tiempo (s)")
    ax.set_xticks(thread_counts)
    ax.legend(fontsize=7)
    ax.grid(True, linestyle="--", alpha=0.5)

# ─── plot 2: speedup vs threads + Amdahl ──────────────────────────────────────

def fit_amdahl(thread_counts, speedups):
    """Estimate f_serial that minimises MSE against measured speedups."""
    best_f, best_err = 0.5, float("inf")
    for f in np.linspace(0.01, 0.99, 1000):
        pred = [1.0 / (f + (1 - f) / n) for n in thread_counts]
        err = sum((p - s) ** 2 for p, s in zip(pred, speedups))
        if err < best_err:
            best_err, best_f = err, f
    return best_f

def plot_speedup(rows, task, ax):
    # Use best (fastest) time per thread count for the speedup curve
    best = best_time_per_threads(rows)
    if not best:
        return
    t1 = best.get(1)
    if t1 is None:
        t1 = min(best.values(), key=lambda r: r["threads"])
    t1_val = t1["time_s"]

    thread_counts = sorted(best.keys())
    speedups = [t1_val / best[t]["time_s"] for t in thread_counts]

    ax.plot(thread_counts, speedups, marker="o", color="steelblue",
            label="Speedup medido (mejor scheduler)", linewidth=2)

    # Ideal linear speedup
    ax.plot(thread_counts, thread_counts, linestyle="--", color="green",
            alpha=0.7, label="Speedup ideal (lineal)")

    # Amdahl fit
    f_serial = fit_amdahl(thread_counts, speedups)
    t_amdahl = np.linspace(1, max(thread_counts), 200)
    s_amdahl = [1.0 / (f_serial + (1 - f_serial) / n) for n in t_amdahl]
    ax.plot(t_amdahl, s_amdahl, linestyle=":", color="red", linewidth=1.5,
            label=f"Ley de Amdahl (f_serial≈{f_serial:.3f})")

    # Annotate degradation point (first thread count where speedup decreases)
    for i in range(1, len(speedups)):
        if speedups[i] < speedups[i - 1]:
            ax.annotate("degradación\nOS overhead",
                        xy=(thread_counts[i], speedups[i]),
                        xytext=(thread_counts[i] + 0.3, speedups[i] - 0.2),
                        fontsize=8, color="darkorange",
                        arrowprops=dict(arrowstyle="->", color="darkorange"))
            break

    ax.set_title(f"Speedup vs hilos — {task.capitalize()}")
    ax.set_xlabel("Número de hilos")
    ax.set_ylabel("Speedup S(n)")
    ax.set_xticks(thread_counts)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run scripts/benchmark.sh first.")
        sys.exit(1)

    rows = load(CSV_PATH)
    tasks = sorted({r["task"] for r in rows})
    os.makedirs(OUT_DIR, exist_ok=True)

    # Plot 1: execution time
    fig, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 5), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        task_rows = [r for r in rows if r["task"] == task]
        plot_exec_time(task_rows, task, ax)
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "exec_time_vs_threads.png")
    fig.savefig(p1, dpi=150)
    print(f"Saved: {p1}")

    # Plot 2: speedup
    fig, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 5), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        task_rows = [r for r in rows if r["task"] == task]
        plot_speedup(task_rows, task, ax)
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "speedup_vs_threads.png")
    fig.savefig(p2, dpi=150)
    print(f"Saved: {p2}")

if __name__ == "__main__":
    main()
