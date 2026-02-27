"""
Runs the GA for 150 generations, saves history and final population,
then generates the fitness convergence plot.
"""
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from phase2 import evaluate_fitness
from phase3 import run_ga
from phase4 import _build_basic_strategy_chromosome


def plot_convergence(history, reference_fitness, out_path="convergence.png"):
    generations = [h["generation"] for h in history]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(generations, [h["max"]    for h in history], color="#1565C0", lw=2,            label="Max")
    ax.plot(generations, [h["mean"]   for h in history], color="#2E7D32", lw=2,            label="Mean")
    ax.plot(generations, [h["median"] for h in history], color="#E65100", lw=1.5, ls="--", label="Median")
    ax.plot(generations, [h["min"]    for h in history], color="#B71C1C", lw=1,   ls=":",  label="Min")

    ax.axhline(
        reference_fitness, color="black", lw=1.5, ls="--",
        label=f"Basic strategy reference ({reference_fitness:.4f})",
    )

    ax.set_xlabel("Generation", fontsize=13)
    ax.set_ylabel("Fitness  (wins + 0.5 × ties) / hands", fontsize=13)
    ax.set_title("Blackjack GA — Fitness Convergence", fontsize=15)
    ax.legend(fontsize=11, loc="lower right")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.set_xlim(0, len(generations) - 1)
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", alpha=0.15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    # Calibrate reference line
    print("Calibrating basic-strategy reference fitness...")
    basic_chrom = _build_basic_strategy_chromosome()
    ref_fitness = np.mean([evaluate_fitness(basic_chrom, n_hands=5000) for _ in range(5)])
    print(f"Reference fitness: {ref_fitness:.4f}\n")

    # Run GA
    print("Running GA (300 generations, 100 individuals, 5000 hands)...")
    population, history = run_ga(
        generations=300,
        pop_size=100,
        n_hands=5000,
        verbose=True,
    )

    # Save results
    np.save("final_population.npy", population)
    with open("history.json", "w") as f:
        json.dump(history, f)
    print("Saved final_population.npy and history.json")

    # Plot
    plot_convergence(history, ref_fitness, out_path="convergence.png")
    print("\nDone.")
