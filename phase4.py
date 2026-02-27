import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — saves to file instead of opening a window
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from phase3 import run_ga

# ---------------------------------------------------------------------------
# Reference line: best known hit/stand-only basic strategy
# Encodes: hit on hard <= 11 always, stand on hard >= 17 always,
# hit hard 12-16 when dealer shows 7-A, stand otherwise.
# Soft: hit soft <= 17, stand soft >= 18.
# ---------------------------------------------------------------------------

def _build_basic_strategy_chromosome():
    """
    Approximate hit/stand basic strategy chromosome (no doubles/splits).
    Returns a (260,) uint8 array.
    """
    chrom = np.zeros(260, dtype=np.uint8)

    # Hard hands (indices 0–169): rows = player total 4–20, cols = dealer A,2–10
    # dealer col 0 = Ace, cols 1–8 = dealer 2–9, col 9 = dealer 10
    for total in range(4, 21):
        for dealer_col in range(10):       # 0=Ace, 1=2, ..., 9=10
            dealer_val = 1 if dealer_col == 0 else dealer_col + 1
            idx = (total - 4) * 10 + dealer_col
            if total <= 11:
                chrom[idx] = 1             # always hit
            elif total >= 17:
                chrom[idx] = 0             # always stand
            else:
                # 12–16: stand vs dealer 2–6, hit vs 7–A
                chrom[idx] = 0 if 2 <= dealer_val <= 6 else 1

    # Soft hands (indices 170–259): rows = soft total 12–20, cols = dealer A,2–10
    for soft_total in range(12, 21):
        for dealer_col in range(10):
            idx = 170 + (soft_total - 12) * 10 + dealer_col
            chrom[idx] = 1 if soft_total <= 17 else 0  # hit soft <=17

    return chrom


# ---------------------------------------------------------------------------
# Figure 1: fitness over generations
# ---------------------------------------------------------------------------

def plot_fitness(history, reference_fitness, out_path="fitness.png"):
    generations = [h["generation"] for h in history]
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(generations, [h["max"]    for h in history], color="#2196F3", lw=2,   label="Max")
    ax.plot(generations, [h["mean"]   for h in history], color="#4CAF50", lw=2,   label="Mean")
    ax.plot(generations, [h["median"] for h in history], color="#FF9800", lw=1.5, linestyle="--", label="Median")
    ax.plot(generations, [h["min"]    for h in history], color="#F44336", lw=1,   linestyle=":",  label="Min")

    ax.axhline(
        reference_fitness, color="black", lw=1.5, linestyle="--",
        label=f"Basic strategy reference ({reference_fitness:.4f})",
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness  (wins + 0.5×ties) / hands")
    ax.set_title("Blackjack GA — Fitness over Generations")
    ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: strategy heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(population, out_path="strategy.png"):
    hit_frequency = population.mean(axis=0)                     # (260,)
    hard_matrix   = hit_frequency[:170].reshape(17, 10)         # player 4–20 × dealer A,2–10
    soft_matrix   = hit_frequency[170:].reshape(9, 10)          # soft 12–20 × dealer A,2–10

    col_labels = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    hard_row_labels = [str(t) for t in range(4, 21)]
    soft_row_labels = [f"A+{t - 11}" for t in range(12, 21)]   # A+1 … A+9

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

    for ax, matrix, row_labels, title in [
        (axes[0], hard_matrix, hard_row_labels, "Hard Hands"),
        (axes[1], soft_matrix, soft_row_labels, "Soft Hands"),
    ]:
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")

        ax.set_xticks(range(10))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("Dealer upcard")
        ax.set_ylabel("Player total")
        ax.set_title(f"Hit frequency — {title}")

        # Annotate each cell with percentage
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                val = matrix[r, c]
                text_color = "white" if val > 0.75 or val < 0.25 else "black"
                ax.text(c, r, f"{int(round(val * 100))}%",
                        ha="center", va="center", fontsize=7, color=text_color)

    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02,
                 label="Hit frequency (1.0 = always hit, 0.0 = always stand)")
    fig.suptitle("Blackjack GA — Evolved Strategy Heatmap", fontsize=14)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from phase2 import evaluate_fitness

    # Calibrate the reference line
    print("Calibrating basic-strategy reference fitness...")
    basic_chrom = _build_basic_strategy_chromosome()
    ref_fitness = np.mean([evaluate_fitness(basic_chrom, n_hands=5000) for _ in range(5)])
    print(f"Reference fitness: {ref_fitness:.4f}\n")

    # Run the full GA
    print("Running GA (50 generations, 100 individuals, 5000 hands)...")
    population, history = run_ga(
        generations=50,
        pop_size=100,
        n_hands=5000,
        verbose=True,
    )

    # Generate figures
    plot_fitness(history, ref_fitness, out_path="fitness.png")
    plot_heatmap(population,           out_path="strategy.png")

    print("\nDone. Outputs: fitness.png, strategy.png")
