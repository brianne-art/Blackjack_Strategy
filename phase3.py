import numpy as np
from multiprocessing import Pool, cpu_count

from phase2 import evaluate_fitness

# ---------------------------------------------------------------------------
# GA operators
# ---------------------------------------------------------------------------

def _select_roulette(population, fitnesses):
    """Return one parent via fitness-proportionate (roulette wheel) selection."""
    probs = fitnesses / fitnesses.sum()
    idx = np.random.choice(len(population), p=probs)
    return population[idx]


def _select_tournament(population, fitnesses, k=5):
    """Return one parent via tournament selection (pick best of k random)."""
    indices = np.random.choice(len(population), size=k, replace=False)
    best = indices[np.argmax(fitnesses[indices])]
    return population[best]


def _crossover(parent1, parent2):
    """Single-point crossover. Returns one child."""
    point = np.random.randint(1, 259)
    return np.concatenate([parent1[:point], parent2[point:]])


def _mutate(child, rate=0.01):
    """Flip each bit independently with probability rate."""
    mask = np.random.rand(len(child)) < rate
    return np.where(mask, 1 - child, child).astype(np.uint8)


# ---------------------------------------------------------------------------
# Parallel fitness evaluation helper (module-level for pickling)
# ---------------------------------------------------------------------------

def _eval_worker(args):
    chromosome, n_hands = args
    return evaluate_fitness(chromosome, n_hands)


# ---------------------------------------------------------------------------
# Main GA loop
# ---------------------------------------------------------------------------

def run_ga(
    generations=50,
    pop_size=100,
    n_hands=5000,
    mutation_rate=0.01,
    elite_count=2,
    n_jobs=None,
    verbose=True,
):
    """
    Run the genetic algorithm and return (population, history).

    history is a list of dicts with keys: generation, min, max, mean, median.
    n_jobs: number of parallel workers (default: all CPU cores).
    """
    if n_jobs is None:
        n_jobs = cpu_count()

    rng = np.random.default_rng()
    population = rng.integers(0, 2, size=(pop_size, 260), dtype=np.uint8)
    history = []
    stagnant_gens = 0
    best_fitness_seen = -1.0

    with Pool(processes=n_jobs) as pool:
        for gen in range(generations):
            # --- Evaluate fitness (parallel) ---
            args = [(population[i], n_hands) for i in range(pop_size)]
            fitnesses = np.array(pool.map(_eval_worker, args))

            # --- Record stats ---
            stats = {
                "generation": gen,
                "min":    fitnesses.min(),
                "max":    fitnesses.max(),
                "mean":   fitnesses.mean(),
                "median": float(np.median(fitnesses)),
            }
            history.append(stats)

            if verbose:
                print(
                    f"Gen {gen:3d} | "
                    f"max={stats['max']:.4f}  "
                    f"mean={stats['mean']:.4f}  "
                    f"median={stats['median']:.4f}  "
                    f"min={stats['min']:.4f}"
                )

            # --- Stagnation / adaptive mutation ---
            if stats["max"] <= best_fitness_seen:
                stagnant_gens += 1
            else:
                best_fitness_seen = stats["max"]
                stagnant_gens = 0

            current_mutation_rate = 0.05 if stagnant_gens >= 10 else mutation_rate

            # --- Diversity check: switch to tournament if converging early ---
            use_tournament = (fitnesses.std() < 0.001) and (gen < 20)
            select_fn = _select_tournament if use_tournament else _select_roulette

            # --- Build next generation ---
            sorted_idx = np.argsort(fitnesses)[::-1]
            elites = [population[i].copy() for i in sorted_idx[:elite_count]]

            new_population = elites[:]
            while len(new_population) < pop_size:
                p1 = select_fn(population, fitnesses)
                p2 = select_fn(population, fitnesses)
                child = _crossover(p1, p2)
                child = _mutate(child, rate=current_mutation_rate)
                new_population.append(child)

            population = np.array(new_population, dtype=np.uint8)

    return population, history


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_tests():
    all_passed = True

    def check_bool(label, value):
        nonlocal all_passed
        if not value:
            all_passed = False
        print(f"[{'PASS' if value else 'FAIL'}] {label}")

    print("Smoke test: 5 generations, 1000 hands, 20 individuals...")
    np.random.seed(0)

    pop, history = run_ga(
        generations=5,
        pop_size=20,
        n_hands=1000,
        verbose=True,
    )

    # GA ran without errors
    check_bool("completed 5 generations", len(history) == 5)

    # Population shape is correct
    check_bool(f"population shape is (20, 260): {pop.shape}", pop.shape == (20, 260))

    # Best fitness in gen 4 >= best fitness in gen 0
    gen0_best = history[0]["max"]
    gen4_best = history[4]["max"]
    check_bool(
        f"best fitness non-decreasing: gen0={gen0_best:.4f}, gen4={gen4_best:.4f}",
        gen4_best >= gen0_best,
    )

    # Stats are in valid range
    for record in history:
        check_bool(
            f"gen {record['generation']} max in [0,1]: {record['max']:.4f}",
            0.0 <= record["max"] <= 1.0,
        )

    print()
    if all_passed:
        print("All tests passed.")
    else:
        print("One or more tests FAILED.")

    return all_passed


if __name__ == "__main__":
    run_tests()
