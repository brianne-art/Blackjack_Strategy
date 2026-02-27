# Blackjack_Strategy
# Blackjack Genetic Algorithm — Implementation Spec

---

## Overview

This project evolves a blackjack playing strategy using a genetic algorithm. A population of strategy chromosomes is evolved over many generations, with fitness determined by simulated blackjack play. The goal is to discover a hit/stand strategy approaching the best achievable win rate without splitting or doubling (~42–44% wins, ~49% combined win+tie score).

> **Scope note:** This implementation encodes only hit/stand decisions. Split and double-down actions are excluded. The commonly cited "49.5% theoretical optimum" applies to full basic strategy including doubles and splits; without those actions, the reachable ceiling for this chromosome encoding is lower (~0.49 fitness score when ties are weighted at 0.5, but with a lower raw win rate). The reference line in the visualization should reflect this.

The implementation is broken into four phases, each independently testable before moving to the next.

---

## Phase 1: Card & Hand Utilities

The foundation of the entire project. Everything else depends on these being correct, so build and test this phase thoroughly before moving on.

### Deck representation

Represent a deck as a list of 52 integers. Card values are: 2–10 at face value, Jack/Queen/King = 10, Ace stored as 11 internally. Shuffle the deck at the start of each hand using `random.shuffle` or `np.random.shuffle`.

```
deck = [2,3,4,5,6,7,8,9,10,10,10,10,11] * 4  # 11 represents Ace
```

> **Ace encoding note:** The deck stores Aces as `11`. The chromosome index scheme uses `dealer_upcard=1` for an Ace. Any function that receives a card value from the deck and uses it as a chromosome index must convert `11 → 1` explicitly. Forgetting this conversion is a common silent bug.

### Deck safety

A single hand can consume many cards (e.g., a player hitting repeatedly plus dealer hits). Reshuffle the deck when fewer than 15 cards remain rather than letting it run empty.

```python
if len(deck) < 15:
    deck = [2,3,4,5,6,7,8,9,10,10,10,10,11] * 4
    random.shuffle(deck)
```

### Hand total function

This is the most error-prone piece of the whole project. Write a function `hand_total(cards)` that takes a list of card integers and returns `(total, is_soft)`.

The logic is:
1. Sum all cards, treating Ace as 11
2. Count the number of aces
3. While total > 21 and there are aces being counted as 11, subtract 10 and decrement the soft ace count
4. `is_soft` is True if at least one ace is still being counted as 11 in the final total

Test cases to verify before proceeding:

| Hand | Expected total | Expected is_soft |
|---|---|---|
| [11, 6] | 17 | True |
| [11, 11] | 12 | True |
| [11, 11, 11] | 13 | True (one ace still as 11: 1+1+11=13) |
| [11, 9] | 20 | True |
| [11, 10] | 21 | True |
| [11, 6, 8] | 15 | False |
| [5, 6, 10] | 21 | False |
| [10, 10, 5] | 25 (bust) | False |

### Suggested test approach

Write these as simple assertions or a unittest block. If any fail, fix `hand_total` before continuing. A bug here will silently corrupt every fitness evaluation downstream.

---

## Phase 2: Blackjack Simulation Engine

With correct hand utilities in place, build the single-hand simulation and the fitness evaluator.

### Strategy lookup

Write a function `get_decision(chromosome, player_total, is_soft, dealer_upcard)` that returns 0 (stand) or 1 (hit) by indexing into the chromosome.

> **Ace conversion:** `dealer_upcard` must be passed as `1` when the dealer's card is an Ace (stored as `11` in the deck). Convert before calling this function: `upcard = 1 if card == 11 else card`.

The index scheme is:

```
# dealer_upcard: Ace=1, 2–10 as face value
# Hard hands: player totals 4–20 (17 values), dealer 1–10 (10 values)
hard_index = (player_total - 4) * 10 + (dealer_upcard - 1)   # range 0–169

# Soft hands: soft totals 12–20 (9 values), dealer 1–10 (10 values)
# Note: soft_total 12 = Ace+Ace, soft_total 20 = Ace+9. Soft 21 is handled by the >= 21 guard.
soft_index = 170 + (soft_total - 12) * 10 + (dealer_upcard - 1)  # range 170–259
```

Handle edge cases explicitly — if total >= 21, return stand (0) immediately without indexing. If somehow total < 4, return hit (1).

### Single hand simulation

Write a function `simulate_hand(chromosome, deck)` that plays one hand and returns `"win"`, `"loss"`, or `"tie"`. The deck is passed in pre-shuffled and cards are popped from it.

The sequence is:
1. Check deck length and reshuffle if fewer than 15 cards remain
2. Deal two cards to player, two to dealer (one dealer card face down)
3. Check for player natural blackjack (two-card 21) — if yes, check dealer blackjack too (tie if both have it, otherwise player wins)
4. Player acts: loop calling `get_decision` until the strategy returns stand or player busts
5. If player busts, return loss immediately
6. Dealer acts: hit until total >= 17 (dealer stands on soft 17)
7. If dealer busts, return win
8. Compare totals: higher is a win, equal is a tie, lower is a loss

### Fitness function

Write `evaluate_fitness(chromosome, n_hands=5000)` that simulates `n_hands` hands and returns:

```
fitness = (wins + 0.5 * ties) / n_hands
```

Shuffle a fresh 52-card deck before each hand. Count wins, losses, and ties. Return the scalar fitness value.

> **Why 5,000 hands:** At 1,000 hands the standard deviation of the fitness score is ~±1.5%, which is larger than the difference between a mediocre and a good strategy. At 5,000 hands this drops to ~±0.7%, providing enough signal for selection to work reliably. Use 1,000 only for the quick smoke-test in Phase 3.

### Suggested test approach

Before wiring this into the GA, verify the following with hard-coded chromosomes and explicit assertions:

1. **Always-stand vs always-hit:** Always-stand (all zeros) should return fitness ~0.35–0.40. Always-hit (all ones) should be noticeably worse, ~0.10–0.20. If these are reversed, the simulation logic is wrong.
2. **Natural blackjack detection:** Deal a hand where the player has [11, 10] and the dealer does not — confirm `"win"` is returned immediately without the player taking any action.
3. **Simultaneous blackjack:** Deal a hand where both player and dealer have [11, 10] — confirm `"tie"`.
4. **Dealer soft-17:** Construct a scenario where the dealer has [11, 6] (soft 17) and confirm the dealer hits rather than standing.

---

## Phase 3: Genetic Algorithm

With a working fitness function, implement the evolution loop.

### Population initialization

```python
population = np.random.randint(0, 2, size=(100, 260), dtype=np.uint8)
```

### Selection

Use roulette wheel (fitness-proportionate) selection. Normalize fitnesses to sum to 1.0 and use `np.random.choice` with the `p` argument to sample parent indices.

```python
probs = fitnesses / fitnesses.sum()
parent_idx = np.random.choice(len(population), p=probs)
```

> **Premature convergence:** With a small population (100), roulette wheel selection can converge prematurely if one individual dominates. If diversity collapses early (e.g., population standard deviation of fitness < 0.001 before generation 20), consider switching to **tournament selection** (pick k=5 random individuals, return the best). Tournament selection is more resistant to one individual crowding out the rest.

### Crossover

Single-point crossover. Pick a random point in `[1, 258]`, take the left segment from parent 1 and the right segment from parent 2.

```python
point = np.random.randint(1, 259)
child = np.concatenate([parent1[:point], parent2[point:]])
```

### Mutation

Flip each bit independently with probability 0.01.

```python
mask = np.random.rand(260) < 0.01
child = np.where(mask, 1 - child, child)
```

> **Adaptive mutation:** If you observe stagnation (best fitness unchanged for 10+ generations), temporarily raise the mutation rate to 0.05 for one generation to re-introduce diversity, then revert to 0.01.

### Elitism

Before building the new population, identify the top 2 individuals by fitness and copy them unchanged into the next generation. Fill the remaining 98 slots with crossover+mutation offspring.

### Generation loop structure

```
initialize population (100 × 260)
for generation in range(50):
    evaluate fitness for all 100 individuals (5,000 hands each)
    record min, max, mean, median fitness for this generation
    sort population by fitness descending
    new_population = [elite_1, elite_2]
    while len(new_population) < 100:
        select parent_1, parent_2 via roulette wheel
        child = crossover(parent_1, parent_2)
        child = mutate(child)
        append child to new_population
    population = new_population
```

> **Performance note:** 100 individuals × 5,000 hands × 50 generations = 25 million simulated hands. In pure Python this will be slow (potentially 10–30 minutes). If speed is a concern, evaluate fitness in parallel using `multiprocessing.Pool` or reduce to 1,000 hands during early generations and 5,000 in later generations.

### Suggested test approach

Run for just 5 generations with 1,000 hands per fitness evaluation (fast but noisy). Confirm that the best fitness in generation 5 is higher than in generation 0. You don't need convergence yet — just verify the loop runs without errors and fitness is not declining on average.

---

## Phase 4: Visualization

Once the GA is producing results, generate the two required figures.

### Figure 1 — Fitness over generations

Plot min, max, mean, and median fitness on the y-axis against generation number on the x-axis. Use distinct colors or line styles for each. Add a horizontal dashed reference line representing the best achievable hit/stand-only strategy (empirically ~0.47–0.49; calibrate this by running a known-good basic strategy chromosome). Label axes and add a legend.

### Figure 2 — Strategy heatmap

After the final generation, compute the hit frequency for each cell across all 100 individuals:

```python
hit_frequency = population.mean(axis=0)  # shape (260,)
hard_matrix = hit_frequency[:170].reshape(17, 10)   # rows=player 4–20, cols=dealer A,2–10
soft_matrix = hit_frequency[170:].reshape(9, 10)    # rows=soft 12–20, cols=dealer A,2–10
```

Display as two subplots using `imshow` with a diverging colormap — `RdBu_r` works well with `vmin=0, vmax=1`. Red = hit (1.0), blue = stand (0.0). Add tick labels: rows labeled with player totals, columns labeled with dealer upcards (A, 2, 3, … 10). Annotate each cell with the percentage as an integer (e.g. "73%").

> **Column ordering:** The chromosome indexes dealer Ace as column 0 (`dealer_upcard - 1` where upcard=1). Label this column "A" and columns 1–9 as "2" through "10" so the heatmap reads left-to-right as A, 2, 3, … 10.

### What to look for

The hard strategy matrix should show a clear band of blue (stand) in the upper rows (hard 17–20) and red (hit) in the lower rows (hard 4–11), with a mixed transition zone in the middle that varies based on the dealer's upcard. The soft strategy matrix should show more hitting overall. Any column or row that is uniformly one color is a potential sign of a bug or insufficient generations.

---

## Appendix: Parameters Summary

| Parameter | Value |
|---|---|
| Population size | 100 |
| Chromosome length | 260 bits |
| Generations | 50–100 |
| Hands per fitness eval | 5,000 (1,000 for smoke tests) |
| Mutation rate | 0.01 per bit (0.05 if stagnating) |
| Crossover | Single-point |
| Selection | Roulette wheel (tournament if converging early) |
| Elitism | Top 2 carry over |
| Hard hand state space | 17 × 10 = 170 bits (player 4–20, dealer A/2–10) |
| Soft hand state space | 9 × 10 = 90 bits (soft 12–20, dealer A/2–10) |
| Actions encoded | Hit / Stand only (no split, no double-down) |
| Ace in deck | Stored as 11; converted to 1 for chromosome indexing |
