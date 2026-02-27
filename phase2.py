import random
import numpy as np

from phase1 import make_deck, hand_total

# ---------------------------------------------------------------------------
# Strategy lookup
# ---------------------------------------------------------------------------

def get_decision(chromosome, player_total, is_soft, dealer_upcard):
    """
    Return 0 (stand) or 1 (hit) by looking up the chromosome.

    dealer_upcard must already be converted: Ace=1, 2-10 as face value.
    Cards come from the deck as 11 for Ace — convert before calling:
        upcard = 1 if card == 11 else card
    """
    if player_total >= 21:
        return 0  # stand
    if player_total < 4:
        return 1  # hit

    if is_soft:
        # Soft totals 12-20 → index 170-259
        soft_total = player_total
        index = 170 + (soft_total - 12) * 10 + (dealer_upcard - 1)
    else:
        # Hard totals 4-20 → index 0-169
        index = (player_total - 4) * 10 + (dealer_upcard - 1)

    return int(chromosome[index])


# ---------------------------------------------------------------------------
# Single hand simulation
# ---------------------------------------------------------------------------

def _reshuffle_if_low(deck):
    if len(deck) < 15:
        deck.clear()
        deck.extend([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4)
        random.shuffle(deck)


def simulate_hand(chromosome, deck):
    """
    Play one hand using chromosome as the strategy.
    Returns "win", "loss", or "tie".
    Modifies deck in place (pops cards); reshuffles before hit cards if fewer
    than 15 remain (initial deal always requires exactly 4 cards).
    """
    # Deal: player gets 2 cards, dealer gets 2 cards
    player = [deck.pop(), deck.pop()]
    dealer = [deck.pop(), deck.pop()]

    # Convert dealer upcard (face-up = dealer[0]) for chromosome indexing
    dealer_upcard = 1 if dealer[0] == 11 else dealer[0]

    # Check natural blackjack (two-card 21)
    player_total, _ = hand_total(player)
    dealer_total, _ = hand_total(dealer)

    if player_total == 21:
        return "tie" if dealer_total == 21 else "win"
    if dealer_total == 21:
        return "loss"

    # Player acts
    while True:
        player_total, is_soft = hand_total(player)
        if get_decision(chromosome, player_total, is_soft, dealer_upcard) == 0:
            break
        player.append(deck.pop())
        player_total, _ = hand_total(player)
        if player_total > 21:
            return "loss"

    # Dealer acts: hit on any total < 17, and also hit on soft 17
    while True:
        dealer_total, is_soft = hand_total(dealer)
        if dealer_total > 17:
            break
        if dealer_total == 17 and not is_soft:
            break
        dealer.append(deck.pop())

    dealer_total, _ = hand_total(dealer)

    if dealer_total > 21:
        return "win"

    player_total, _ = hand_total(player)
    if player_total > dealer_total:
        return "win"
    elif player_total == dealer_total:
        return "tie"
    else:
        return "loss"


# ---------------------------------------------------------------------------
# Fitness function
# ---------------------------------------------------------------------------

def evaluate_fitness(chromosome, n_hands=5000):
    """
    Simulate n_hands hands and return fitness = (wins + 0.5 * ties) / n_hands.
    """
    wins = ties = 0
    deck = make_deck()

    for _ in range(n_hands):
        _reshuffle_if_low(deck)
        result = simulate_hand(chromosome, deck)
        if result == "win":
            wins += 1
        elif result == "tie":
            ties += 1

    return (wins + 0.5 * ties) / n_hands


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _make_fixed_deck(cards):
    """Return a list that pops cards in the given order (last element first)."""
    return list(reversed(cards))


def run_tests():
    all_passed = True

    def check(label, result, expected):
        nonlocal all_passed
        passed = result == expected
        if not passed:
            all_passed = False
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {label}: got {result!r}, expected {expected!r}")

    # ------------------------------------------------------------------
    # Test 1: always-stand vs always-hit (statistical)
    # ------------------------------------------------------------------
    always_stand = np.zeros(260, dtype=np.uint8)
    always_hit   = np.ones(260,  dtype=np.uint8)

    random.seed(42)
    fitness_stand = evaluate_fitness(always_stand, n_hands=5000)
    fitness_hit   = evaluate_fitness(always_hit,   n_hands=5000)

    stand_ok = 0.30 <= fitness_stand <= 0.45
    hit_ok   = 0.05 <= fitness_hit   <= 0.25
    order_ok = fitness_stand > fitness_hit

    def check_bool(label, value):
        nonlocal all_passed
        if not value:
            all_passed = False
        print(f"[{'PASS' if value else 'FAIL'}] {label}")

    check_bool(
        f"always-stand fitness in [0.30, 0.45]: {fitness_stand:.4f}",
        stand_ok,
    )
    check_bool(
        f"always-hit fitness in [0.05, 0.25]: {fitness_hit:.4f}",
        hit_ok,
    )
    check_bool(
        f"always-stand ({fitness_stand:.4f}) > always-hit ({fitness_hit:.4f})",
        order_ok,
    )

    # ------------------------------------------------------------------
    # Test 2: player natural blackjack (dealer does not have it)
    # Deck pops from the end, so deal order: player[0], player[1], dealer[0], dealer[1]
    # player=[11,10]=21, dealer=[5,7]=12
    # ------------------------------------------------------------------
    chromosome = np.zeros(260, dtype=np.uint8)
    deck = _make_fixed_deck([11, 10, 5, 7])
    result = simulate_hand(chromosome, deck)
    check("player natural BJ (dealer no BJ) → win", result, "win")

    # ------------------------------------------------------------------
    # Test 3: simultaneous blackjack → tie
    # player=[11,10]=21, dealer=[11,10]=21
    # ------------------------------------------------------------------
    deck = _make_fixed_deck([11, 10, 11, 10])
    result = simulate_hand(chromosome, deck)
    check("simultaneous BJ → tie", result, "tie")

    # ------------------------------------------------------------------
    # Test 4: dealer natural blackjack, player no BJ → loss
    # player=[5,7]=12, dealer=[11,10]=21
    # ------------------------------------------------------------------
    deck = _make_fixed_deck([5, 7, 11, 10])
    result = simulate_hand(chromosome, deck)
    check("dealer natural BJ, player no BJ → loss", result, "loss")

    # ------------------------------------------------------------------
    # Test 5: dealer soft-17 — dealer must hit
    # always-stand chromosome: player stands on any total
    # player=[10,8]=18 (stands), dealer=[11,6]=soft-17 (must hit)
    # Give dealer a 2 as the next card → dealer total=19, player loses
    # ------------------------------------------------------------------
    deck = _make_fixed_deck([10, 8, 11, 6, 2])
    result = simulate_hand(chromosome, deck)
    # Dealer hits soft-17 → gets 2 → total=19; player has 18 → loss
    check("dealer hits soft-17 (player 18 loses to dealer 19)", result, "loss")

    # ------------------------------------------------------------------
    # Test 6: player bust → loss immediately (dealer not acted)
    # always-hit chromosome; player=[5,7]=12, next cards push player over 21
    # dealer=[2,3]=5 (irrelevant since player busts)
    # ------------------------------------------------------------------
    always_hit_chrom = np.ones(260, dtype=np.uint8)
    deck = _make_fixed_deck([5, 7, 2, 3, 6, 9])  # player hits 5+7=12, +6=18, +9=27
    result = simulate_hand(always_hit_chrom, deck)
    check("player bust → loss", result, "loss")

    # ------------------------------------------------------------------
    # Test 7: dealer bust → win
    # player=[10,8]=18 (stands), dealer=[10,6]=16 (hits), +10=26 (bust)
    # ------------------------------------------------------------------
    deck = _make_fixed_deck([10, 8, 10, 6, 10])
    result = simulate_hand(chromosome, deck)
    check("dealer bust → win", result, "win")

    print()
    if all_passed:
        print("All tests passed.")
    else:
        print("One or more tests FAILED.")

    return all_passed


if __name__ == "__main__":
    run_tests()
