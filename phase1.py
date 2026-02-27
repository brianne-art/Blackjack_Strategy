import random


def make_deck():
    """Return a fresh shuffled 52-card deck. Aces stored as 11."""
    deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4
    random.shuffle(deck)
    return deck


def hand_total(cards):
    """
    Return (total, is_soft) for a list of card integers.
    Aces are stored as 11; this function collapses them to 1 as needed.
    is_soft is True if at least one Ace is still counted as 11.
    """
    total = sum(cards)
    soft_aces = cards.count(11)

    while total > 21 and soft_aces > 0:
        total -= 10
        soft_aces -= 1

    is_soft = soft_aces > 0
    return total, is_soft


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_tests():
    test_cases = [
        ([11, 6],       17, True),
        ([11, 11],      12, True),
        ([11, 11, 11],  13, True),   # 1+1+11=13, one ace still as 11
        ([11, 9],       20, True),
        ([11, 10],      21, True),
        ([11, 6, 8],    15, False),
        ([5, 6, 10],    21, False),
        ([10, 10, 5],   25, False),
    ]

    all_passed = True
    for hand, expected_total, expected_soft in test_cases:
        total, is_soft = hand_total(hand)
        passed = (total == expected_total) and (is_soft == expected_soft)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] hand={hand} → total={total} (expected {expected_total}), "
              f"is_soft={is_soft} (expected {expected_soft})")

    print()
    if all_passed:
        print("All tests passed.")
    else:
        print("One or more tests FAILED.")

    return all_passed


if __name__ == "__main__":
    run_tests()
