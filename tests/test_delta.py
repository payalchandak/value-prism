"""Hand-checked Δ rows.

Tags are listed in A, B, N, J order. Score: promotes = +1, neutral = 0,
violates = −1. Δ = score(choice 1) − score(choice 2).
"""

from valueprism.data import load_cases

# 065d7abf
#   choice 1: neutral, promotes, promotes, neutral
#   choice 2: promotes, violates, violates, neutral
#   Δ = (0-1, 1-(-1), 1-(-1), 0-0) = (-1, 2, 2, 0)
# 0b5a2bcb
#   choice 1: violates, promotes, promotes, violates
#   choice 2: promotes, neutral, neutral, promotes
#   Δ = (-1-1, 1-0, 1-0, -1-1) = (-2, 1, 1, -2)
# 0c790bb1
#   choice 1: promotes, neutral, violates, neutral
#   choice 2: violates, neutral, promotes, neutral
#   Δ = (1-(-1), 0-0, -1-1, 0-0) = (2, 0, -2, 0)
HAND = {
    "065d7abf-3671-43a2-8375-79fb6e87ed78": (-1, 2, 2, 0),
    "0b5a2bcb-0946-461a-8ba1-8f29213f01c5": (-2, 1, 1, -2),
    "0c790bb1-dd0d-4724-8502-e8613968dd04": (2, 0, -2, 0),
}


def test_delta_matches_hand_checked_cases():
    cases, delta = load_cases()
    assert list(delta.columns) == ["A", "B", "N", "J"]
    assert delta.shape == (len(cases), 4)
    assert list(delta.index) == list(cases["case_id"])
    for case_id, expected in HAND.items():
        assert tuple(int(v) for v in delta.loc[case_id].tolist()) == expected
