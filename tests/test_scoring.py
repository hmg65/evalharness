from evalharness.scoring import (
    ContainsScorer, LengthScorer, NotRefusalScorer, RegexScorer,
    build_scorers, weighted_score,
)


def test_contains_scorer():
    s = ContainsScorer()
    assert s.score("p", "42", "The answer is 42.")[0] == 1.0
    assert s.score("p", "42", "I think it is 41")[0] == 0.0
    assert s.score("p", None, "anything")[0] == 0.0


def test_regex_scorer():
    s = RegexScorer(pattern=r"O\(1\)")
    assert s.score("p", None, "average O(1) lookup")[0] == 1.0
    assert s.score("p", None, "linear time")[0] == 0.0


def test_length_scorer_band():
    s = LengthScorer(min_words=3, max_words=5)
    assert s.score("p", None, "one two three four")[0] == 1.0
    assert s.score("p", None, "one")[0] < 1.0


def test_not_refusal_scorer():
    s = NotRefusalScorer()
    assert s.score("p", None, "I am not sure about that")[0] == 0.0
    assert s.score("p", None, "The treaty was signed in 1919.")[0] == 1.0


def test_weighted_score_combines():
    scorers = build_scorers([
        {"type": "contains", "weight": 2.0},
        {"type": "not_refusal", "weight": 1.0},
    ])
    value, detail = weighted_score(scorers, "p", "42", "It is 42.")
    assert value == 1.0
    assert set(detail) == {"contains", "not_refusal"}
    value, _ = weighted_score(scorers, "p", "42", "I am not sure, maybe 41")
    assert abs(value - 0.0) < 1e-9


def test_unknown_scorer_rejected():
    try:
        build_scorers([{"type": "does_not_exist"}])
    except ValueError as exc:
        assert "unknown scorer" in str(exc)
    else:
        raise AssertionError("expected ValueError")
