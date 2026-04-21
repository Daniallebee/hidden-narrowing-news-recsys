import pytest

from hidden_narrowing.metrics import (
    cross_topic_rate,
    history_congruent_share,
    mrr,
    ndcg_at_k,
    topical_concentration,
    topical_entropy,
)


def test_ndcg_and_mrr():
    labels = [0, 1, 0]
    scores = [0.2, 0.9, 0.1]
    assert ndcg_at_k(labels, scores, k=10) == pytest.approx(1.0)
    assert mrr(labels, scores) == pytest.approx(1.0)


def test_topical_concentration():
    subcats = ["newsus", "newsus", "newsworld", "newsopinion"]
    assert topical_concentration(subcats) == pytest.approx(0.5)


def test_topical_entropy():
    subcats = ["newsus", "newspolitics", "newsworld", "newsopinion"]
    assert topical_entropy(subcats, support_size=4) == pytest.approx(1.0)


def test_cross_topic_and_history_congruent_share():
    subcats = ["newsus", "newsworld", "newsus", "newspolitics"]
    assert cross_topic_rate(subcats, "newsus") == pytest.approx(0.5)
    assert history_congruent_share(subcats, "newsus") == pytest.approx(0.5)
