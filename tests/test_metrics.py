import pytest

from hidden_narrowing.metrics import (
    ideological_concentration,
    intra_list_diversity,
    mrr,
    ndcg_at_k,
    source_coverage,
)


def test_ndcg_and_mrr():
    labels = [0, 1, 0]
    scores = [0.2, 0.9, 0.1]
    assert ndcg_at_k(labels, scores, k=10) == pytest.approx(1.0)
    assert mrr(labels, scores) == pytest.approx(1.0)


def test_ideological_concentration():
    ideologies = [-1.0, 0.0, 1.0]
    assert ideological_concentration(ideologies) == pytest.approx(0.0)


def test_intra_list_diversity():
    ideologies = [-1.0, 0.0, 1.0]
    # pairwise distances: 1,2,1 => mean 4/3
    assert intra_list_diversity(ideologies) == pytest.approx(4 / 3)


def test_source_coverage():
    domains = ["a.com", "b.com", "a.com", "c.com"]
    assert source_coverage(domains) == pytest.approx(0.75)
