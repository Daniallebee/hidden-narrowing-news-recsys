from hidden_narrowing.metrics import bootstrap_ci_paired


def test_bootstrap_ci_paired_returns_bounds():
    values = {"baseline": [0.1, 0.2, 0.3], "breadth": [0.2, 0.3, 0.4]}
    out = bootstrap_ci_paired(values, samples=200, seed=42)
    assert set(out.keys()) == {"baseline", "breadth"}
    for lo, hi in out.values():
        assert lo <= hi
