from __future__ import annotations

import math
import random


def ndcg_at_k(labels: list[int], scores: list[float], k: int = 10) -> float:
    if not labels:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    ranked = [labels[i] for i in order]
    dcg = sum(((2**rel - 1) / math.log2(i + 2)) for i, rel in enumerate(ranked))

    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum(((2**rel - 1) / math.log2(i + 2)) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(labels: list[int], scores: list[float]) -> float:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    for rank, i in enumerate(order, start=1):
        if labels[i] > 0:
            return 1.0 / rank
    return 0.0


def hit_at_k(labels: list[int], scores: list[float], k: int = 10) -> float:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return 1.0 if any(labels[i] > 0 for i in order) else 0.0


def _clean_ideos(values) -> list[float]:
    return [float(v) for v in values if v is not None]


def average_ideology(ideologies) -> float:
    vals = _clean_ideos(ideologies)
    return sum(vals) / len(vals) if vals else 0.0


def ideological_concentration(ideologies) -> float:
    return abs(average_ideology(ideologies))


def intra_list_diversity(ideologies) -> float:
    vals = _clean_ideos(ideologies)
    if len(vals) < 2:
        return 0.0
    pairs = []
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            pairs.append(abs(vals[i] - vals[j]))
    return sum(pairs) / len(pairs) if pairs else 0.0


def cross_cutting_exposure_rate(ideologies, user_ideology: float | None) -> float:
    vals = _clean_ideos(ideologies)
    if not vals or user_ideology is None or abs(user_ideology) < 0.2:
        return 0.0
    opp = [1 if (user_ideology > 0 > v) or (user_ideology < 0 < v) else 0 for v in vals]
    return sum(opp) / len(opp)


def source_coverage(domains) -> float:
    clean = [d for d in domains if isinstance(d, str) and d.strip()]
    return len(set(clean)) / len(clean) if clean else 0.0


def bootstrap_ci_paired(values_by_condition: dict[str, list[float]], samples: int = 1000, seed: int = 42, ci: float = 0.95) -> dict[str, tuple[float, float]]:
    conditions = list(values_by_condition.keys())
    n = min((len(v) for v in values_by_condition.values()), default=0)
    if n == 0:
        return {c: (0.0, 0.0) for c in conditions}
    rng = random.Random(seed)
    results = {c: [] for c in conditions}
    for _ in range(samples):
        idx = [rng.randrange(n) for _ in range(n)]
        for c in conditions:
            vals = values_by_condition[c]
            results[c].append(sum(vals[i] for i in idx) / n)

    alpha = (1 - ci) / 2
    lo_ix = max(0, int(alpha * samples) - 1)
    hi_ix = min(samples - 1, int((1 - alpha) * samples) - 1)
    out = {}
    for c in conditions:
        s = sorted(results[c])
        out[c] = (float(s[lo_ix]), float(s[hi_ix]))
    return out
