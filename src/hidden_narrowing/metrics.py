from __future__ import annotations

import math


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
