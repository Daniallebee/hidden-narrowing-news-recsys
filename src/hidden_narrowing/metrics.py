from __future__ import annotations

import math
import random
from collections import Counter


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


def topical_concentration(subcategories: list[str]) -> float:
    clean = [s for s in subcategories if isinstance(s, str) and s.strip()]
    if not clean:
        return 0.0
    counts = Counter(clean)
    return max(counts.values()) / len(clean)


def subcategory_coverage(subcategories: list[str]) -> float:
    clean = [s for s in subcategories if isinstance(s, str) and s.strip()]
    return float(len(set(clean))) if clean else 0.0


def topical_entropy(subcategories: list[str], support_size: int | None = None) -> float:
    clean = [s for s in subcategories if isinstance(s, str) and s.strip()]
    if not clean:
        return 0.0
    counts = Counter(clean)
    total = sum(counts.values())
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p, 2)
    denom_k = support_size if support_size is not None else len(counts)
    denom_k = max(denom_k, len(counts))
    denom = math.log(max(1, denom_k), 2)
    return ent / denom if denom > 0 else 0.0


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na <= 0 or nb <= 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def semantic_diversity(news_ids: list[str], article_vectors: dict[str, dict[str, float]]) -> float:
    vecs = [article_vectors.get(nid, {}) for nid in news_ids]
    vecs = [v for v in vecs if v]
    if len(vecs) < 2:
        return 0.0
    distances: list[float] = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            distances.append(1.0 - cosine_similarity(vecs[i], vecs[j]))
    return sum(distances) / len(distances) if distances else 0.0


def cross_topic_rate(subcategories: list[str], user_dominant_subcategory: str | None) -> float:
    clean = [s for s in subcategories if isinstance(s, str) and s.strip()]
    if not clean or not user_dominant_subcategory:
        return 0.0
    return sum(1 for s in clean if s != user_dominant_subcategory) / len(clean)


def history_congruent_share(subcategories: list[str], user_dominant_subcategory: str | None) -> float:
    clean = [s for s in subcategories if isinstance(s, str) and s.strip()]
    if not clean or not user_dominant_subcategory:
        return 0.0
    return sum(1 for s in clean if s == user_dominant_subcategory) / len(clean)


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
