from __future__ import annotations

import math


def _dot(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def _norm(a: dict[str, float]) -> float:
    return math.sqrt(sum(v * v for v in a.values()))


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _dot(a, b) / (na * nb)


def score_candidates_cosine(user_vector: dict[str, float], candidate_news_ids: list[str], article_vectors: dict[str, dict[str, float]]) -> list[float]:
    return [cosine_similarity(user_vector, article_vectors.get(nid, {})) for nid in candidate_news_ids]


def rank_candidates(candidates_rows: list[dict]) -> list[dict]:
    return sorted(candidates_rows, key=lambda r: float(r.get("relevance_score", 0.0)), reverse=True)


def train_logistic_regression_baseline(*args, **kwargs):
    raise NotImplementedError("Logistic regression baseline is not implemented in this scaffold.")
