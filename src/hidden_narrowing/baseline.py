from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


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


def rank_candidates(candidates_rows: list[dict], score_field: str = "relevance_score") -> list[dict]:
    return sorted(candidates_rows, key=lambda r: float(r.get(score_field, 0.0)), reverse=True)


@dataclass
class BaselineModel:
    mode: str
    model: Any | None = None
    feature_names: list[str] | None = None


def _build_feature_row(
    user_vec: dict[str, float],
    candidate_news: dict,
    user_history_ids: list[str],
    history_set: set[str],
    article_vectors: dict[str, dict[str, float]],
    news_by_id: dict[str, dict],
    popularity: dict[str, int],
) -> list[float]:
    candidate_id = candidate_news["NewsID"]
    candidate_vec = article_vectors.get(candidate_id, {})
    cosine = cosine_similarity(user_vec, candidate_vec)

    category_match = 0.0
    subcategory_match = 0.0
    if user_history_ids:
        hist_news = [news_by_id.get(nid, {}) for nid in user_history_ids]
        candidate_cat = candidate_news.get("Category")
        candidate_subcat = candidate_news.get("SubCategory")
        if candidate_cat:
            category_match = 1.0 if any(n.get("Category") == candidate_cat for n in hist_news) else 0.0
        if candidate_subcat:
            subcategory_match = 1.0 if any(n.get("SubCategory") == candidate_subcat for n in hist_news) else 0.0

    pop = float(popularity.get(candidate_id, 0))
    hist_len = float(len(history_set))
    ideology_available = 1.0 if candidate_news.get("ideology_score") is not None else 0.0
    return [cosine, category_match, subcategory_match, pop, hist_len, ideology_available]


def build_training_examples(
    impressions: list[dict],
    histories_by_user: dict[str, list[str]],
    article_vectors: dict[str, dict[str, float]],
    news_by_id: dict[str, dict],
    popularity: dict[str, int],
) -> tuple[list[list[float]], list[int]]:
    X: list[list[float]] = []
    y: list[int] = []
    for row in impressions:
        user_id = row["UserID"]
        nid = row["NewsID"]
        user_hist = histories_by_user.get(user_id, [])
        user_vec = _mean_vectors([article_vectors.get(hid, {}) for hid in user_hist])
        feats = _build_feature_row(
            user_vec=user_vec,
            candidate_news=news_by_id.get(nid, {"NewsID": nid}),
            user_history_ids=user_hist,
            history_set=set(user_hist),
            article_vectors=article_vectors,
            news_by_id=news_by_id,
            popularity=popularity,
        )
        X.append(feats)
        y.append(int(row.get("clicked", 0)))
    return X, y


def _mean_vectors(vectors: list[dict[str, float]]) -> dict[str, float]:
    valid = [v for v in vectors if v]
    if not valid:
        return {}
    acc: dict[str, float] = {}
    for vec in valid:
        for k, v in vec.items():
            acc[k] = acc.get(k, 0.0) + v
    n = float(len(valid))
    return {k: v / n for k, v in acc.items()}


def train_logistic_regression_baseline(
    impressions: list[dict],
    histories_by_user: dict[str, list[str]],
    article_vectors: dict[str, dict[str, float]],
    news_by_id: dict[str, dict],
    popularity: dict[str, int],
) -> BaselineModel:
    X, y = build_training_examples(impressions, histories_by_user, article_vectors, news_by_id, popularity)
    if len(X) < 20 or len(set(y)) < 2:
        return BaselineModel(mode="cosine")

    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        model.fit(X, y)
        return BaselineModel(
            mode="logistic_regression",
            model=model,
            feature_names=[
                "cosine_similarity",
                "category_match",
                "subcategory_match",
                "article_popularity",
                "user_history_length",
                "candidate_ideology_available",
            ],
        )
    except Exception:
        return BaselineModel(mode="cosine")


def score_candidates(
    model: BaselineModel,
    user_id: str,
    user_history_ids: list[str],
    candidate_ids: list[str],
    article_vectors: dict[str, dict[str, float]],
    news_by_id: dict[str, dict],
    popularity: dict[str, int],
) -> list[float]:
    user_vec = _mean_vectors([article_vectors.get(hid, {}) for hid in user_history_ids])
    if model.mode != "logistic_regression" or model.model is None:
        return score_candidates_cosine(user_vec, candidate_ids, article_vectors)

    X = []
    for nid in candidate_ids:
        X.append(
            _build_feature_row(
                user_vec=user_vec,
                candidate_news=news_by_id.get(nid, {"NewsID": nid}),
                user_history_ids=user_history_ids,
                history_set=set(user_history_ids),
                article_vectors=article_vectors,
                news_by_id=news_by_id,
                popularity=popularity,
            )
        )
    probs = model.model.predict_proba(X)
    return [float(p[1]) for p in probs]
