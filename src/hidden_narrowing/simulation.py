from __future__ import annotations

import csv
import math
import random
from collections import Counter
from pathlib import Path

from . import baseline, features, metrics, rerank
from .data_mind import PUBLIC_AFFAIRS_SUBCATEGORIES


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _softmax_sample(ids: list[str], scores: list[float], temperature: float, rng: random.Random) -> str:
    if not ids:
        return ""
    if temperature <= 0:
        return ids[max(range(len(scores)), key=lambda i: scores[i])]
    mx = max(scores) if scores else 0.0
    exps = [math.exp((s - mx) / temperature) for s in scores]
    denom = sum(exps) or 1.0
    p = [e / denom for e in exps]
    x = rng.random()
    c = 0.0
    for i, pr in enumerate(p):
        c += pr
        if x <= c:
            return ids[i]
    return ids[-1]


def run_repeated_rounds(
    user_ids: list[str],
    candidate_pool: list[str],
    histories_by_user: dict[str, list[str]],
    news_by_id: dict[str, dict],
    article_vectors: dict[str, dict[str, float]],
    model: baseline.BaselineModel,
    popularity: Counter,
    results_path: Path,
    rounds: int = 5,
    top_k: int = 10,
    lambda_breadth: float = 0.35,
    temperature: float = 0.5,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []

    for user_id in user_ids:
        init_hist = histories_by_user.get(user_id, [])
        if len(init_hist) < 1:
            continue

        for condition in ["baseline", "breadth_aware"]:
            hist = list(init_hist)
            for rd in range(1, rounds + 1):
                profile = features.build_user_subcategory_profile(hist, news_by_id, PUBLIC_AFFAIRS_SUBCATEGORIES)
                user_dom = profile["dominant_subcategory"]

                rel_scores = baseline.score_candidates(
                    model=model,
                    user_id=user_id,
                    user_history_ids=hist,
                    candidate_ids=candidate_pool,
                    article_vectors=article_vectors,
                    news_by_id=news_by_id,
                    popularity=popularity,
                )
                candidates = []
                for nid, s in zip(candidate_pool, rel_scores):
                    n = news_by_id.get(nid, {})
                    candidates.append(
                        {
                            "NewsID": nid,
                            "relevance_score": float(s),
                            "SubCategory": n.get("SubCategory", "").strip().lower(),
                        }
                    )

                if condition == "baseline":
                    ranked = baseline.rank_candidates(candidates)[:top_k]
                else:
                    ranked = rerank.greedy_breadth_rerank(
                        candidates,
                        article_vectors=article_vectors,
                        top_k=top_k,
                        lambda_breadth=lambda_breadth,
                        user_dominant_subcategory=user_dom,
                    )

                rec_ids = [r["NewsID"] for r in ranked]
                rec_subcats = [r.get("SubCategory", "") for r in ranked]

                click_scores = [float(r["relevance_score"]) for r in ranked]
                clicked_news_id = _softmax_sample(rec_ids, click_scores, temperature, rng)
                clicked_subcategory = news_by_id.get(clicked_news_id, {}).get("SubCategory", "").strip().lower()
                hist.append(clicked_news_id)

                rows.append(
                    {
                        "user_id": user_id,
                        "round": rd,
                        "condition": condition,
                        "topical_concentration": metrics.topical_concentration(rec_subcats),
                        "subcategory_coverage": metrics.subcategory_coverage(rec_subcats),
                        "topical_entropy": metrics.topical_entropy(rec_subcats, support_size=len(PUBLIC_AFFAIRS_SUBCATEGORIES)),
                        "semantic_diversity": metrics.semantic_diversity(rec_ids, article_vectors),
                        "cross_topic_rate": metrics.cross_topic_rate(rec_subcats, user_dom),
                        "history_congruent_share": metrics.history_congruent_share(rec_subcats, user_dom),
                        "clicked_news_id": clicked_news_id,
                        "clicked_subcategory": clicked_subcategory,
                    }
                )

    _write_csv(results_path, rows)
    return rows
