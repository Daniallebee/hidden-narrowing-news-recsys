from __future__ import annotations

from . import metrics


def greedy_breadth_rerank(
    candidates_rows: list[dict],
    article_vectors: dict[str, dict[str, float]],
    top_k: int = 10,
    lambda_breadth: float = 0.35,
    user_dominant_subcategory: str | None = None,
) -> list[dict]:
    remaining = [dict(r) for r in candidates_rows]
    selected: list[dict] = []

    while remaining and len(selected) < min(top_k, len(candidates_rows)):
        selected_ids = [r.get("NewsID", "") for r in selected]
        selected_subcats = [r.get("SubCategory", "") for r in selected]

        best_i, best_final = 0, -10**9
        for i, row in enumerate(remaining):
            rel = float(row.get("relevance_score", 0.0))
            nid = row.get("NewsID", "")
            subcat = row.get("SubCategory", "")

            if selected_ids:
                dists = [1.0 - metrics.cosine_similarity(article_vectors.get(nid, {}), article_vectors.get(sid, {})) for sid in selected_ids]
                semantic_diversity_gain = sum(dists) / len(dists) if dists else 0.0
            else:
                semantic_diversity_gain = 0.0

            cross_topic_gain = 1.0 if user_dominant_subcategory and subcat and subcat != user_dominant_subcategory else 0.0
            subcategory_novelty = 1.0 if subcat and subcat not in set(selected_subcats) else 0.0

            new_subcats = selected_subcats + ([subcat] if subcat else [])
            topical_concentration_penalty = metrics.topical_concentration(new_subcats)

            breadth_term = (
                0.35 * semantic_diversity_gain
                + 0.35 * cross_topic_gain
                + 0.15 * subcategory_novelty
                - 0.15 * topical_concentration_penalty
            )
            final_score = rel + lambda_breadth * breadth_term
            if final_score > best_final:
                best_i, best_final = i, final_score

        chosen = remaining.pop(best_i)
        chosen["final_score"] = best_final
        selected.append(chosen)
    return selected
