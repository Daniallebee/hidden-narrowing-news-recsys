from __future__ import annotations


def _mean(xs: list[float]) -> float:
    clean = [x for x in xs if x is not None]
    return sum(clean) / len(clean) if clean else 0.0


def _cross_gain(user_ideology: float | None, item_ideology: float | None) -> float:
    if user_ideology is None or item_ideology is None or abs(user_ideology) < 0.2:
        return 0.0
    return 1.0 if (user_ideology > 0 > item_ideology) or (user_ideology < 0 < item_ideology) else 0.0


def greedy_breadth_rerank(candidates_rows: list[dict], top_k: int = 10, lambda_breadth: float = 0.35, user_ideology: float | None = None) -> list[dict]:
    remaining = [dict(r) for r in candidates_rows]
    selected: list[dict] = []

    while remaining and len(selected) < min(top_k, len(candidates_rows)):
        selected_ideos = [r.get("ideology_score") for r in selected if r.get("ideology_score") is not None]
        cur_mean = _mean(selected_ideos)
        selected_domains = {r.get("domain") for r in selected if r.get("domain")}

        best_i, best_final = 0, -10**9
        for i, row in enumerate(remaining):
            ideo = row.get("ideology_score")
            rel = float(row.get("relevance_score", 0.0))
            dom = row.get("domain")

            diversity_gain = abs(ideo - cur_mean) if ideo is not None else 0.0
            cross_cut = _cross_gain(user_ideology, ideo)
            source_novelty = 1.0 if dom and dom not in selected_domains else 0.0
            new_mean = _mean(selected_ideos + ([ideo] if ideo is not None else []))
            concentration_penalty = max(0.0, abs(new_mean) - abs(cur_mean))

            breadth_term = (
                0.35 * diversity_gain
                + 0.35 * cross_cut
                + 0.10 * source_novelty
                - 0.20 * concentration_penalty
            )
            final_score = rel + lambda_breadth * breadth_term
            if final_score > best_final:
                best_i, best_final = i, final_score

        chosen = remaining.pop(best_i)
        chosen["final_score"] = best_final
        selected.append(chosen)
    return selected
