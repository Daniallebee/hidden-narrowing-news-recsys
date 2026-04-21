from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from . import baseline, data_mind, features, metrics, rerank
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


def _markdown_table(rows: list[dict], headers: list[str]) -> str:
    if not rows:
        return "None"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def _build_histories_by_user(histories_rows: list[dict]) -> dict[str, list[str]]:
    by_user: dict[str, list[str]] = defaultdict(list)
    for row in histories_rows:
        by_user[row["UserID"]].append(row["NewsID"])
    return dict(by_user)


def _resolve_embedding(news_rows: list[dict], method: str) -> tuple[dict, dict[str, dict[str, float]], str]:
    if method == "sentence-transformer":
        try:
            meta, vectors = features.build_sentence_transformer_features(news_rows)
            return meta, vectors, "sentence-transformer"
        except Exception:
            meta, vectors = features.build_tfidf_features(news_rows)
            return meta, vectors, "tfidf"
    meta, vectors = features.build_tfidf_features(news_rows)
    return meta, vectors, "tfidf"


def run_experiment(
    train_news_path: Path,
    train_behaviors_path: Path,
    dev_news_path: Path,
    dev_behaviors_path: Path,
    results_dir: Path,
    reports_dir: Path,
    top_k: int = 10,
    lambda_breadth: float = 0.35,
    embedding_method: str = "tfidf",
    bootstrap_samples: int = 1000,
    lambda_values: list[float] | None = None,
    dataset_label: str = "sample",
    max_train_impressions: int | None = None,
    max_dev_impressions: int | None = None,
    max_users: int | None = None,
    slice_name: str = "public_affairs",
    include_newscrime: bool = False,
) -> tuple[list[dict], list[dict]]:
    lambda_values = lambda_values or [0.15, 0.35, 0.60]

    train = data_mind.parse_mind(str(train_news_path), str(train_behaviors_path), slice_name=slice_name, include_newscrime=include_newscrime)
    dev = data_mind.parse_mind(str(dev_news_path), str(dev_behaviors_path), slice_name=slice_name, include_newscrime=include_newscrime)

    if max_users is not None and max_users > 0:
        keep_users = {r["UserID"] for r in train.behaviors[:max_users] if r.get("UserID")}
        train.behaviors = [r for r in train.behaviors if r.get("UserID") in keep_users]
        train.impressions = [r for r in train.impressions if r.get("UserID") in keep_users]
        train.histories = [r for r in train.histories if r.get("UserID") in keep_users]
        dev.behaviors = [r for r in dev.behaviors if r.get("UserID") in keep_users]
        dev.impressions = [r for r in dev.impressions if r.get("UserID") in keep_users]
        dev.histories = [r for r in dev.histories if r.get("UserID") in keep_users]

    if max_train_impressions is not None and max_train_impressions > 0:
        train.impressions = train.impressions[:max_train_impressions]
    if max_dev_impressions is not None and max_dev_impressions > 0:
        dev.impressions = dev.impressions[:max_dev_impressions]

    merged_news = {n["NewsID"]: n for n in (train.news + dev.news)}
    _, article_vectors, used_embedding_method = _resolve_embedding(list(merged_news.values()), method=embedding_method)
    news_by_id = merged_news

    histories_by_user_train = _build_histories_by_user(train.histories)
    popularity = Counter([r["NewsID"] for r in train.impressions if int(r.get("clicked", 0)) == 1])

    model = baseline.train_logistic_regression_baseline(
        impressions=train.impressions,
        histories_by_user=histories_by_user_train,
        article_vectors=article_vectors,
        news_by_id=news_by_id,
        popularity=popularity,
    )

    imps_by_id: dict[str, list[dict]] = defaultdict(list)
    for imp in dev.impressions:
        imps_by_id[imp["ImpressionID"]].append(imp)

    metric_names = [
        "ndcg_10",
        "mrr",
        "hit_10",
        "topical_concentration",
        "subcategory_coverage",
        "topical_entropy",
        "semantic_diversity",
        "cross_topic_rate",
        "history_congruent_share",
    ]

    static_rows: list[dict] = []
    for imp_id, group in imps_by_id.items():
        user_id = group[0]["UserID"]
        user_hist = histories_by_user_train.get(user_id, [])
        if not user_hist:
            continue

        profile = features.build_user_subcategory_profile(user_hist, news_by_id, PUBLIC_AFFAIRS_SUBCATEGORIES | ({"newscrime"} if include_newscrime else set()))
        user_dominant_subcategory = profile["dominant_subcategory"]

        candidate_ids = [g["NewsID"] for g in group]
        rel_scores = baseline.score_candidates(
            model=model,
            user_id=user_id,
            user_history_ids=user_hist,
            candidate_ids=candidate_ids,
            article_vectors=article_vectors,
            news_by_id=news_by_id,
            popularity=popularity,
        )

        candidates = []
        for g, score in zip(group, rel_scores):
            row_news = news_by_id.get(g["NewsID"], {})
            item = dict(g)
            item["relevance_score"] = float(score)
            item["SubCategory"] = row_news.get("SubCategory", "").strip().lower()
            candidates.append(item)

        baseline_ranked = baseline.rank_candidates(candidates, score_field="relevance_score")[:top_k]
        breadth_ranked = rerank.greedy_breadth_rerank(
            candidates,
            article_vectors=article_vectors,
            top_k=top_k,
            lambda_breadth=lambda_breadth,
            user_dominant_subcategory=user_dominant_subcategory,
        )

        for condition, frame, score_col in [
            ("baseline", baseline_ranked, "relevance_score"),
            ("breadth_aware", breadth_ranked, "final_score"),
        ]:
            labels = [int(r["clicked"]) for r in frame]
            scores = [float(r.get(score_col, 0.0)) for r in frame]
            subcats = [r.get("SubCategory", "") for r in frame]
            ids = [r.get("NewsID", "") for r in frame]
            static_rows.append(
                {
                    "ImpressionID": imp_id,
                    "UserID": user_id,
                    "condition": condition,
                    "ndcg_10": metrics.ndcg_at_k(labels, scores, top_k),
                    "mrr": metrics.mrr(labels, scores),
                    "hit_10": metrics.hit_at_k(labels, scores, top_k),
                    "topical_concentration": metrics.topical_concentration(subcats),
                    "subcategory_coverage": metrics.subcategory_coverage(subcats),
                    "topical_entropy": metrics.topical_entropy(subcats, support_size=len(PUBLIC_AFFAIRS_SUBCATEGORIES)),
                    "semantic_diversity": metrics.semantic_diversity(ids, article_vectors),
                    "cross_topic_rate": metrics.cross_topic_rate(subcats, user_dominant_subcategory),
                    "history_congruent_share": metrics.history_congruent_share(subcats, user_dominant_subcategory),
                }
            )

    _write_csv(results_dir / "static_metrics.csv", static_rows)

    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in static_rows:
        by_condition[row["condition"]].append(row)

    summary_rows: list[dict] = []
    for metric_name in metric_names:
        condition_vals: dict[str, list[float]] = {}
        for condition, rows in by_condition.items():
            condition_vals[condition] = [float(r[metric_name]) for r in rows]
        cis = metrics.bootstrap_ci_paired(condition_vals, samples=bootstrap_samples, seed=42)
        for condition, vals in condition_vals.items():
            if not vals:
                continue
            summary_rows.append(
                {
                    "metric": metric_name,
                    "condition": condition,
                    "mean": sum(vals) / len(vals),
                    "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                    "bootstrap_ci_low": cis[condition][0],
                    "bootstrap_ci_high": cis[condition][1],
                    "n": len(vals),
                }
            )

    _write_csv(results_dir / "static_metrics_summary.csv", summary_rows)

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "# Results Summary",
        "",
        f"Dataset: **{dataset_label}**",
        "",
        (
            "These outputs validate the pipeline only and should not be interpreted as research findings."
            if dataset_label == "sample"
            else "These outputs are generated from MIND-style evaluation data and may be used for analysis subject to the stated limitations."
        ),
        "",
        "## Operationalization",
        "- This experiment uses topical-semantic breadth within political/public-affairs news exposure.",
        "- Source-level ideology labels were not used in the main MIND-based experiment because MIND URL fields did not expose original publisher domains in usable form.",
        "- Synthetic/sample results remain validation-only.",
        f"- Real-data analysis uses slice: `{slice_name}`{', with newscrime robustness' if include_newscrime else ''}.",
        "",
        "## Methodology",
        f"- Baseline model: {model.mode} (LogisticRegression(max_iter=1000, class_weight='balanced') with cosine fallback).",
        f"- Embedding method requested: {embedding_method}; used: {used_embedding_method}.",
        "- Breadth-aware reranking: final_score = relevance_score + lambda_breadth * breadth_term.",
        "- breadth_term = 0.35*semantic_diversity_gain + 0.35*cross_topic_gain + 0.15*subcategory_novelty - 0.15*topical_concentration_penalty.",
        f"- lambda sensitivity tested: {', '.join(str(x) for x in lambda_values)} (main comparison baseline vs breadth_aware).",
        "",
        "## Evaluation Setup",
        f"- Number of articles (dev): {len(dev.news)}",
        f"- Number of users (dev impressions): {len(set(r['UserID'] for r in dev.impressions))}",
        f"- Number of impressions (dev candidate rows): {len(dev.impressions)}",
        "",
        "## Static Evaluation (summary)",
        _markdown_table(summary_rows, ["metric", "condition", "mean", "std", "bootstrap_ci_low", "bootstrap_ci_high", "n"]),
    ]

    (reports_dir / "results_summary.md").write_text("\n".join(report_lines), encoding="utf-8")
    return static_rows, summary_rows
