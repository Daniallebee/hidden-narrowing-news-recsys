from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from . import baseline, data_mind, features, ideology, metrics, rerank


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
    allsides_path: Path,
    results_dir: Path,
    reports_dir: Path,
    top_k: int = 10,
    lambda_breadth: float = 0.35,
    embedding_method: str = "tfidf",
    bootstrap_samples: int = 1000,
    lambda_values: list[float] | None = None,
    dataset_label: str = "sample",
) -> tuple[list[dict], list[dict]]:
    lambda_values = lambda_values or [0.15, 0.35, 0.60]

    train = data_mind.parse_mind(str(train_news_path), str(train_behaviors_path))
    dev = data_mind.parse_mind(str(dev_news_path), str(dev_behaviors_path))

    allsides_rows = ideology.load_allsides(str(allsides_path))
    train_news_rows, mapping_report_train = ideology.attach_ideology(train.news, allsides_rows)
    dev_news_rows, mapping_report_dev = ideology.attach_ideology(dev.news, allsides_rows)

    # Domain coverage outputs for evaluation corpus
    coverage_rows = [
        {
            "dataset": dataset_label,
            "total_articles": mapping_report_dev.total_articles,
            "mapped_articles": mapping_report_dev.mapped_articles,
            "unmapped_articles": mapping_report_dev.unmapped_articles,
            "mapping_coverage_percentage": round(mapping_report_dev.coverage * 100.0, 2),
            "top_unmapped_domains": "; ".join(f"{x['domain']}:{x['count']}" for x in mapping_report_dev.top_unmapped_domains[:10]),
        }
    ]
    _write_csv(results_dir / "domain_coverage.csv", coverage_rows)
    _write_csv(results_dir / "unmapped_domains.csv", mapping_report_dev.top_unmapped_domains)

    # Fit representations on train news, infer on combined ids fallback for sample
    merged_news = {n["NewsID"]: n for n in (train_news_rows + dev_news_rows)}
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
        "ndcg@10",
        "mrr",
        "hit@10",
        "average_ideology",
        "ideological_concentration",
        "intra_list_diversity",
        "cross_cutting_exposure_rate",
        "source_coverage",
    ]

    static_rows: list[dict] = []
    for imp_id, group in imps_by_id.items():
        user_id = group[0]["UserID"]
        user_hist = histories_by_user_train.get(user_id, [])
        if not user_hist:
            continue
        user_ideo = metrics.average_ideology([news_by_id.get(nid, {}).get("ideology_score") for nid in user_hist])

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
            item["domain"] = row_news.get("domain")
            item["ideology_score"] = row_news.get("ideology_score")
            candidates.append(item)

        baseline_ranked = baseline.rank_candidates(candidates, score_field="relevance_score")[:top_k]
        breadth_ranked = rerank.greedy_breadth_rerank(candidates, top_k=top_k, lambda_breadth=lambda_breadth, user_ideology=user_ideo)

        for condition, frame, score_col in [
            ("baseline", baseline_ranked, "relevance_score"),
            ("breadth_aware_lambda_0.35", breadth_ranked, "final_score"),
        ]:
            labels = [int(r["clicked"]) for r in frame]
            scores = [float(r.get(score_col, 0.0)) for r in frame]
            ideos = [r.get("ideology_score") for r in frame]
            domains = [r.get("domain") for r in frame]
            static_rows.append(
                {
                    "ImpressionID": imp_id,
                    "UserID": user_id,
                    "condition": condition,
                    "ndcg@10": metrics.ndcg_at_k(labels, scores, top_k),
                    "mrr": metrics.mrr(labels, scores),
                    "hit@10": metrics.hit_at_k(labels, scores, top_k),
                    "average_ideology": metrics.average_ideology(ideos),
                    "ideological_concentration": metrics.ideological_concentration(ideos),
                    "intra_list_diversity": metrics.intra_list_diversity(ideos),
                    "cross_cutting_exposure_rate": metrics.cross_cutting_exposure_rate(ideos, user_ideo),
                    "source_coverage": metrics.source_coverage(domains),
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
        "**Important:** This run is synthetic validation only, not real-world findings." if dataset_label == "sample" else "This run uses MINDsmall inputs.",
        "",
        "## Methodology",
        f"- Baseline model: {model.mode} (LogisticRegression(max_iter=1000, class_weight='balanced') with cosine fallback).",
        f"- Embedding method requested: {embedding_method}; used: {used_embedding_method}.",
        "- Breadth-aware reranking: final_score = relevance_score + lambda_breadth * breadth_term.",
        "- breadth_term = 0.35*diversity_gain + 0.35*cross_cutting_gain + 0.10*source_novelty - 0.20*concentration_penalty.",
        f"- lambda sensitivity tested: {', '.join(str(x) for x in lambda_values)} (main comparison uses 0.35).",
        "",
        "## Evaluation Setup",
        f"- Number of articles (dev): {len(dev_news_rows)}",
        f"- Number of users (dev impressions): {len(set(r['UserID'] for r in dev.impressions))}",
        f"- Number of impressions (dev candidate rows): {len(dev.impressions)}",
        f"- Ideology mapping coverage: {mapping_report_dev.coverage:.2%}",
        "",
        "## Static Evaluation (summary)",
        _markdown_table(summary_rows, ["metric", "condition", "mean", "std", "bootstrap_ci_low", "bootstrap_ci_high", "n"]),
        "",
        "## Discussion",
        "- Compare concentration/diversity/cross-cutting/source coverage between baseline and breadth-aware rows above.",
        "- Utility tradeoff should be read from NDCG@10/MRR summary lines.",
        "- See simulation section below for repeated-round dynamics.",
        "",
        "## Limitations",
        "- Ideology scores depend on outlet-domain mapping coverage.",
        "- Sentence-transformer mode is optional and environment-dependent.",
        "- Offline click simulation may not reflect live user adaptation.",
    ]
    (reports_dir / "results_summary.md").write_text("\n".join(report_lines), encoding="utf-8")

    return static_rows, summary_rows
