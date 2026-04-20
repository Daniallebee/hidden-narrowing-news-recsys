from __future__ import annotations

import csv
from collections import defaultdict
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


def run_sample_pipeline(news_path: Path, behaviors_path: Path, allsides_path: Path, results_dir: Path, reports_dir: Path, top_k: int = 10, lambda_breadth: float = 0.35) -> tuple[list[dict], list[dict]]:
    parsed = data_mind.parse_mind(str(news_path), str(behaviors_path))
    allsides_rows = ideology.load_allsides(str(allsides_path))
    news_rows, mapping_report = ideology.attach_ideology(parsed.news, allsides_rows)

    _, article_vectors = features.build_tfidf_features(news_rows)
    user_vectors = features.build_user_vectors(parsed.histories, article_vectors)
    news_by_id = {n["NewsID"]: n for n in news_rows}

    imps_by_id: dict[str, list[dict]] = defaultdict(list)
    for imp in parsed.impressions:
        imps_by_id[imp["ImpressionID"]].append(imp)

    metrics_rows: list[dict] = []
    for imp_id, group in imps_by_id.items():
        user_id = group[0]["UserID"]
        user_vec = user_vectors.get(user_id)
        if not user_vec:
            continue

        candidate_ids = [g["NewsID"] for g in group]
        rel_scores = baseline.score_candidates_cosine(user_vec, candidate_ids, article_vectors)

        candidates = []
        for g, score in zip(group, rel_scores):
            item = dict(g)
            item["relevance_score"] = score
            item["domain"] = news_by_id.get(g["NewsID"], {}).get("domain")
            item["ideology_score"] = news_by_id.get(g["NewsID"], {}).get("ideology_score")
            candidates.append(item)

        hist_ids = [h["NewsID"] for h in parsed.histories if h["UserID"] == user_id]
        user_ideo = metrics.average_ideology([news_by_id.get(nid, {}).get("ideology_score") for nid in hist_ids])

        baseline_ranked = baseline.rank_candidates(candidates)[:top_k]
        breadth_ranked = rerank.greedy_breadth_rerank(candidates, top_k=top_k, lambda_breadth=lambda_breadth, user_ideology=user_ideo)

        for method, frame, score_col in [
            ("baseline", baseline_ranked, "relevance_score"),
            ("breadth_rerank", breadth_ranked, "final_score"),
        ]:
            labels = [int(r["clicked"]) for r in frame]
            scores = [float(r.get(score_col, 0.0)) for r in frame]
            ideos = [r.get("ideology_score") for r in frame]
            domains = [r.get("domain") for r in frame]
            metrics_rows.append(
                {
                    "ImpressionID": imp_id,
                    "UserID": user_id,
                    "method": method,
                    "ndcg@10": metrics.ndcg_at_k(labels, scores, 10),
                    "mrr": metrics.mrr(labels, scores),
                    "average_ideology": metrics.average_ideology(ideos),
                    "ideological_concentration": metrics.ideological_concentration(ideos),
                    "intra_list_diversity": metrics.intra_list_diversity(ideos),
                    "cross_cutting_exposure_rate": metrics.cross_cutting_exposure_rate(ideos, user_ideo),
                    "source_coverage": metrics.source_coverage(domains),
                    "mapping_coverage": mapping_report.coverage,
                }
            )

    # summary means by method
    summary: list[dict] = []
    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in metrics_rows:
        by_method[r["method"]].append(r)
    numeric_keys = [
        "ndcg@10",
        "mrr",
        "average_ideology",
        "ideological_concentration",
        "intra_list_diversity",
        "cross_cutting_exposure_rate",
        "source_coverage",
        "mapping_coverage",
    ]
    for method, rows in by_method.items():
        s = {"method": method}
        for k in numeric_keys:
            s[k] = sum(float(r[k]) for r in rows) / len(rows)
        summary.append(s)

    _write_csv(results_dir / "static_metrics.csv", metrics_rows)
    _write_csv(results_dir / "static_metrics_summary.csv", summary)

    reports_dir.mkdir(parents=True, exist_ok=True)
    report = [
        "# Static Evaluation Summary",
        "",
        f"Ideology mapping coverage: {mapping_report.coverage:.2%}",
        "",
        "## Mean metrics by method",
        "",
        _markdown_table(summary, ["method"] + numeric_keys),
        "",
        "## Top unmapped domains",
        "",
        _markdown_table(mapping_report.top_unmapped_domains, ["domain", "count"]),
    ]
    (reports_dir / "results_summary.md").write_text("\n".join(report), encoding="utf-8")

    return metrics_rows, summary
