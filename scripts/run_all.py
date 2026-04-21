#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hidden_narrowing.config import DATA_DIR, FIGURES_DIR, FIXTURES_DIR, REPORTS_DIR, RESULTS_DIR
from hidden_narrowing.pipeline import run_experiment


def _validate_mind_dir(path: Path) -> tuple[Path, Path]:
    news = path / "news.tsv"
    beh = path / "behaviors.tsv"
    if not news.exists() or not beh.exists():
        raise FileNotFoundError(
            f"Missing MINDsmall files in {path}. Expected files: news.tsv and behaviors.tsv. "
            "Place datasets at data/raw/MINDsmall_train and data/raw/MINDsmall_dev."
        )
    return news, beh


def _append_simulation_summary(report_path: Path, sim_path: Path) -> None:
    import csv
    from collections import defaultdict

    if not sim_path.exists() or sim_path.stat().st_size == 0:
        return
    rows = list(csv.DictReader(sim_path.open("r", encoding="utf-8")))
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    lines = ["", "## Simulation Summary", "", "| condition | concentration_mean | diversity_mean | cross_cutting_rate_mean | source_coverage_mean |", "| --- | --- | --- | --- | --- |"]
    for cond, vals in by_cond.items():
        def mean(k):
            xs=[float(v[k]) for v in vals]
            return sum(xs)/len(xs) if xs else 0.0
        lines.append(f"| {cond} | {mean('concentration'):.4f} | {mean('diversity'):.4f} | {mean('cross_cutting_rate'):.4f} | {mean('source_coverage'):.4f} |")
    # Read static summary for utility/exposure directionality
    static_path = sim_path.parent / "static_metrics_summary.csv"
    static_rows = list(csv.DictReader(static_path.open("r", encoding="utf-8"))) if static_path.exists() else []
    def mean_static(metric: str, condition: str) -> float:
        row = next((r for r in static_rows if r.get("metric") == metric and r.get("condition") == condition), None)
        return float(row["mean"]) if row else 0.0
    b = "baseline"
    r = "breadth_aware_lambda_0.35"
    conc_dec = mean_static("ideological_concentration", r) < mean_static("ideological_concentration", b)
    div_inc = mean_static("intra_list_diversity", r) > mean_static("intra_list_diversity", b)
    cc_inc = mean_static("cross_cutting_exposure_rate", r) > mean_static("cross_cutting_exposure_rate", b)
    src_inc = mean_static("source_coverage", r) > mean_static("source_coverage", b)
    ndcg_drop = mean_static("ndcg@10", r) < mean_static("ndcg@10", b)
    mrr_drop = mean_static("mrr", r) < mean_static("mrr", b)

    sim_base = next((vals for cond, vals in by_cond.items() if cond == "baseline"), [])
    sim_rerank = next((vals for cond, vals in by_cond.items() if cond == "breadth_aware"), [])
    def mean_from_rows(vs, k):
        xs=[float(v[k]) for v in vs]
        return sum(xs)/len(xs) if xs else 0.0
    stronger_narrowing_baseline = mean_from_rows(sim_base, "concentration") > mean_from_rows(sim_rerank, "concentration")

    lines += [
        "",
        "### Interpretation",
        f"- Concentration decreased under breadth-aware: {'yes' if conc_dec else 'no'}.",
        f"- Diversity increased under breadth-aware: {'yes' if div_inc else 'no'}.",
        f"- Cross-cutting exposure increased under breadth-aware: {'yes' if cc_inc else 'no'}.",
        f"- Source coverage increased under breadth-aware: {'yes' if src_inc else 'no'}.",
        f"- NDCG@10 declined under breadth-aware: {'yes' if ndcg_drop else 'no'}.",
        f"- MRR declined under breadth-aware: {'yes' if mrr_drop else 'no'}.",
        f"- Repeated rounds show stronger narrowing under baseline: {'yes' if stronger_narrowing_baseline else 'no'}.",
    ]
    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hidden narrowing full pipeline.")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--mind-train-dir", type=Path)
    parser.add_argument("--mind-dev-dir", type=Path)
    parser.add_argument("--allsides-path", type=Path, default=DATA_DIR / "external" / "allsides_media_bias.csv")
    parser.add_argument("--embedding-method", choices=["tfidf", "sentence-transformer"], default="tfidf")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--lambda-breadth", type=float, default=0.35)
    parser.add_argument("--lambda-values", type=float, nargs="*", default=[0.15, 0.35, 0.60])
    parser.add_argument("--max-train-impressions", type=int)
    parser.add_argument("--max-dev-impressions", type=int)
    parser.add_argument("--max-users", type=int)
    args = parser.parse_args()

    if args.sample:
        train_news = FIXTURES_DIR / "news.tsv"
        train_behaviors = FIXTURES_DIR / "behaviors.tsv"
        dev_news = train_news
        dev_behaviors = train_behaviors
        dataset_label = "sample"
        if not args.allsides_path.exists():
            args.allsides_path = FIXTURES_DIR / "allsides_media_bias.csv"
    else:
        if not args.mind_train_dir or not args.mind_dev_dir:
            parser.error("Use --sample, or provide --mind-train-dir and --mind-dev-dir.")
        train_news, train_behaviors = _validate_mind_dir(args.mind_train_dir)
        dev_news, dev_behaviors = _validate_mind_dir(args.mind_dev_dir)
        dataset_label = "MINDsmall"
        if not args.allsides_path.exists():
            raise FileNotFoundError(
                f"AllSides mapping file not found: {args.allsides_path}. "
                "Create data/external/allsides_media_bias.csv from the provided template."
            )

    run_experiment(
        train_news_path=train_news,
        train_behaviors_path=train_behaviors,
        dev_news_path=dev_news,
        dev_behaviors_path=dev_behaviors,
        allsides_path=args.allsides_path,
        results_dir=RESULTS_DIR,
        reports_dir=REPORTS_DIR,
        lambda_breadth=args.lambda_breadth,
        embedding_method=args.embedding_method,
        bootstrap_samples=args.bootstrap_samples,
        lambda_values=args.lambda_values,
        dataset_label=dataset_label,
        max_train_impressions=args.max_train_impressions,
        max_dev_impressions=args.max_dev_impressions,
        max_users=args.max_users,
    )

    sim_cmd = [sys.executable, str(ROOT / "scripts" / "run_simulation.py")]
    if args.sample:
        sim_cmd.append("--sample")
    else:
        sim_cmd += ["--mind-train-dir", str(args.mind_train_dir), "--mind-dev-dir", str(args.mind_dev_dir), "--allsides-path", str(args.allsides_path)]
    subprocess.run(sim_cmd, check=True)

    plot_cmd = [sys.executable, str(ROOT / "scripts" / "make_plots.py"), "--results-dir", str(RESULTS_DIR), "--figures-dir", str(FIGURES_DIR)]
    if args.sample:
        plot_cmd.append("--sample")
    subprocess.run(plot_cmd, check=True)

    _append_simulation_summary(REPORTS_DIR / "results_summary.md", RESULTS_DIR / "simulation_round_metrics.csv")

    print("Full pipeline complete.")


if __name__ == "__main__":
    main()
