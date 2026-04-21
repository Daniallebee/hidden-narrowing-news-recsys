#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hidden_narrowing.config import FIGURES_DIR, FIXTURES_DIR, REPORTS_DIR, RESULTS_DIR
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

    lines = [
        "",
        "## Simulation Summary",
        "",
        "| condition | topical_concentration_mean | semantic_diversity_mean | cross_topic_rate_mean | subcategory_coverage_mean |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cond, vals in by_cond.items():
        def mean(k: str) -> float:
            xs = [float(v[k]) for v in vals]
            return sum(xs) / len(xs) if xs else 0.0

        lines.append(
            f"| {cond} | {mean('topical_concentration'):.4f} | {mean('semantic_diversity'):.4f} | {mean('cross_topic_rate'):.4f} | {mean('subcategory_coverage'):.4f} |"
        )
    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hidden narrowing full pipeline.")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--mind-train-dir", type=Path)
    parser.add_argument("--mind-dev-dir", type=Path)
    parser.add_argument("--slice", choices=["public_affairs", "all"], default="public_affairs")
    parser.add_argument("--include-newscrime", action="store_true")
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
    else:
        if not args.mind_train_dir or not args.mind_dev_dir:
            parser.error("Use --sample, or provide --mind-train-dir and --mind-dev-dir.")
        train_news, train_behaviors = _validate_mind_dir(args.mind_train_dir)
        dev_news, dev_behaviors = _validate_mind_dir(args.mind_dev_dir)
        dataset_label = "MINDsmall"

    effective_slice = "all" if args.sample and args.slice == "public_affairs" else args.slice

    run_experiment(
        train_news_path=train_news,
        train_behaviors_path=train_behaviors,
        dev_news_path=dev_news,
        dev_behaviors_path=dev_behaviors,
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
        slice_name=effective_slice,
        include_newscrime=args.include_newscrime,
    )

    sim_cmd = [sys.executable, str(ROOT / "scripts" / "run_simulation.py"), "--slice", effective_slice]
    if args.include_newscrime:
        sim_cmd.append("--include-newscrime")
    if args.sample:
        sim_cmd.append("--sample")
    else:
        sim_cmd += ["--mind-train-dir", str(args.mind_train_dir), "--mind-dev-dir", str(args.mind_dev_dir)]
    subprocess.run(sim_cmd, check=True)

    plot_cmd = [sys.executable, str(ROOT / "scripts" / "make_plots.py"), "--results-dir", str(RESULTS_DIR), "--figures-dir", str(FIGURES_DIR)]
    if args.sample:
        plot_cmd.append("--sample")
    subprocess.run(plot_cmd, check=True)

    _append_simulation_summary(REPORTS_DIR / "results_summary.md", RESULTS_DIR / "simulation_round_metrics.csv")

    print("Full pipeline complete.")


if __name__ == "__main__":
    main()
