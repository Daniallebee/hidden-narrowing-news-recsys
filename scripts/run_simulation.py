#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hidden_narrowing import baseline, data_mind, features, ideology
from hidden_narrowing.config import FIXTURES_DIR, RESULTS_DIR
from hidden_narrowing.simulation import run_repeated_rounds


def _validate_dir(path: Path) -> tuple[Path, Path]:
    news = path / "news.tsv"
    beh = path / "behaviors.tsv"
    if not news.exists() or not beh.exists():
        raise FileNotFoundError(
            f"Missing MINDsmall files in {path}. Expected news.tsv and behaviors.tsv. "
            "Place MINDsmall_train or MINDsmall_dev folders under data/raw/."
        )
    return news, beh


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated-round simulation.")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--mind-train-dir", type=Path)
    parser.add_argument("--mind-dev-dir", type=Path)
    parser.add_argument("--allsides-path", type=Path, default=FIXTURES_DIR / "allsides_media_bias.csv")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.5)
    args = parser.parse_args()

    if args.sample:
        train_news = FIXTURES_DIR / "news.tsv"
        train_beh = FIXTURES_DIR / "behaviors.tsv"
        dev_news = train_news
        dev_beh = train_beh
    else:
        if not args.mind_train_dir or not args.mind_dev_dir:
            parser.error("Provide --mind-train-dir and --mind-dev-dir, or use --sample.")
        train_news, train_beh = _validate_dir(args.mind_train_dir)
        dev_news, dev_beh = _validate_dir(args.mind_dev_dir)

    train = data_mind.parse_mind(str(train_news), str(train_beh))
    dev = data_mind.parse_mind(str(dev_news), str(dev_beh))
    allsides = ideology.load_allsides(str(args.allsides_path))
    train_news_rows, _ = ideology.attach_ideology(train.news, allsides)
    dev_news_rows, _ = ideology.attach_ideology(dev.news, allsides)

    merged_news = {n["NewsID"]: n for n in train_news_rows + dev_news_rows}
    _, article_vectors = features.build_tfidf_features(list(merged_news.values()))
    histories_by_user = {}
    for h in train.histories:
        histories_by_user.setdefault(h["UserID"], []).append(h["NewsID"])

    popularity = Counter([r["NewsID"] for r in train.impressions if int(r.get("clicked", 0)) == 1])
    model = baseline.train_logistic_regression_baseline(train.impressions, histories_by_user, article_vectors, merged_news, popularity)

    user_ids = sorted(histories_by_user.keys())
    candidate_pool = sorted({r["NewsID"] for r in dev.impressions})

    run_repeated_rounds(
        user_ids=user_ids,
        candidate_pool=candidate_pool,
        histories_by_user=histories_by_user,
        news_by_id=merged_news,
        article_vectors=article_vectors,
        model=model,
        popularity=popularity,
        results_path=RESULTS_DIR / "simulation_round_metrics.csv",
        rounds=args.rounds,
        temperature=args.temperature,
    )
    print("Simulation complete.")


if __name__ == "__main__":
    main()
