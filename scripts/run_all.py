#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hidden_narrowing.config import FIXTURES_DIR, REPORTS_DIR, RESULTS_DIR
from hidden_narrowing.pipeline import run_sample_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hidden narrowing research pipelines.")
    parser.add_argument("--sample", action="store_true", help="Run synthetic sample pipeline from tests/fixtures.")
    args = parser.parse_args()

    if args.sample:
        run_sample_pipeline(
            news_path=FIXTURES_DIR / "news.tsv",
            behaviors_path=FIXTURES_DIR / "behaviors.tsv",
            allsides_path=FIXTURES_DIR / "allsides_media_bias.csv",
            results_dir=RESULTS_DIR,
            reports_dir=REPORTS_DIR,
        )
        print("Sample pipeline complete.")
        return

    parser.error("No mode selected. Use --sample for the synthetic pipeline.")


if __name__ == "__main__":
    main()
