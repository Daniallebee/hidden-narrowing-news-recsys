#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hidden_narrowing.config import DATA_DIR, RESULTS_DIR
from hidden_narrowing.data_mind import parse_news
from hidden_narrowing.ideology import load_allsides


def _write_csv(path: Path, rows: list[dict], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MIND domain coverage against AllSides mapping.")
    parser.add_argument("--mind-dir", required=True, type=Path, help="Directory containing MIND news.tsv.")
    parser.add_argument(
        "--allsides-path",
        type=Path,
        default=DATA_DIR / "external" / "allsides_media_bias.csv",
        help="AllSides-style mapping CSV path.",
    )
    args = parser.parse_args()

    news_path = args.mind_dir / "news.tsv"
    if not news_path.exists():
        raise FileNotFoundError(f"Missing news.tsv in {args.mind_dir}")

    news_rows = parse_news(str(news_path))
    domain_counts = Counter([(r.get("domain") or "<missing>").lower() for r in news_rows])

    mapped_domains: set[str] = set()
    if args.allsides_path.exists():
        mapped_domains = {r.get("domain", "") for r in load_allsides(str(args.allsides_path))}
    else:
        print(f"Warning: mapping file not found at {args.allsides_path}; all domains treated as unmapped.")

    coverage_rows: list[dict] = []
    unmapped_rows: list[dict] = []
    for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
        is_mapped = domain in mapped_domains
        row = {"domain": domain, "article_count": count, "is_mapped": int(is_mapped)}
        coverage_rows.append(row)
        if not is_mapped:
            unmapped_rows.append({"domain": domain, "article_count": count})

    _write_csv(RESULTS_DIR / "domain_coverage.csv", coverage_rows, ["domain", "article_count", "is_mapped"])
    _write_csv(RESULTS_DIR / "mapping_priority_domains.csv", unmapped_rows, ["domain", "article_count"])
    _write_csv(RESULTS_DIR / "unmapped_domains.csv", unmapped_rows, ["domain", "article_count"])

    print(f"Total unique domains: {len(domain_counts)}")
    print(f"Mapped domains: {sum(1 for r in coverage_rows if r['is_mapped'])}")
    print("Top unmapped domains:")
    for row in unmapped_rows[:10]:
        print(f"- {row['domain']}: {row['article_count']}")


if __name__ == "__main__":
    main()
