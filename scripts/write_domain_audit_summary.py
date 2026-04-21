#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hidden_narrowing.config import DATA_DIR, REPORTS_DIR
from hidden_narrowing.domain_audit_report import (
    compute_split_domain_stats,
    render_domain_audit_summary,
    write_domain_audit_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write markdown summary for MINDsmall domain audit outputs.")
    parser.add_argument("--train-dir", type=Path, default=DATA_DIR / "raw" / "MINDsmall_train")
    parser.add_argument("--dev-dir", type=Path, default=DATA_DIR / "raw" / "MINDsmall_dev")
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=DATA_DIR / "external" / "allsides_media_bias.csv",
    )
    parser.add_argument("--download-status", default="unknown", help="Status line for download step.")
    parser.add_argument("--data-ready-status", default="unknown", help="Status line for check_data_ready step.")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "domain_audit_summary.md")
    args = parser.parse_args()

    train_stats = compute_split_domain_stats("train", args.train_dir)
    dev_stats = compute_split_domain_stats("dev", args.dev_dir)

    content = render_domain_audit_summary(
        download_status_lines=[
            f"MINDsmall download status: {args.download_status}",
            f"Data readiness check status: {args.data_ready_status}",
        ],
        train_news_exists=(args.train_dir / "news.tsv").exists(),
        dev_news_exists=(args.dev_dir / "news.tsv").exists(),
        train_stats=train_stats,
        dev_stats=dev_stats,
        mapping_path=args.mapping_path,
    )
    write_domain_audit_summary(args.output, content)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
