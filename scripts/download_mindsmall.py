#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hidden_narrowing.config import DATA_DIR
from hidden_narrowing.data_helpers import download_and_extract_zip

MIND_URLS = {
    "MINDsmall_train": "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
    "MINDsmall_dev": "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
}




def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract official MINDsmall train/dev splits.")
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract even if files already exist.")
    args = parser.parse_args()

    print("MIND data is for research use. Do not commit downloaded or extracted dataset files.")

    raw_dir = DATA_DIR / "raw"
    targets = {
        "MINDsmall_train": raw_dir / "MINDsmall_train",
        "MINDsmall_dev": raw_dir / "MINDsmall_dev",
    }

    statuses = {}
    for name, url in MIND_URLS.items():
        print(f"[{name}] preparing dataset in {targets[name]}")
        status = download_and_extract_zip(name=name, url=url, dataset_dir=targets[name], force=args.force)
        if status == "skipped":
            print(f"[{name}] already present; skipping download (use --force to re-download).")
        else:
            print(f"[{name}] downloaded and extracted.")
        statuses[name] = status

    print("Done.")
    for name, status in statuses.items():
        print(f"- {name}: {status}")


if __name__ == "__main__":
    main()
