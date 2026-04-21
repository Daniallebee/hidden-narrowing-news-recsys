#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hidden_narrowing.config import DATA_DIR
from hidden_narrowing.data_helpers import find_missing_data_files


def main() -> None:
    missing = find_missing_data_files(DATA_DIR)
    if not missing:
        print("READY")
        return

    print("MISSING")
    for path in missing:
        print(f"- {path}")


if __name__ == "__main__":
    main()
