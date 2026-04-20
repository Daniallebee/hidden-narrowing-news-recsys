#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hidden_narrowing.config import FIGURES_DIR, RESULTS_DIR
from hidden_narrowing.plots import make_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate result figures.")
    parser.add_argument("--sample", action="store_true", help="Compatibility flag.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    paths = make_plots(results_dir=args.results_dir, figures_dir=args.figures_dir)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
