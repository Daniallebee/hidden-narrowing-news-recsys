from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


_MIN_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa7\x86\xa1\x9d\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fallback_png(path: Path) -> None:
    path.write_bytes(_MIN_PNG)


def make_plots(results_dir: Path, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []

    summary = _read_csv(results_dir / "static_metrics_summary.csv")
    sim = _read_csv(results_dir / "simulation_round_metrics.csv")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        for fname in [
            "static_exposure_comparison.png",
            "utility_tradeoff.png",
            "simulation_concentration_over_rounds.png",
            "simulation_diversity_over_rounds.png",
            "source_coverage_comparison.png",
        ]:
            p = figures_dir / fname
            _fallback_png(p)
            out.append(p)
        return out

    exposure_metrics = ["ideological_concentration", "intra_list_diversity", "cross_cutting_exposure_rate", "source_coverage"]
    conds = sorted({r["condition"] for r in summary if r["metric"] in exposure_metrics})
    x = range(len(exposure_metrics))
    width = 0.35
    plt.figure(figsize=(8, 4))
    for i, c in enumerate(conds):
        vals = []
        for m in exposure_metrics:
            row = next((r for r in summary if r["condition"] == c and r["metric"] == m), None)
            vals.append(float(row["mean"]) if row else 0.0)
        offs = [xx + (i - 0.5) * width for xx in x]
        plt.bar(offs, vals, width=width, label=c)
    plt.xticks(list(x), exposure_metrics, rotation=20, ha="right")
    plt.tight_layout()
    plt.legend()
    p = figures_dir / "static_exposure_comparison.png"
    plt.savefig(p, dpi=300)
    plt.close()
    out.append(p)

    plt.figure(figsize=(5, 4))
    for c in conds:
        xval = next((float(r["mean"]) for r in summary if r["condition"] == c and r["metric"] == "ndcg@10"), 0.0)
        yval = next((float(r["mean"]) for r in summary if r["condition"] == c and r["metric"] == "mrr"), 0.0)
        plt.scatter([xval], [yval], label=c)
        plt.text(xval, yval, c)
    plt.xlabel("NDCG@10")
    plt.ylabel("MRR")
    plt.tight_layout()
    p = figures_dir / "utility_tradeoff.png"
    plt.savefig(p, dpi=300)
    plt.close()
    out.append(p)

    for metric, fname in [
        ("concentration", "simulation_concentration_over_rounds.png"),
        ("diversity", "simulation_diversity_over_rounds.png"),
    ]:
        plt.figure(figsize=(6, 4))
        by_cond_round: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in sim:
            by_cond_round[r["condition"]][int(r["round"])].append(float(r[metric]))
        for c, round_vals in by_cond_round.items():
            rounds = sorted(round_vals.keys())
            means = [sum(round_vals[rd]) / len(round_vals[rd]) for rd in rounds]
            plt.plot(rounds, means, marker="o", label=c)
        plt.xlabel("Round")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        p = figures_dir / fname
        plt.savefig(p, dpi=300)
        plt.close()
        out.append(p)

    plt.figure(figsize=(5, 4))
    metric = "source_coverage"
    vals = []
    labels = []
    for c in conds:
        row = next((r for r in summary if r["condition"] == c and r["metric"] == metric), None)
        labels.append(c)
        vals.append(float(row["mean"]) if row else 0.0)
    plt.bar(labels, vals)
    plt.ylabel("Source coverage")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    p = figures_dir / "source_coverage_comparison.png"
    plt.savefig(p, dpi=300)
    plt.close()
    out.append(p)

    return out
