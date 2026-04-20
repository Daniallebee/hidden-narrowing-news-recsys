from __future__ import annotations

import csv
from dataclasses import dataclass

BIAS_TO_SCORE = {
    "Left": -1.0,
    "Lean Left": -0.5,
    "Center": 0.0,
    "Lean Right": 0.5,
    "Right": 1.0,
}


@dataclass
class IdeologyMappingReport:
    total_articles: int
    mapped_articles: int
    coverage: float
    top_unmapped_domains: list[dict]


def load_allsides(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {"outlet", "domain", "bias_label", "bias_score"}
        if not expected.issubset(set(reader.fieldnames or [])):
            raise ValueError("Missing required columns in AllSides-style file")
        for row in reader:
            row = dict(row)
            row["domain"] = row.get("domain", "").lower().strip()
            row["ideology_score"] = BIAS_TO_SCORE.get(row.get("bias_label", ""))
            rows.append(row)
    return rows


def attach_ideology(news_rows: list[dict], allsides_rows: list[dict]) -> tuple[list[dict], IdeologyMappingReport]:
    by_domain = {r["domain"]: r for r in allsides_rows}
    merged = []
    for n in news_rows:
        n2 = dict(n)
        source = by_domain.get((n.get("domain") or "").lower())
        if source:
            n2["bias_label"] = source.get("bias_label")
            n2["bias_score"] = source.get("bias_score")
            n2["ideology_score"] = source.get("ideology_score")
        else:
            n2["bias_label"] = None
            n2["bias_score"] = None
            n2["ideology_score"] = None
        merged.append(n2)

    total = len(merged)
    mapped = sum(1 for r in merged if r.get("ideology_score") is not None)
    coverage = mapped / total if total else 0.0

    counts: dict[str, int] = {}
    for r in merged:
        if r.get("ideology_score") is None:
            d = r.get("domain") or "<missing>"
            counts[d] = counts.get(d, 0) + 1
    top_unmapped = [{"domain": k, "count": v} for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]]

    return merged, IdeologyMappingReport(total, mapped, coverage, top_unmapped)
