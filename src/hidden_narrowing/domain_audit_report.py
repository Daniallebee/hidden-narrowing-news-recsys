from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from hidden_narrowing.data_mind import parse_news


@dataclass(frozen=True)
class SplitDomainStats:
    split_name: str
    article_count: int
    unique_domain_count: int
    top_domains: list[tuple[str, int]]


def compute_split_domain_stats(split_name: str, mind_dir: Path, top_n: int = 10) -> SplitDomainStats:
    news_path = mind_dir / "news.tsv"
    if not news_path.exists():
        raise FileNotFoundError(f"Missing news.tsv in {mind_dir}")

    news_rows = parse_news(str(news_path))
    domain_counts = Counter([(r.get("domain") or "<missing>").lower() for r in news_rows])
    return SplitDomainStats(
        split_name=split_name,
        article_count=len(news_rows),
        unique_domain_count=len(domain_counts),
        top_domains=domain_counts.most_common(top_n),
    )


def render_domain_audit_summary(
    *,
    download_status_lines: list[str],
    train_news_exists: bool,
    dev_news_exists: bool,
    train_stats: SplitDomainStats,
    dev_stats: SplitDomainStats,
    mapping_path: Path,
) -> str:
    mapping_status = "present" if mapping_path.exists() else "missing"
    lines = [
        "# MINDsmall Domain Audit Summary",
        "",
        "## Download and readiness status",
        *[f"- {line}" for line in download_status_lines],
        f"- train news.tsv present: {train_news_exists}",
        f"- dev news.tsv present: {dev_news_exists}",
        "",
        "## Split statistics",
        f"- Number of train articles: {train_stats.article_count}",
        f"- Number of dev articles: {dev_stats.article_count}",
        f"- Number of unique train domains: {train_stats.unique_domain_count}",
        f"- Number of unique dev domains: {dev_stats.unique_domain_count}",
        "",
        "## Top domains by article count",
        "",
        "### Train top domains",
    ]
    lines.extend([f"- {domain}: {count}" for domain, count in train_stats.top_domains])
    lines.extend(["", "### Dev top domains"])
    lines.extend([f"- {domain}: {count}" for domain, count in dev_stats.top_domains])
    lines.extend(
        [
            "",
            "## Mapping file status",
            f"- Mapping file at `{mapping_path}`: {mapping_status}",
            "",
            "## Next steps",
            "- Create `data/external/allsides_media_bias.csv` using `data/external/allsides_media_bias_template.csv`.",
            "- Use `results/unmapped_domains.csv` and `results/mapping_priority_domains.csv` to prioritize high-volume outlets.",
            "- Re-run `python scripts/audit_domains.py --mind-dir data/raw/MINDsmall_train` and `python scripts/audit_domains.py --mind-dir data/raw/MINDsmall_dev` after updating mappings.",
            "",
        ]
    )
    return "\n".join(lines)


def write_domain_audit_summary(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
