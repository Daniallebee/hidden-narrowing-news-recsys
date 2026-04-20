from __future__ import annotations

import csv
from dataclasses import dataclass
from urllib.parse import urlparse

NEWS_COLUMNS = [
    "NewsID",
    "Category",
    "SubCategory",
    "Title",
    "Abstract",
    "URL",
    "TitleEntities",
    "AbstractEntities",
]
BEHAVIOR_COLUMNS = ["ImpressionID", "UserID", "Time", "History", "Impressions"]


@dataclass
class ParsedMindData:
    news: list[dict]
    behaviors: list[dict]
    impressions: list[dict]
    histories: list[dict]


def extract_domain(url: str) -> str | None:
    if not isinstance(url, str) or not url.strip():
        return None
    host = urlparse(url).netloc.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host or None


def _read_tsv(path: str, columns: list[str]) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for raw in reader:
            padded = (raw + [""] * len(columns))[: len(columns)]
            rows.append({k: v for k, v in zip(columns, padded)})
    return rows


def parse_news(news_path: str) -> list[dict]:
    rows = _read_tsv(news_path, NEWS_COLUMNS)
    for r in rows:
        r["domain"] = extract_domain(r.get("URL", ""))
    return rows


def parse_impressions(impression_tokens: str) -> list[tuple[str, int]]:
    if not impression_tokens:
        return []
    out: list[tuple[str, int]] = []
    for token in impression_tokens.split():
        if "-" not in token:
            continue
        nid, label = token.rsplit("-", 1)
        try:
            out.append((nid, int(label)))
        except ValueError:
            continue
    return out


def parse_behaviors(behaviors_path: str) -> list[dict]:
    return _read_tsv(behaviors_path, BEHAVIOR_COLUMNS)


def build_impressions_rows(behaviors: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for b in behaviors:
        for rank, (nid, clicked) in enumerate(parse_impressions(b.get("Impressions", ""))):
            rows.append(
                {
                    "ImpressionID": b.get("ImpressionID"),
                    "UserID": b.get("UserID"),
                    "candidate_rank": rank,
                    "NewsID": nid,
                    "clicked": clicked,
                }
            )
    return rows


def build_histories_rows(behaviors: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for b in behaviors:
        for idx, nid in enumerate([x for x in b.get("History", "").split() if x]):
            rows.append(
                {
                    "ImpressionID": b.get("ImpressionID"),
                    "UserID": b.get("UserID"),
                    "history_rank": idx,
                    "NewsID": nid,
                }
            )
    return rows


def parse_mind(news_path: str, behaviors_path: str) -> ParsedMindData:
    news = parse_news(news_path)
    behaviors = parse_behaviors(behaviors_path)
    impressions = build_impressions_rows(behaviors)
    histories = build_histories_rows(behaviors)
    return ParsedMindData(news=news, behaviors=behaviors, impressions=impressions, histories=histories)
