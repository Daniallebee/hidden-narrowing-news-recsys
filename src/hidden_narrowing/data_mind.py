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

PUBLIC_AFFAIRS_SUBCATEGORIES = {"newsus", "newspolitics", "newsworld", "newsopinion"}


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


def filter_by_slice(parsed: ParsedMindData, slice_name: str = "public_affairs", include_newscrime: bool = False) -> ParsedMindData:
    if slice_name == "all":
        return parsed
    if slice_name != "public_affairs":
        raise ValueError(f"Unknown slice '{slice_name}'. Supported: public_affairs, all")

    subcats = set(PUBLIC_AFFAIRS_SUBCATEGORIES)
    if include_newscrime:
        subcats.add("newscrime")

    filtered_news = [
        n
        for n in parsed.news
        if n.get("Category", "").strip().lower() == "news" and n.get("SubCategory", "").strip().lower() in subcats
    ]
    keep_ids = {n["NewsID"] for n in filtered_news}

    filtered_impressions = [r for r in parsed.impressions if r.get("NewsID") in keep_ids]
    filtered_histories = [r for r in parsed.histories if r.get("NewsID") in keep_ids]
    keep_users = {r["UserID"] for r in filtered_impressions if r.get("UserID")}
    filtered_behaviors = [r for r in parsed.behaviors if r.get("UserID") in keep_users]

    return ParsedMindData(
        news=filtered_news,
        behaviors=filtered_behaviors,
        impressions=filtered_impressions,
        histories=filtered_histories,
    )


def parse_mind(news_path: str, behaviors_path: str, slice_name: str = "all", include_newscrime: bool = False) -> ParsedMindData:
    news = parse_news(news_path)
    behaviors = parse_behaviors(behaviors_path)
    parsed = ParsedMindData(
        news=news,
        behaviors=behaviors,
        impressions=build_impressions_rows(behaviors),
        histories=build_histories_rows(behaviors),
    )
    return filter_by_slice(parsed, slice_name=slice_name, include_newscrime=include_newscrime)
