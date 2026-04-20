from pathlib import Path

from hidden_narrowing.data_mind import extract_domain, parse_impressions, parse_mind
from hidden_narrowing.ideology import attach_ideology, load_allsides

FIX = Path(__file__).parent / "fixtures"


def test_parse_impressions():
    tokens = "N12345-0 N23456-1 N34567-0"
    parsed = parse_impressions(tokens)
    assert parsed == [("N12345", 0), ("N23456", 1), ("N34567", 0)]


def test_extract_domain():
    assert extract_domain("https://www.example.com/news/story") == "example.com"
    assert extract_domain("") is None


def test_parse_mind_outputs():
    parsed = parse_mind(str(FIX / "news.tsv"), str(FIX / "behaviors.tsv"))
    assert len(parsed.news) > 0
    assert len(parsed.behaviors) > 0
    assert len(parsed.impressions) > 0
    assert len(parsed.histories) > 0
    assert "domain" in parsed.news[0]


def test_ideology_mapping_scores_and_coverage():
    parsed = parse_mind(str(FIX / "news.tsv"), str(FIX / "behaviors.tsv"))
    allsides = load_allsides(str(FIX / "allsides_media_bias.csv"))
    merged, report = attach_ideology(parsed.news, allsides)

    left_score = next(r["ideology_score"] for r in merged if r.get("domain") == "leftnews.com")
    right_score = next(r["ideology_score"] for r in merged if r.get("domain") == "rightreport.com")
    assert left_score == -1.0
    assert right_score == 1.0
    assert 0 < report.coverage < 1
