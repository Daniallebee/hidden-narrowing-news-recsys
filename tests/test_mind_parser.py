from pathlib import Path

from hidden_narrowing.data_mind import PUBLIC_AFFAIRS_SUBCATEGORIES, extract_domain, parse_impressions, parse_mind

FIX = Path(__file__).parent / "fixtures"


def test_parse_impressions():
    tokens = "N12345-0 N23456-1 N34567-0"
    parsed = parse_impressions(tokens)
    assert parsed == [("N12345", 0), ("N23456", 1), ("N34567", 0)]


def test_extract_domain():
    assert extract_domain("https://www.example.com/news/story") == "example.com"
    assert extract_domain("") is None


def test_parse_mind_outputs():
    parsed = parse_mind(str(FIX / "news.tsv"), str(FIX / "behaviors.tsv"), slice_name="all")
    assert len(parsed.news) > 0
    assert len(parsed.behaviors) > 0
    assert len(parsed.impressions) > 0
    assert len(parsed.histories) > 0
    assert "domain" in parsed.news[0]


def test_public_affairs_slice_filtering(tmp_path: Path):
    news = tmp_path / "news.tsv"
    behaviors = tmp_path / "behaviors.tsv"
    news.write_text(
        "N1\tnews\tnewsus\tA\tB\thttps://a.com\t[]\t[]\n"
        "N2\tnews\tsports\tA\tB\thttps://b.com\t[]\t[]\n"
        "N3\tnews\tnewspolitics\tA\tB\thttps://c.com\t[]\t[]\n",
        encoding="utf-8",
    )
    behaviors.write_text("1\tU1\t2020-01-01\tN1 N2\tN1-1 N2-0 N3-0\n", encoding="utf-8")

    parsed = parse_mind(str(news), str(behaviors), slice_name="public_affairs")
    assert parsed.news
    assert all(n["Category"].strip().lower() == "news" for n in parsed.news)
    assert all(n["SubCategory"].strip().lower() in PUBLIC_AFFAIRS_SUBCATEGORIES for n in parsed.news)
