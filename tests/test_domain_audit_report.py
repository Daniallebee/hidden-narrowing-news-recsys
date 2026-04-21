from pathlib import Path

from hidden_narrowing.domain_audit_report import compute_split_domain_stats, render_domain_audit_summary


def test_compute_split_domain_stats(tmp_path: Path):
    mind_dir = tmp_path / "MINDsmall_train"
    mind_dir.mkdir(parents=True)
    news = mind_dir / "news.tsv"
    news.write_text(
        "\t".join(["N1", "c", "s", "t", "a", "https://www.cnn.com/x", "", ""]) + "\n"
        + "\t".join(["N2", "c", "s", "t", "a", "https://cnn.com/y", "", ""]) + "\n"
        + "\t".join(["N3", "c", "s", "t", "a", "https://foxnews.com/z", "", ""]) + "\n",
        encoding="utf-8",
    )

    stats = compute_split_domain_stats("train", mind_dir)

    assert stats.article_count == 3
    assert stats.unique_domain_count == 2
    assert stats.top_domains[0] == ("cnn.com", 2)


def test_render_domain_audit_summary_mentions_mapping_missing(tmp_path: Path):
    train = compute_split_domain_stats("train", Path("tests/fixtures"))
    dev = compute_split_domain_stats("dev", Path("tests/fixtures"))
    content = render_domain_audit_summary(
        download_status_lines=["MINDsmall download status: downloaded"],
        train_news_exists=True,
        dev_news_exists=True,
        train_stats=train,
        dev_stats=dev,
        mapping_path=tmp_path / "missing.csv",
    )

    assert "Mapping file" in content
    assert "missing" in content
    assert "Number of train articles" in content
    assert "Number of dev articles" in content
