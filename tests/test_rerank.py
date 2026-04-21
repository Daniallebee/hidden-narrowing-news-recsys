from hidden_narrowing.rerank import greedy_breadth_rerank


def test_breadth_rerank_promotes_cross_topic_item():
    candidates = [
        {"NewsID": "N1", "relevance_score": 0.90, "SubCategory": "newsus"},
        {"NewsID": "N2", "relevance_score": 0.88, "SubCategory": "newsus"},
        {"NewsID": "N3", "relevance_score": 0.82, "SubCategory": "newsworld"},
    ]
    article_vectors = {
        "N1": {"a": 1.0},
        "N2": {"a": 1.0},
        "N3": {"b": 1.0},
    }
    reranked = greedy_breadth_rerank(
        candidates,
        article_vectors=article_vectors,
        top_k=3,
        lambda_breadth=0.6,
        user_dominant_subcategory="newsus",
    )
    assert "N3" in [r["NewsID"] for r in reranked[:2]]
