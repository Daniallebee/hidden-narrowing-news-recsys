from hidden_narrowing.rerank import greedy_breadth_rerank


def test_breadth_rerank_promotes_cross_cutting_item():
    candidates = [
        {"NewsID": "N1", "relevance_score": 0.90, "ideology_score": 0.9, "domain": "right.com"},
        {"NewsID": "N2", "relevance_score": 0.88, "ideology_score": 0.8, "domain": "right2.com"},
        {"NewsID": "N3", "relevance_score": 0.82, "ideology_score": -0.8, "domain": "left.com"},
    ]
    reranked = greedy_breadth_rerank(candidates, top_k=3, lambda_breadth=0.6, user_ideology=0.9)
    assert "N3" in [r["NewsID"] for r in reranked[:2]]
