from collections import Counter

from hidden_narrowing import baseline


def _toy():
    article_vectors = {
        "N1": {"a": 1.0},
        "N2": {"b": 1.0},
        "N3": {"a": 0.5, "b": 0.5},
    }
    news_by_id = {
        "N1": {"NewsID": "N1", "Category": "Politics", "SubCategory": "A", "ideology_score": -1.0},
        "N2": {"NewsID": "N2", "Category": "Politics", "SubCategory": "B", "ideology_score": 1.0},
        "N3": {"NewsID": "N3", "Category": "World", "SubCategory": "C", "ideology_score": None},
    }
    histories = {"U1": ["N1"], "U2": ["N2"]}
    impressions = [
        {"UserID": "U1", "NewsID": "N1", "clicked": 1},
        {"UserID": "U1", "NewsID": "N2", "clicked": 0},
        {"UserID": "U2", "NewsID": "N2", "clicked": 1},
        {"UserID": "U2", "NewsID": "N1", "clicked": 0},
    ] * 6
    pop = Counter({"N1": 3, "N2": 3, "N3": 1})
    return impressions, histories, article_vectors, news_by_id, pop


def test_feature_construction_shape():
    impressions, histories, article_vectors, news_by_id, pop = _toy()
    X, y = baseline.build_training_examples(impressions, histories, article_vectors, news_by_id, pop)
    assert len(X) == len(y)
    assert len(X[0]) == 6


def test_training_fallback_behavior():
    model = baseline.train_logistic_regression_baseline([], {}, {}, {}, Counter())
    assert model.mode == "cosine"
