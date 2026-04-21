from hidden_narrowing.features import build_user_subcategory_profile


def test_dominant_historical_subcategory_and_frequency_vector():
    news_by_id = {
        "N1": {"SubCategory": "newsus"},
        "N2": {"SubCategory": "newsus"},
        "N3": {"SubCategory": "newsworld"},
    }
    profile = build_user_subcategory_profile(
        ["N1", "N2", "N3"],
        news_by_id,
        allowed_subcategories={"newsus", "newsworld", "newspolitics", "newsopinion"},
    )
    assert profile["dominant_subcategory"] == "newsus"
    assert profile["subcategory_frequency"]["newsus"] == 2 / 3
