from pathlib import Path

from hidden_narrowing import baseline
from hidden_narrowing.plots import make_plots
from hidden_narrowing.simulation import run_repeated_rounds


def test_simulation_output_format(tmp_path: Path):
    user_ids = ["U1"]
    pool = ["N1", "N2"]
    histories = {"U1": ["N1"]}
    news_by_id = {
        "N1": {"NewsID": "N1", "domain": "a.com", "ideology_score": -1.0},
        "N2": {"NewsID": "N2", "domain": "b.com", "ideology_score": 1.0},
    }
    article_vectors = {"N1": {"a": 1.0}, "N2": {"b": 1.0}}
    model = baseline.BaselineModel(mode="cosine")
    rows = run_repeated_rounds(user_ids, pool, histories, news_by_id, article_vectors, model, {}, tmp_path / "sim.csv", rounds=2)
    assert rows
    expected = {"user_id", "round", "condition", "avg_ideology", "concentration", "diversity", "cross_cutting_rate", "source_coverage", "clicked_news_id", "clicked_ideology"}
    assert expected.issubset(set(rows[0].keys()))


def test_plot_generation(tmp_path: Path):
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    results_dir.mkdir()
    (results_dir / "static_metrics_summary.csv").write_text(
        "metric,condition,mean,std,bootstrap_ci_low,bootstrap_ci_high,n\n"
        "ndcg@10,baseline,0.5,0.1,0.4,0.6,10\n"
        "mrr,baseline,0.4,0.1,0.3,0.5,10\n"
        "source_coverage,baseline,0.4,0.1,0.3,0.5,10\n"
        "ideological_concentration,baseline,0.3,0.1,0.2,0.4,10\n"
        "intra_list_diversity,baseline,0.6,0.1,0.5,0.7,10\n"
        "cross_cutting_exposure_rate,baseline,0.2,0.1,0.1,0.3,10\n"
        "ndcg@10,breadth_aware_lambda_0.35,0.48,0.1,0.38,0.58,10\n"
        "mrr,breadth_aware_lambda_0.35,0.38,0.1,0.28,0.48,10\n"
        "source_coverage,breadth_aware_lambda_0.35,0.7,0.1,0.6,0.8,10\n"
        "ideological_concentration,breadth_aware_lambda_0.35,0.1,0.1,0.0,0.2,10\n"
        "intra_list_diversity,breadth_aware_lambda_0.35,0.9,0.1,0.8,1.0,10\n"
        "cross_cutting_exposure_rate,breadth_aware_lambda_0.35,0.5,0.1,0.4,0.6,10\n",
        encoding="utf-8",
    )
    (results_dir / "simulation_round_metrics.csv").write_text(
        "user_id,round,condition,avg_ideology,concentration,diversity,cross_cutting_rate,source_coverage,clicked_news_id,clicked_ideology\n"
        "U1,1,baseline,0.2,0.2,0.5,0.1,0.5,N1,-1\n"
        "U1,2,baseline,0.3,0.3,0.4,0.1,0.4,N1,-1\n"
        "U1,1,breadth_aware,0.0,0.0,0.7,0.4,0.8,N2,1\n"
        "U1,2,breadth_aware,0.0,0.0,0.8,0.5,0.9,N2,1\n",
        encoding="utf-8",
    )
    paths = make_plots(results_dir, figures_dir)
    assert len(paths) == 5
    assert all(p.exists() for p in paths)
