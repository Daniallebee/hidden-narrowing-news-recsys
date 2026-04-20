import subprocess
import sys
from pathlib import Path


def test_run_all_sample_end_to_end():
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root / "scripts" / "run_all.py"), "--sample", "--bootstrap-samples", "200"]
    subprocess.run(cmd, check=True)

    assert (root / "reports" / "results_summary.md").exists()
    assert (root / "results" / "static_metrics_summary.csv").exists()
    assert (root / "results" / "simulation_round_metrics.csv").exists()
    for name in [
        "static_exposure_comparison.png",
        "utility_tradeoff.png",
        "simulation_concentration_over_rounds.png",
        "simulation_diversity_over_rounds.png",
        "source_coverage_comparison.png",
    ]:
        assert (root / "figures" / name).exists()
