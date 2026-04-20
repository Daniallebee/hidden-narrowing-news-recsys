from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = ROOT_DIR / "figures"
FIXTURES_DIR = ROOT_DIR / "tests" / "fixtures"

DEFAULT_TOP_K = 10
DEFAULT_LAMBDA_BREADTH = 0.35
