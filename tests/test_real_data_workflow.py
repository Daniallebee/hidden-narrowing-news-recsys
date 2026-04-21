import csv
import subprocess
import sys
from pathlib import Path


def test_audit_domains_with_fixtures(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(root / "scripts" / "audit_domains.py"),
        "--mind-dir",
        str(root / "tests" / "fixtures"),
        "--allsides-path",
        str(root / "tests" / "fixtures" / "allsides_media_bias.csv"),
    ]
    subprocess.run(cmd, check=True)
    assert (root / "results" / "domain_coverage.csv").exists()
    assert (root / "results" / "unmapped_domains.csv").exists()


def test_run_all_sample_report_label():
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root / "scripts" / "run_all.py"), "--sample", "--bootstrap-samples", "200"]
    subprocess.run(cmd, check=True)
    report = (root / "reports" / "results_summary.md").read_text(encoding="utf-8")
    assert "These outputs validate the pipeline only and should not be interpreted as research findings." in report


def test_run_all_with_max_limits():
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(root / "scripts" / "run_all.py"),
        "--sample",
        "--max-train-impressions",
        "20",
        "--max-dev-impressions",
        "20",
        "--max-users",
        "2",
        "--bootstrap-samples",
        "200",
    ]
    subprocess.run(cmd, check=True)
    summary_path = root / "results" / "static_metrics_summary.csv"
    rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8")))
    assert rows


def test_missing_real_data_directory_message():
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root / "scripts" / "run_all.py"), "--mind-train-dir", "missing_train", "--mind-dev-dir", "missing_dev"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    combined = f"{proc.stdout}\n{proc.stderr}"
    assert "Missing MINDsmall files" in combined


def test_run_all_real_data_args_with_slice_public_affairs():
    root = Path(__file__).resolve().parents[1]
    fixture_dir = root / "tests" / "fixtures"
    cmd = [
        sys.executable,
        str(root / "scripts" / "run_all.py"),
        "--mind-train-dir",
        str(fixture_dir),
        "--mind-dev-dir",
        str(fixture_dir),
        "--slice",
        "public_affairs",
        "--bootstrap-samples",
        "50",
    ]
    subprocess.run(cmd, check=True)
    assert (root / "results" / "static_metrics_summary.csv").exists()
