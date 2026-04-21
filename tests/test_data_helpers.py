import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from hidden_narrowing.data_helpers import download_and_extract_zip, find_missing_data_files


def _write_fake_mind_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("news.tsv", "N1\tv\n")
        zf.writestr("behaviors.tsv", "1\tU1\n")


def test_download_helper_skips_if_files_exist(tmp_path: Path):
    dataset_dir = tmp_path / "MINDsmall_train"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "news.tsv").write_text("n", encoding="utf-8")
    (dataset_dir / "behaviors.tsv").write_text("b", encoding="utf-8")

    status = download_and_extract_zip(name="MINDsmall_train", url="unused", dataset_dir=dataset_dir, force=False)

    assert status == "skipped"


def test_download_helper_force_reextracts(tmp_path: Path):
    dataset_dir = tmp_path / "MINDsmall_train"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "news.tsv").write_text("old", encoding="utf-8")
    (dataset_dir / "behaviors.tsv").write_text("old", encoding="utf-8")
    (dataset_dir / "stale.txt").write_text("stale", encoding="utf-8")

    zip_source = tmp_path / "source.zip"
    _write_fake_mind_zip(zip_source)

    def fake_urlretrieve(_url: str, destination: Path):
        Path(destination).write_bytes(zip_source.read_bytes())

    original = urllib.request.urlretrieve
    urllib.request.urlretrieve = fake_urlretrieve
    try:
        status = download_and_extract_zip(name="MINDsmall_train", url="unused", dataset_dir=dataset_dir, force=True)
    finally:
        urllib.request.urlretrieve = original

    assert status == "downloaded"
    assert (dataset_dir / "news.tsv").exists()
    assert (dataset_dir / "behaviors.tsv").exists()
    assert not (dataset_dir / "stale.txt").exists()


def test_check_data_ready_reports_missing(tmp_path: Path):
    missing = find_missing_data_files(tmp_path)
    assert len(missing) == 5
    assert tmp_path / "raw" / "MINDsmall_train" / "news.tsv" in missing


def test_check_data_ready_script_missing_output():
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root / "scripts" / "check_data_ready.py")]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "MISSING" in proc.stdout or "READY" in proc.stdout


def test_real_data_doc_mentions_download_helper():
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs" / "REAL_DATA_RUN.md").read_text(encoding="utf-8")
    assert "python scripts/download_mindsmall.py" in text
