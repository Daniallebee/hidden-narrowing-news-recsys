from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REQUIRED_MIND_FILES = ("news.tsv", "behaviors.tsv")
REQUIRED_DATA_RELATIVE_PATHS = [
    Path("raw/MINDsmall_train/news.tsv"),
    Path("raw/MINDsmall_train/behaviors.tsv"),
    Path("raw/MINDsmall_dev/news.tsv"),
    Path("raw/MINDsmall_dev/behaviors.tsv"),
    Path("external/allsides_media_bias.csv"),
]


def required_files_present(dataset_dir: Path, required_files: tuple[str, ...] = REQUIRED_MIND_FILES) -> bool:
    return all((dataset_dir / fname).exists() for fname in required_files)


def download_and_extract_zip(
    *,
    name: str,
    url: str,
    dataset_dir: Path,
    force: bool = False,
    required_files: tuple[str, ...] = REQUIRED_MIND_FILES,
) -> str:
    if required_files_present(dataset_dir, required_files) and not force:
        return "skipped"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = dataset_dir.parent

    with tempfile.TemporaryDirectory(prefix=f"{name}_", dir=raw_dir) as tmp_dir:
        archive_path = Path(tmp_dir) / f"{name}.zip"
        urllib.request.urlretrieve(url, archive_path)

        if force:
            for entry in dataset_dir.iterdir():
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()

        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dataset_dir)

    if not required_files_present(dataset_dir, required_files):
        missing = [fname for fname in required_files if not (dataset_dir / fname).exists()]
        raise RuntimeError(f"[{name}] extraction completed but missing required files: {missing}")

    return "downloaded"


def find_missing_data_files(base_data_dir: Path) -> list[Path]:
    missing: list[Path] = []
    for rel_path in REQUIRED_DATA_RELATIVE_PATHS:
        path = base_data_dir / rel_path
        if not path.exists():
            missing.append(path)
    return missing
