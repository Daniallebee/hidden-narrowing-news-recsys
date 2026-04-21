# Real MINDsmall Run Guide

This guide describes how to run the full experiment on real MINDsmall files while keeping the repository safe and reproducible.

## 1) Download MINDsmall (one command)

```bash
python scripts/download_mindsmall.py
```

This helper downloads official MINDsmall train/dev zip files and extracts them into:

- `data/raw/MINDsmall_train/`
- `data/raw/MINDsmall_dev/`

Use `--force` to re-download and re-extract.

```bash
python scripts/download_mindsmall.py --force
```

## 2) Optional domain audit (validation only)

```bash
python scripts/audit_domains.py --mind-dir data/raw/MINDsmall_train
```

## 3) Fast debug run (public-affairs slice)

```bash
python scripts/run_all.py --mind-train-dir data/raw/MINDsmall_train --mind-dev-dir data/raw/MINDsmall_dev --slice public_affairs --max-train-impressions 5000 --max-dev-impressions 1000 --bootstrap-samples 200
```

## 4) Fuller run (public-affairs slice)

```bash
python scripts/run_all.py --mind-train-dir data/raw/MINDsmall_train --mind-dev-dir data/raw/MINDsmall_dev --slice public_affairs --bootstrap-samples 1000
```

## Optional robustness setting

Add `--include-newscrime` to include `newscrime` in the slice.

## Optional readiness check

```bash
python scripts/check_data_ready.py
```

## Safety warning

- **Do not commit raw MIND data.**
- **Do not commit extracted MIND files.**
- Keep generated large artifacts out of version control.
