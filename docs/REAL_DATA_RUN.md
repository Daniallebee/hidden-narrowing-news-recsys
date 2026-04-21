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

## 2) Optional domain audit before running

```bash
python scripts/audit_domains.py --mind-dir data/raw/MINDsmall_train
```

## 3) Create outlet-to-bias mapping file

Create:

- `data/external/allsides_media_bias.csv`

Use `data/external/allsides_media_bias_template.csv` as the schema template.

## 4) Fast debug run

```bash
python scripts/run_all.py --mind-train-dir data/raw/MINDsmall_train --mind-dev-dir data/raw/MINDsmall_dev --max-train-impressions 2000 --max-dev-impressions 500 --bootstrap-samples 200
```

## 5) Fuller run

```bash
python scripts/run_all.py --mind-train-dir data/raw/MINDsmall_train --mind-dev-dir data/raw/MINDsmall_dev --bootstrap-samples 1000
```

## Optional readiness check

```bash
python scripts/check_data_ready.py
```

## Safety warning

- **Do not commit raw MIND data.**
- **Do not commit extracted MIND files.**
- **Do not commit private mapping files.**
- Keep generated large artifacts out of version control.
