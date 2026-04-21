# Real MINDsmall Run Guide

This guide describes how to run the full experiment on real MINDsmall files while keeping the repository safe and reproducible.

## 1) Download MINDsmall

Download the MIND dataset from Microsoft MIND resources (use the **MINDsmall** train and dev splits).

## 2) Place files in the expected structure

Create this exact layout:

- `data/raw/MINDsmall_train/news.tsv`
- `data/raw/MINDsmall_train/behaviors.tsv`
- `data/raw/MINDsmall_dev/news.tsv`
- `data/raw/MINDsmall_dev/behaviors.tsv`

## 3) Create outlet-to-bias mapping file

Place your mapping at:

- `data/external/allsides_media_bias.csv`

Use `data/external/allsides_media_bias_template.csv` as a schema template.

## 4) Optional domain audit before full run

Audit domain coverage in each split:

```bash
python scripts/audit_domains.py --mind-dir data/raw/MINDsmall_train
python scripts/audit_domains.py --mind-dir data/raw/MINDsmall_dev
```

The script writes:

- `results/domain_coverage.csv`
- `results/unmapped_domains.csv`

## 5) Run the full real-data experiment

```bash
python scripts/run_all.py --mind-train-dir data/raw/MINDsmall_train --mind-dev-dir data/raw/MINDsmall_dev --bootstrap-samples 1000
```

## 6) Faster debug run

```bash
python scripts/run_all.py --mind-train-dir data/raw/MINDsmall_train --mind-dev-dir data/raw/MINDsmall_dev --bootstrap-samples 200
```

You can also reduce workload for quick checks:

```bash
python scripts/run_all.py --mind-train-dir data/raw/MINDsmall_train --mind-dev-dir data/raw/MINDsmall_dev --max-train-impressions 2000 --max-dev-impressions 500 --bootstrap-samples 200
```

## Safety warning

- **Do not commit raw MIND data.**
- **Do not commit private mapping files.**
- Keep generated large artifacts out of version control.
