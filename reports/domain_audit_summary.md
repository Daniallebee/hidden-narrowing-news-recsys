# Domain Audit Summary (MINDsmall)

Date: 2026-04-21 (UTC)
Project: Resisting Hidden Narrowing and Echo-Chamber Formation in Virtualized Political News Exposure through Breadth-Aware Re-Ranking

## Workflow status

- **MINDsmall download attempt:** **Failed in sandbox** due to network/proxy error (`Tunnel connection failed: 403 Forbidden`) when running `python scripts/download_mindsmall.py`.
- **Train/dev file existence check (`python scripts/check_data_ready.py`):** **Missing** all required MINDsmall files.
- **Domain audits (`python scripts/audit_domains.py ...`):** **Blocked** because `news.tsv` is missing in both train and dev directories.
- **AllSides source file (`data/external/allsides_media_bias.csv`):** **Missing** (expected at this stage before private mapping creation).

## Required MINDsmall files status

| File | Exists |
|---|---|
| `data/raw/MINDsmall_train/news.tsv` | No |
| `data/raw/MINDsmall_train/behaviors.tsv` | No |
| `data/raw/MINDsmall_dev/news.tsv` | No |
| `data/raw/MINDsmall_dev/behaviors.tsv` | No |

## Aggregate audit metrics

Because the dataset files are not available in this sandbox run, article/domain metrics cannot be computed yet.

| Metric | Value |
|---|---|
| Number of train articles | N/A (download blocked) |
| Number of dev articles | N/A (download blocked) |
| Number of unique train domains | N/A (download blocked) |
| Number of unique dev domains | N/A (download blocked) |

## Top 50 source domains by article frequency

### Train
Not available yet. Run after successful download and extraction:

```bash
python scripts/audit_domains.py --mind-dir data/raw/MINDsmall_train
```

### Dev
Not available yet. Run after successful download and extraction:

```bash
python scripts/audit_domains.py --mind-dir data/raw/MINDsmall_dev
```

## Prioritized list of domains to map first

Not available yet because domain extraction from `news.tsv` could not run without MINDsmall files.

## Exact next step for ideology mapping file

1. Re-run dataset download in an environment that can access the MINDsmall URLs:
   ```bash
   python scripts/download_mindsmall.py
   ```
2. Confirm readiness:
   ```bash
   python scripts/check_data_ready.py
   ```
3. Generate domain audits for train and dev:
   ```bash
   python scripts/audit_domains.py --mind-dir data/raw/MINDsmall_train
   python scripts/audit_domains.py --mind-dir data/raw/MINDsmall_dev
   ```
4. Use the resulting aggregate domain counts to create the private ideology mapping file at:
   - `data/external/allsides_media_bias.csv`

   Start with highest-frequency unmapped domains first (descending total article count across train+dev), then re-run readiness checks.
