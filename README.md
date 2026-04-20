# Resisting Hidden Narrowing in Political News Recommendation

This repository contains the **initial research scaffold** for the paper:

> **“Resisting Hidden Narrowing and Echo-Chamber Formation in Virtualized Political News Exposure through Breadth-Aware Re-Ranking.”**

## Project purpose

The project investigates whether a breadth-aware re-ranking intervention can reduce ideological narrowing and echo-chamber formation in political news exposure while preserving recommendation usefulness.

## Research design (current stage)

This stage implements a reproducible offline experiment scaffold with:

1. A baseline relevance-only recommender.
2. A breadth-aware re-ranking method that modifies baseline rankings.
3. Utility and exposure-structure metrics.
4. Synthetic fixtures to run and test the pipeline without downloading real MIND data.

Utility metrics:
- NDCG@10
- MRR

Exposure metrics:
- Average ideology
- Ideological concentration
- Intra-list diversity
- Cross-cutting exposure rate
- Source coverage

## Repository structure

- `src/hidden_narrowing/`: Core modules (data parser, ideology mapping, features, baseline, reranker, metrics, pipeline).
- `scripts/`: Entry points for data prep/training/evaluation/simulation/plots.
- `tests/`: Unit tests and synthetic fixtures.
- `data/`: Data directories (`raw`, `external`, `processed`).
- `results/`, `reports/`, `figures/`: Outputs.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run tests

```bash
pytest
```

## Run synthetic end-to-end pipeline

```bash
python scripts/run_all.py --sample
```

Expected outputs:
- `results/static_metrics.csv`
- `results/static_metrics_summary.csv`
- `reports/results_summary.md`

## Data policy warning

Do **not** commit raw MIND dataset files to this repository. The scaffold currently runs only on synthetic fixtures in `tests/fixtures/`.
