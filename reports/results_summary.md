# Results Summary

Dataset: **sample**

**Important:** This run is synthetic validation only, not real-world findings.

## Methodology
- Baseline model: cosine (LogisticRegression(max_iter=1000, class_weight='balanced') with cosine fallback).
- Embedding method requested: tfidf; used: tfidf.
- Breadth-aware reranking: final_score = relevance_score + lambda_breadth * breadth_term.
- breadth_term = 0.35*diversity_gain + 0.35*cross_cutting_gain + 0.10*source_novelty - 0.20*concentration_penalty.
- lambda sensitivity tested: 0.15, 0.35, 0.6 (main comparison uses 0.35).

## Evaluation Setup
- Number of articles (dev): 8
- Number of users (dev impressions): 3
- Number of impressions (dev candidate rows): 12
- Ideology mapping coverage: 87.50%

## Static Evaluation (summary)
| metric | condition | mean | std | bootstrap_ci_low | bootstrap_ci_high | n |
| --- | --- | --- | --- | --- | --- | --- |
| ndcg@10 | baseline | 0.7103099178571526 | 0.2117009020001536 | 0.5 | 1.0 | 3 |
| ndcg@10 | breadth_aware_lambda_0.35 | 0.8333333333333334 | 0.23570226039551584 | 0.5 | 1.0 | 3 |
| mrr | baseline | 0.611111111111111 | 0.28327886186626583 | 0.3333333333333333 | 1.0 | 3 |
| mrr | breadth_aware_lambda_0.35 | 0.7777777777777777 | 0.3142696805273545 | 0.3333333333333333 | 1.0 | 3 |
| hit@10 | baseline | 1.0 | 0.0 | 1.0 | 1.0 | 3 |
| hit@10 | breadth_aware_lambda_0.35 | 1.0 | 0.0 | 1.0 | 1.0 | 3 |
| average_ideology | baseline | -0.05555555555555556 | 0.41154256638969594 | -0.625 | 0.3333333333333333 | 3 |
| average_ideology | breadth_aware_lambda_0.35 | -0.05555555555555556 | 0.41154256638969594 | -0.625 | 0.3333333333333333 | 3 |
| ideological_concentration | baseline | 0.3611111111111111 | 0.20506698694768613 | 0.125 | 0.5277777777777778 | 3 |
| ideological_concentration | breadth_aware_lambda_0.35 | 0.3611111111111111 | 0.20506698694768613 | 0.125 | 0.5277777777777778 | 3 |
| intra_list_diversity | baseline | 0.888888888888889 | 0.3215510250775062 | 0.5833333333333334 | 1.3333333333333333 | 3 |
| intra_list_diversity | breadth_aware_lambda_0.35 | 0.888888888888889 | 0.3215510250775062 | 0.5833333333333334 | 1.3333333333333333 | 3 |
| cross_cutting_exposure_rate | baseline | 0.3333333333333333 | 0.31180478223116176 | 0.0 | 0.5833333333333334 | 3 |
| cross_cutting_exposure_rate | breadth_aware_lambda_0.35 | 0.3333333333333333 | 0.31180478223116176 | 0.0 | 0.5833333333333334 | 3 |
| source_coverage | baseline | 0.75 | 0.0 | 0.75 | 0.75 | 3 |
| source_coverage | breadth_aware_lambda_0.35 | 0.75 | 0.0 | 0.75 | 0.75 | 3 |

## Discussion
- Compare concentration/diversity/cross-cutting/source coverage between baseline and breadth-aware rows above.
- Utility tradeoff should be read from NDCG@10/MRR summary lines.
- See simulation section below for repeated-round dynamics.

## Limitations
- Ideology scores depend on outlet-domain mapping coverage.
- Sentence-transformer mode is optional and environment-dependent.
- Offline click simulation may not reflect live user adaptation.
## Simulation Summary

| condition | concentration_mean | diversity_mean | cross_cutting_rate_mean | source_coverage_mean |
| --- | --- | --- | --- | --- |
| baseline | 0.0714 | 1.0000 | 0.3524 | 0.6250 |
| breadth_aware | 0.0714 | 1.0000 | 0.2190 | 0.6250 |

### Interpretation
- Concentration decreased under breadth-aware: no.
- Diversity increased under breadth-aware: no.
- Cross-cutting exposure increased under breadth-aware: no.
- Source coverage increased under breadth-aware: no.
- NDCG@10 declined under breadth-aware: no.
- MRR declined under breadth-aware: no.
- Repeated rounds show stronger narrowing under baseline: no.