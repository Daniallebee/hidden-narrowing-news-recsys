# Results Summary

Dataset: **sample**

These outputs validate the pipeline only and should not be interpreted as research findings.

## Operationalization
- This experiment uses topical-semantic breadth within political/public-affairs news exposure.
- Source-level ideology labels were not used in the main MIND-based experiment because MIND URL fields did not expose original publisher domains in usable form.
- Synthetic/sample results remain validation-only.
- Real-data analysis uses slice: `all`.

## Methodology
- Baseline model: cosine (LogisticRegression(max_iter=1000, class_weight='balanced') with cosine fallback).
- Embedding method requested: tfidf; used: tfidf.
- Breadth-aware reranking: final_score = relevance_score + lambda_breadth * breadth_term.
- breadth_term = 0.35*semantic_diversity_gain + 0.35*cross_topic_gain + 0.15*subcategory_novelty - 0.15*topical_concentration_penalty.
- lambda sensitivity tested: 0.15, 0.35, 0.6 (main comparison baseline vs breadth_aware).

## Evaluation Setup
- Number of articles (dev): 8
- Number of users (dev impressions): 3
- Number of impressions (dev candidate rows): 12

## Static Evaluation (summary)
| metric | condition | mean | std | bootstrap_ci_low | bootstrap_ci_high | n |
| --- | --- | --- | --- | --- | --- | --- |
| ndcg_10 | baseline | 0.7103099178571526 | 0.2117009020001536 | 0.5 | 1.0 | 3 |
| ndcg_10 | breadth_aware | 0.5205354372149502 | 0.08303257879894044 | 0.43067655807339306 | 0.6309297535714575 | 3 |
| mrr | baseline | 0.611111111111111 | 0.28327886186626583 | 0.3333333333333333 | 1.0 | 3 |
| mrr | breadth_aware | 0.3611111111111111 | 0.10393492741038726 | 0.25 | 0.5 | 3 |
| hit_10 | baseline | 1.0 | 0.0 | 1.0 | 1.0 | 3 |
| hit_10 | breadth_aware | 1.0 | 0.0 | 1.0 | 1.0 | 3 |
| topical_concentration | baseline | 0.25 | 0.0 | 0.25 | 0.25 | 3 |
| topical_concentration | breadth_aware | 0.25 | 0.0 | 0.25 | 0.25 | 3 |
| subcategory_coverage | baseline | 4.0 | 0.0 | 4.0 | 4.0 | 3 |
| subcategory_coverage | breadth_aware | 4.0 | 0.0 | 4.0 | 4.0 | 3 |
| topical_entropy | baseline | 1.0 | 0.0 | 1.0 | 1.0 | 3 |
| topical_entropy | breadth_aware | 1.0 | 0.0 | 1.0 | 1.0 | 3 |
| semantic_diversity | baseline | 0.9772254131687231 | 0.018755695274956647 | 0.9540630272324707 | 0.9925377374245663 | 3 |
| semantic_diversity | breadth_aware | 0.9772254131687231 | 0.018755695274956647 | 0.9540630272324707 | 0.9925377374245663 | 3 |
| cross_topic_rate | baseline | 0.0 | 0.0 | 0.0 | 0.0 | 3 |
| cross_topic_rate | breadth_aware | 0.0 | 0.0 | 0.0 | 0.0 | 3 |
| history_congruent_share | baseline | 0.0 | 0.0 | 0.0 | 0.0 | 3 |
| history_congruent_share | breadth_aware | 0.0 | 0.0 | 0.0 | 0.0 | 3 |
## Simulation Summary

| condition | topical_concentration_mean | semantic_diversity_mean | cross_topic_rate_mean | subcategory_coverage_mean |
| --- | --- | --- | --- | --- |
| baseline | 0.1250 | 0.9755 | 0.0000 | 8.0000 |
| breadth_aware | 0.1250 | 0.9755 | 0.0000 | 8.0000 |