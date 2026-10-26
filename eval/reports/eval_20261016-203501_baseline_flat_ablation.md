# Eval report — eval_20261016-203501_baseline_flat_ablation

## Config

```json
{
  "preset": "baseline_flat",
  "strategy": "flat",
  "rerank": false,
  "routing": false,
  "expand": false,
  "rewrite": false,
  "retrieval_only": false,
  "top_n": 8,
  "context_budget": 6000,
  "llm_model": "gemini-3.5-flash",
  "judge_model": "gemini-3.5-flash",
  "embed_model": "BAAI/bge-base-en-v1.5",
  "rerank_model": null,
  "n_questions": 52
}
```

## Summary

| Metric | Value |
|---|---|
| hit@1 | 0.604 |
| hit@3 | 0.771 |
| hit@5 | 0.854 |
| hit@8 | 0.896 |
| mrr | 0.709 |
| groups_covered_multi | 1.0 |
| context_recall | 0.896 |
| evidence_in_context | 0.833 |
| avg_context_tokens | 3351 |
| correct | 0.921 |
| correct_or_partial | 0.947 |
| faithful | 1.0 |
| citation_on_gold_page | 0.85 |
| refusal_accuracy | 1.0 |
| false_refusals | 3 |
| latency_p50_s | 41.2 |
| errors | 8 |

## By category

| Category | n | Correct | Context recall |
|---|---|---|---|
| broad | 2 | 1.0 | 1.0 |
| comparison | 2 | None | 1.0 |
| explanatory | 7 | 1.0 | 1.0 |
| factual | 3 | 1.0 | 0.667 |
| figure | 2 | None | 1.0 |
| multi_hop | 2 | 1.0 | 1.0 |
| numeric | 26 | 0.88 | 0.846 |
| table | 4 | None | 1.0 |
| unanswerable | 4 | 1.0 | None |

Failures: {'retrieval_miss': 3}

## Per question

| id | category | ctx recall | judgement | faithful | failure | latency s |
|---|---|---|---|---|---|---|
| q01 | numeric | True | correct | True |  | 40.7 |
| q02 | numeric | True | correct | True |  | 36.1 |
| q03 | factual | True | correct | True |  | 39.7 |
| q04 | numeric | True | correct | True |  | 34.5 |
| q05 | numeric | True | correct | True |  | 40.8 |
| q06 | numeric | False | incorrect | True | retrieval_miss | 39.2 |
| q07 | numeric | True | correct | True |  | 53.1 |
| q08 | factual | True | correct | True |  | 40.9 |
| q09 | numeric | False | correct | True |  | 40.2 |
| q10 | numeric | True | correct | True |  | 52.8 |
| q11 | numeric | True | correct | True |  | 40.3 |
| q12 | factual | False | correct | True |  | 41.2 |
| q13 | numeric | True | correct | True |  | 40.9 |
| q14 | numeric | True | correct | True |  | 37.0 |
| q15 | numeric | True | correct | True |  | 40.6 |
| q16 | numeric | False | partially_correct | True | retrieval_miss | 58.8 |
| q17 | numeric | True | correct | True |  | 37.7 |
| q18 | numeric | True | correct | True |  | 70.8 |
| q19 | numeric | True | correct | True |  | 39.7 |
| q20 | numeric | True | correct | True |  | 54.0 |
| q21 | numeric | True | correct | True |  | 39.0 |
| q22 | numeric | True | correct | True |  | 38.5 |
| q23 | numeric | False | incorrect | True | retrieval_miss | 59.7 |
| q24 | numeric | True | correct | True |  | 35.9 |
| q25 | numeric | True | correct | True |  | 43.8 |
| q26 | numeric | True | correct | True |  | 35.4 |
| q27 | numeric | True | correct | True |  | 37.4 |
| q28 | numeric | True | correct | True |  | 73.9 |
| q29 | explanatory | True | correct | True |  | 55.3 |
| q30 | explanatory | True | correct | True |  | 62.4 |
| q31 | explanatory | True | correct | True |  | 38.1 |
| q32 | explanatory | True | correct | True |  | 37.1 |
| q33 | explanatory | True | correct | True |  | 41.2 |
| q34 | explanatory | True | correct | True |  | 36.6 |
| q35 | explanatory | True | correct | True |  | 35.8 |
| q36 | table | True | None | None |  | 43.5 |
| q37 | table | True | None | None |  | 35.6 |
| q38 | table | True | generation_failed | None |  | 432.7 |
| q39 | table | True | generation_failed | None |  | 432.7 |
| q40 | numeric | True | generation_failed | None |  | 432.7 |
| q41 | figure | True | generation_failed | None |  | 432.7 |
| q42 | figure | True | generation_failed | None |  | 432.7 |
| q43 | comparison | True | generation_failed | None |  | 432.7 |
| q44 | multi_hop | True | generation_failed | None |  | 432.7 |
| q45 | comparison | True | generation_failed | None |  | 432.7 |
| q46 | broad | True | correct | True |  | 163.8 |
| q47 | broad | True | correct | True |  | 151.4 |
| q48 | multi_hop | True | correct | True |  | 44.9 |
| q49 | unanswerable | None | True | True |  | 36.7 |
| q50 | unanswerable | None | True | True |  | 44.7 |
| q51 | unanswerable | None | True | True |  | 38.8 |
| q52 | unanswerable | None | True | True |  | 76.6 |
