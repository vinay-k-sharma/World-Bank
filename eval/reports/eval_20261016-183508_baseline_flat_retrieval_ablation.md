# Eval report — eval_20261016-183508_baseline_flat_retrieval_ablation

## Config

```json
{
  "preset": "baseline_flat",
  "strategy": "flat",
  "rerank": false,
  "routing": false,
  "expand": false,
  "rewrite": false,
  "retrieval_only": true,
  "top_n": 8,
  "context_budget": 6000,
  "llm_model": null,
  "judge_model": null,
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

## Per question

| id | category | first hit | context recall | evidence in ctx | top retrieved |
|---|---|---|---|---|---|
| q01 | numeric | 1 | True | True | p59:text, p28-29:text, p59-60:text |
| q02 | numeric | 1 | True | True | p154-155:text, p153-154:text, p155:text |
| q03 | factual | 1 | True | True | p17:text, p175-179:text, p17-18:text |
| q04 | numeric | 2 | True | True | p180:text, p179-180:text, p179:text |
| q05 | numeric | 1 | True | True | p130:text, p130-131:text, p131:text |
| q06 | numeric | None | False | False | p197-198:text, p62:text, p245-246:text |
| q07 | numeric | 1 | True | True | p66:text, p40:text, p198:text |
| q08 | factual | 1 | True | True | p46:text, p41-42:text, p28:text |
| q09 | numeric | None | False | False | p49-50:text, p46-47:text, p47-48:text |
| q10 | numeric | 1 | True | True | p100-101:text, p102:text, p94-95:text |
| q11 | numeric | 7 | True | True | p190-191:text, p120:text, p104-105:text |
| q12 | factual | None | False | False | p30:text, p77:text, p77-78:text |
| q13 | numeric | 2 | True | True | p129:text, p133:text, p132-133:text |
| q14 | numeric | 1 | True | True | p93-94:text, p93-94:text, p96-97:text |
| q15 | numeric | 5 | True | True | p220-221:text, p17-18:text, p52:text |
| q16 | numeric | None | False | False | p99-100:text, p101-102:text, p99:text |
| q17 | numeric | 1 | True | True | p143-144:text, p142:text, p39-40:text |
| q18 | numeric | 1 | True | True | p159:text, p217:text, p216-217:text |
| q19 | numeric | 1 | True | True | p107:text, p108-109:text, p107-108:text |
| q20 | numeric | 1 | True | True | p86:text, p81-83:text, p87:text |
| q21 | numeric | 1 | True | True | p31-32:text, p101:text, p32:text |
| q22 | numeric | 1 | True | True | p91:text, p92:text, p95-96:text |
| q23 | numeric | None | False | False | p94-95:text, p117-118:text, p85:text |
| q24 | numeric | 1 | True | True | p30-31:text, p31:text, p94-95:text |
| q25 | numeric | 2 | True | True | p137:text, p136-137:text, p135:text |
| q26 | numeric | 4 | True | False | p49-50:text, p50-51:text, p46-47:text |
| q27 | numeric | 1 | True | True | p33-34:text, p93-94:text, p78:text |
| q28 | numeric | 1 | True | True | p200-201:text, p200:text, p193:text |
| q29 | explanatory | 2 | True | True | p148:text, p147:text, p147-148:text |
| q30 | explanatory | 2 | True | True | p45:text, p78-79:text, p53-54:text |
| q31 | explanatory | 1 | True | True | p107:text, p83:text, p46:text |
| q32 | explanatory | 1 | True | True | p85:text, p85:text, p81-83:text |
| q33 | explanatory | 5 | True | True | p52:text, p117-118:text, p110:text |
| q34 | explanatory | 2 | True | True | p45:text, p36:text, p87:text |
| q35 | explanatory | 1 | True | True | p136-137:text, p135-136:text, p135:text |
| q36 | table | 1 | True | True | p105-106:text, p101:text, p100-101:text |
| q37 | table | 1 | True | True | p144-145:text, p144:text, p145:text |
| q38 | table | 3 | True | False | p110:text, p109-110:text, p113:text |
| q39 | table | 4 | True | True | p85:text, p83-84:text, p81-83:text |
| q40 | numeric | 2 | True | True | p129:text, p134:text, p132-133:text |
| q41 | figure | 1 | True | True | p146:text, p147-148:text, p145-146:text |
| q42 | figure | 1 | True | True | p190-191:text, p191-192:text, p179-180:text |
| q43 | comparison | 1 | True | False | p41-42:text, p19-21:text, p81:text |
| q44 | multi_hop | 1 | True | True | p41:text, p41-42:text, p92-93:text |
| q45 | comparison | 1 | True | True | p28-29:text, p78-79:text, p59:text |
| q46 | broad | 1 | True | True | p159-160:text, p155:text, p129-130:text |
| q47 | broad | 6 | True | True | p95:text, p46-47:text, p78-79:text |
| q48 | multi_hop | 1 | True | True | p108-109:text, p112-113:text, p107-108:text |
| q49 | unanswerable | None | None | None | p86:text, p78:text, p106:text |
| q50 | unanswerable | None | None | None | p85-86:text, p86:text, p85:text |
| q51 | unanswerable | None | None | None | p90-91:text, p25-26:text, p96-97:text |
| q52 | unanswerable | None | None | None | p108-109:text, p198:text, p109:text |
