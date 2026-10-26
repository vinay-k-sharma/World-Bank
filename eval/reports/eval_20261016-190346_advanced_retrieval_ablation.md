# Eval report — eval_20261016-190346_advanced_retrieval_ablation

## Config

```json
{
  "preset": "advanced",
  "strategy": "hierarchical",
  "rerank": true,
  "routing": true,
  "expand": true,
  "rewrite": false,
  "retrieval_only": true,
  "top_n": 8,
  "context_budget": 6000,
  "llm_model": null,
  "judge_model": null,
  "embed_model": "BAAI/bge-base-en-v1.5",
  "rerank_model": "BAAI/bge-reranker-base",
  "n_questions": 52
}
```

## Summary

| Metric | Value |
|---|---|
| hit@1 | 0.875 |
| hit@3 | 0.958 |
| hit@5 | 1.0 |
| hit@8 | 1.0 |
| mrr | 0.92 |
| groups_covered_multi | 1.0 |
| context_recall | 1.0 |
| evidence_in_context | 0.979 |
| avg_context_tokens | 3903 |

## Per question

| id | category | first hit | context recall | evidence in ctx | top retrieved |
|---|---|---|---|---|---|
| q01 | numeric | 1 | True | True | p59:text, p29:text, p59:text |
| q02 | numeric | 1 | True | True | p154-155:text, p154:text, p152-155:summary |
| q03 | factual | 1 | True | True | p17:text, p77:figure, p17-18:summary |
| q04 | numeric | 1 | True | True | p179-181:summary, p179:text, p180:figure |
| q05 | numeric | 1 | True | True | p130:text, p130-140:summary, p130:text |
| q06 | numeric | 1 | True | True | p147:text, p142-148:summary, p127:text |
| q07 | numeric | 1 | True | True | p66:text, p60-68:summary, p40:text |
| q08 | factual | 1 | True | True | p46:text, p46:text, p40:text |
| q09 | numeric | 4 | True | False | p47-52:summary, p47:text, p51:text |
| q10 | numeric | 1 | True | True | p100-101:text, p102:text, p242:table |
| q11 | numeric | 1 | True | True | p79:text, p79:figure, p75-81:summary |
| q12 | factual | 1 | True | True | p75:text, p75-76:summary, p77:figure |
| q13 | numeric | 1 | True | True | p133:figure, p127-130:summary, p129:text |
| q14 | numeric | 1 | True | True | p93-94:text, p92-95:summary, p91-98:summary |
| q15 | numeric | 3 | True | True | p17-18:text, p17-18:summary, p64:text |
| q16 | numeric | 2 | True | True | p103:figure, p191-196:summary, p193:text |
| q17 | numeric | 1 | True | True | p143:abstract, p143:text, p144:figure |
| q18 | numeric | 1 | True | True | p159:text, p155-159:summary, p127-130:summary |
| q19 | numeric | 1 | True | True | p107-108:summary, p107:text, p107:text |
| q20 | numeric | 1 | True | True | p86:text, p87-90:summary, p87:text |
| q21 | numeric | 1 | True | True | p31:text, p100-101:text, p25-30:summary |
| q22 | numeric | 1 | True | True | p91:text, p91-92:summary, p92:figure |
| q23 | numeric | 1 | True | True | p35:text, p34-37:summary, p25-30:summary |
| q24 | numeric | 1 | True | True | p30-31:text, p30-34:summary, p31:figure |
| q25 | numeric | 1 | True | True | p135-138:summary, p137:text, p135:text |
| q26 | numeric | 3 | True | True | p47:abstract, p47-52:summary, p37-53:summary |
| q27 | numeric | 4 | True | True | p27:text, p78:figure, p25-30:summary |
| q28 | numeric | 1 | True | True | p200:text, p201:figure, p203:text |
| q29 | explanatory | 1 | True | True | p147:text, p142-148:summary, p130-140:summary |
| q30 | explanatory | 1 | True | True | p78:text, p79:figure, p79:text |
| q31 | explanatory | 1 | True | True | p107:text, p107-108:summary, p109:text |
| q32 | explanatory | 1 | True | True | p85:text, p85:figure, p85:text |
| q33 | explanatory | 1 | True | True | p50-51:text, p118:figure, p49:figure |
| q34 | explanatory | 1 | True | True | p36:text, p35:figure, p87-88:text |
| q35 | explanatory | 1 | True | True | p136:text, p135-138:summary, p138:text |
| q36 | table | 1 | True | True | p106:table, p106:table, p100-101:text |
| q37 | table | 1 | True | True | p145:table, p144:figure, p145:table |
| q38 | table | 1 | True | True | p113:table, p109:figure, p110:text |
| q39 | table | 1 | True | True | p89:table, p89:table, p84-86:summary |
| q40 | numeric | 2 | True | True | p139:figure, p134:text, p138:text |
| q41 | figure | 1 | True | True | p146:figure, p142-148:summary, p148:text |
| q42 | figure | 1 | True | True | p191:figure, p190-191:text, p191-196:summary |
| q43 | comparison | 1 | True | True | p75:abstract, p78:figure, p80:table |
| q44 | multi_hop | 1 | True | True | p41:text, p41:text, p41:abstract |
| q45 | comparison | 1 | True | True | p29:text, p59:text, p59:text |
| q46 | broad | 1 | True | True | p127-130:summary, p155:text, p129-130:text |
| q47 | broad | 1 | True | True | p53:text, p110-113:summary, p78-81:summary |
| q48 | multi_hop | 1 | True | True | p107-113:summary, p108-110:summary, p107:abstract |
| q49 | unanswerable | None | None | None | p78:figure, p113:table, p244:table |
| q50 | unanswerable | None | None | None | p86:text, p34-37:summary, p86:text |
| q51 | unanswerable | None | None | None | p245:summary, p105:table, p93:figure |
| q52 | unanswerable | None | None | None | p196:text, p241-244:summary, p17-18:summary |
