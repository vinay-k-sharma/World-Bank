# Eval report — eval_20261016-183509_structure_retrieval_ablation

## Config

```json
{
  "preset": "structure",
  "strategy": "hierarchical",
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
| hit@1 | 0.708 |
| hit@3 | 0.917 |
| hit@5 | 0.938 |
| hit@8 | 0.958 |
| mrr | 0.817 |
| groups_covered_multi | 1.0 |
| context_recall | 0.958 |
| evidence_in_context | 0.917 |
| avg_context_tokens | 2257 |

## Per question

| id | category | first hit | context recall | evidence in ctx | top retrieved |
|---|---|---|---|---|---|
| q01 | numeric | 1 | True | True | p59:text, p59:text, p59:text |
| q02 | numeric | 2 | True | True | p155:text, p154:figure, p154-155:text |
| q03 | factual | 1 | True | True | p17:text, p17:text, p17-18:summary |
| q04 | numeric | 2 | True | True | p180:figure, p179:text, p179-181:summary |
| q05 | numeric | 1 | True | True | p130:text, p130:text, p130-140:summary |
| q06 | numeric | 1 | True | True | p147:text, p182:text, p60-68:summary |
| q07 | numeric | 2 | True | True | p40:text, p60-68:summary, p39:figure |
| q08 | factual | 1 | True | True | p46:text, p46:figure, p46:text |
| q09 | numeric | None | False | False | p47:text, p51:text, p47:text |
| q10 | numeric | 1 | True | True | p100-101:text, p102:text, p242:table |
| q11 | numeric | 1 | True | True | p79:text, p104:text, p57:text |
| q12 | factual | 2 | True | True | p77:figure, p75-76:summary, p77:figure |
| q13 | numeric | 2 | True | True | p129:text, p133:figure, p133:text |
| q14 | numeric | 1 | True | True | p92-95:summary, p19-20:text, p93-94:text |
| q15 | numeric | None | False | False | p17-18:text, p52:text, p220:table |
| q16 | numeric | 2 | True | True | p99:text, p193:text, p102:text |
| q17 | numeric | 1 | True | True | p143:abstract, p144:figure, p142:text |
| q18 | numeric | 1 | True | True | p159:text, p155-159:summary, p217:text |
| q19 | numeric | 2 | True | True | p108-110:summary, p107-108:summary, p107-113:summary |
| q20 | numeric | 1 | True | True | p86:text, p87-90:summary, p87:figure |
| q21 | numeric | 2 | True | True | p32:text, p31:text, p102:text |
| q22 | numeric | 1 | True | True | p91:text, p91-92:summary, p75-121:summary |
| q23 | numeric | 1 | True | False | p34-37:summary, p92-93:text, p92-95:summary |
| q24 | numeric | 1 | True | True | p30-31:text, p31:figure, p108-109:text |
| q25 | numeric | 1 | True | True | p135-138:summary, p135:text, p137:table |
| q26 | numeric | 8 | True | True | p49:text, p47:abstract, p51:text |
| q27 | numeric | 1 | True | True | p33:figure, p78:figure, p27:text |
| q28 | numeric | 3 | True | True | p201:figure, p191-192:text, p200:text |
| q29 | explanatory | 1 | True | True | p147:text, p142-148:summary, p148:text |
| q30 | explanatory | 2 | True | True | p79:figure, p78:text, p45:text |
| q31 | explanatory | 1 | True | True | p107-108:summary, p107:text, p107:text |
| q32 | explanatory | 1 | True | True | p85:text, p85:figure, p110:text |
| q33 | explanatory | 1 | True | True | p50-51:text, p52:text, p52:text |
| q34 | explanatory | 1 | True | True | p36:text, p87:text, p87-88:text |
| q35 | explanatory | 1 | True | True | p136:text, p135-138:summary, p135:abstract |
| q36 | table | 1 | True | True | p106:table, p106:table, p105:table |
| q37 | table | 1 | True | True | p145:table, p145:table, p144:text |
| q38 | table | 1 | True | True | p113:table, p113:table, p106:table |
| q39 | table | 1 | True | True | p89:table, p89:table, p90:table |
| q40 | numeric | 4 | True | True | p139:figure, p138:text, p129:text |
| q41 | figure | 1 | True | True | p146:figure, p148:text, p142-148:summary |
| q42 | figure | 1 | True | True | p191:figure, p191:figure, p190-191:text |
| q43 | comparison | 1 | True | False | p41-45:summary, p81:table, p80:table |
| q44 | multi_hop | 1 | True | True | p41:text, p19-20:text, p19-20:summary |
| q45 | comparison | 1 | True | True | p59:text, p28-29:text, p59:text |
| q46 | broad | 1 | True | True | p155:text, p159-160:summary, p129-130:text |
| q47 | broad | 1 | True | True | p53:text, p110-113:summary, p54:text |
| q48 | multi_hop | 1 | True | True | p108:figure, p107:abstract, p107-108:summary |
| q49 | unanswerable | None | None | None | p78:figure, p244:table, p120:table |
| q50 | unanswerable | None | None | None | p85:figure, p241:table, p35:figure |
| q51 | unanswerable | None | None | None | p32:text, p97:table, p97:table |
| q52 | unanswerable | None | None | None | p108-109:text, p241-244:summary, p25-68:summary |
