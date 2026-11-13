Embedder: `bge-small-en-v1.5` · corpus: 56 journals, 35 messages, 5 users · 60 questions, 20 privacy probes, 10 no-memory inputs

| Configuration | Hit@1 | Hit@2 (memories the prompt uses) | Hit@3 | MRR | Hit@3 direct | Hit@3 paraphrase | Privacy-probe leaks | Secrets retrieved | Small talk → memories | Unrelated question → memories |
|---|---|---|---|---|---|---|---|---|---|---|
| Updated pipeline (current app.py, MEMORY_MIN_SIM=0) | 87% | 98% | 98% | 0.93 | 97% | 100% | 0% | 0/20 | 0% | 100% |
| Updated pipeline with MEMORY_MIN_SIM=0.55 | 85% | 97% | 97% | 0.91 | 97% | 97% | 0% | 0/20 | 0% | 0% |
| Previous pipeline (commit 81ed922) | 33% | 53% | 63% | 0.47 | 66% | 61% | 0% | 0/20 | 0% | 100% |
| Previous, without synonym expansion | 48% | 62% | 75% | 0.59 | 79% | 71% | 0% | 0/20 | 0% | 100% |
| Previous, without score boosts | 63% | 70% | 88% | 0.73 | 90% | 87% | 0% | 0/20 | 0% | 100% |
| Previous, without recency weight | 73% | 83% | 87% | 0.79 | 90% | 84% | 0% | 0/20 | 0% | 100% |
| Previous, without boosts and recency weight | 28% | 43% | 88% | 0.51 | 90% | 87% | 0% | 0/20 | 0% | 100% |
| Previous, without second journal pass | 37% | 55% | 65% | 0.49 | 66% | 65% | 0% | 0/20 | 0% | 100% |
| Previous, without privacy filters | 38% | 57% | 67% | 0.51 | 69% | 65% | 0% | 0/20 | 0% | 100% |
| Previous, without privacy filters and boosts | 72% | 78% | 78% | 0.75 | 86% | 71% | 100% | 19/20 | 0% | 100% |
| Baseline: dense top-3 (same privacy rules) | 80% | 95% | 98% | 0.89 | 97% | 100% | 0% | 0/20 | 0% | 40% |
| Baseline: BM25 keyword top-3 (same privacy rules) | 52% | 67% | 73% | 0.61 | 76% | 71% | 0% | 0/20 | 0% | 100% |

Previous pipeline across 5 PYTHONHASHSEED values: Hit@3 60%–68% (the updated pipeline has no synonym step and gives the same result every run).
Share of returned memories that are the twin's messages to a third person (another chat): Updated pipeline 0%, Previous pipeline 0%, Previous, without boosts and recency weight 0%.
