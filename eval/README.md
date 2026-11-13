# Retrieval evaluation

Measures how well the AI twin finds the right memory, whether it ever exposes another
person's private journal or chat, and what each part of the retrieval pipeline contributes.
The evaluation led to the retrieval changes in `app.py` described below.

## What is measured

- **Benchmark** ([`benchmark.py`](benchmark.py)): a synthetic family of five (no real user data) with 56 private journal entries and 35 chat messages, plus:
  - 60 questions, each labelled with the memories that answer it (29 worded like the memory, 31 paraphrased);
  - 20 privacy probes, which ask a twin about something that exists **only** in another person's private journal;
  - 10 inputs that should pull in no memories (5 small-talk inputs, 5 unrelated questions).
- **Pipelines** ([`pipeline.py`](pipeline.py)):
  - an offline copy of the previous retrieval (commit `81ed922`), with each part switchable for ablations;
  - the updated retrieval now in `app.py`.
  - [`check_app.py`](check_app.py) runs the real functions from `app.py` against a fake database built from the benchmark, and confirms they give the same numbers as the offline copy.
- **Metrics:**
  - Hit@1 / Hit@3 / MRR on the memories the twin actually receives (at most 3);
  - privacy-probe leaks (another person's journal, or a chat the twin was not in);
  - the twin's messages from *another* chat being shown to the current asker;
  - how often inputs with nothing to recall still pull in memories.

## Run

```bash
pip install sentence-transformers faiss-cpu rank_bm25 spacy nltk pytz
python -m spacy download en_core_web_sm
python -m eval.run_eval                                        # local model, no API key
GEMINI_API_KEY=... python -m eval.run_eval --embedder gemini   # the embeddings the live app uses
PYTHONHASHSEED=0 python eval/check_app.py                      # app.py matches the evaluated pipeline
```

Results are written to [`results/`](results/).

## Results (local embedding model `bge-small-en-v1.5`)

Full table with all ablations: [`results/local.md`](results/local.md).

| Configuration | Hit@1 | Hit@3 | MRR | Privacy-probe leaks | Messages from another chat shown | Unrelated question → memories |
|---|---|---|---|---|---|---|
| **Updated pipeline (current `app.py`)** | **87%** | **98%** | **0.93** | 0/20 | 0% | 100% |
| Updated, with `MEMORY_MIN_SIM=0.55` | 85% | 97% | 0.91 | 0/20 | 0% | 0% |
| Previous pipeline (`81ed922`) | 33% | 63% | 0.47 | 0/20 | see finding 3 | 100% |
| Previous, without privacy filters and boosts | 72% | 78% | 0.75 | **19/20 secrets retrieved** | – | 100% |
| BM25 keyword baseline, same privacy rules | 52% | 73% | 0.61 | 0/20 | – | 100% |

### Findings and changes

1. **Ranking favoured recent journals over relevant ones.** The previous scoring added +1.0 for the twin's own journals and multiplied every score by a recency weight of about 0.5–0.95. Relevant and irrelevant memories differ in similarity by only about 0.2, so the newest journals won almost regardless of the question. Removing either the boosts or the recency weight alone raised Hit@3 from 63% to about 88%.
   **Change:** rank by cosine similarity; own journal / own words are now tie-breakers (+0.03 / +0.02), and recency changes a score by at most 10%.
2. **Privacy filters work, and they are what does the work.** No probe surfaced another person's journal, before or after the change. With the filters and the boosts both switched off, 19 of the 20 secrets were retrieved.
3. **The twin repeated its own messages from other chats.** The previous rules let a twin recall anything it had said in *any* chat. In the benchmark, the father asks Mina's twin "Has Rafi accepted the Singapore job?" and the updated ranking retrieved Mina's message to Rafi: *"Bhaiya, are you really going to Singapore? Tell me first before you tell Abbu."* The old ranking hid this only by accident, because its journal boost pushed chat messages down; 19% of all memories returned contained such messages once the ranking was fixed.
   **Change:** chat messages are recalled only from the twin's chat with the current asker. The twin's general knowledge still comes from its own journals.
4. **Results changed between restarts.** Synonym expansion iterated over Python sets, whose order depends on `PYTHONHASHSEED` (random per process): Hit@3 ranged from 60% to 68% across five seeds. Expansion also added unrelated senses, for example "Are you scared of dogs?" was embedded as "Are you scared of dogs? firedog wiener tag".
   **Change:** queries are embedded as written, so retrieval is deterministic.
5. **The thresholds filtered almost nothing.** After the +1.0 boost, the twin's own journals always cleared the 0.30 threshold, so unrelated questions ("What's the capital of Peru?") still pulled in three memories.
   **Change:** an optional cosine cut-off, `MEMORY_MIN_SIM` (default 0, off). At 0.55 the local model drops unrelated questions to 0% while keeping Hit@3 at 97%. That value was picked on this benchmark and for this model, so calibrate it with `--embedder gemini` before turning it on in production.
6. **Smaller fixes found while reading the code:**
   - every message triggered one extra embedding API call whose result was never used (removed);
   - the second journal pass read only the 100 most recent journals of *all* users before filtering by owner, so older entries dropped out once the app held more than 100 journals. That pass is removed; the main search now covers the whole index with the same privacy filters.

### Personality traits (not evaluated, code fixes only)

The Big Five profile is generated by an LLM; measuring its quality needs real LLM calls and is not part of this benchmark. Three problems in the code were fixed, and `check_app.py` checks the first two:

- the prompt took 80 chat messages first and cut everything at 1,500 characters, so journals were usually cut off. Journals now come first with half of a 4,000-character budget;
- if the LLM failed three times, a neutral 50/50/50 profile was cached for the full TTL. The fallback is still used for that reply but is no longer cached;
- `temperature` was lowered from 0.7 to 0.2, so the same text gives more consistent traits.

### Caveats

- **These numbers use a local model,** not the Gemini embeddings the live app uses. Findings 1–4 follow from the structure of the code. Absolute numbers, and the right value for `MEMORY_MIN_SIM`, need the Gemini run.
- **The benchmark is small and synthetic,** written by the project author, with one fixed set of questions. Treat differences of a few percentage points as noise.
- **The updated pipeline was designed after seeing these results.** A fresh, independently written question set would be a fairer test of it.
- **The cosine conversion assumes unit-length embeddings.** It is computed from FAISS's squared L2 distance as `1 - d/2`. This holds for `bge-small` with normalisation; confirm it for the Gemini embeddings during the Gemini run (cosines should stay between -1 and 1).
