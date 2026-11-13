"""
Evaluate the AI-twin memory retrieval on the synthetic benchmark.

    python -m eval.run_eval                       # local embedding model (no API key)
    GEMINI_API_KEY=... python -m eval.run_eval --embedder gemini   # the live app's embeddings

Reports, for the deployed pipeline and for ablations / baselines:
  * Hit@1, Hit@3, MRR on the (at most 3) memories the twin actually receives
  * privacy: other people's private journals or threads the twin is not part of
  * how often small talk / unrelated questions still pull in memories
Writes eval/results/<embedder>.json and .md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from . import benchmark as B
from .pipeline import Config, FixedMemoryPipeline, Item, MemoryPipeline, build_items, is_lightweight

NOW = datetime(2026, 11, 13, 12, 0)  # fixed reference date -> reproducible recency weights
OUT = Path(__file__).parent / "results"


class LocalEmbedder:
    name = "bge-small-en-v1.5"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def query(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)

    documents = query


class GeminiEmbedder:
    name = "gemini-embedding-001"

    def __init__(self):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            sys.exit("Set GEMINI_API_KEY in your environment first.")
        self.model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=key)

    def query(self, texts):
        return [self.model.embed_query(t) for t in texts]

    def documents(self, texts):
        return self.model.embed_documents(texts)


class Cached:
    """Embed each distinct text once (ablations re-use the same texts)."""

    def __init__(self, inner):
        self.inner, self.name, self.q, self.d = inner, inner.name, {}, {}

    def _get(self, cache, fn, texts):
        missing = [t for t in dict.fromkeys(texts) if t not in cache]
        if missing:
            for t, v in zip(missing, fn(missing)):
                cache[t] = np.asarray(v, dtype="float32")
        return np.stack([cache[t] for t in texts])

    def query(self, texts):
        return self._get(self.q, self.inner.query, texts)

    def documents(self, texts):
        return self._get(self.d, self.inner.documents, texts)


def violations(asker: str, twin: str, item: Item) -> dict:
    return {
        # someone else's private journal
        "foreign_journal": item.item_type == "journal" and set(item.user_id) != {twin},
        # a chat the twin was not part of
        "foreign_thread": item.item_type == "conversation" and twin not in item.user_id,
        # the twin's own messages to a third person, shown to the asker
        # (allowed by the previous pipeline, blocked by the updated one)
        "cross_thread": item.item_type == "conversation" and twin in item.user_id and asker not in item.user_id,
    }


def score(retrieve) -> dict:
    hits1 = hits2 = hits3 = rr = 0.0
    by_cat: dict[str, list[float]] = {}
    for qid, asker, twin, text, gold, cat in B.QUERIES:
        ids = [m["item"].item_id for m in retrieve(asker, twin, text)]
        rank = next((r for r, i in enumerate(ids, 1) if i in gold), None)
        hits1 += rank == 1
        hits2 += rank is not None and rank <= 2
        hits3 += rank is not None and rank <= 3
        rr += 1 / rank if rank else 0.0
        by_cat.setdefault(cat, []).append(float(rank is not None and rank <= 3))
    n = len(B.QUERIES)

    probe_leaks = secret_hits = 0
    for qid, asker, twin, text, owner, secret in B.PRIVACY_PROBES:
        items = [m["item"] for m in retrieve(asker, twin, text)]
        probe_leaks += any(violations(asker, twin, it)["foreign_journal"] or violations(asker, twin, it)["foreign_thread"]
                           for it in items)
        secret_hits += any(it.item_id == secret for it in items)

    returned = cross = 0
    for qid, asker, twin, text, *_ in B.QUERIES + [(p[0], p[1], p[2], p[3]) for p in B.PRIVACY_PROBES]:
        items = [m["item"] for m in retrieve(asker, twin, text)]
        returned += len(items)
        cross += sum(violations(asker, twin, it)["cross_thread"] for it in items)

    small_talk = [(a, t, x) for _, a, t, x in B.NO_MEMORY if is_lightweight(x)]
    unrelated = [(a, t, x) for _, a, t, x in B.NO_MEMORY if not is_lightweight(x)]
    st = sum(bool(retrieve(a, t, x)) for a, t, x in small_talk)
    un = sum(bool(retrieve(a, t, x)) for a, t, x in unrelated)
    return {
        "hit@1": hits1 / n, "hit@2": hits2 / n, "hit@3": hits3 / n, "mrr": rr / n,
        "hit@3_direct": float(np.mean(by_cat["direct"])),
        "hit@3_paraphrase": float(np.mean(by_cat["paraphrase"])),
        "privacy_probe_leak_rate": probe_leaks / len(B.PRIVACY_PROBES),
        "privacy_secret_retrieved": secret_hits,
        "cross_thread_share": cross / max(returned, 1),
        "small_talk_false_include": st / len(small_talk),
        "unrelated_false_include": un / len(unrelated),
        "n_queries": n, "n_probes": len(B.PRIVACY_PROBES), "n_no_memory": len(B.NO_MEMORY),
    }


def accessible(asker: str, twin: str, it: Item) -> bool:
    """What the app's privacy rules allow a twin to see (used by the simple baselines)."""
    if it.item_type == "journal":
        return set(it.user_id) == {twin}
    if twin not in it.user_id:
        return False
    return it.speaker_id == twin or it.speaker_id == asker


def dense_baseline(items, embedder):
    vecs = embedder.documents([it.content for it in items])

    def retrieve(asker, twin, text):
        if is_lightweight(text):
            return []
        q = embedder.query([text])[0]
        sims = vecs @ q
        order = [i for i in np.argsort(-sims) if accessible(asker, twin, items[i])]
        return [{"item": items[i]} for i in order[:3] if sims[i] >= 0.5]
    return retrieve


def bm25_baseline(items):
    from rank_bm25 import BM25Okapi
    tok = lambda s: [w for w in "".join(c.lower() if c.isalnum() else " " for c in s).split() if len(w) > 2]  # noqa: E731
    bm = BM25Okapi([tok(it.content) for it in items])

    def retrieve(asker, twin, text):
        if is_lightweight(text):
            return []
        s = bm.get_scores(tok(text))
        order = [i for i in np.argsort(-s) if accessible(asker, twin, items[i]) and s[i] > 0]
        return [{"item": items[i]} for i in order[:3]]
    return retrieve


def main():
    if os.environ.get("PYTHONHASHSEED") is None:  # make the main table reproducible (see hash_seed_runs)
        os.execve(sys.executable, [sys.executable, "-m", "eval.run_eval", *sys.argv[1:]],
                  {**os.environ, "PYTHONHASHSEED": "0"})
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=["local", "gemini"], default="local")
    ap.add_argument("--seed-check", action="store_true", help="internal: print previous-pipeline metrics as JSON")
    ap.add_argument("--min-sim", type=float, default=0.55, help="similarity cut-off to report alongside the default (0)")
    args = ap.parse_args()

    embedder = Cached(LocalEmbedder() if args.embedder == "local" else GeminiEmbedder())
    items = build_items(B, NOW)
    names = {u: n for u, n, _ in B.USERS}

    def pipe(**off):
        p = MemoryPipeline(items, names, embedder, NOW, Config(**off))
        return p.retrieve

    if args.seed_check:
        print(json.dumps(score(pipe())))
        return

    def updated(**kw):
        return FixedMemoryPipeline(items, names, embedder, NOW, **kw).retrieve

    rows = {
        "Updated pipeline (current app.py, MEMORY_MIN_SIM=0)": score(updated()),
        f"Updated pipeline with MEMORY_MIN_SIM={args.min_sim}": score(updated(min_sim=args.min_sim)),
        "Previous pipeline (commit 81ed922)": score(pipe()),
        "Previous, without synonym expansion": score(pipe(synonyms=False)),
        "Previous, without score boosts": score(pipe(boosts=False)),
        "Previous, without recency weight": score(pipe(temporal=False)),
        "Previous, without boosts and recency weight": score(pipe(boosts=False, temporal=False)),
        "Previous, without second journal pass": score(pipe(second_pass=False)),
        "Previous, without privacy filters": score(pipe(privacy=False)),
        "Previous, without privacy filters and boosts": score(pipe(privacy=False, boosts=False)),
        "Baseline: dense top-3 (same privacy rules)": score(dense_baseline(items, embedder)),
        "Baseline: BM25 keyword top-3 (same privacy rules)": score(bm25_baseline(items)),
    }

    # The synonym expansion iterates over Python sets, whose order changes with
    # PYTHONHASHSEED (random per process by default). Measure the run-to-run spread.
    seeds = {}
    for seed in range(5):
        env = {**os.environ, "PYTHONHASHSEED": str(seed)}
        out = subprocess.run([sys.executable, "-m", "eval.run_eval", "--embedder", args.embedder, "--seed-check"],
                             env=env, capture_output=True, text=True, check=True, cwd=Path(__file__).parents[1])
        seeds[seed] = json.loads(out.stdout.strip().splitlines()[-1])

    OUT.mkdir(exist_ok=True)
    result = {"embedder": embedder.name, "reference_date": NOW.isoformat(), "rows": rows, "hash_seed_runs": seeds,
              "corpus": {"journals": len(B.JOURNALS), "messages": len(B.CONVERSATIONS), "users": len(B.USERS)}}
    (OUT / f"{args.embedder}.json").write_text(json.dumps(result, indent=2))

    pct = lambda v: f"{100 * v:.0f}%"  # noqa: E731
    lines = [f"Embedder: `{embedder.name}` · corpus: {len(B.JOURNALS)} journals, {len(B.CONVERSATIONS)} messages, "
             f"{len(B.USERS)} users · {len(B.QUERIES)} questions, {len(B.PRIVACY_PROBES)} privacy probes, "
             f"{len(B.NO_MEMORY)} no-memory inputs\n",
             "| Configuration | Hit@1 | Hit@2 (memories the prompt uses) | Hit@3 | MRR | Hit@3 direct | Hit@3 paraphrase | Privacy-probe leaks | "
             "Secrets retrieved | Small talk → memories | Unrelated question → memories |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        lines.append(f"| {name} | {pct(r['hit@1'])} | {pct(r['hit@2'])} | {pct(r['hit@3'])} | {r['mrr']:.2f} | {pct(r['hit@3_direct'])} | "
                     f"{pct(r['hit@3_paraphrase'])} | {pct(r['privacy_probe_leak_rate'])} | "
                     f"{r['privacy_secret_retrieved']}/{r['n_probes']} | {pct(r['small_talk_false_include'])} | "
                     f"{pct(r['unrelated_false_include'])} |")
    h3 = [s["hit@3"] for s in seeds.values()]
    lines.append(f"\nPrevious pipeline across 5 PYTHONHASHSEED values: Hit@3 {pct(min(h3))}–{pct(max(h3))} "
                 "(the updated pipeline has no synonym step and gives the same result every run).")
    lines.append("Share of returned memories that are the twin's messages to a third person (another chat): "
                 + ", ".join(f"{k.split(' (')[0]} {pct(rows[k]['cross_thread_share'])}"
                             for k in ["Updated pipeline (current app.py, MEMORY_MIN_SIM=0)",
                                       "Previous pipeline (commit 81ed922)",
                                       "Previous, without boosts and recency weight"]) + ".")
    table = "\n".join(lines)
    (OUT / f"{args.embedder}.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
