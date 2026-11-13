"""Check that app.py's own retrieval code matches the evaluated pipeline.

Runs the real find_relevant_memories / should_include_memories / generate_personality_traits
source from app.py against an in-memory fake database and FAISS index built from the
benchmark (no MongoDB, no API keys), and compares the metrics with eval/pipeline.py.

    PYTHONHASHSEED=0 python eval/check_app.py
"""
import ast, asyncio, json, logging, re, sys
from datetime import datetime, timedelta
from typing import Any, List, Optional, Tuple
import numpy as np, pytz, faiss
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from eval import benchmark as B
from eval.run_eval import LocalEmbedder, Cached, NOW, score
from eval.pipeline import build_items, FixedMemoryPipeline

src = (__import__("pathlib").Path(__file__).resolve().parents[1] / "app.py").read_text()
tree = ast.parse(src)
want = {"find_relevant_memories", "should_include_memories", "_is_lightweight_input", "_as_uid_list",
        "as_utc_aware", "generate_personality_traits", "_normalize_traits"}
code = "\n\n".join(ast.get_source_segment(src, n) for n in tree.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in want)

E = Cached(LocalEmbedder())
items = build_items(B, NOW)
real_now = datetime.now(pytz.UTC)
shift = real_now - NOW.replace(tzinfo=pytz.UTC)   # app uses the real clock; shift benchmark dates to match

class Doc:
    def __init__(self, c, m): self.page_content, self.metadata = c, m

class FakeFAISS:
    def __init__(self):
        v = E.documents([it.content for it in items]).astype("float32")
        self.index = faiss.IndexFlatL2(v.shape[1]); self.index.add(v)
        self.docs = [Doc(it.content, {"item_id": it.item_id, "item_type": it.item_type, "user_id": it.user_id,
                                      "speaker_id": it.speaker_id, "target_id": it.target_id,
                                      "timestamp": (it.timestamp.replace(tzinfo=pytz.UTC) + shift)}) for it in items]
    def similarity_search_with_score(self, q, k):
        d, i = self.index.search(E.query([q]).astype("float32"), min(k, self.index.ntotal))
        return [(self.docs[j], float(x)) for x, j in zip(d[0], i[0])]

class Coll:
    def __init__(self, rows, key): self.rows = {r[key]: r for r in rows}; self.key = key; self.updates = []
    async def find_one(self, q):
        (k, v), = q.items()
        if k == self.key: return self.rows.get(v)
        return next((r for r in self.rows.values() if r.get(k) == v), None)
    async def update_one(self, *a, **kw): self.updates.append(a)
    def find(self, q):
        rows = [r for r in self.rows.values() if all(
            (r.get(k) == v) if not isinstance(v, dict) else (any(x in r.get(k, []) for x in v["$in"]))
            for k, v in q.items())]
        class Cur:
            def sort(s, *a): rows.sort(key=lambda r: r["timestamp"], reverse=True); return s
            def limit(s, n): s.rows = rows[:n]; return s
            def __aiter__(s): s.it = iter(s.rows); return s
            async def __anext__(s):
                try: return next(s.it)
                except StopIteration: raise StopAsyncIteration
        return Cur()

names = {u: n for u, n, _ in B.USERS}
users = Coll([{"user_id": u, "display_name": n} for u, n, _ in B.USERS], "user_id")
journals = Coll([{"entry_id": it.item_id, "user_id": it.user_id, "content": it.content, "timestamp": it.timestamp}
                 for it in items if it.item_type == "journal"], "entry_id")
convs = Coll([{"conversation_id": it.item_id, "user_id": it.user_id, "content": it.content, "timestamp": it.timestamp,
               "speaker_id": it.speaker_id, "speaker_name": names[it.speaker_id], "target_id": it.target_id}
              for it in items if it.item_type == "conversation"], "conversation_id")
personalities = Coll([], "user_id")

async def _noop(): return None
ns = dict(List=List, Optional=Optional, Tuple=Tuple, Any=Any, np=np, pytz=pytz, datetime=datetime, timedelta=timedelta,
          asyncio=asyncio, re=re, json=json, logger=logging.getLogger("smoke"), faiss_store=FakeFAISS(),
          ensure_faiss_store=_noop, get_mongo_client=_noop, users_col=users, journals_col=journals,
          conversations_col=convs, personalities_col=personalities, MEMORY_MIN_SIM=0.0,
          PERSONALITY_CACHE_TTL_H=24, ANALYSIS_MODEL="x")
exec(code, ns)

def retrieve(asker, twin, text):
    ok, mems = asyncio.run(ns["should_include_memories"](text, asker, twin))
    by_content = {it.content: it for it in items}
    return [{"item": by_content[m["content"]]} for m in mems]

app_r = score(retrieve)
off_r = score(FixedMemoryPipeline(items, names, E, NOW).retrieve)
for k in ["hit@1", "hit@3", "privacy_probe_leak_rate", "cross_thread_share"]:
    print(f"{k:26s} app.py {app_r[k]:.3f}   offline copy {off_r[k]:.3f}")

# Personality: failing LLM must not be cached; journals must be in the prompt
captured = {}
class FailingClient:
    class chat:
        class completions:
            @staticmethod
            async def create(**kw):
                captured["prompt"] = kw["messages"][1]["content"]; raise RuntimeError("api down")
async def _client(): return FailingClient()
ns["get_openai_client"] = _client
traits = asyncio.run(ns["generate_personality_traits"]("u1"))
print("fallback traits returned:", list(traits["core_traits"])[:2], "| cached writes:", len(personalities.updates))
print("journal text in prompt:", "Fujifilm" in captured["prompt"], "| prompt chars:", len(captured["prompt"]))
