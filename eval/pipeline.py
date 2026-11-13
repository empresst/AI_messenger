"""
Offline copy of the AI-twin memory retrieval in app.py, without MongoDB or a server.

Mirrors (as of commit 81ed922):
  preprocess_input          -> preprocess()
  find_relevant_memories    -> MemoryPipeline.candidates()
  semantic_own_journals     -> MemoryPipeline.own_journal_pass()
  should_include_memories   -> MemoryPipeline.retrieve()

Same index layout as the app: the FAISS index embeds the raw text of every message
and journal, while the stored embeddings used by the second journal pass embed the
synonym-expanded text. `Config` switches individual parts off for ablations.
If you change the retrieval logic in app.py, update this file to match.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import faiss
import numpy as np

_NLP = None


def _nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def preprocess(user_input: str) -> str:
    """Same as app.preprocess_input: append up to 10 WordNet synonyms of nouns/verbs."""
    from nltk.corpus import wordnet
    try:
        doc = _nlp()(user_input)
        key_terms = [t.text.lower() for t in doc if t.pos_ in ["NOUN", "VERB"] and not t.is_stop]
        extra_terms = []
        for term in key_terms:
            try:
                syns = wordnet.synsets(term)
            except Exception:
                syns = []
            synonyms = set()
            for syn in syns:
                for lemma in syn.lemmas():
                    w = lemma.name().replace("_", " ")
                    if w != term and len(w.split()) <= 2:
                        synonyms.add(w)
            extra_terms.extend(list(synonyms)[:3])  # set order depends on PYTHONHASHSEED
        if extra_terms:
            user_input += " " + " ".join(set(extra_terms[:10]))
        return user_input
    except Exception:
        return user_input


_LIGHTWEIGHT = {
    "?", "??", "???", "...", "ok", "okay", "k", "kk", "yes", "y", "no", "n",
    "hi", "hey", "hello", "yo", "sup", "hmm", "hmmm", "lol", "haha", "hehe",
    "thanks", "ty", "np", "cool", "nice", "sure", "yup", "yeah", "nah",
    "what", "what?", "what's up", "whats up", "wut", "huh", "right", "true",
    "same", "idk", "oh", "ah", "mhm", "hm", "and?", "so?", "well?",
}


def is_lightweight(user_input: str) -> bool:
    t = re.sub(r"\s+", " ", (user_input or "").strip().lower())
    if not t or len(t) <= 2 or t in _LIGHTWEIGHT:
        return True
    if re.fullmatch(r"(what|hey|hi|ok|okay|yeah|yup|nah|huh|oh)\?{0,3}!{0,3}", t):
        return True
    return bool(re.fullmatch(r"[\W_]+", t))


@dataclass
class Config:
    synonyms: bool = True        # WordNet query expansion
    boosts: bool = True          # own journal +1.0, own speech +0.75, partner +0.2, name +0.15
    temporal: bool = True        # recency weight
    second_pass: bool = True     # extra cosine pass over the twin's own journals
    privacy: bool = True         # ownership / thread filters
    gating: bool = True          # lightweight-input skip and score thresholds


@dataclass
class Item:
    item_id: str
    item_type: str               # "journal" | "conversation"
    content: str
    user_id: list
    timestamp: datetime
    speaker_id: str | None = None
    target_id: str | None = None


class MemoryPipeline:
    def __init__(self, items: list[Item], names: dict[str, str], embedder, now: datetime, cfg: Config):
        self.items, self.names, self.now, self.cfg = items, names, now, cfg
        self.embed = embedder.query          # app.py: embeddings.embed_query
        embed = self.embed
        self.pre = preprocess if cfg.synonyms else (lambda s: s)
        # FAISS index over raw text (what app.py's FAISS.from_documents / add_documents stores)
        vecs = np.asarray(embedder.documents([it.content for it in items]), dtype="float32")
        self.index = faiss.IndexFlatL2(vecs.shape[1])
        self.index.add(vecs)
        # Stored embeddings of synonym-expanded text (what the second journal pass reads)
        self.stored = np.asarray(embed([self.pre(it.content) for it in items]), dtype="float32")

    # find_relevant_memories
    def candidates(self, speaker_id: str, user_id: str, text: str, max_memories: int = 8) -> list[dict]:
        cfg = self.cfg
        q = np.asarray(self.embed([self.pre(text)]), dtype="float32")
        k = min(max(max_memories * 8, 24), len(self.items))
        dist, idx = self.index.search(q, k)
        speaker_name, target_name = self.names[speaker_id], self.names[user_id]
        mems = []
        for d, i in zip(dist[0], idx[0]):
            it = self.items[i]
            uids = list(it.user_id)
            is_own_journal = is_own_speech = False
            if cfg.privacy:
                if user_id not in uids:
                    continue
                if it.item_type == "journal":
                    if set(uids) != {user_id}:
                        continue
                    is_own_journal = True
                else:
                    is_own_speech = it.speaker_id == user_id
                    if not is_own_speech:
                        if it.speaker_id != speaker_id:
                            continue
                        if not (it.target_id == user_id or (speaker_id in uids and user_id in uids)):
                            continue
            else:
                is_own_journal = it.item_type == "journal" and set(uids) == {user_id}
                is_own_speech = it.item_type == "conversation" and it.speaker_id == user_id

            adjusted = 1.0 - float(d)
            if cfg.boosts:
                if is_own_journal:
                    adjusted += 1.0
                elif is_own_speech:
                    adjusted += 0.75
                else:
                    adjusted += 0.2
                low = it.content.lower()
                if speaker_name.lower() in low or target_name.lower() in low:
                    adjusted += 0.15
            if cfg.temporal:
                days_old = (self.now - it.timestamp).days
                adjusted *= 1 / (1 + np.log1p(max(days_old, 1) / 30))
            if cfg.gating and adjusted < 0.3:
                continue
            mems.append({"item": it, "score": float(adjusted), "is_own_journal": is_own_journal})
        mems.sort(key=lambda m: m["score"], reverse=True)
        return mems[:max_memories]

    # semantic_own_journals
    def own_journal_pass(self, user_id: str, text: str, max_n: int = 3, min_sim: float = 0.28) -> list[dict]:
        q = np.asarray(self.embed([self.pre(text)]), dtype="float32")[0]
        q = q / max(np.linalg.norm(q), 1e-9)
        hits = []
        # app.py reads the 100 most recent journal embeddings of ALL users, then filters by owner
        journals = sorted((i for i, it in enumerate(self.items) if it.item_type == "journal"),
                          key=lambda i: self.items[i].timestamp, reverse=True)[:100]
        for i in journals:
            it = self.items[i]
            if self.cfg.privacy and set(it.user_id) != {user_id}:
                continue
            v = self.stored[i]
            sim = float(np.dot(q, v / max(np.linalg.norm(v), 1e-9)))
            if self.cfg.gating and sim < min_sim:
                continue
            hits.append({"item": it, "score": sim + 0.45, "is_own_journal": set(it.user_id) == {user_id}})
        hits.sort(key=lambda h: -h["score"])
        return hits[:max_n]

    # should_include_memories
    def retrieve(self, speaker_id: str, user_id: str, text: str) -> list[dict]:
        if self.cfg.gating and is_lightweight(text):
            return []
        rel = []
        for m in self.candidates(speaker_id, user_id, text, max_memories=8):
            thr = 0.30 if (m["is_own_journal"] or m["item"].item_type == "journal") else 0.50
            if not self.cfg.gating or m["score"] >= thr:
                rel.append(m)
        if self.cfg.second_pass:
            seen = {m["item"].content.strip().lower() for m in rel}
            for j in self.own_journal_pass(user_id, text):
                key = j["item"].content.strip().lower()
                if key not in seen:
                    rel.append(j)
                    seen.add(key)
        rel.sort(key=lambda m: (0 if m["is_own_journal"] else 1, -m["score"]))
        return rel[:3]


def build_items(benchmark, now: datetime) -> list[Item]:
    def ts(days_ago, hour):
        return (now - timedelta(days=days_ago)).replace(hour=hour, minute=0)

    items = [Item(eid, "journal", text, [owner], ts(d, h)) for eid, owner, d, h, text in benchmark.JOURNALS]
    items += [Item(cid, "conversation", text, [sp, tg], ts(d, h), speaker_id=sp, target_id=tg)
              for cid, sp, tg, d, h, text in benchmark.CONVERSATIONS]
    return items


class FixedMemoryPipeline(MemoryPipeline):
    """The updated ranking now in app.py (find_relevant_memories / should_include_memories).

    * rank by cosine similarity, from FAISS L2 distance on unit vectors: cos = 1 - d / 2
    * own journal / own words only as small tie-breakers (+0.03 / +0.02)
    * recency can change a score by at most 10%
    * no synonym expansion (deterministic, and it added noise)
    * candidates: the whole index, then the privacy filters; chat messages only from the
      twin's thread with the current asker (the twin's messages to others stay in that chat)
    * keep memories with cos >= min_sim, return the top 3 by score
    """

    def __init__(self, items, names, embedder, now, min_sim: float = 0.0, own_bonus: bool = True, recency: bool = True):
        super().__init__(items, names, embedder, now, Config(synonyms=False))
        self.min_sim, self.own_bonus, self.recency = min_sim, own_bonus, recency

    def retrieve(self, speaker_id, user_id, text):
        if is_lightweight(text):
            return []
        q = np.asarray(self.embed([text]), dtype="float32")
        dist, idx = self.index.search(q, len(self.items))
        out = []
        for d, i in zip(dist[0], idx[0]):
            if 1.0 - float(d) / 2.0 < self.min_sim:
                continue
            if len(out) >= 12:  # app.py stops after max_memories * 4 survivors
                break
            it = self.items[i]
            uids = list(it.user_id)
            if user_id not in uids:
                continue
            if it.item_type == "journal":
                if set(uids) != {user_id}:
                    continue
                bonus = 0.03
            elif set(uids) != {speaker_id, user_id}:
                continue  # app.py: only the thread between the twin and the current chat partner
            elif it.speaker_id == user_id:
                bonus = 0.02
            else:
                bonus = 0.0
            cos = 1.0 - float(d) / 2.0
            s = cos + (bonus if self.own_bonus else 0.0)
            if self.recency:
                days_old = (self.now - it.timestamp).days
                s *= 0.9 + 0.1 / (1 + np.log1p(max(days_old, 1) / 30))
            out.append({"item": it, "score": s, "is_own_journal": it.item_type == "journal"})
        out.sort(key=lambda m: -m["score"])
        return out[:3]
