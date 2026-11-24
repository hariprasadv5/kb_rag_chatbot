"""
common.py — shared building blocks for the Company Knowledge-Base Chatbot (Module 2).

Everything the four m2_*.py deliverables share lives here so each script stays
focused on its own technique:

  * Settings (read from .env)
  * Lazy clients: OpenAI (wrapped for LangSmith), Chroma, Cohere
  * Embeddings, tokenizer, BM25, Reciprocal Rank Fusion (RRF)
  * A Python evaluator for Chroma `where` filters (so BM25 obeys the same filters)
  * Grounded answer generation with numbered citations
  * Audit log (logs/audit.jsonl) — every answer is traceable to its sources

Heavy dependencies (chromadb, openai, cohere, langsmith) are imported lazily so the
pure-Python helpers (chunking, RRF, filters, BM25) can be unit-tested offline.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parent

try:  # .env is optional
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


# --------------------------------------------------------------------------- settings
class Settings:
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    chroma_dir = os.getenv("CHROMA_DIR", str(ROOT / "chroma_db"))
    collection = os.getenv("CHROMA_COLLECTION", "company_kb")
    raptor_collection = os.getenv("RAPTOR_COLLECTION", "company_kb_raptor")
    reranker = os.getenv("RERANKER", "cohere")  # cohere | cross-encoder | none
    cohere_model = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
    cross_encoder_model = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    dense_k = int(os.getenv("DENSE_K", 50))        # stage 1: bi-encoder candidates
    bm25_k = int(os.getenv("BM25_K", 50))          # stage 1: keyword candidates
    rrf_k = int(os.getenv("RRF_K", 60))            # RRF constant (Cormack et al. 2009)
    final_k = int(os.getenv("FINAL_K", 5))         # stage 2: after cross-encoder rerank
    min_rerank_score = float(os.getenv("MIN_RERANK_SCORE", 0.05))  # below → abstain
    default_tenant = os.getenv("DEFAULT_TENANT", "northwind")
    audit_log = ROOT / "logs" / "audit.jsonl"


S = Settings()


# --------------------------------------------------------------------------- tracing
try:
    from langsmith import traceable  # type: ignore
except ImportError:  # LangSmith not installed → no-op decorator

    def traceable(*args, **kwargs):  # type: ignore
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        return lambda f: f


def current_run_id() -> str | None:
    """LangSmith run id of the active trace (for the audit log)."""
    try:
        from langsmith.run_helpers import get_current_run_tree

        rt = get_current_run_tree()
        return str(rt.trace_id) if rt else None
    except Exception:
        return None


# --------------------------------------------------------------------------- clients
_openai = None
_chroma = None
_cohere = None
_cross_encoder = None


def get_openai():
    global _openai
    if _openai is None:
        from openai import OpenAI

        client = OpenAI()
        try:  # every OpenAI call shows up as a child run in LangSmith
            from langsmith.wrappers import wrap_openai

            client = wrap_openai(client)
        except ImportError:
            pass
        _openai = client
    return _openai


def get_chroma():
    global _chroma
    if _chroma is None:
        import chromadb

        _chroma = chromadb.PersistentClient(path=S.chroma_dir)
    return _chroma


def get_collection(name: str | None = None):
    return get_chroma().get_or_create_collection(
        name=name or S.collection, metadata={"hnsw:space": "cosine"}
    )


def get_cohere():
    global _cohere
    if _cohere is None:
        import cohere

        _cohere = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
    return _cohere


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(S.cross_encoder_model)
    return _cross_encoder


# --------------------------------------------------------------------------- embeddings / LLM
@traceable(name="embed", run_type="embedding")
def embed(texts: list[str], batch_size: int = 96) -> list[list[float]]:
    out: list[list[float]] = []
    client = get_openai()
    for i in range(0, len(texts), batch_size):
        batch = [t.replace("\n", " ")[:30000] for t in texts[i : i + batch_size]]
        resp = client.embeddings.create(model=S.embed_model, input=batch)
        out.extend(d.embedding for d in resp.data)
    return out


def chat(messages: list[dict], temperature: float = 0.0, json_mode: bool = False, model: str | None = None) -> str:
    kwargs: dict[str, Any] = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = get_openai().chat.completions.create(
        model=model or S.chat_model, messages=messages, temperature=temperature, **kwargs
    )
    return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- text utils
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")
STOPWORDS = set(
    "a an the and or of to in on for with is are was were be by at as it its this that from "
    "how what when where which who whom why do does did can could should would will i we you "
    "our your my me us they them their there here about into than then so if not no".split()
)


def tokenize(text: str) -> list[str]:
    """Keyword tokenizer that keeps exact IDs intact ("INC-2041", "RB-DB-004", "v2.3")
    and ALSO emits their parts, so "inc-2041" and "2041" both match."""
    toks: list[str] = []
    for m in _TOKEN_RE.findall(text.lower()):
        if m in STOPWORDS:
            continue
        toks.append(m)
        if re.search(r"[-_./]", m):
            toks.extend(p for p in re.split(r"[-_./]", m) if p and p not in STOPWORDS)
    return toks


def sha1(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8", "ignore"))
        h.update(b"\x1f")
    return h.hexdigest()


def date_to_int(d: str | date | datetime | None) -> int:
    """'2025-07-01' → 20250701 (Chroma range filters need numbers, not strings)."""
    if d is None or d == "":
        return 0
    if isinstance(d, (date, datetime)):
        return int(d.strftime("%Y%m%d"))
    return int(str(d)[:10].replace("-", ""))


# --------------------------------------------------------------------------- BM25
class SimpleBM25:
    """Okapi BM25 (k1=1.5, b=0.75). Used when `rank_bm25` is not installed;
    identical scoring to rank_bm25.BM25Okapi for practical purposes."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = (sum(self.doc_len) / len(corpus)) if corpus else 0.0
        self.tf = [Counter(d) for d in corpus]
        df: Counter = Counter()
        for d in corpus:
            df.update(set(d))
        n = len(corpus)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def get_scores(self, query: list[str]) -> list[float]:
        scores = []
        for tf, dl in zip(self.tf, self.doc_len):
            s = 0.0
            for q in query:
                f = tf.get(q)
                if not f:
                    continue
                s += self.idf.get(q, 0.0) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                )
            scores.append(s)
        return scores


def make_bm25(tokenized_corpus: list[list[str]]):
    try:
        from rank_bm25 import BM25Okapi

        return BM25Okapi(tokenized_corpus)
    except ImportError:
        return SimpleBM25(tokenized_corpus)


# --------------------------------------------------------------------------- RRF
def rrf_fuse(ranked_lists: Iterable[list[str]], k: int = 60, weights: list[float] | None = None) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = Σ w_i / (k + rank_i(d)), rank starting at 1.
    Rank-based, so BM25 scores and cosine similarities never need normalising."""
    scores: dict[str, float] = {}
    lists = list(ranked_lists)
    weights = weights or [1.0] * len(lists)
    for w, lst in zip(weights, lists):
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# --------------------------------------------------------------------------- where-filter evaluator
def matches_where(meta: dict, where: dict | None) -> bool:
    """Evaluate a Chroma-style `where` dict against one metadata dict in Python.
    Lets the BM25 side apply *exactly* the same filter as the vector side."""
    if not where:
        return True
    for key, cond in where.items():
        if key == "$and":
            if not all(matches_where(meta, c) for c in cond):
                return False
        elif key == "$or":
            if not any(matches_where(meta, c) for c in cond):
                return False
        else:
            val = meta.get(key)
            if not isinstance(cond, dict):
                cond = {"$eq": cond}
            for op, ref in cond.items():
                if op == "$eq" and val != ref:
                    return False
                if op == "$ne" and val == ref:
                    return False
                if op == "$in" and val not in ref:
                    return False
                if op == "$nin" and val in ref:
                    return False
                if op in ("$gt", "$gte", "$lt", "$lte"):
                    if val is None:
                        return False
                    if op == "$gt" and not val > ref:
                        return False
                    if op == "$gte" and not val >= ref:
                        return False
                    if op == "$lt" and not val < ref:
                        return False
                    if op == "$lte" and not val <= ref:
                        return False
    return True


# --------------------------------------------------------------------------- docs
def as_doc(doc_id: str, text: str, meta: dict, score: float | None = None, **extra) -> dict:
    """Documents are plain dicts shaped the way LangSmith renders retriever outputs."""
    d = {"id": doc_id, "page_content": text, "metadata": dict(meta or {}), "type": "Document"}
    if score is not None:
        d["score"] = round(float(score), 4)
    d.update(extra)
    return d


def cite_label(meta: dict) -> str:
    bits = [meta.get("title") or Path(str(meta.get("source", "?"))).name]
    if meta.get("page"):
        bits.append(f"p.{meta['page']}")
    if meta.get("section"):
        bits.append(f"§ {meta['section']}")
    if meta.get("ticket_id"):
        bits.append(meta["ticket_id"])
    if meta.get("date"):
        bits.append(str(meta["date"]))
    return " — ".join(bits)


# --------------------------------------------------------------------------- reranking
def cohere_rerank_with_retry(query, texts, top_n, retries=4):
    """Trial keys allow ~10 rerank calls/minute: wait and retry instead of silently skipping rerank."""
    for attempt in range(retries + 1):
        try:
            return get_cohere().rerank(model=S.cohere_model, query=query, documents=texts, top_n=top_n)
        except Exception as e:
            if "TooManyRequests" not in type(e).__name__ or attempt == retries:
                raise
            wait = 7 * (attempt + 1)
            print(f"[rerank] Cohere rate limit hit; waiting {wait}s (retry {attempt + 1}/{retries})")
            time.sleep(wait)
@traceable(name="rerank", run_type="retriever")
def rerank(query: str, docs: list[dict], top_n: int | None = None, method: str | None = None) -> list[dict]:
    """Stage 2: cross-encoder scores (query, passage) pairs jointly → much sharper than
    the bi-encoder used in stage 1, but too slow to run over the whole corpus."""
    top_n = top_n or S.final_k
    method = (method or S.reranker).lower()
    if not docs:
        return []
    texts = [d["page_content"] for d in docs]
    try:
        if method == "cohere":
            res = cohere_rerank_with_retry(query, texts, top_n)
            ranked = [(r.index, r.relevance_score) for r in res.results]
        elif method in ("cross-encoder", "cross_encoder", "local"):
            scores = get_cross_encoder().predict([(query, t) for t in texts])
            # sigmoid → 0..1 so MIN_RERANK_SCORE means the same thing for both rerankers
            scores = [1 / (1 + math.exp(-float(s))) for s in scores]
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_n]
        else:
            return [dict(d, rerank_score=None) for d in docs[:top_n]]
    except Exception as e:  # never fail the user because the reranker is down
        print(f"[rerank] {method} failed ({e!r}); falling back to fused order")
        return [dict(d, rerank_score=None) for d in docs[:top_n]]
    return [dict(docs[i], rerank_score=round(float(s), 4)) for i, s in ranked]


# --------------------------------------------------------------------------- answer generation
ANSWER_SYSTEM = """You are the internal Company Knowledge-Base assistant.
Rules:
1. Answer ONLY from the numbered context passages. Do not use outside knowledge.
2. Cite every factual sentence with the passage number(s) in square brackets, e.g. [1] or [2][3].
3. If passages conflict, prefer the most recent effective date and say that an older version exists.
4. Quote exact IDs, numbers, amounts and dates exactly as written.
5. If the context does not contain the answer, reply exactly:
   "I couldn't find this in the knowledge base." and suggest which team might know.
6. Never claim a value "remained", "stayed the same" or is "unchanged" unless BOTH the old and
   the new value appear in the context. If one version's value is missing, say it is not in the
   retrieved passages.
7. If the question is about a specific company, customer, product or person that the passages do
   not mention, do NOT answer with another company's information; follow rule 5 instead.
Keep answers concise (max ~150 words) and use bullet points for steps."""

NO_ANSWER = "I couldn't find this in the knowledge base."


def build_context(docs: list[dict]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        m = d["metadata"]
        head = f"[{i}] {cite_label(m)} | dept={m.get('department','?')} | type={m.get('doc_type','?')}"
        parts.append(f"{head}\n{d['page_content']}")
    return "\n\n".join(parts)


@traceable(name="generate_answer", run_type="chain")
def generate_answer(question: str, docs: list[dict], history: list[dict] | None = None) -> dict:
    if not docs:
        return {"answer": NO_ANSWER, "citations": [], "abstained": True}
    messages = [{"role": "system", "content": ANSWER_SYSTEM}]
    for turn in (history or [])[-6:]:
        messages.append(turn)
    messages.append(
        {"role": "user", "content": f"Context passages:\n\n{build_context(docs)}\n\nQuestion: {question}"}
    )
    text = chat(messages)
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(docs)})
    citations = [
        {
            "n": n,
            "label": cite_label(docs[n - 1]["metadata"]),
            "source": docs[n - 1]["metadata"].get("source"),
            "chunk_id": docs[n - 1]["id"],
        }
        for n in used
    ]
    abstained = NO_ANSWER.lower()[:30] in text.lower()
    return {"answer": text, "citations": citations, "abstained": abstained}


def format_answer(result: dict) -> str:
    lines = [result["answer"].strip()]
    if result.get("citations"):
        lines.append("\nSources:")
        for c in result["citations"]:
            lines.append(f"  [{c['n']}] {c['label']}  ({c['source']})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- audit trail
def audit(event: dict) -> None:
    """Append-only JSONL audit trail: question, filters, retrieved chunk ids, answer,
    citations, latency and the LangSmith trace id to jump straight to the trace."""
    S.audit_log.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": datetime.now(timezone.utc).isoformat(), "trace_id": current_run_id(), **event}
    with S.audit_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = round((time.perf_counter() - self.t0) * 1000, 1)


def print_docs(docs: list[dict], title: str = "") -> None:
    if title:
        print(f"\n=== {title} ===")
    for i, d in enumerate(docs, start=1):
        m = d["metadata"]
        score = d.get("rerank_score", d.get("score"))
        snippet = d["page_content"].replace("\n", " ")[:110]
        print(f"{i:>2}. [{score}] {cite_label(m)} ({m.get('department')}/{m.get('doc_type')})\n    {snippet}…")


def load_all_chunks(collection, where: dict | None = None, include_embeddings: bool = False, page: int = 1000) -> dict:
    """Page through a Chroma collection (needed to build the BM25 index / RAPTOR tree)."""
    include = ["documents", "metadatas"] + (["embeddings"] if include_embeddings else [])
    out = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
    offset = 0
    while True:
        kwargs = dict(include=include, limit=page, offset=offset)
        if where:
            kwargs["where"] = where
        batch = collection.get(**kwargs)
        if not batch["ids"]:
            break
        out["ids"] += batch["ids"]
        out["documents"] += batch["documents"]
        out["metadatas"] += batch["metadatas"]
        if include_embeddings:
            out["embeddings"] += [list(e) for e in batch["embeddings"]]
        offset += len(batch["ids"])
        if len(batch["ids"]) < page:
            break
    return out
