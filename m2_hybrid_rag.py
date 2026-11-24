"""
m2_hybrid_rag.py — Deliverable 2: hybrid retrieval (BM25 + semantic) fused with RRF,
then two-stage reranking, then grounded answer with citations.

    query ─┬─ bi-encoder (OpenAI embeddings, Chroma)  top-50 ─┐
           └─ BM25 keyword search                     top-50 ─┴─ RRF fusion → top-50
                                                                    → cross-encoder rerank → top-5
                                                                    → LLM answer with [n] citations

Retrieval modes (compare them with --compare):
    dense          semantic only            (misses exact IDs like "INC-2041")
    bm25           keyword only             (misses paraphrases like "time off" vs "PTO")
    hybrid         BM25 + dense with RRF
    hybrid_rerank  hybrid → cross-encoder top-5        ← default / production
    multi_query    LLM rewrites the question 3 ways, hybrid for each, RRF, rerank
    hyde           LLM writes a hypothetical answer, embed THAT for dense search, rerank

Usage
    python m2_hybrid_rag.py "How many PTO days do I get?"
    python m2_hybrid_rag.py "What happened in INC-2057?" --compare
    python m2_hybrid_rag.py "how do I fail over the orders database" --mode multi_query
    python m2_hybrid_rag.py --chat                                   # multi-turn with memory
"""
from __future__ import annotations

import argparse
import json

from common import (
    S, Timer, as_doc, audit, chat, embed, format_answer, generate_answer, get_collection,
    load_all_chunks, make_bm25, matches_where, print_docs, rerank, rrf_fuse, tokenize, traceable,
    NO_ANSWER,
)

MODES = ["dense", "bm25", "hybrid", "hybrid_rerank", "multi_query", "hyde"]


class HybridRetriever:
    def __init__(self, collection_name: str | None = None):
        self.col = get_collection(collection_name)
        self._build_bm25()

    # ------------------------------------------------------------------ indexes
    def _build_bm25(self) -> None:
        """BM25 lives in memory, built from the same chunks stored in Chroma, so both
        retrievers always see identical text + metadata (rebuild after re-ingestion)."""
        data = load_all_chunks(self.col)
        self.ids = data["ids"]
        self.texts = data["documents"]
        self.metas = data["metadatas"]
        self.by_id = {i: (t, m) for i, t, m in zip(self.ids, self.texts, self.metas)}
        self.bm25 = make_bm25([tokenize(t) for t in self.texts]) if self.texts else None
        if not self.ids:
            raise SystemExit("Collection is empty — run `python m2_chroma_ingestion.py` first.")

    def doc(self, doc_id: str, score: float | None = None, **extra) -> dict:
        t, m = self.by_id[doc_id]
        return as_doc(doc_id, t, m, score, **extra)

    # ------------------------------------------------------------------ stage 1 retrievers
    @traceable(name="dense_search", run_type="retriever")
    def dense(self, query: str, k: int = S.dense_k, where: dict | None = None, query_vector=None) -> list[dict]:
        vec = query_vector or embed([query])[0]
        kwargs = {"query_embeddings": [vec], "n_results": min(k, len(self.ids)), "include": ["distances"]}
        if where:
            kwargs["where"] = where
        res = self.col.query(**kwargs)
        return [self.doc(i, 1 - d, retriever="dense") for i, d in zip(res["ids"][0], res["distances"][0])]

    @traceable(name="bm25_search", run_type="retriever")
    def sparse(self, query: str, k: int = S.bm25_k, where: dict | None = None) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked:
            if scores[i] <= 0:
                break
            if matches_where(self.metas[i], where):  # same filter as the vector side
                out.append(self.doc(self.ids[i], scores[i], retriever="bm25"))
            if len(out) >= k:
                break
        return out

    @traceable(name="rrf_fusion", run_type="retriever")
    def hybrid(self, query: str, k: int = S.dense_k, where: dict | None = None, query_vector=None,
               weights: tuple[float, float] = (1.0, 1.0)) -> list[dict]:
        dense = self.dense(query, k, where, query_vector)
        sparse = self.sparse(query, k, where)
        fused = rrf_fuse([[d["id"] for d in dense], [d["id"] for d in sparse]], k=S.rrf_k, weights=list(weights))
        drank = {d["id"]: r for r, d in enumerate(dense, 1)}
        srank = {d["id"]: r for r, d in enumerate(sparse, 1)}
        return [self.doc(i, s, retriever="rrf", dense_rank=drank.get(i), bm25_rank=srank.get(i)) for i, s in fused[:k]]

    # ------------------------------------------------------------------ advanced query modes
    @traceable(name="multi_query_rewrite", run_type="llm")
    def rewrite_queries(self, query: str, n: int = 3) -> list[str]:
        prompt = (
            f"Rewrite the employee question below into {n} different search queries for an internal "
            "knowledge base (policies, runbooks, contracts, tickets). Use synonyms and the formal terms "
            "a policy document would use; keep any IDs exactly. Return JSON: {\"queries\": [...]}\n\n"
            f"Question: {query}"
        )
        try:
            qs = json.loads(chat([{"role": "user", "content": prompt}], temperature=0.3, json_mode=True))["queries"]
        except Exception:
            qs = []
        return [query] + [q for q in qs if isinstance(q, str)][:n]

    @traceable(name="multi_query_retrieve", run_type="retriever")
    def multi_query(self, query: str, k: int = S.dense_k, where: dict | None = None) -> list[dict]:
        queries = self.rewrite_queries(query)
        vectors = embed(queries)
        lists = [self.hybrid(q, k, where, v) for q, v in zip(queries, vectors)]
        fused = rrf_fuse([[d["id"] for d in lst] for lst in lists], k=S.rrf_k)
        return [self.doc(i, s, retriever="multi_query") for i, s in fused[:k]]

    @traceable(name="hyde_retrieve", run_type="retriever")
    def hyde(self, query: str, k: int = S.dense_k, where: dict | None = None) -> list[dict]:
        """HyDE: answers look more like documents than questions do, so embed a
        hypothetical answer. BM25 still uses the real question (keeps exact IDs)."""
        fake = chat([{"role": "user", "content":
                      "Write a short passage (80-120 words) from an internal company policy, runbook or "
                      f"contract that would answer this question. Invent plausible details.\n\nQuestion: {query}"}],
                    temperature=0.2)
        return self.hybrid(query, k, where, query_vector=embed([fake])[0])

    # ------------------------------------------------------------------ public API
    @traceable(name="retrieve", run_type="retriever")
    def retrieve(self, query: str, mode: str = "hybrid_rerank", where: dict | None = None,
                 top_n: int = S.final_k) -> list[dict]:
        if mode == "dense":
            return self.dense(query, top_n, where)
        if mode == "bm25":
            return self.sparse(query, top_n, where)
        if mode == "hybrid":
            return self.hybrid(query, S.dense_k, where)[:top_n]
        if mode == "hybrid_rerank":
            return rerank(query, self.hybrid(query, S.dense_k, where), top_n)
        if mode == "multi_query":
            return rerank(query, self.multi_query(query, S.dense_k, where), top_n)
        if mode == "hyde":
            return rerank(query, self.hyde(query, S.dense_k, where), top_n)
        raise ValueError(f"unknown mode {mode}; choose from {MODES}")


# ---------------------------------------------------------------------- end-to-end pipeline
@traceable(name="condense_question", run_type="llm")
def condense_question(question: str, history: list[dict]) -> str:
    """Turn a follow-up ("and for contractors?") into a standalone search query."""
    if not history:
        return question
    convo = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history[-6:])
    return chat([{"role": "user", "content":
                  f"Conversation so far:\n{convo}\n\nRewrite the follow-up as a standalone question. "
                  f"Return only the question.\nFollow-up: {question}"}]).strip()


@traceable(name="kb_pipeline", run_type="chain")
def ask(question: str, retriever: HybridRetriever, mode: str = "hybrid_rerank", where: dict | None = None,
        history: list[dict] | None = None, post_filter=None) -> dict:
    with Timer() as t_total:
        standalone = condense_question(question, history or [])
        with Timer() as t_ret:
            docs = retriever.retrieve(standalone, mode=mode, where=where)
            if post_filter:  # e.g. latest_only() from m2_metadata_filtering
                docs = post_filter(docs)
        # abstain when even the best reranked passage is weak → no hallucinated answer
        top = docs[0].get("rerank_score") if docs else None
        if not docs or (top is not None and top < S.min_rerank_score):
            result = {"answer": NO_ANSWER, "citations": [], "abstained": True}
        else:
            result = generate_answer(standalone, docs, history)
    result.update(
        question=question, standalone_question=standalone, mode=mode, filters=where,
        retrieved=[{"id": d["id"], "source": d["metadata"].get("source"), "ticket_id": d["metadata"].get("ticket_id"),
                    "score": d.get("rerank_score", d.get("score"))} for d in docs],
        retrieval_ms=t_ret.ms, total_ms=t_total.ms,
    )
    audit({k: result[k] for k in ("question", "standalone_question", "mode", "filters", "retrieved",
                                  "answer", "citations", "abstained", "retrieval_ms", "total_ms")})
    return result


def compare(question: str, retriever: HybridRetriever, where=None, modes=None) -> None:
    for m in modes or MODES:
        with Timer() as t:
            docs = retriever.retrieve(question, mode=m, where=where)
        print_docs(docs, f"{m}  ({t.ms} ms)")


def chat_loop(retriever: HybridRetriever, mode: str, where=None) -> None:
    history: list[dict] = []
    print(f"Knowledge-base chat (mode={mode}). Type 'exit' to quit, ':mode <name>' to switch.")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit"):
            break
        if q.startswith(":mode"):
            mode = q.split()[-1]
            print(f"mode → {mode}")
            continue
        res = ask(q, retriever, mode, where, history)
        print("\nBot:", format_answer(res), f"\n({res['total_ms']} ms)")
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": res["answer"]}]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?")
    ap.add_argument("--mode", default="hybrid_rerank", choices=MODES)
    ap.add_argument("--tenant", default=S.default_tenant, help="tenant isolation (always applied)")
    ap.add_argument("--compare", action="store_true", help="show top-5 from every retrieval mode")
    ap.add_argument("--chat", action="store_true")
    a = ap.parse_args()

    r = HybridRetriever()
    where = {"tenant": a.tenant}
    if a.chat:
        chat_loop(r, a.mode, where)
    elif a.question and a.compare:
        compare(a.question, r, where)
    elif a.question:
        res = ask(a.question, r, a.mode, where)
        print_docs([r.doc(x["id"], x["score"]) for x in res["retrieved"]], f"retrieved ({a.mode})")
        print("\n" + format_answer(res) + f"\n\n(retrieval {res['retrieval_ms']} ms, total {res['total_ms']} ms)")
    else:
        ap.print_help()
