"""
m2_raptor_rag.py — Deliverable 3: RAPTOR (Recursive Abstractive Processing for
Tree-Organized Retrieval, Sarthi et al. 2024) for long-corpus summary questions.

Flat chunk retrieval fails on questions like "Summarise our incident history this year"
or "What changed between the 2024 and 2026 leave policy?" because no single chunk holds
the answer. RAPTOR builds a tree on top of the leaf chunks:

    level 0  leaf chunks (from m2_chroma_ingestion)
    level 1  cluster leaves by embedding (GMM, soft membership) → LLM summary per cluster
    level 2  cluster the level-1 summaries → summaries of summaries
    ...      until ≤ 3 nodes remain or MAX_LEVELS reached

Summaries are embedded and stored in a separate Chroma collection with
`level`, `children`, `sources` metadata, built per TENANT (no cross-tenant summaries).

Retrieval:
    collapsed  (default) search leaves + every summary level together, best k by similarity
    tree       start at the top level, descend into the children of the best nodes

Usage
    python m2_raptor_rag.py build [--tenant northwind]
    python m2_raptor_rag.py show
    python m2_raptor_rag.py ask "Summarise all SEV1 and SEV2 incidents and their root causes"
    python m2_raptor_rag.py ask "What changed in the leave policy between 2024 and 2026?" --mode tree
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from common import (
    S, Timer, as_doc, audit, chat, embed, format_answer, generate_answer, get_chroma, get_collection,
    load_all_chunks, print_docs, rerank, sha1, traceable,
)

MAX_LEVELS = 3
MAX_CLUSTER_CHARS = 12000        # cap text sent to the summariser per cluster
SOFT_THRESHOLD = 0.25            # GMM probability above which a chunk also joins a cluster
REDUCED_DIM = 10                 # dimensionality reduction before clustering


# ---------------------------------------------------------------------- clustering
def reduce_dims(X: np.ndarray, dim: int = REDUCED_DIM) -> np.ndarray:
    """UMAP if installed (what the paper uses), else PCA. Both just make GMM tractable."""
    n = X.shape[0]
    dim = max(2, min(dim, n - 2))
    try:
        import umap

        return umap.UMAP(n_neighbors=max(2, min(15, n - 1)), n_components=dim, metric="cosine",
                         random_state=42).fit_transform(X)
    except Exception:
        from sklearn.decomposition import PCA

        return PCA(n_components=min(dim, n - 1, X.shape[1]), random_state=42).fit_transform(X)


def cluster(X: np.ndarray, max_clusters: int | None = None, seed: int = 42) -> list[list[int]]:
    """Gaussian Mixture, number of clusters chosen by BIC; soft assignment means one chunk
    can belong to several clusters (a policy paragraph about travel AND expenses)."""
    from sklearn.mixture import GaussianMixture

    n = X.shape[0]
    if n <= 3:
        return [list(range(n))]
    Z = reduce_dims(X)
    max_clusters = max_clusters or max(2, min(12, n // 3))
    best, best_bic = None, float("inf")
    for k in range(1, max_clusters + 1):
        gm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed, reg_covar=1e-4).fit(Z)
        bic = gm.bic(Z)
        if bic < best_bic:
            best, best_bic = gm, bic
    probs = best.predict_proba(Z)
    clusters: list[list[int]] = [[] for _ in range(best.n_components)]
    for i, p in enumerate(probs):
        members = [c for c in np.where(p >= SOFT_THRESHOLD)[0]] or [int(p.argmax())]
        for c in members:
            clusters[c].append(i)
    return [c for c in clusters if c]


# ---------------------------------------------------------------------- summarisation
@traceable(name="raptor_summarise", run_type="llm")
def summarise(texts: list[str], level: int) -> str:
    joined, total = [], 0
    for t in texts:
        if total + len(t) > MAX_CLUSTER_CHARS:
            break
        joined.append(t)
        total += len(t)
    prompt = (
        "You are building a hierarchical summary of a company knowledge base. Write a dense, factual "
        "summary (150-250 words) of the passages below. Preserve every policy/ticket/contract ID, number, "
        "amount, date, version and owner. Name the source documents. If passages conflict or represent "
        "different versions, state both with their dates.\n\n" + "\n\n---\n\n".join(joined)
    )
    return chat([{"role": "user", "content": prompt}], temperature=0.0)


# ---------------------------------------------------------------------- build tree
@traceable(name="raptor_build", run_type="chain")
def build(tenant: str) -> None:
    leaves_col = get_collection()
    leaves = load_all_chunks(leaves_col, where={"tenant": tenant}, include_embeddings=True)
    if not leaves["ids"]:
        raise SystemExit("No leaf chunks — run m2_chroma_ingestion.py first.")
    tree_col = get_collection(S.raptor_collection)
    old = tree_col.get(where={"tenant": tenant})
    if old["ids"]:
        tree_col.delete(ids=old["ids"])

    # current layer: ids, texts, embeddings, metadata
    ids, texts, metas = leaves["ids"], leaves["documents"], leaves["metadatas"]
    X = np.array(leaves["embeddings"], dtype=float)
    print(f"level 0: {len(ids)} leaf chunks")

    for level in range(1, MAX_LEVELS + 1):
        if len(ids) <= 3:
            break
        groups = cluster(X)
        if len(groups) == 1 and level > 1:
            groups = [list(range(len(ids)))]
        new_ids, new_texts, new_metas = [], [], []
        for g in groups:
            child_ids = [ids[i] for i in g]
            summary = summarise([texts[i] for i in g], level)
            srcs = sorted({s for i in g for s in str(metas[i].get("sources", metas[i].get("source", ""))).split(" | ") if s})
            depts = sorted({metas[i].get("department", "") for i in g})
            dates = [metas[i].get("date_int", 0) for i in g]
            new_ids.append(f"raptor-L{level}-" + sha1(*child_ids)[:16])
            new_texts.append(f"[Summary L{level} of {len(g)} passages]\n{summary}")
            new_metas.append({
                "tenant": tenant, "level": level, "node_type": "summary",
                "department": depts[0] if len(depts) == 1 else "mixed",
                "doc_type": "summary", "title": f"RAPTOR summary L{level}",
                "children": json.dumps(child_ids), "sources": " | ".join(srcs)[:2000],
                "source": " | ".join(srcs)[:500], "date_int": max(dates), "date": str(max(dates)),
            })
        vecs = embed(new_texts)
        tree_col.upsert(ids=new_ids, documents=new_texts, metadatas=new_metas, embeddings=vecs)
        print(f"level {level}: {len(groups)} clusters → {len(new_ids)} summaries")
        if len(new_ids) == len(ids):  # no compression → stop
            break
        ids, texts, metas, X = new_ids, new_texts, new_metas, np.array(vecs, dtype=float)
    print(f"RAPTOR tree stored in '{S.raptor_collection}' ({tree_col.count()} nodes total)")


# ---------------------------------------------------------------------- retrieval
def _query(col, vec, k, where):
    if col.count() == 0:
        return []
    res = col.query(query_embeddings=[vec], n_results=min(k, col.count()), where=where,
                    include=["documents", "metadatas", "distances"])
    return [as_doc(i, d, m, 1 - dist) for i, d, m, dist in
            zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]


@traceable(name="raptor_collapsed_retrieve", run_type="retriever")
def collapsed_retrieve(query: str, tenant: str, k: int = 8) -> list[dict]:
    """Collapsed tree: leaves and all summary levels compete in one similarity ranking.
    The paper found this beats layer-by-layer traversal in most settings."""
    vec = embed([query])[0]
    where = {"tenant": tenant}
    pool = _query(get_collection(), vec, k * 3, where) + _query(get_collection(S.raptor_collection), vec, k * 3, where)
    pool.sort(key=lambda d: d["score"], reverse=True)
    return rerank(query, pool[: k * 3], top_n=k)


@traceable(name="raptor_tree_retrieve", run_type="retriever")
def tree_retrieve(query: str, tenant: str, beam: int = 2, k: int = 8) -> list[dict]:
    """Top-down traversal: best `beam` nodes at the top level, then best children, down to leaves."""
    vec = np.array(embed([query])[0])
    tree = get_collection(S.raptor_collection)
    leaves = get_collection()
    top_level = max([m["level"] for m in tree.get(where={"tenant": tenant}, include=["metadatas"])["metadatas"]] or [0])
    if top_level == 0:
        return collapsed_retrieve(query, tenant, k)
    selected = _query(tree, vec.tolist(), beam, {"$and": [{"tenant": tenant}, {"level": top_level}]})
    collected = list(selected)
    for _ in range(top_level):
        child_ids = [c for d in selected for c in json.loads(d["metadata"].get("children", "[]"))]
        if not child_ids:
            break
        src = tree if child_ids[0].startswith("raptor-") else leaves
        got = src.get(ids=child_ids, include=["documents", "metadatas", "embeddings"])
        scored = []
        for i, doc, m, e in zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"]):
            e = np.array(e)
            sim = float(e @ vec / (np.linalg.norm(e) * np.linalg.norm(vec) + 1e-9))
            scored.append(as_doc(i, doc, m, sim))
        selected = sorted(scored, key=lambda d: d["score"], reverse=True)[: beam * 2]
        collected += selected
    return rerank(query, collected, top_n=k)


@traceable(name="raptor_pipeline", run_type="chain")
def ask(question: str, tenant: str, mode: str = "collapsed") -> dict:
    with Timer() as t:
        docs = tree_retrieve(question, tenant) if mode == "tree" else collapsed_retrieve(question, tenant)
        result = generate_answer(question, docs)
    result.update(question=question, mode=f"raptor_{mode}", retrieved=[
        {"id": d["id"], "source": d["metadata"].get("source"), "level": d["metadata"].get("level", 0),
         "score": d.get("rerank_score", d.get("score"))} for d in docs], total_ms=t.ms)
    audit({k: result.get(k) for k in ("question", "mode", "retrieved", "answer", "citations", "total_ms")})
    return result, docs


def show(tenant: str) -> None:
    tree = get_collection(S.raptor_collection).get(where={"tenant": tenant}, include=["documents", "metadatas"])
    for i, d, m in sorted(zip(tree["ids"], tree["documents"], tree["metadatas"]), key=lambda x: -x[2]["level"]):
        print(f"\n[L{m['level']}] {i}  children={len(json.loads(m['children']))}  dept={m['department']}")
        print("   sources:", m["sources"][:160])
        print("  ", d.replace("\n", " ")[:300], "…")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "show", "ask"])
    ap.add_argument("question", nargs="?")
    ap.add_argument("--tenant", default=S.default_tenant)
    ap.add_argument("--mode", choices=["collapsed", "tree"], default="collapsed")
    a = ap.parse_args()
    if a.command == "build":
        build(a.tenant)
    elif a.command == "show":
        show(a.tenant)
    else:
        if not a.question:
            raise SystemExit("ask needs a question")
        res, docs = ask(a.question, a.tenant, a.mode)
        print_docs(docs, f"RAPTOR {a.mode} retrieval")
        print("\n" + format_answer(res) + f"\n\n({res['total_ms']} ms)")
