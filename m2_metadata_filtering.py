"""
m2_metadata_filtering.py — Deliverable 4: precision retrieval with metadata filters.

Every chunk carries: tenant, department, doc_type, date (+ date_int), access, source, page/section,
ticket_id. Filters are applied INSIDE both retrievers (Chroma `where` for vectors, the same
predicate in Python for BM25) — i.e. pre-filtering, not post-filtering, so the top-50 candidate
pool is spent only on eligible documents.

Three ways to get filters:
  1. explicit     --department hr --doc-type policy --since 2025-01-01
  2. self-query   --auto   (LLM extracts filters from the question, validated against the
                            values that really exist in the collection)
  3. guardrails   tenant is ALWAYS enforced (multi-tenant isolation); --latest-only keeps only
                  the newest version of each policy title family

Usage
    python m2_metadata_filtering.py "How many PTO days do we get?" --department hr --since 2025-01-01
    python m2_metadata_filtering.py "What engineering tickets were SEV1 this year?" --auto
    python m2_metadata_filtering.py "How many leave days do employees get?" --compare --department hr --since 2026-01-01
    python m2_metadata_filtering.py --benchmark       # precision@5 with vs. without filters on the golden set
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from common import ROOT, S, chat, date_to_int, format_answer, print_docs, traceable
from m2_hybrid_rag import HybridRetriever, ask


# ---------------------------------------------------------------------- filter builder
def build_where(
    tenant: str,
    departments: list[str] | None = None,
    doc_types: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    access: list[str] | None = None,
    ticket_id: str | None = None,
) -> dict:
    """Compose a Chroma `where` clause. Chroma needs `$and` when there is more than one
    condition, and numeric values for range operators — hence `date_int`."""
    conds: list[dict] = [{"tenant": tenant}]  # tenant isolation is non-negotiable
    if departments:
        conds.append({"department": {"$in": departments}} if len(departments) > 1 else {"department": departments[0]})
    if doc_types:
        conds.append({"doc_type": {"$in": doc_types}} if len(doc_types) > 1 else {"doc_type": doc_types[0]})
    if date_from:
        conds.append({"date_int": {"$gte": date_to_int(date_from)}})
    if date_to:
        conds.append({"date_int": {"$lte": date_to_int(date_to)}})
    if access:
        conds.append({"access": {"$in": access}})
    if ticket_id:
        conds.append({"ticket_id": ticket_id})
    return conds[0] if len(conds) == 1 else {"$and": conds}


def facet_values(retriever: HybridRetriever, tenant: str) -> dict[str, list[str]]:
    """Distinct values present for this tenant — used to validate LLM-extracted filters."""
    vals: dict[str, set] = {"department": set(), "doc_type": set()}
    for m in retriever.metas:
        if m.get("tenant") == tenant:
            for k in vals:
                if m.get(k):
                    vals[k].add(m[k])
    return {k: sorted(v) for k, v in vals.items()}


# ---------------------------------------------------------------------- self-query
@traceable(name="self_query_extract", run_type="llm")
def extract_filters(question: str, facets: dict[str, list[str]], today: str | None = None) -> dict:
    """LLM 'self-query': pull structured filters out of natural language.
    Anything not in the allowed lists is dropped — the LLM cannot invent a department."""
    today = today or date.today().isoformat()
    prompt = f"""Extract search filters from an employee's question to a company knowledge base.
Today is {today}.
Allowed departments: {facets['department']}
Allowed doc_types: {facets['doc_type']}  (policy, runbook, contract, ticket, catalog, ...)
Return JSON with keys:
  "departments": list (subset of allowed, empty if not clearly implied),
  "doc_types": list (subset of allowed, empty if not clearly implied),
  "date_from": "YYYY-MM-DD" or null,  "date_to": "YYYY-MM-DD" or null,
  "ticket_id": exact ticket ID like INC-1234 if mentioned else null,
  "search_query": the question rewritten WITHOUT the filter words.
Only set a filter when the question clearly asks for it ("this year", "HR policy", "runbook", "tickets").
Question: {question}"""
    try:
        raw = json.loads(chat([{"role": "user", "content": prompt}], json_mode=True))
    except Exception:
        return {"departments": [], "doc_types": [], "date_from": None, "date_to": None, "ticket_id": None,
                "search_query": question}
    raw["departments"] = [d for d in raw.get("departments") or [] if d in facets["department"]]
    raw["doc_types"] = [d for d in raw.get("doc_types") or [] if d in facets["doc_type"]]
    raw["search_query"] = raw.get("search_query") or question
    return raw


def latest_only(docs: list[dict]) -> list[dict]:
    """Keep only the newest document within each policy family (same policy ID / title stem),
    so superseded versions (e.g. Leave Policy v3.1 2024) never compete with the current one."""
    import re

    def family(m):
        # "Leave Policy v4.0 (2026)" and "Leave Policy v3.1 (2024)" → "leave policy"
        if m.get("doc_type") == "ticket":
            return m.get("ticket_id") or m.get("source", "")
        t = re.sub(r"\bv?\d+(\.\d+)*\b", " ", m.get("title", "").lower())
        return re.sub(r"[^a-z]+", " ", t).strip()

    newest: dict[str, int] = {}
    for d in docs:
        f = family(d["metadata"])
        newest[f] = max(newest.get(f, 0), d["metadata"].get("date_int", 0))
    return [d for d in docs if d["metadata"].get("date_int", 0) == newest[family(d["metadata"])]]


# ---------------------------------------------------------------------- benchmark
def precision_at_k(docs: list[dict], expected: list[str], k: int = 5) -> float:
    if not docs:
        return 0.0
    hits = sum(1 for d in docs[:k] if is_relevant(d["metadata"], expected))
    return hits / min(k, len(docs))


def is_relevant(meta: dict, expected: list[str]) -> bool:
    src = Path(str(meta.get("source", ""))).name
    return src in expected or (meta.get("ticket_id") in expected if meta.get("ticket_id") else False)


def benchmark(retriever: HybridRetriever, tenant: str) -> None:
    rows = [json.loads(l) for l in (ROOT / "eval" / "golden_set.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("filters") and r.get("expected_sources")]
    print(f"{'question':<58} {'P@5 none':>9} {'P@5 filt':>9}")
    tot_a = tot_b = 0.0
    for r in rows:
        f = r["filters"]
        where_none = {"tenant": r.get("tenant", tenant)}
        where_filt = build_where(r.get("tenant", tenant), f.get("departments"), f.get("doc_types"),
                                 f.get("date_from"), f.get("date_to"))
        a = precision_at_k(retriever.retrieve(r["question"], "hybrid", where_none), r["expected_sources"])
        b = precision_at_k(retriever.retrieve(r["question"], "hybrid", where_filt), r["expected_sources"])
        tot_a, tot_b = tot_a + a, tot_b + b
        print(f"{r['question'][:57]:<58} {a:>9.2f} {b:>9.2f}")
    n = max(1, len(rows))
    print(f"{'MEAN':<58} {tot_a/n:>9.2f} {tot_b/n:>9.2f}")


# ---------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?")
    ap.add_argument("--tenant", default=S.default_tenant)
    ap.add_argument("--department", action="append", help="repeatable: --department hr --department finance")
    ap.add_argument("--doc-type", action="append")
    ap.add_argument("--since", help="YYYY-MM-DD")
    ap.add_argument("--until", help="YYYY-MM-DD")
    ap.add_argument("--access", action="append", help="caller's access groups, e.g. --access all --access legal")
    ap.add_argument("--auto", action="store_true", help="LLM self-query filter extraction")
    ap.add_argument("--latest-only", action="store_true", help="drop superseded policy versions")
    ap.add_argument("--mode", default="hybrid_rerank")
    ap.add_argument("--compare", action="store_true", help="show unfiltered vs filtered results")
    ap.add_argument("--benchmark", action="store_true")
    a = ap.parse_args()

    r = HybridRetriever()
    if a.benchmark:
        benchmark(r, a.tenant)
        raise SystemExit
    if not a.question:
        ap.print_help()
        raise SystemExit

    query = a.question
    if a.auto:
        f = extract_filters(a.question, facet_values(r, a.tenant))
        print("self-query filters:", json.dumps(f, indent=2))
        where = build_where(a.tenant, f["departments"], f["doc_types"], f.get("date_from"), f.get("date_to"),
                            a.access, f.get("ticket_id"))
        query = f["search_query"]
    else:
        where = build_where(a.tenant, a.department, a.doc_type, a.since, a.until, a.access)
    print("where =", json.dumps(where))

    if a.compare:
        print_docs(r.retrieve(query, a.mode, {"tenant": a.tenant}), "UNFILTERED (tenant only)")
        print_docs(r.retrieve(query, a.mode, where), "FILTERED")
    else:
        res = ask(query, r, a.mode, where, post_filter=latest_only if a.latest_only else None)
        print_docs([r.doc(x["id"], x["score"]) for x in res["retrieved"]], "retrieved")
        print("\n" + format_answer(res))
