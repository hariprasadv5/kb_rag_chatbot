"""Offline tests — no API keys, no Chroma needed.  Run:  pytest -q"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import SimpleBM25, date_to_int, matches_where, rrf_fuse, tokenize  # noqa: E402
from m2_chroma_ingestion import (  # noqa: E402
    DATA, LOADERS, RAW, build_records, iter_sources, recursive_split,
)
from m2_eval_langsmith import keys_of, score  # noqa: E402
from m2_metadata_filtering import build_where, latest_only  # noqa: E402
from m2_raptor_rag import cluster  # noqa: E402


# ---------------------------------------------------------------- tokenizer / BM25
def test_tokenizer_keeps_ids_and_parts():
    toks = tokenize("What happened in INC-2057 on orders-db?")
    assert "inc-2057" in toks and "2057" in toks and "orders-db" in toks
    assert "what" not in toks


def test_bm25_exact_id_wins():
    corpus = ["Ticket INC-2041 connection pool exhausted", "Ticket INC-2057 primary database down",
              "Leave policy paid time off 22 days"]
    bm = SimpleBM25([tokenize(c) for c in corpus])
    s = bm.get_scores(tokenize("INC-2057"))
    assert int(np.argmax(s)) == 1


# ---------------------------------------------------------------- RRF
def test_rrf_rewards_agreement():
    fused = rrf_fuse([["a", "b", "c"], ["b", "d", "a"]], k=60)
    assert fused[0][0] in ("a", "b")
    assert {d for d, _ in fused} == {"a", "b", "c", "d"}
    assert abs(dict(fused)["b"] - (1 / 62 + 1 / 61)) < 1e-12


# ---------------------------------------------------------------- filters
def test_build_where_and_python_evaluator_agree():
    w = build_where("northwind", ["hr"], ["policy"], "2025-01-01", None)
    assert w == {"$and": [{"tenant": "northwind"}, {"department": "hr"}, {"doc_type": "policy"},
                          {"date_int": {"$gte": 20250101}}]}
    new = {"tenant": "northwind", "department": "hr", "doc_type": "policy", "date_int": 20260301}
    old = dict(new, date_int=20240115)
    other_tenant = dict(new, tenant="contoso")
    assert matches_where(new, w) and not matches_where(old, w) and not matches_where(other_tenant, w)


def test_where_in_operator():
    w = build_where("northwind", ["hr", "finance"])
    assert matches_where({"tenant": "northwind", "department": "finance"}, w)
    assert not matches_where({"tenant": "northwind", "department": "legal"}, w)


def test_date_to_int():
    assert date_to_int("2026-03-01") == 20260301


def test_latest_only_drops_superseded_version():
    docs = [{"metadata": {"title": "Leave Policy v4.0 (2026)", "date_int": 20260301, "doc_type": "policy"}},
            {"metadata": {"title": "Leave Policy v3.1 (2024)", "date_int": 20240115, "doc_type": "policy"}},
            {"metadata": {"title": "Travel & Expense Policy", "date_int": 20250701, "doc_type": "policy"}}]
    kept = [d["metadata"]["title"] for d in latest_only(docs)]
    assert kept == ["Leave Policy v4.0 (2026)", "Travel & Expense Policy"]


# ---------------------------------------------------------------- chunking + loaders
def test_recursive_split_respects_size():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(20))
    chunks = recursive_split(text, 500, 80, ["\n\n", "\n", ". ", " "])
    assert len(chunks) > 5 and all(len(c) <= 500 for c in chunks)


def test_all_sample_documents_load_and_chunk():
    import yaml

    if not (DATA / "manifest.yaml").exists():
        import make_sample_corpus

        make_sample_corpus.main()
    manifest = yaml.safe_load((DATA / "manifest.yaml").read_text())
    total = 0
    for source, loader, meta in iter_sources(manifest):
        recs = build_records(source, loader(), meta)
        assert recs, source
        for r in recs:
            md = r["metadata"]
            assert md["tenant"] and md["department"] and md["doc_type"] and md["date_int"] > 20000000
            assert all(isinstance(v, (str, int, float, bool)) for v in md.values())  # Chroma-safe
        total += len(recs)
    assert total > 20


def test_tickets_are_atomic_with_ids():
    sections = LOADERS[".csv"](RAW / "northwind/support/tickets.csv")
    recs = build_records("tickets.csv", sections, {"title": "Support Tickets", "tenant": "northwind",
                                                     "department": "support", "doc_type": "ticket", "date": "2026-07-02"})
    assert len(recs) == len(sections)
    assert {r["metadata"]["ticket_id"] for r in recs} >= {"INC-2041", "INC-2057"}
    assert recs[0]["metadata"]["date_int"] == 20251203  # ticket's own date, not the file's


def test_contract_split_on_clauses():
    from m2_chroma_ingestion import load_docx

    secs = load_docx(RAW / "northwind/legal/msa_acme_retail.docx")
    assert any("14.2" in s["text"] and s["section"].startswith("Section 14") for s in secs)


def test_runbook_steps_numbered():
    from m2_chroma_ingestion import load_docx

    secs = load_docx(RAW / "northwind/engineering/runbook_db_failover.docx")
    proc = next(s for s in secs if s["section"] == "Failover procedure")
    assert "Step 1." in proc["text"] and "Step 5." in proc["text"]


# ---------------------------------------------------------------- RAPTOR clustering
def test_gmm_clusters_separate_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.05, (12, 32)) + 1
    b = rng.normal(0, 0.05, (12, 32)) - 1
    groups = cluster(np.vstack([a, b]))
    assert len(groups) >= 2
    assert all(len(g) > 0 for g in groups)


# ---------------------------------------------------------------- metrics
def test_metrics():
    out = {"retrieved": [{"source": "data/raw/x/tickets.csv", "ticket_id": "INC-2057"},
                         {"source": "data/raw/x/tickets.csv", "ticket_id": "INC-2041"},
                         {"source": "data/raw/a/leave_policy_2026.pdf"}],
           "abstained": False, "citations": [{"n": 1}], "total_ms": 900, "retrieval_ms": 300, "answer": "x"}
    m = score(out, ["INC-2057"])
    assert m["precision@5"] == 1 / 3 and m["recall@5"] == 1 and m["mrr"] == 1 and m["abstention_correct"] == 1
    neg = score(dict(out, abstained=True, citations=[]), [])
    assert neg["abstention_correct"] == 1 and neg["recall@5"] is None
    assert keys_of({"source": "a/x.pdf | b/y.docx"}) == {"x.pdf", "y.docx"}
