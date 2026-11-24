"""End-to-end pipeline test with fakes (in-memory Chroma, hashing embeddings, stub LLM).
Exercises ingestion → hybrid/RRF → filters → RAPTOR → eval loop without any API key."""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402


# ------------------------------------------------------------------ fakes
def fake_embed(texts, batch_size=96):
    out = []
    for t in texts:
        v = np.zeros(256)
        for tok in common.tokenize(t):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256] += 1
        n = np.linalg.norm(v)
        out.append((v / n if n else v).tolist())
    return out


def fake_chat(messages, temperature=0.0, json_mode=False, model=None):
    content = messages[-1]["content"]
    if json_mode and "queries" in content:
        return json.dumps({"queries": ["paid time off days", "annual leave entitlement"]})
    if json_mode and "search filters" in content:
        return json.dumps({"departments": ["hr", "made-up"], "doc_types": ["policy"], "date_from": "2026-01-01",
                           "date_to": None, "ticket_id": None, "search_query": "PTO days"})
    if json_mode:
        return json.dumps({"score": 1})
    if "Context passages" in content:
        return "Employees get 22 days of PTO [1]."
    return "summary text " + content[:200]


class FakeCollection:
    def __init__(self):
        self.rows = {}

    def count(self):
        return len(self.rows)

    def upsert(self, ids, documents, metadatas, embeddings):
        for i, d, m, e in zip(ids, documents, metadatas, embeddings):
            self.rows[i] = (d, m, np.array(e))

    def delete(self, ids):
        for i in ids:
            self.rows.pop(i, None)

    def _filter(self, where):
        return [(i, r) for i, r in self.rows.items() if common.matches_where(r[1], where)]

    def get(self, ids=None, where=None, include=(), limit=None, offset=0):
        items = [(i, self.rows[i]) for i in ids if i in self.rows] if ids else self._filter(where)
        items = items[offset: offset + limit] if limit else items[offset:]
        return {"ids": [i for i, _ in items], "documents": [r[0] for _, r in items],
                "metadatas": [r[1] for _, r in items], "embeddings": [r[2] for _, r in items]}

    def query(self, query_embeddings, n_results, where=None, include=()):
        q = np.array(query_embeddings[0])
        items = self._filter(where)
        scored = sorted(items, key=lambda x: -float(x[1][2] @ q))[:n_results]
        return {"ids": [[i for i, _ in scored]], "documents": [[r[0] for _, r in scored]],
                "metadatas": [[r[1] for _, r in scored]], "distances": [[1 - float(r[2] @ q) for _, r in scored]]}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    cols = {}

    def get_collection(name=None):
        return cols.setdefault(name or common.S.collection, FakeCollection())

    import m2_chroma_ingestion as ing
    import m2_hybrid_rag as hy
    import m2_metadata_filtering as mf
    import m2_raptor_rag as rp
    import m2_eval_langsmith as ev

    mp = pytest.MonkeyPatch()
    common.S.audit_log = tmp_path_factory.mktemp("logs") / "audit.jsonl"
    common.S.reranker = "none"
    for mod in (common, ing, hy, mf, rp, ev):
        for name, fn in (("embed", fake_embed), ("chat", fake_chat), ("get_collection", get_collection)):
            if hasattr(mod, name):
                mp.setattr(mod, name, fn)
    ing.run()
    yield {"hy": hy, "mf": mf, "rp": rp, "ev": ev, "cols": cols, "ing": ing}
    mp.undo()


def test_ingestion_is_idempotent(env):
    n = env["cols"][common.S.collection].count()
    res = env["ing"].run()
    assert all(r["status"] == "unchanged" for r in res)
    assert env["cols"][common.S.collection].count() == n


def test_hybrid_modes_and_exact_id(env):
    r = env["hy"].HybridRetriever()
    where = {"tenant": "northwind"}
    for mode in env["hy"].MODES:
        docs = r.retrieve("What happened in INC-2057?", mode, where)
        assert docs, mode
    top = r.retrieve("What happened in INC-2057?", "bm25", where)[0]
    assert top["metadata"].get("ticket_id") == "INC-2057"
    hyb = r.hybrid("What happened in INC-2057?", 50, where)
    assert any(d["metadata"].get("ticket_id") == "INC-2057" for d in hyb[:3])


def test_tenant_isolation(env):
    r = env["hy"].HybridRetriever()
    docs = r.retrieve("Contoso leave carry-over", "hybrid", {"tenant": "northwind"})
    assert all(d["metadata"]["tenant"] == "northwind" for d in docs)


def test_ask_writes_audit_and_citations(env):
    r = env["hy"].HybridRetriever()
    res = env["hy"].ask("How many PTO days?", r, "hybrid_rerank", {"tenant": "northwind"})
    assert res["citations"] and res["citations"][0]["n"] == 1
    line = json.loads(common.S.audit_log.read_text().splitlines()[-1])
    assert line["question"] == "How many PTO days?" and line["retrieved"]


def test_metadata_filters_and_self_query(env):
    mf, r = env["mf"], env["hy"].HybridRetriever()
    where = mf.build_where("northwind", ["hr"], ["policy"], "2026-01-01")
    docs = r.retrieve("paid time off", "hybrid", where)
    assert docs and all(d["metadata"]["source"].endswith("leave_policy_2026.pdf") for d in docs)
    f = mf.extract_filters("HR policy on PTO since 2026", mf.facet_values(r, "northwind"))
    assert f["departments"] == ["hr"]  # 'made-up' department dropped by validation


def test_raptor_build_and_retrieve(env):
    rp = env["rp"]
    rp.build("northwind")
    tree = env["cols"][common.S.raptor_collection]
    assert tree.count() >= 1
    res, docs = rp.ask("Summarise incidents", "northwind", "collapsed")
    assert docs
    res2, docs2 = rp.ask("Summarise incidents", "northwind", "tree")
    assert docs2


def test_eval_loop(env, monkeypatch):
    ev = env["ev"]
    monkeypatch.setattr(ev, "_retriever", env["hy"].HybridRetriever())
    monkeypatch.setattr(sys, "argv", ["x", "--modes", "hybrid", "filtered", "raptor", "--limit", "5"])
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    ev.main()
    summary = (ev.REPORTS / "eval_summary.md").read_text()
    assert "| hybrid |" in summary and "| filtered |" in summary and "| raptor |" in summary
