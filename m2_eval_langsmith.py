"""
m2_eval_langsmith.py — Deliverable 6: evaluation + LangSmith trace dashboard
(precision, recall, latency) with replayable failure cases.

What it does
  1. Runs the golden set (eval/golden_set.jsonl) through every retrieval mode.
  2. Scores each answer:  precision@5, recall@5, MRR, hit@5, abstention correctness,
     citation presence, latency (and optional LLM-judge correctness with --judge).
  3. Writes reports/eval_results.csv + reports/eval_summary.md (works without LangSmith).
  4. With LANGSMITH_API_KEY set:
       --log-feedback  every question is a real trace in project $LANGSMITH_PROJECT, tagged with
                       its mode; the scores are attached as FEEDBACK → chart them in a dashboard.
       --experiments   also runs LangSmith `evaluate()` per mode on dataset "kb-golden" →
                       side-by-side experiment comparison in the LangSmith UI.
  5. Every failing example is saved to reports/failures.jsonl AND the LangSmith dataset
     "kb-failures" (with the trace link) → replay them after a fix with --replay.

Usage
    python m2_eval_langsmith.py                                  # all modes, local report
    python m2_eval_langsmith.py --modes hybrid_rerank dense bm25 --log-feedback
    python m2_eval_langsmith.py --experiments --judge
    python m2_eval_langsmith.py --replay                         # re-run only previous failures
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from common import ROOT, S, chat

GOLDEN = ROOT / "eval" / "golden_set.jsonl"
REPORTS = ROOT / "reports"
FAILURES = REPORTS / "failures.jsonl"
DATASET = "kb-golden"
FAIL_DATASET = "kb-failures"
ALL_MODES = ["dense", "bm25", "hybrid", "hybrid_rerank", "multi_query", "hyde", "filtered", "raptor"]
K = 5

_retriever = None


def retriever():
    global _retriever
    if _retriever is None:
        from m2_hybrid_rag import HybridRetriever

        _retriever = HybridRetriever()
    return _retriever


# ---------------------------------------------------------------------- the system under test
def run_pipeline(inputs: dict, mode: str, langsmith_extra: dict | None = None) -> dict:
    from m2_hybrid_rag import ask
    from m2_metadata_filtering import build_where

    q, tenant = inputs["question"], inputs.get("tenant", S.default_tenant)
    extra = {"langsmith_extra": langsmith_extra} if langsmith_extra else {}
    if mode == "raptor":
        from m2_raptor_rag import ask as raptor_ask

        res, _ = raptor_ask(q, tenant, "collapsed", **extra)
        res.setdefault("retrieval_ms", res.get("total_ms"))
    elif mode == "filtered":
        f = inputs.get("filters") or {}
        where = build_where(tenant, f.get("departments"), f.get("doc_types"), f.get("date_from"), f.get("date_to"))
        res = ask(q, retriever(), "hybrid_rerank", where, **extra)
    else:
        res = ask(q, retriever(), mode, {"tenant": tenant}, **extra)
    return {
        "answer": res["answer"], "abstained": res.get("abstained", False),
        "citations": res.get("citations", []), "retrieved": res.get("retrieved", [])[:K],
        "retrieval_ms": res.get("retrieval_ms"), "total_ms": res.get("total_ms"),
    }


# ---------------------------------------------------------------------- metrics
def keys_of(item: dict) -> set[str]:
    """Identifiers a retrieved chunk can match: file names (RAPTOR summaries list several
    joined by ' | ') and ticket IDs."""
    keys = {Path(s.strip()).name for s in str(item.get("source") or "").split(" | ") if s.strip()}
    if item.get("ticket_id"):
        keys.add(item["ticket_id"])
    return keys


def score(outputs: dict, expected: list[str], reference: str = "", judge: bool = False, question: str = "") -> dict:
    retrieved = outputs["retrieved"][:K]
    rel = [bool(keys_of(r) & set(expected)) for r in retrieved]
    found = {e for e in expected for r in retrieved if e in keys_of(r)}
    negative = not expected
    m = {
        "precision@5": (sum(rel) / len(retrieved)) if (retrieved and not negative) else None,
        "recall@5": (len(found) / len(expected)) if not negative else None,
        "hit@5": float(any(rel)) if not negative else None,
        "mrr": next((1 / i for i, r in enumerate(rel, 1) if r), 0.0) if not negative else None,
        "abstention_correct": float(outputs["abstained"] == negative),
        "cited": float(bool(outputs["citations"])) if not outputs["abstained"] else None,
        "latency_s": round((outputs.get("total_ms") or 0) / 1000, 3),
        "retrieval_latency_s": round((outputs.get("retrieval_ms") or 0) / 1000, 3),
    }
    if judge and not negative:
        m["correctness"] = llm_judge(question, reference, outputs["answer"])
    return m


def llm_judge(question: str, reference: str, answer: str) -> float:
    prompt = (f"Question: {question}\nReference answer: {reference}\nCandidate answer: {answer}\n\n"
              "Does the candidate contain the key facts of the reference without contradicting it? "
              'Return JSON {"score": 1} for yes, {"score": 0.5} for partially, {"score": 0} for no.')
    try:
        return float(json.loads(chat([{"role": "user", "content": prompt}], json_mode=True))["score"])
    except Exception:
        return 0.0


def is_failure(m: dict) -> bool:
    if m["abstention_correct"] < 1:
        return True
    if m["recall@5"] is not None and m["recall@5"] < 1:
        return True
    return m.get("correctness") is not None and m["correctness"] < 1


# ---------------------------------------------------------------------- LangSmith helpers
def ls_client():
    if not os.getenv("LANGSMITH_API_KEY"):
        return None
    from langsmith import Client

    return Client()


def ensure_dataset(client, name: str, rows: list[dict], description: str):
    if client.has_dataset(dataset_name=name):
        return client.read_dataset(dataset_name=name)
    ds = client.create_dataset(name, description=description)
    client.create_examples(
        inputs=[{"question": r["question"], "tenant": r["tenant"], "filters": r.get("filters")} for r in rows],
        outputs=[{"expected_sources": r["expected_sources"], "reference": r["reference"], "category": r["category"]} for r in rows],
        metadata=[{"id": r["id"], "category": r["category"]} for r in rows],
        dataset_id=ds.id,
    )
    return ds


def run_experiments(client, rows: list[dict], modes: list[str], judge: bool) -> None:
    """LangSmith-native experiments: one experiment per mode on the same dataset, so the
    UI's 'Compare experiments' view shows precision/recall/latency side by side."""
    try:
        from langsmith import evaluate
    except ImportError:
        from langsmith.evaluation import evaluate

    ensure_dataset(client, DATASET, rows, "Company KB golden questions (Module 2)")

    def make_evaluator(key):
        def _ev(run, example):
            m = score(run.outputs, example.outputs["expected_sources"], example.outputs.get("reference", ""),
                      judge and key == "correctness", example.inputs["question"])
            return {"key": key, "score": m.get(key)}
        _ev.__name__ = key.replace("@", "_at_")
        return _ev

    keys = ["precision@5", "recall@5", "mrr", "abstention_correct", "cited", "latency_s"] + (["correctness"] if judge else [])
    for mode in modes:
        print(f"\n▶ LangSmith experiment: {mode}")
        evaluate(lambda inputs, mode=mode: run_pipeline(inputs, mode), data=DATASET,
                 evaluators=[make_evaluator(k) for k in keys], experiment_prefix=f"kb-{mode}",
                 metadata={"mode": mode, "reranker": S.reranker, "embed_model": S.embed_model,
                           "dense_k": S.dense_k, "final_k": S.final_k}, max_concurrency=2)


# ---------------------------------------------------------------------- main loop
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", nargs="+", default=ALL_MODES, choices=ALL_MODES)
    ap.add_argument("--log-feedback", action="store_true", help="trace every run + attach scores as feedback")
    ap.add_argument("--experiments", action="store_true", help="also run LangSmith evaluate() experiments")
    ap.add_argument("--judge", action="store_true", help="LLM-as-judge answer correctness")
    ap.add_argument("--replay", action="store_true", help="only re-run examples from reports/failures.jsonl")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    rows = [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]
    if a.replay:
        failed = {(json.loads(l)["id"], json.loads(l)["mode"]) for l in FAILURES.read_text().splitlines() if l.strip()}
        ids = {i for i, _ in failed}
        rows = [r for r in rows if r["id"] in ids]
        a.modes = sorted({m for _, m in failed} & set(a.modes)) or a.modes
        print(f"Replaying {len(rows)} failed examples across modes {a.modes}")
    if a.limit:
        rows = rows[: a.limit]

    client = ls_client()
    if (a.log_feedback or a.experiments) and client is None:
        print("⚠ LANGSMITH_API_KEY not set — running locally only.")
    REPORTS.mkdir(exist_ok=True)

    results, failures, feedback_queue = [], [], []
    for mode in a.modes:
        for r in rows:
            run_id = uuid.uuid4()
            extra = None
            if client and a.log_feedback:
                extra = {"run_id": run_id, "tags": ["eval", mode, r["category"]],
                         "metadata": {"mode": mode, "category": r["category"], "example_id": r["id"],
                                      "reranker": S.reranker}}
            try:
                out = run_pipeline({"question": r["question"], "tenant": r["tenant"], "filters": r.get("filters")}, mode, extra)
            except Exception as e:
                print(f"  ✖ {mode} {r['id']}: {e!r}")
                continue
            m = score(out, r["expected_sources"], r["reference"], a.judge, r["question"])
            row = {"mode": mode, "id": r["id"], "category": r["category"], "question": r["question"], **m,
                   "answer": out["answer"][:300].replace("\n", " "),
                   "retrieved": " ; ".join(sorted({k for it in out["retrieved"] for k in keys_of(it)}))}
            results.append(row)
            if extra:
                feedback_queue.append((run_id, m))
            if is_failure(m):
                failures.append({**r, "mode": mode, "metrics": m, "answer": out["answer"],
                                 "retrieved": out["retrieved"], "run_id": str(run_id) if extra else None})
            flag = "✔" if not is_failure(m) else "✖"
            print(f"  {flag} {mode:<13} {r['id']} P@5={_f(m['precision@5'])} R@5={_f(m['recall@5'])} "
                  f"abst={m['abstention_correct']:.0f} {m['latency_s']:.2f}s  {r['question'][:55]}")

    # ---- feedback → LangSmith (scores become chartable per trace)
    if client and feedback_queue:
        time.sleep(2)  # let the tracer flush the runs before attaching feedback
        if hasattr(client, "flush"):
            try:
                client.flush()
            except Exception:
                pass
        for run_id, m in feedback_queue:
            for key, val in m.items():
                if val is not None:
                    try:
                        client.create_feedback(run_id, key=key, score=float(val))
                    except Exception as e:
                        print("feedback error:", e)
        print(f"Attached feedback to {len(feedback_queue)} traces in project "
              f"'{os.getenv('LANGSMITH_PROJECT', 'default')}'.")

    # ---- failures → replayable dataset
    with FAILURES.open("w") as f:
        for x in failures:
            f.write(json.dumps(x, default=str) + "\n")
    if client and failures:
        ds = (client.read_dataset(dataset_name=FAIL_DATASET) if client.has_dataset(dataset_name=FAIL_DATASET)
              else client.create_dataset(FAIL_DATASET, description="Replayable failure cases from KB evals"))
        client.create_examples(
            inputs=[{"question": x["question"], "tenant": x["tenant"], "filters": x.get("filters")} for x in failures],
            outputs=[{"expected_sources": x["expected_sources"], "reference": x["reference"]} for x in failures],
            metadata=[{"mode": x["mode"], "id": x["id"], "run_id": x["run_id"], "category": x["category"]} for x in failures],
            dataset_id=ds.id,
        )
        print(f"Saved {len(failures)} failures to LangSmith dataset '{FAIL_DATASET}'.")

    write_reports(results)
    if client and a.experiments:
        run_experiments(client, rows, a.modes, a.judge)


def _f(x):
    return "  - " if x is None else f"{x:.2f}"


def write_reports(results: list[dict]) -> None:
    if not results:
        print("No results.")
        return
    with (REPORTS / "eval_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    def avg(vals):
        vals = [v for v in vals if v is not None]
        return mean(vals) if vals else None

    by_mode = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    lines = ["# Knowledge-base retrieval evaluation", "",
             f"Golden set: {len({r['id'] for r in results})} questions · k={K} · reranker={S.reranker}", "",
             "| mode | precision@5 | recall@5 | MRR | abstention ok | cited | p50 latency (s) | p95 latency (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for mode, rs in by_mode.items():
        lat = sorted(r["latency_s"] for r in rs)
        p95 = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))]
        lines.append(f"| {mode} | {_f(avg(r['precision@5'] for r in rs))} | {_f(avg(r['recall@5'] for r in rs))} | "
                     f"{_f(avg(r['mrr'] for r in rs))} | {_f(avg(r['abstention_correct'] for r in rs))} | "
                     f"{_f(avg(r['cited'] for r in rs))} | {median(lat):.2f} | {p95:.2f} |")
    lines += ["", "## Recall@5 by question category", "",
              "| mode | " + " | ".join(sorted({r['category'] for r in results})) + " |",
              "|---|" + "---|" * len({r['category'] for r in results})]
    cats = sorted({r["category"] for r in results})
    for mode, rs in by_mode.items():
        cells = []
        for c in cats:
            v = avg(r["recall@5"] if r["recall@5"] is not None else r["abstention_correct"] for r in rs if r["category"] == c)
            cells.append(_f(v))
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    lines += ["", "_negative / tenant-isolation columns show abstention correctness instead of recall._"]
    (REPORTS / "eval_summary.md").write_text("\n".join(lines))
    print("\n" + "\n".join(lines[4 : 6 + len(by_mode)]))
    print(f"\nReports: {REPORTS/'eval_summary.md'}, {REPORTS/'eval_results.csv'}, {FAILURES}")


if __name__ == "__main__":
    main()
