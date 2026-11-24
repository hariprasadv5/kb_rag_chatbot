# Company Knowledge-Base Chatbot — Module 2 Capstone (RAG Engineer)

A production-style RAG chatbot over internal policies, runbooks, contracts and tickets:
type-aware chunking → ChromaDB → hybrid BM25 + semantic retrieval fused with RRF → cross-encoder
rerank (top-50 → top-5) → grounded answers with citations, with metadata filtering, Multi-Query,
HyDE, RAPTOR, an audit log, and LangSmith tracing + evaluation.

## Deliverables map

| # | Deliverable (from the brief) | File(s) | How to show it works |
|---|---|---|---|
| 1 | `m2_chroma_ingestion.py` — ingestion and indexing pipeline | `m2_chroma_ingestion.py` | `--dry-run --show 2` output + chunk stats; `collection.count()` |
| 2 | `m2_hybrid_rag.py` — hybrid retrieval with RRF and reranking | `m2_hybrid_rag.py` | `--compare` on an exact-ID and a paraphrase question |
| 3 | `m2_raptor_rag.py` — recursive summarization tree retrieval | `m2_raptor_rag.py` | `build`, `show`, `ask` on a summary question |
| 4 | `m2_metadata_filtering.py` — precision retrieval with metadata | `m2_metadata_filtering.py` | `--compare`, `--auto`, `--benchmark` (P@5 with vs. without filters) |
| 5 | Flowise template `query_rerank_redismem` | `flowise/query_rerank_redismem.json` (+ `_basic` fallback) | Imported chatflow answering with sources; Redis keys present |
| 6 | LangSmith trace dashboard (precision, recall, latency) | `m2_eval_langsmith.py`, `eval/golden_set.jsonl` | Dashboard screenshot + experiment comparison + `reports/eval_summary.md` |

Supporting files: `common.py` (shared clients, BM25, RRF, filter evaluator, citation prompt, audit log),
`make_sample_corpus.py` (demo corpus), `tests/` (21 offline tests), `docker-compose.yml`, `.env.example`.

```
kb_chatbot/
├── common.py                    shared core
├── m2_chroma_ingestion.py       D1
├── m2_hybrid_rag.py             D2
├── m2_raptor_rag.py             D3
├── m2_metadata_filtering.py     D4
├── m2_eval_langsmith.py         D6
├── flowise/                     D5 (+ generator script)
├── make_sample_corpus.py
├── data/raw/<tenant>/<department>/…   documents      data/manifest.yaml  metadata
├── eval/golden_set.jsonl        22 labelled questions (exact-ID, paraphrase, time, procedure,
│                                multi-doc, summary, negative, tenant-isolation)
├── reports/  logs/audit.jsonl   generated
└── tests/
```

---

## Step-by-step method

Work through the steps in order; each one ends with a **✅ Done when** check so you know the
deliverable is complete before moving on. Suggested timing for the 8-hour module is at the end.

### Step 0 — Accounts and tools (15 min)

1. API keys: **OpenAI** (embeddings + chat), **Cohere** (rerank; free trial key is fine), **LangSmith** (smith.langchain.com → Settings → API Keys).
2. Python 3.10+; Node 18+ **or** Docker (for Flowise and Redis).

✅ Done when you have the three keys.

### Step 1 — Project setup (15 min)

```bash
cd kb_chatbot
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # paste your keys into .env
pytest -q                                                # 21 tests, no API calls
```

✅ Done when `pytest` shows **21 passed**.

### Step 2 — Prepare the corpus (15 min)

```bash
python make_sample_corpus.py
```

This writes 10 documents for the fictional company *Northwind* (PDF, DOCX, HTML, PPTX, XLSX, CSV
tickets) plus one document for a second tenant, *Contoso*, and `data/manifest.yaml`.
The corpus deliberately contains: exact IDs (`INC-2057`, `CTR-2025-0412`), an **old and a new
version** of the leave policy (18 vs 22 days) and another tenant's policy — so every technique has
something to prove.

**Using your own documents:** drop files into `data/raw/<tenant>/<department>/` and add an entry per
file in `data/manifest.yaml`:

```yaml
- path: northwind/hr/handbook.pdf
  department: hr          # used by metadata filters
  doc_type: policy        # policy | runbook | contract | ticket | catalog | web → picks the chunking strategy
  date: 2026-01-01        # effective date → time-bounded search
  tenant: northwind       # isolation boundary
  title: Employee Handbook
  access: all             # e.g. legal → restrict with --access
```

Unlisted files are still ingested, with metadata inferred from the folder path. Web pages go under
`urls:` in the manifest or via `--url`.

✅ Done when `data/raw` and `data/manifest.yaml` exist.

### Step 3 — Deliverable 1: ingestion pipeline (45 min)

```bash
python m2_chroma_ingestion.py --dry-run --show 2     # parse + chunk only, no API cost
python m2_chroma_ingestion.py                        # embed + upsert into ./chroma_db
python m2_chroma_ingestion.py                        # run again → every source "unchanged"
```

What it does and what to explain in your write-up:

| Doc type | Strategy | Why |
|---|---|---|
| policy | 1,200 chars, 200 overlap, paragraph → line → sentence splitting | rules must not be cut mid-sentence |
| runbook | numbered steps kept together (`Step n.`), no overlap | a procedure is useless if split |
| contract | split on clause numbers (`14.2 …`) first | answers cite a clause |
| ticket | one ticket = one chunk, `ticket_id` in metadata | exact-ID lookups stay atomic |
| catalog (xlsx) | rows as `Header: value; …` | each row describes itself |
| web/html | heading sections; nav/footer/scripts removed | removes boilerplate noise |

Every chunk also gets a **contextual header** (`Title — Section`) and metadata: `tenant, department,
doc_type, date, date_int, access, source, page/section, ticket_id, doc_hash`. IDs are deterministic,
so re-runs are idempotent; a changed document has its old chunks deleted before the new ones are added.

✅ Done when the run ends with `Collection 'company_kb' now holds N chunks` (≈38 for the sample corpus)
and the second run reports `unchanged`.

### Step 4 — Deliverable 2: hybrid RAG with RRF and reranking (60 min)

```bash
python m2_hybrid_rag.py "What happened in INC-2057?" --compare
python m2_hybrid_rag.py "How much holiday can I roll over?" --compare
python m2_hybrid_rag.py "How do I promote the standby during an orders-db failover?"
python m2_hybrid_rag.py --chat            # multi-turn: ask a follow-up like "and what is the RTO?"
```

Pipeline: bi-encoder top-50 (Chroma) ∥ BM25 top-50 → **RRF** `score = Σ 1/(60 + rank)` →
Cohere `rerank-v3.5` → top-5 → GPT answer that must cite `[n]` → audit line in `logs/audit.jsonl`.

What to point out in `--compare`:
- `INC-2057`: **bm25** finds it immediately; **dense** may rank other tickets higher → the reason for hybrid search.
- "roll over holiday": **dense** finds "carry over PTO"; **bm25** struggles → the reason for semantic search.
- `hybrid_rerank` gets both right; `multi_query` / `hyde` help with vague questions.
- If the best reranked passage scores below `MIN_RERANK_SCORE`, the bot **abstains** instead of hallucinating.

No Cohere key? Set `RERANKER=cross-encoder` (needs `pip install sentence-transformers`) or `RERANKER=none`.

✅ Done when answers show `Sources:` with document/page/section, and `--compare` shows the
exact-ID vs paraphrase difference. Save both outputs for your write-up.

### Step 5 — Deliverable 4: metadata filtering (45 min)

```bash
# filtered vs unfiltered — the 2024 policy (18 days) disappears from the results
python m2_metadata_filtering.py "How many PTO days do employees get?" --department hr --since 2026-01-01 --compare

# self-query: the LLM extracts filters, which are validated against real values
python m2_metadata_filtering.py "Which SEV1 tickets did we have since January 2026?" --auto

# keep only the newest version of each policy
python m2_metadata_filtering.py "How long is parental leave?" --latest-only

# restricted documents (contract is access=legal)
python m2_metadata_filtering.py "What is the liability cap?" --access all            # contract excluded
python m2_metadata_filtering.py "What is the liability cap?" --access all --access legal

# measured improvement
python m2_metadata_filtering.py --benchmark
```

Key design points: filters are applied **before** retrieval in both the vector and BM25 sides
(`common.matches_where` evaluates the same `where` dict in Python), dates are stored as `date_int`
because Chroma range operators need numbers, and **tenant is always enforced**.

✅ Done when `--benchmark` shows a higher mean P@5 for the filtered column. Screenshot it.

### Step 6 — Deliverable 3: RAPTOR (45 min)

```bash
python m2_raptor_rag.py build
python m2_raptor_rag.py show
python m2_raptor_rag.py ask "Summarise the SEV1 and SEV2 incidents and their root causes"
python m2_raptor_rag.py ask "What changed in the leave policy between 2024 and 2026?" --mode tree
python m2_hybrid_rag.py "What changed in the leave policy between 2024 and 2026?"   # compare with flat retrieval
```

How it works: leaf chunks → dimensionality reduction (UMAP if installed, else PCA) → Gaussian Mixture
clustering with BIC-selected k and soft membership → LLM summary per cluster → embed → repeat on
the summaries (up to 3 levels). Retrieval is **collapsed-tree** by default (leaves and summaries
compete in one ranking, then rerank) or **tree** traversal (top-down beam). Trees are built per tenant.

✅ Done when `show` prints L1/L2 summary nodes with their sources and the summary questions get a
broader, cited answer than flat hybrid retrieval.

### Step 7 — Deliverable 6: LangSmith tracing, evaluation and dashboard (75 min)

**7a. Tracing.** With `LANGSMITH_TRACING=true`, run any question from Step 4. In LangSmith, open
project **kb-chatbot** → you should see a `kb_pipeline` trace with children
`condense_question → retrieve → dense_search / bm25_search → rrf_fusion → rerank → generate_answer`,
each showing inputs, retrieved documents, scores and latency. The `trace_id` in `logs/audit.jsonl`
links each answer to its trace (your audit trail).

**7b. Evaluation with feedback** (makes precision/recall/latency chartable):

```bash
python m2_eval_langsmith.py --log-feedback                      # all 8 modes × 22 questions
python m2_eval_langsmith.py --log-feedback --modes hybrid_rerank dense bm25   # cheaper
```

Every question becomes a trace tagged with its `mode` and `category`, and the scores are attached
as feedback: `precision@5, recall@5, mrr, hit@5, abstention_correct, cited, latency_s`.
A local copy goes to `reports/eval_summary.md` and `reports/eval_results.csv`.

**7c. Experiments** (side-by-side comparison):

```bash
python m2_eval_langsmith.py --experiments --judge --modes hybrid_rerank hybrid dense bm25 filtered
```

This uploads the golden set as dataset **kb-golden** and runs one experiment per mode. In
LangSmith → Datasets → kb-golden → select the experiments → **Compare**.

**7d. Build the dashboard.** In LangSmith open **Dashboards → New dashboard** (exact menu labels
can differ slightly between LangSmith versions), name it *KB Chatbot — Retrieval Quality*, and add
charts on project `kb-chatbot`:

| Chart | Metric | Group by / filter |
|---|---|---|
| Precision@5 by mode | Feedback `precision@5` (avg) | group by metadata `mode` |
| Recall@5 by mode | Feedback `recall@5` (avg) | group by metadata `mode` |
| Latency | Run latency P50 and P99 | filter run name = `kb_pipeline`, group by `mode` |
| Abstention correctness | Feedback `abstention_correct` | group by metadata `category` |
| Reranker latency | Run latency | filter run name = `rerank` |
| Cost / tokens | Total tokens | over time |

**7e. Replayable failure cases.** Failures (recall < 1, wrong abstention, or judge < 1) are written to
`reports/failures.jsonl` and to the LangSmith dataset **kb-failures** with the trace id. After
changing a chunk size, prompt or `k`, replay just those:

```bash
python m2_eval_langsmith.py --replay --log-feedback
```

✅ Done when you have: the dashboard screenshot (precision, recall, latency), the experiment-compare
screenshot, `reports/eval_summary.md`, and at least one failure you replayed after a fix.

### Step 8 — Deliverable 5: Flowise template `query_rerank_redismem` (60 min)

1. **Serve the Chroma collection** that the Python pipeline built (same folder):
   ```bash
   chroma run --path ./chroma_db --port 8000
   ```
2. **Start Redis and Flowise** — either:
   - `docker compose up -d` (Flowise at http://localhost:3000; inside Docker use Redis host `redis` and Chroma URL `http://host.docker.internal:8000`), or
   - `docker run -d -p 6379:6379 redis:7-alpine` and `npx flowise start` (then `localhost` works for both).
3. **Credentials** (Flowise → Credentials → Add): *OpenAI API*, *Cohere API*, *Redis API* (host/port) or *Redis URL* (`redis://localhost:6379`).
4. **Import:** Chatflows → Add New → ⚙ settings (top right) → **Load Chatflow** → `flowise/query_rerank_redismem.json`.
5. On each node, pick the credential; check Chroma = collection `company_kb`, URL as above, metadata filter `{"tenant":"northwind"}`, Top K 50; Cohere Rerank Top K = 5; Redis memory window 6.
6. **Save**, open the chat bubble and test:
   - "How much holiday can I roll over?" → answer with sources from the 2026 policy
   - follow-up "and when does it expire?" → memory resolves "it"
   - `redis-cli KEYS '*'` shows the session keys (proof memory lives in Redis).
7. **Trace Flowise in LangSmith:** chatflow ⚙ → Configuration → Analyse Chatflow → LangSmith → same project.

The flow is: OpenAI Embeddings → Chroma (vector store) → **Multi Query Retriever** (3 rewrites) →
**Cohere Rerank Retriever** (top 5) → **Conversational Retrieval QA Chain** ← **Redis-Backed Chat Memory**.

> **If the import shows a broken link between Multi Query and Cohere Rerank:** some Flowise
> versions only let the Cohere Rerank node accept a *Vector Store Retriever*. Load
> `flowise/query_rerank_redismem_basic.json` instead (Chroma retriever → Cohere Rerank → chain + Redis
> memory). In your write-up, note that multi-query is then provided by the chain's
> question-rephrasing step plus the Python `multi_query` mode. Regenerate both files any time with
> `python flowise/build_flowise_template.py`.

✅ Done when the chatflow answers with source documents, follow-ups work, and Redis holds the session.

### Step 9 — Final check and submission (30 min)

- [ ] `pytest -q` → 21 passed
- [ ] D1 ingestion log (chunk stats per type) + idempotent re-run
- [ ] D2 `--compare` outputs for one exact-ID and one paraphrase question
- [ ] D3 `show` output + one RAPTOR answer
- [ ] D4 `--benchmark` table
- [ ] D5 Flowise JSON + screenshot of a cited answer and a follow-up
- [ ] D6 LangSmith dashboard screenshot (precision, recall, latency) + experiment comparison + `reports/eval_summary.md`
- [ ] A short write-up: which mode won, a failure case, how you fixed it and the replayed result

Zip the folder **without** `.env`, `.venv/` and `chroma_db/`.

---

## Suggested 8-hour schedule

| Time | Work |
|---|---|
| 0:00–0:45 | Steps 0–2: setup, tests, corpus |
| 0:45–1:30 | Step 3: ingestion |
| 1:30–2:30 | Step 4: hybrid + rerank |
| 2:30–3:15 | Step 5: metadata filtering |
| 3:15–4:00 | Step 6: RAPTOR |
| 4:00–5:15 | Step 7: LangSmith tracing, eval, dashboard |
| 5:15–6:15 | Step 8: Flowise |
| 6:15–7:30 | Tuning loop: change one knob (`DENSE_K`, chunk size, `MIN_RERANK_SCORE`, reranker), re-run eval, replay failures |
| 7:30–8:00 | Step 9: checklist and write-up |

## Troubleshooting

| Problem | Fix |
|---|---|
| `Collection is empty` | Run `python m2_chroma_ingestion.py` first |
| BM25 results look stale | The BM25 index is built when the retriever starts; restart after re-ingesting |
| Cohere errors / rate limits | `RERANKER=cross-encoder` or `none`; the pipeline falls back automatically |
| Bot abstains too often | Lower `MIN_RERANK_SCORE` (e.g. 0.01); raise it if it answers off-topic |
| Flowise can't reach Chroma | Is `chroma run` up? In Docker use `http://host.docker.internal:8000` |
| Flowise answers differ from Python | Flowise embeddings model must be `text-embedding-3-small` (same as ingestion) |
| No feedback on dashboard | Check `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT`; feedback can take a minute to appear |
| `hnsw:space` or Chroma API warnings | Pin `chromadb` to the version used at ingestion time |
