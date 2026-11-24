"""
m2_chroma_ingestion.py — Deliverable 1: ingestion and indexing pipeline.

    load (PDF / web page / DOCX / PPTX / XLSX / CSV tickets / MD)
      → clean
      → chunk with a strategy chosen per document type
      → attach metadata (tenant, department, doc_type, date, source, page/section …)
      → embed (OpenAI) → upsert into ChromaDB (idempotent, change-detected)

Usage
    python m2_chroma_ingestion.py                    # ingest everything in data/manifest.yaml
    python m2_chroma_ingestion.py --dry-run          # parse + chunk only, print stats (no API calls)
    python m2_chroma_ingestion.py --reset            # drop the collection and rebuild
    python m2_chroma_ingestion.py --url https://... --department hr --doc-type web
    python m2_chroma_ingestion.py --show 3           # print 3 sample chunks per document
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from common import ROOT, S, date_to_int, embed, get_chroma, get_collection, sha1, traceable

DATA = ROOT / "data"
RAW = DATA / "raw"

# ============================================================================ chunk strategy
# Sizes are in characters (~4 chars per token). Chosen per document TYPE, because the
# natural unit of meaning differs:
#   policy   → a clause/section; medium chunks, generous overlap so rules aren't cut mid-sentence
#   runbook  → a procedure; keep numbered steps together, never split inside a step
#   contract → a clause ("4.2 …"); split on clause numbers first, larger chunks
#   ticket   → one ticket = one chunk (never split, never merged) so IDs stay atomic
#   catalog  → spreadsheet rows grouped, header repeated in every chunk
#   web      → heading sections
CHUNK_PROFILES: dict[str, dict] = {
    "policy":   {"size": 1200, "overlap": 200, "separators": ["\n\n", "\n", ". ", " "]},
    "runbook":  {"size": 1000, "overlap": 0,   "separators": ["\n\n", "\nStep ", "\n", ". "]},
    "contract": {"size": 1400, "overlap": 150, "separators": [r"\n(?=\d+\.\d+ )", r"(?<=\.) (?=\d+\.\d+ )", "\n\n", ". "]},
    "ticket":   {"size": 10**9, "overlap": 0,  "separators": []},
    "catalog":  {"size": 1500, "overlap": 0,   "separators": ["\n"]},
    "web":      {"size": 1000, "overlap": 150, "separators": ["\n\n", "\n", ". ", " "]},
    "default":  {"size": 1000, "overlap": 150, "separators": ["\n\n", "\n", ". ", " "]},
}


def recursive_split(text: str, size: int, overlap: int, separators: list[str]) -> list[str]:
    """Recursive character splitter (same idea as LangChain's RecursiveCharacterTextSplitter):
    try the coarsest separator first; any piece still too big is split with the next one.
    Separators may be regexes. Adjacent small pieces are merged up to `size` with `overlap`."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    if not separators:  # hard cut as last resort
        step = max(1, size - overlap)
        return [text[i : i + size] for i in range(0, len(text), step)]

    sep, rest = separators[0], separators[1:]
    is_regex = sep.startswith("\\") or "(?" in sep
    pieces = re.split(sep, text) if is_regex else text.split(sep)
    joiner = "\n" if is_regex else sep
    pieces = [p for p in pieces if p and p.strip()]
    if len(pieces) == 1:
        return recursive_split(text, size, overlap, rest)

    # split oversize pieces further
    atoms: list[str] = []
    for p in pieces:
        atoms.extend(recursive_split(p, size, overlap, rest) if len(p) > size else [p])

    # merge atoms into chunks with overlap
    chunks: list[str] = []
    cur = ""
    for a in atoms:
        candidate = f"{cur}{joiner}{a}" if cur else a
        if len(candidate) <= size:
            cur = candidate
        else:
            if cur:
                chunks.append(cur.strip())
            tail = cur[-overlap:] if overlap and cur else ""
            # start the overlap at a word boundary
            if tail and " " in tail:
                tail = tail[tail.index(" ") + 1 :]
            cur = f"{tail}{joiner}{a}" if tail and len(tail) + len(a) < size else a
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


# ============================================================================ loaders
# Each loader returns a list of "sections": {"text": str, **extra_metadata}


def clean(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # de-hyphenate line breaks (PDF)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> list[dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    out = []
    for i, page in enumerate(reader.pages, start=1):
        txt = clean(page.extract_text() or "")
        if txt:
            out.append({"text": txt, "page": i})
    return out


def load_docx(path: Path) -> list[dict]:
    """Walk the body in order (paragraphs AND tables), grouping text under headings."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(str(path))
    sections: list[dict] = []
    heading, buf, step_no = "Introduction", [], 0

    def flush():
        if buf:
            sections.append({"text": clean("\n".join(buf)), "section": heading})

    for el in d.element.body.iterchildren():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(el, d)
            style = (p.style.name or "").lower() if p.style is not None else ""
            text = p.text.strip()
            if not text:
                continue
            if style.startswith("heading") or style == "title":
                flush()
                buf, step_no = [], 0
                heading = text
            elif "list number" in style:
                step_no += 1
                buf.append(f"Step {step_no}. {text}")
            elif "list" in style:
                buf.append(f"- {text}")
            else:
                buf.append(text)
        elif tag == "tbl":
            t = Table(el, d)
            rows = [[c.text.strip() for c in r.cells] for r in t.rows]
            if rows:
                header = rows[0]
                for r in rows[1:]:
                    buf.append("; ".join(f"{h}: {v}" for h, v in zip(header, r)))
    flush()
    return sections


def load_pptx(path: Path) -> list[dict]:
    from pptx import Presentation

    out = []
    for i, slide in enumerate(Presentation(str(path)).slides, start=1):
        texts = []
        for shp in slide.shapes:
            if shp.has_text_frame:
                texts.extend(p.text for p in shp.text_frame.paragraphs if p.text.strip())
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                texts.append(f"Speaker notes: {notes}")
        title = slide.shapes.title.text if slide.shapes.title is not None else f"Slide {i}"
        out.append({"text": clean("\n".join(texts)), "page": i, "section": title})
    return out


def load_xlsx(path: Path) -> list[dict]:
    """Every row becomes 'Header: value; …' so a chunk is self-describing."""
    from openpyxl import load_workbook

    out = []
    wb = load_workbook(str(path), read_only=True, data_only=True)
    for ws in wb.worksheets:
        rows = [r for r in ws.iter_rows(values_only=True) if any(v is not None for v in r)]
        if not rows:
            continue
        header = [str(h) for h in rows[0]]
        lines = ["; ".join(f"{h}: {v}" for h, v in zip(header, r) if v is not None) for r in rows[1:]]
        out.append({"text": "\n".join(lines), "section": ws.title})
    return out


def _html_sections(html: str) -> list[dict]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    sections, heading, buf = [], title or "Page", []
    body = soup.body or soup
    for el in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "td", "th"]):
        if el.find_parent(["li", "td"]) is not None and el.name in ("p",):
            continue
        txt = el.get_text(" ", strip=True)
        if not txt:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            if buf:
                sections.append({"text": clean("\n".join(buf)), "section": heading})
            heading, buf = txt, []
        else:
            buf.append(("- " if el.name == "li" else "") + txt)
    if buf:
        sections.append({"text": clean("\n".join(buf)), "section": heading})
    return sections


def load_html(path: Path) -> list[dict]:
    return _html_sections(path.read_text(encoding="utf-8", errors="ignore"))


def load_url(url: str) -> list[dict]:
    import requests

    r = requests.get(url, timeout=30, headers={"User-Agent": "kb-ingestion-bot/1.0"})
    r.raise_for_status()
    return _html_sections(r.text)


def load_md(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    parts = re.split(r"(?m)^(#{1,4} .+)$", text)
    out, heading = [], path.stem
    for p in parts:
        if re.match(r"^#{1,4} ", p):
            heading = p.lstrip("# ").strip()
        elif p.strip():
            out.append({"text": clean(p), "section": heading})
    return out


def load_tickets_csv(path: Path) -> list[dict]:
    """One ticket per section; ticket_id becomes metadata for exact lookups."""
    out = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = row.get("ticket_id") or row.get("id") or ""
            text = "\n".join(f"{k}: {v}" for k, v in row.items() if v)
            out.append({"text": f"Ticket {tid}\n{text}", "ticket_id": tid, "ticket_date": row.get("date", "")})
    return out


LOADERS = {
    ".pdf": load_pdf, ".docx": load_docx, ".pptx": load_pptx, ".xlsx": load_xlsx,
    ".html": load_html, ".htm": load_html, ".md": load_md, ".txt": load_md, ".csv": load_tickets_csv,
}


# ============================================================================ pipeline
def infer_meta(path: Path) -> dict:
    """Fallback metadata when a file is not listed in the manifest:
    data/raw/<tenant>/<department>/<file>."""
    rel = path.relative_to(RAW).parts
    tenant = rel[0] if len(rel) >= 3 else S.default_tenant
    dept = rel[-2] if len(rel) >= 2 else "general"
    name = path.stem.lower()
    doc_type = next((t for t in ("runbook", "contract", "policy", "ticket") if t in name), "default")
    if path.suffix == ".csv":
        doc_type = "ticket"
    mdate = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
    return {"tenant": tenant, "department": dept, "doc_type": doc_type, "date": mdate,
            "title": path.stem.replace("_", " ").title(), "access": "all"}


def chunk_document(sections: list[dict], meta: dict) -> list[dict]:
    """Split sections with the doc-type profile; prepend a short context header to each
    chunk (title + section) — "contextual chunk headers" noticeably improve both BM25 and
    dense recall because a chunk like "Up to 5 days may be carried over" is otherwise anonymous."""
    profile = CHUNK_PROFILES.get(meta["doc_type"], CHUNK_PROFILES["default"])
    chunks = []
    for sec in sections:
        extra = {k: v for k, v in sec.items() if k != "text"}
        for piece in recursive_split(sec["text"], profile["size"], profile["overlap"], profile["separators"]):
            header = f"{meta['title']}"
            if extra.get("section"):
                header += f" — {extra['section']}"
            chunks.append({"text": f"{header}\n{piece}", **extra})
    return chunks


def build_records(source: str, sections: list[dict], meta: dict) -> list[dict]:
    doc_hash = sha1(json.dumps(sections, sort_keys=True, default=str))
    records = []
    for i, ch in enumerate(chunk_document(sections, meta)):
        md = {
            "source": source,
            "title": meta["title"],
            "tenant": meta["tenant"],
            "department": meta["department"],
            "doc_type": meta["doc_type"],
            "date": str(meta["date"]),
            "date_int": date_to_int(meta.get("ticket_date") or meta["date"]),
            "access": meta.get("access", "all"),
            "chunk_index": i,
            "doc_hash": doc_hash,
            "ingested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        for k in ("page", "section", "ticket_id"):
            if ch.get(k) not in (None, ""):
                md[k] = ch[k]
        if ch.get("ticket_date"):  # tickets carry their own date
            md["date"] = ch["ticket_date"]
            md["date_int"] = date_to_int(ch["ticket_date"])
        records.append({"id": sha1(source, str(i), ch["text"])[:24], "text": ch["text"], "metadata": md})
    return records


def iter_sources(manifest: dict, only: str | None = None):
    listed = set()
    for d in manifest.get("documents", []):
        p = RAW / d["path"]
        listed.add(p.resolve())
        if only and only not in str(p):
            continue
        yield str(p.relative_to(ROOT)), (lambda p=p: LOADERS[p.suffix.lower()](p)), dict(d)
    # any file dropped into data/raw but not in the manifest is still ingested
    for p in sorted(RAW.rglob("*")):
        if p.is_file() and p.suffix.lower() in LOADERS and p.resolve() not in listed:
            if only and only not in str(p):
                continue
            yield str(p.relative_to(ROOT)), (lambda p=p: LOADERS[p.suffix.lower()](p)), infer_meta(p)
    for u in manifest.get("urls", []) or []:
        if only and only not in u["url"]:
            continue
        meta = {"tenant": S.default_tenant, "department": "general", "doc_type": "web",
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "title": u["url"], "access": "all", **u}
        yield u["url"], (lambda url=u["url"]: load_url(url)), meta


@traceable(name="ingest_source", run_type="chain")
def ingest_source(source: str, loader, meta: dict, collection, dry_run: bool = False, show: int = 0) -> dict:
    sections = loader()
    records = build_records(source, sections, meta)
    stats = {"source": source, "doc_type": meta["doc_type"], "sections": len(sections), "chunks": len(records),
             "avg_chars": int(sum(len(r["text"]) for r in records) / max(1, len(records)))}
    if show:
        for r in records[:show]:
            print(f"\n--- {r['id']} {json.dumps({k: r['metadata'][k] for k in ('department','doc_type','date') }, default=str)}")
            print(r["text"][:600])
    if dry_run or not records:
        stats["status"] = "dry-run"
        return stats

    # change detection: skip if this exact content is already indexed
    existing = collection.get(where={"source": source}, include=["metadatas"])
    if existing["ids"] and existing["metadatas"][0].get("doc_hash") == records[0]["metadata"]["doc_hash"] \
            and len(existing["ids"]) == len(records):
        stats["status"] = "unchanged"
        return stats
    if existing["ids"]:  # document changed → remove stale chunks first
        collection.delete(ids=existing["ids"])

    vectors = embed([r["text"] for r in records])
    collection.upsert(
        ids=[r["id"] for r in records],
        documents=[r["text"] for r in records],
        metadatas=[r["metadata"] for r in records],
        embeddings=vectors,
    )
    stats["status"] = "indexed"
    return stats


@traceable(name="ingestion_pipeline", run_type="chain")
def run(dry_run=False, reset=False, only=None, show=0, extra_urls=None) -> list[dict]:
    manifest = yaml.safe_load((DATA / "manifest.yaml").read_text()) if (DATA / "manifest.yaml").exists() else {}
    if extra_urls:
        manifest.setdefault("urls", [])
        manifest["urls"] = (manifest["urls"] or []) + extra_urls
    collection = None
    if not dry_run:
        if reset:
            try:
                get_chroma().delete_collection(S.collection)
                print(f"Dropped collection '{S.collection}'")
            except Exception:
                pass
        collection = get_collection()

    results = []
    for source, loader, meta in iter_sources(manifest, only):
        try:
            st = ingest_source(source, loader, meta, collection, dry_run=dry_run, show=show)
        except Exception as e:
            st = {"source": source, "status": f"ERROR {e!r}", "chunks": 0, "doc_type": meta.get("doc_type")}
        results.append(st)
        print(f"{st['status']:>10}  {st.get('chunks',0):>3} chunks  avg {st.get('avg_chars','-'):>5} chars  [{st.get('doc_type')}]  {source}")

    by_type = defaultdict(int)
    for r in results:
        by_type[r.get("doc_type")] += r.get("chunks", 0)
    print(f"\nTotal chunks: {sum(by_type.values())}  by type: {dict(by_type)}")
    if collection is not None:
        print(f"Collection '{S.collection}' now holds {collection.count()} chunks at {S.chroma_dir}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="parse and chunk only; no embeddings / Chroma")
    ap.add_argument("--reset", action="store_true", help="drop and rebuild the collection")
    ap.add_argument("--only", help="only ingest sources whose path/URL contains this text")
    ap.add_argument("--show", type=int, default=0, help="print N sample chunks per document")
    ap.add_argument("--url", action="append", help="extra web page(s) to ingest")
    ap.add_argument("--department", default="general")
    ap.add_argument("--doc-type", default="web")
    ap.add_argument("--tenant", default=S.default_tenant)
    ap.add_argument("--date", default=None)
    a = ap.parse_args()
    urls = [{"url": u, "department": a.department, "doc_type": a.doc_type, "tenant": a.tenant,
             "date": a.date or datetime.now(timezone.utc).strftime("%Y-%m-%d"), "title": u} for u in (a.url or [])]
    run(dry_run=a.dry_run, reset=a.reset, only=a.only, show=a.show, extra_urls=urls)
