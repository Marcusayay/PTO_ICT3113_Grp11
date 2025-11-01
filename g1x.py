# === One-click: Build KB + FAISS from Marker (JSON+MD) with table parsing =====
# - Auto-installs deps
# - Parses Marker JSON 'Table' blocks (via HTML) -> DataFrames
# - Adds table row-sentences to embeddings
# - Saves long-form tables to data/kb_tables.parquet
# - Caches per-doc; if nothing changed, keeps existing KB/index

import sys, subprocess, warnings, re, json, hashlib, time
from pathlib import Path
from io import StringIO

# 1) Ensure dependencies
for pkg in ["sentence-transformers", "faiss-cpu", "pandas", "pyarrow", "numpy", "lxml", "tqdm"]:
    try:
        __import__(pkg.split("-")[0])
    except Exception:
        print(f"📦 Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import numpy as np, pandas as pd, faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---------- helpers ----------
def _file_hash_key(p: Path) -> str:
    try:
        s = p.stat()
        return hashlib.md5(f"{p.resolve()}|{s.st_size}|{int(s.st_mtime)}".encode()).hexdigest()
    except FileNotFoundError:
        return ""

def _safe_read(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue
    return ""

def _strip_md_basic(md: str) -> str:
    md = re.sub(r"```.*?```", " ", md, flags=re.DOTALL)     # code fences
    md = re.sub(r"!\[[^\]]*\]\([^\)]*\)", " ", md)          # images
    md = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", md)       # links
    md = re.sub(r"<[^>]+>", " ", md)                        # html tags
    md = re.sub(r"\s+", " ", md)
    return md.strip()

def _extract_text_from_marker_json(jtxt: str) -> str:
    # Best-effort: prefer 'markdown', else join pages[].text, else collect strings
    try:
        data = json.loads(jtxt)
    except Exception:
        return ""
    if isinstance(data, dict) and isinstance(data.get("markdown"), str):
        return _strip_md_basic(data["markdown"])
    pages = data.get("pages") if isinstance(data, dict) else None
    if isinstance(pages, list):
        segs = []
        for p in pages:
            if isinstance(p, dict):
                t = p.get("text")
                if isinstance(t, str) and t.strip():
                    segs.append(t.strip())
        if segs:
            return _strip_md_basic("\n\n".join(segs))
    # fallback: collect strings
    collected = []
    def walk(n):
        if isinstance(n, dict):
            for v in n.values(): walk(v)
        elif isinstance(n, list):
            for v in n: walk(v)
        elif isinstance(n, str):
            s = n.strip()
            if len(s) >= 20:
                collected.append(s)
    walk(data)
    return _strip_md_basic("\n\n".join(collected)) if collected else ""

def _chunk_text(text: str, max_chars: int = 1600, overlap: int = 200):
    if not text: return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf, cur = [], [], 0
    def flush():
        nonlocal buf, cur
        if not buf: return
        s = "\n\n".join(buf).strip()
        step = max_chars - overlap
        for i in range(0, len(s), step):
            piece = s[i:i+step].strip()
            if piece: chunks.append(piece)
        buf.clear(); cur = 0
    for p in paras:
        if cur + len(p) + 2 <= max_chars:
            buf.append(p); cur += len(p) + 2
        else:
            flush(); buf.append(p); cur = len(p)
    flush()
    return chunks

def _discover_docs(in_dir: Path):
    docs = {}
    for f in sorted(in_dir.iterdir()):
        if not f.is_dir():
            continue
        nested = f / f.name
        md = list(f.glob("*.md")) + (list(nested.glob("*.md")) if nested.is_dir() else [])
        js = list(f.glob("*.json")) + (list(nested.glob("*.json")) if nested.is_dir() else [])
        jl = list(f.glob("*.jsonl")) + (list(nested.glob("*.jsonl")) if nested.is_dir() else [])
        if md or js or jl:
            docs[f.name] = {"md": sorted(md), "json": sorted(js), "jsonl": sorted(jl), "root": f}
    return docs

# ---- JSON table parsing (from 'html' field of Table blocks) ----
def _coerce_numbers_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            # remove thousands separators
            s = df[c].astype(str).str.replace(",", "", regex=False)
            # try numeric; keep strings where not numeric (as strings)
            num = pd.to_numeric(s, errors="coerce")
            df[c] = np.where(num.notna(), num, s)
    return df

def _extract_tables_from_marker_json_blocks(jtxt: str):
    """
    Parse Marker JSON and return a list of dicts with tables and their source page:
      [{"df": pandas.DataFrame, "page": int | None}, ...]
    We walk the block tree, track the nearest /page/{n}/ id, and attach it to table blocks.
    """
    try:
        data = json.loads(jtxt)
    except Exception:
        return []
    out: list[dict] = []

    def _page_from_id(node: dict, fallback: 'Optional[int]') -> 'Optional[int]':
        # prefer the node's own id; else fallback from parent context
        node_id = node.get("id") if isinstance(node.get("id"), str) else ""
        m = re.search(r"/page/(\d+)/", node_id or "")
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
        return fallback

    def walk(node, current_page: 'Optional[int]' = None):
        if isinstance(node, dict):
            current_page = _page_from_id(node, current_page)
            if node.get("block_type") == "Table" and isinstance(node.get("html"), str):
                html = node["html"]
                try:
                    dfs = pd.read_html(StringIO(html))
                    for df in dfs:
                        out.append({"df": _coerce_numbers_df(df), "page": current_page})
                except Exception:
                    pass
            # descend
            for v in node.values():
                walk(v, current_page)
        elif isinstance(node, list):
            for v in node:
                walk(v, current_page)

    walk(data)
    return out

# ---- NEW: Extract text spans with page numbers from Marker JSON ----
def _extract_text_spans_with_pages(jtxt: str):
    """
    Walk Marker JSON and yield per-page text spans from textual blocks.
    Returns list of dicts: [{"page": int | None, "text": str}, ...]
    """
    try:
        data = json.loads(jtxt)
    except Exception:
        return []

    spans: list[dict] = []

    def _page_from_id(node: dict, fallback: 'Optional[int]') -> 'Optional[int]':
        node_id = node.get("id") if isinstance(node.get("id"), str) else ""
        m = re.search(r"/page/(\d+)/", node_id or "")
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
        return fallback

    def _strip_html(s: str) -> str:
        s = re.sub(r"<[^>]+>", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    TEXT_BLOCKS = {"Text", "SectionHeader", "Paragraph", "Heading", "ListItem", "Caption", "Footer", "Header"}

    def walk(node, current_page: 'Optional[int]' = None):
        if isinstance(node, dict):
            current_page = _page_from_id(node, current_page)
            bt = node.get("block_type")
            if isinstance(bt, str) and bt in TEXT_BLOCKS:
                html = node.get("html")
                if isinstance(html, str) and html.strip():
                    txt = _strip_html(html)
                    if txt:
                        spans.append({"page": current_page, "text": txt})
            for v in node.values():
                walk(v, current_page)
        elif isinstance(node, list):
            for v in node:
                walk(v, current_page)

    walk(data)
    return spans

# --- NEW: Load JSONL files safely ---
def _load_jsonl(path: Path) -> list:
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rows.append(json.loads(s))
                except Exception:
                    continue
    except Exception:
        return []
    return rows

def _markdown_tables_find(md_text: str):
    lines = md_text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        if '|' in lines[i]:
            j = i + 1
            if j < n and re.search(r'^\s*\|?\s*:?-{3,}', lines[j]):
                k = j + 1
                while k < n and '|' in lines[k] and lines[k].strip():
                    k += 1
                yield "\n".join(lines[i:k])
                i = k; continue
        i += 1

def _markdown_table_to_df(table_md: str) -> pd.DataFrame | None:
    rows = [r.strip() for r in table_md.strip().splitlines() if r.strip()]
    if len(rows) < 2: return None
    def split_row(r: str):
        r = r.strip()
        if r.startswith('|'): r = r[1:]
        if r.endswith('|'): r = r[:-1]
        return [c.strip() for c in r.split('|')]
    cols = split_row(rows[0])
    if len(split_row(rows[1])) != len(cols): return None
    data = []
    for r in rows[2:]:
        cells = split_row(r)
        if len(cells) < len(cols): cells += [""] * (len(cols) - len(cells))
        if len(cells) > len(cols): cells = cells[:len(cols)]
        data.append(cells)
    try:
        df = pd.DataFrame(data, columns=cols)
        return _coerce_numbers_df(df)
    except Exception:
        return None

def _table_rows_to_sentences(df: pd.DataFrame, doc_name: str, table_id: int):
    sents = []
    if df.shape[1] == 0: return sents
    label = df.columns[0]
    for ridx, row in df.reset_index(drop=True).iterrows():
        parts = [str(row[label])]
        for c in df.columns[1:]:
            parts.append(f"{c}: {row[c]}")
        sents.append(f"[{doc_name}] table#{table_id} row#{ridx} :: " + " | ".join(parts))
    return sents

# --- Table signature for fuzzy matching Markdown tables to JSON tables ---
def _table_signature(df: pd.DataFrame) -> str:
    """
    Build a fuzzy signature for a table to match MD tables back to JSON tables.
    Uses: first-column header, set of year-like columns, and a few numeric cell samples.
    """
    try:
        cols = [str(c).strip() for c in df.columns]
        first_col = cols[0] if cols else ""
        # collect 4-digit year columns
        years = sorted({c for c in cols if re.fullmatch(r"\d{4}", str(c))})
        # flatten numeric values (best-effort) and take first 8
        nums = []
        for c in df.columns:
            s = pd.to_numeric(pd.Series(df[c]).astype(str).str.replace(",", "", regex=False), errors="coerce")
            vals = [float(x) for x in s.dropna().tolist()]
            nums.extend(vals)
        nums = [round(x, 3) for x in nums[:8]]
        return "|".join([
            f"first:{first_col.lower()}",
            "years:" + ",".join(years),
            "nums:" + ",".join(map(str, nums))
        ])
    except Exception:
        return ""

# ---- embeddings & index ----
def _encode(texts, model_name):
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(embs, dtype="float32")

def _build_faiss(embs):
    d = int(embs.shape[1])
    idx = faiss.IndexFlatIP(d)  # cosine via normalized inner product
    idx.add(embs)
    return idx

# ---------- main (notebook-friendly) ----------
def build_marker_kb_with_tables(
    in_dir="./All",
    out_dir="./data_marker",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    max_chars=1600,
    overlap=200,
):
    in_path, out_path = Path(in_dir), Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    kb_parquet     = out_path / "kb_chunks.parquet"
    kb_texts_npy   = out_path / "kb_texts.npy"
    kb_meta_json   = out_path / "kb_meta.json"
    kb_index_path  = out_path / "kb_index.faiss"
    kb_index_meta  = out_path / "kb_index_meta.json"
    kb_tables_parq = out_path / "kb_tables.parquet"
    kb_outline_parq = out_path / "kb_outline.parquet"

    cache = {}
    if kb_meta_json.exists():
        try:
            cache = json.loads(kb_meta_json.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    # Discover docs
    docs = _discover_docs(in_path)
    if not docs:
        raise RuntimeError(f"No Marker artefacts found under: {in_path}")
    print(f"🔎 Found {len(docs)} docs under {in_path}")

    # --- collect and persist Marker *_meta.json outlines for provenance/navigation ---
    outline_rows = []
    for doc_name, art in docs.items():
        root = art.get("root", in_path / doc_name)
        # look for "*_meta.json" in root and nested same-name subfolder
        candidates = list(root.glob("*_meta.json"))
        nested_same = root / doc_name
        if nested_same.is_dir():
            candidates += list(nested_same.glob("*_meta.json"))
        for meta_path in candidates:
            try:
                data = json.loads(_safe_read(meta_path))
                toc = data.get("table_of_contents") or data.get("toc") or []
                for i, item in enumerate(toc):
                    outline_rows.append({
                        "doc_name": doc_name,
                        "source_path": str(meta_path),
                        "order": int(i),
                        "title": item.get("title"),
                        "page_id": item.get("page_id"),
                        "polygon": item.get("polygon"),
                    })
            except Exception:
                # ignore malformed meta files
                pass
    if outline_rows:
        pd.DataFrame(outline_rows).to_parquet(kb_outline_parq, engine="pyarrow", index=False)
        print(f"📑 Saved outline → {kb_outline_parq} (rows={len(outline_rows)})")
    else:
        print("ℹ️ No *_meta.json outlines found.")

    # Track new chunks & long-form tables
    rows_meta, chunk_texts = [], []
    tables_long = []

    # map of table signature -> page number (from JSON-origin tables)
    json_table_sig_to_page: dict[str, int] = {}

    changed_any = False
    for name, art in tqdm(docs.items(), desc="Processing docs"):
        md_files, json_files = art["md"], art["json"]
        jsonl_files = art.get("jsonl", [])
        keys = [_file_hash_key(p) for p in (md_files + json_files + jsonl_files)]
        doc_key = hashlib.md5("|".join(keys).encode()).hexdigest()

        # If unchanged, skip reprocessing this doc
        if cache.get(name, {}).get("cache_key") == doc_key:
            continue
        changed_any = True

        # 1) JSON → tables + narrative text (with page numbers)
        table_id = 0
        for jp in json_files:
            jtxt = _safe_read(jp)

            # Tables via HTML blocks (with page capture)
            table_blocks = _extract_tables_from_marker_json_blocks(jtxt)
            for tb in table_blocks:
                df = tb["df"]
                page_no = tb.get("page")
                # record signature->page for later MD matching
                try:
                    sig = _table_signature(df)
                    if page_no is not None and sig:
                        json_table_sig_to_page[sig] = int(page_no)
                except Exception:
                    pass
                # row-sentences for retrieval (append a page hint to the sentence)
                for sent in _table_rows_to_sentences(df, name, table_id):
                    if page_no is not None:
                        sent = f"[page {page_no}] " + sent
                    rows_meta.append({
                        "doc": name,
                        "path": str(jp),
                        "modality": "table_row",
                        "chunk": len(chunk_texts),
                        "cache_key": doc_key,
                        "page": int(page_no) if page_no is not None else None,
                    })
                    chunk_texts.append(sent)
                # long-form cells for analytics
                for ridx, row in df.reset_index(drop=True).iterrows():
                    for col in df.columns:
                        _val = row[col]
                        _val_str = "" if pd.isna(_val) else str(_val)
                        try:
                            _val_num = pd.to_numeric(_val_str.replace(",", ""), errors="coerce")
                        except Exception:
                            _val_num = np.nan
                        tables_long.append({
                            "doc_name": name,
                            "source_path": str(jp),
                            "table_id": table_id,
                            "row_id": int(ridx),
                            "column": str(col),
                            "value_str": _val_str,
                            "value_num": float(_val_num) if pd.notna(_val_num) else None,
                            "page": int(page_no) if page_no is not None else None,
                        })
                table_id += 1

            # Narrative text per page
            spans = _extract_text_spans_with_pages(jtxt)
            # group by page and chunk each page separately
            by_page = {}
            for sp in spans:
                by_page.setdefault(sp.get("page"), []).append(sp["text"])
            for page_no, texts in by_page.items():
                page_text = _strip_md_basic("\n\n".join(texts))
                for i, ch in enumerate(_chunk_text(page_text, max_chars, overlap)):
                    rows_meta.append({
                        "doc": name,
                        "path": str(jp),
                        "modality": "json",
                        "chunk": len(chunk_texts),
                        "cache_key": doc_key,
                        "page": int(page_no) if page_no is not None else None,
                    })
                    chunk_texts.append(ch)

        # 1b) JSONL (extractor outputs) → rows + optional summary
        for jlp in jsonl_files:
            records = _load_jsonl(jlp)
            if not records:
                continue
            ctx = None
            data_recs = []
            for r in records:
                if isinstance(r, dict) and "_context" in r:
                    ctx = r.get("_context")
                elif isinstance(r, dict):
                    data_recs.append(r)

            page_no = None
            if isinstance(ctx, dict):
                p = ctx.get("page")
                if isinstance(p, int):
                    page_no = p

            df_jl = None
            if data_recs:
                try:
                    df_jl = pd.DataFrame(data_recs)
                    if "_meta" in df_jl.columns:
                        try:
                            df_jl = df_jl.drop(columns=["_meta"])  # purely metadata
                        except Exception:
                            pass
                    df_jl = _coerce_numbers_df(df_jl)
                except Exception:
                    df_jl = None

            if df_jl is not None and not df_jl.empty:
                # Emit retrieval sentences for each row
                for sent in _table_rows_to_sentences(df_jl, name, table_id):
                    if page_no is not None:
                        sent = f"[page {page_no}] " + sent
                    rows_meta.append({
                        "doc": name,
                        "path": str(jlp),
                        "modality": "jsonl_row",
                        "chunk": len(chunk_texts),
                        "cache_key": doc_key,
                        "page": page_no,
                    })
                    chunk_texts.append(sent)

                # Persist long-form for analytics
                for ridx, row in df_jl.reset_index(drop=True).iterrows():
                    for col in df_jl.columns:
                        _val = row[col]
                        _val_str = "" if pd.isna(_val) else str(_val)
                        try:
                            _val_num = pd.to_numeric(_val_str.replace(",", ""), errors="coerce")
                        except Exception:
                            _val_num = np.nan
                        tables_long.append({
                            "doc_name": name,
                            "source_path": str(jlp),
                            "table_id": table_id,
                            "row_id": int(ridx),
                            "column": str(col),
                            "value_str": _val_str,
                            "value_num": float(_val_num) if pd.notna(_val_num) else None,
                            "page": page_no,
                        })
                table_id += 1

            # Also add a concise summary sentence if available in context
            if isinstance(ctx, dict) and isinstance(ctx.get("summary"), str) and ctx["summary"].strip():
                rows_meta.append({
                    "doc": name,
                    "path": str(jlp),
                    "modality": "jsonl_summary",
                    "chunk": len(chunk_texts),
                    "cache_key": doc_key,
                    "page": page_no,
                })
                chunk_texts.append(f"[{name}] {ctx['summary'].strip()}")

        # 2) Markdown → tables + non-table text
        for mp in md_files:
            md = _safe_read(mp)

            # tables from MD
            for tblock in _markdown_tables_find(md):
                df = _markdown_table_to_df(tblock)
                if df is None: 
                    continue
                # try to infer page by matching this MD table to a JSON table signature
                md_page = None
                try:
                    md_sig = _table_signature(df)
                    if md_sig and md_sig in json_table_sig_to_page:
                        md_page = int(json_table_sig_to_page[md_sig])
                except Exception:
                    md_page = None

                for sent in _table_rows_to_sentences(df, name, table_id):
                    rows_meta.append({
                        "doc": name,
                        "path": str(mp),
                        "modality": "table_row",
                        "chunk": len(chunk_texts),
                        "cache_key": doc_key,
                        "page": md_page  # may be None if unmatched
                    })
                    chunk_texts.append(sent)
                for ridx, row in df.reset_index(drop=True).iterrows():
                    for col in df.columns:
                        _val = row[col]
                        _val_str = "" if pd.isna(_val) else str(_val)
                        try:
                            _val_num = pd.to_numeric(_val_str.replace(",", ""), errors="coerce")
                        except Exception:
                            _val_num = np.nan
                        tables_long.append({
                            "doc_name": name,
                            "source_path": str(jp if "jp" in locals() else mp),
                            "table_id": table_id,
                            "row_id": int(ridx),
                            "column": str(col),
                            "value_str": _val_str,
                            "value_num": float(_val_num) if pd.notna(_val_num) else None,
                            "page": md_page,  # keep None if not found
                        })
                table_id += 1

            # non-table text (remove table blocks first to avoid dupes)
            md_no_tables = md
            for tblock in _markdown_tables_find(md):
                md_no_tables = md_no_tables.replace(tblock, "")
            for i, ch in enumerate(_chunk_text(_strip_md_basic(md_no_tables), max_chars, overlap)):
                rows_meta.append({"doc": name, "path": str(mp), "modality": "md", "chunk": len(chunk_texts), "cache_key": doc_key, "page": None})
                chunk_texts.append(ch)

        # update per-doc cache (count new chunks we just added for this doc)
        added_for_doc = sum(1 for r in rows_meta if r["cache_key"] == doc_key)
        cache[name] = {"cache_key": doc_key, "chunk_count": added_for_doc, "updated_at": int(time.time())}

    # If nothing changed and KB exists → keep existing artifacts
    if not changed_any and kb_parquet.exists() and kb_texts_npy.exists() and kb_index_path.exists():
        print("✅ No changes detected. Keeping existing KB and FAISS index.")
        df_existing = pd.read_parquet(kb_parquet)
        texts_existing = np.load(kb_texts_npy, allow_pickle=True)
        return {
            "docs_processed": len(docs),
            "chunks_total": int(len(texts_existing)),
            "tables_long_rows": (pd.read_parquet(kb_tables_parq).shape[0] if kb_tables_parq.exists() else 0),
            "paths": {
                "kb_chunks_parquet": str(kb_parquet),
                "kb_texts_npy": str(kb_texts_npy),
                "kb_meta_json": str(kb_meta_json),
                "kb_tables_parquet": str(kb_tables_parq) if kb_tables_parq.exists() else None,
                "kb_outline_parquet": str(kb_outline_parq) if kb_outline_parq.exists() else None,
                "kb_index_faiss": str(kb_index_path),
                "kb_index_meta_json": str(kb_index_meta),
            }
        }

    # Persist KB + tables
    total = len(chunk_texts)
    print(f"🧾 Total new/updated text chunks (incl. table rows): {total}")
    df = pd.DataFrame(rows_meta)
    np.save(kb_texts_npy, np.array(chunk_texts, dtype=object))
    df.to_parquet(kb_parquet, engine="pyarrow", index=False)
    pd.DataFrame(tables_long).to_parquet(kb_tables_parq, engine="pyarrow", index=False) if tables_long else None
    kb_meta_json.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    if tables_long:
        print(f"📑 Saved structured tables → {kb_tables_parq} (rows={len(tables_long)})")
    else:
        print("📑 No structured tables detected this run.")

    if total == 0:
        print("⚠️ No new chunks produced. Skipping embedding/index rebuild.")
        return {
            "docs_processed": len(docs),
            "chunks_total": int(pd.read_parquet(kb_parquet).shape[0]),
            "tables_long_rows": (pd.read_parquet(kb_tables_parq).shape[0] if kb_tables_parq.exists() else 0),
            "paths": {
                "kb_chunks_parquet": str(kb_parquet),
                "kb_texts_npy": str(kb_texts_npy),
                "kb_meta_json": str(kb_meta_json),
                "kb_tables_parquet": str(kb_tables_parq) if kb_tables_parq.exists() else None,
                "kb_outline_parquet": str(kb_outline_parq) if kb_outline_parq.exists() else None,
                "kb_index_faiss": str(kb_index_path) if kb_index_path.exists() else None,
                "kb_index_meta_json": str(kb_index_meta) if kb_index_meta.exists() else None,
            }
        }

    # Embeddings + FAISS
    print("🧠 Encoding embeddings …")
    embs = _encode(chunk_texts, model_name)
    print(f"✅ Embeddings shape: {embs.shape}")

    print("📦 Building FAISS index …")
    idx = _build_faiss(embs)
    faiss.write_index(idx, str(kb_index_path))
    kb_index_meta.write_text(json.dumps({
        "model": model_name,
        "dim": int(embs.shape[1]),
        "total_vectors": int(embs.shape[0]),
        "metric": "cosine (via inner product on normalized vectors)"
    }, indent=2), encoding="utf-8")

    print(f"🎉 Done. KB + index saved to: {out_path}")
    return {
        "docs_processed": len(docs),
        "chunks_total": int(total),
        "tables_long_rows": len(tables_long),
        "paths": {
            "kb_chunks_parquet": str(kb_parquet),
            "kb_texts_npy": str(kb_texts_npy),
            "kb_meta_json": str(kb_meta_json),
            "kb_tables_parquet": str(kb_tables_parq) if tables_long else None,
            "kb_outline_parquet": str(kb_outline_parq),
            "kb_index_faiss": str(kb_index_path),
            "kb_index_meta_json": str(kb_index_meta),
        }
    }

# ▶ Run now (edit paths if needed)
summary = build_marker_kb_with_tables()
summary