from __future__ import annotations

"""
Stage1.py — Ingestion Pipeline

Builds a Knowledge Base (KB) + Vector Store with metadata.
"""
import os, re, json, uuid, pathlib
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np

# --- optional deps ---
try:
    import faiss
    _HAVE_FAISS = True
except Exception:
    _HAVE_FAISS = False

try:
    from pypdf import PdfReader
    _HAVE_PDF = True
except Exception:
    _HAVE_PDF = False

DATA_DIR = os.environ.get("AGENT_CFO_DATA_DIR", "All")
OUT_DIR = os.environ.get("AGENT_CFO_OUT_DIR", "data")
EMBED_BACKEND = os.environ.get("AGENT_CFO_EMBED_BACKEND", "st")
CHUNK_TOKENS = 450
CHUNK_OVERLAP = 80

pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# --- Utilities & Constants ---
_YEAR_PAT = re.compile(r"\b(20\d{2})\b")
_Q_PAT = re.compile(r"([1-4])Q(\d{2})", re.I)
_FY_PAT = re.compile(r"\bFY\s?(20\d{2})\b", re.I)
_QY_PAT_1 = re.compile(r"\b([1-4])\s*Q\s*(20\d{2}|\d{2})\b", re.I)
_QY_PAT_2 = re.compile(r"\bQ\s*([1-4])\s*(20\d{2}|\d{2})\b", re.I)
_QY_PAT_3 = re.compile(r"\b([1-4])Q\s*(20\d{2}|\d{2})\b", re.I)
_FY_PAT_2 = re.compile(r"\bF[Yy]\s*(20\d{2})\b")

MAX_TABLE_WINDOWS_PER_PAGE = 3
DEFAULT_WINDOW_LINES = 18

SHEET_SECTION_PATTERNS = [
    (r"^\s*(?:1\.)?\s*highlights\b|^highlights$", "highlights/summary"),
    (r"expenses|opex", "Operating expenses (Opex)"),
    (r"net\s*interest", "Net interest income"),
    (r"non[- ]?interest|fee|commission", "Non-interest/fee income"),
    (r"cost\s*[-/ ]?to\s*[-/ ]?income|\bcti\b", "Cost-to-income (CTI)"),
    (r"npl|coverage\s+ratios", "Asset quality (NPL)"),
    (r"loans", "Loans"), (r"deposits", "Deposits"), (r"capital|cet\s*1", "Capital & CET1"),
    (r"return\s+on\s+equity|\broe\b", "Returns (ROE/ROA)"), (r"profit|pbt|pat", "Profit"),
]

def sheet_section_label(sheet_name: Optional[str]) -> Optional[str]:
    s = (sheet_name or "").strip()
    if not s: return None
    for pat, label in SHEET_SECTION_PATTERNS:
        if re.search(pat, s, flags=re.IGNORECASE): return label
    return None

def infer_period_from_text(text: str, filename_year: Optional[int] = None) -> Tuple[Optional[int], Optional[int]]:
    if not text: return (None, None)
    head = "\n".join([ln.strip() for ln in text.splitlines() if ln.strip()][:8])
    candidates: list[tuple[int, int]] = []
    for pat in (_QY_PAT_1, _QY_PAT_2, _QY_PAT_3):
        for m in pat.finditer(head):
            q, yy_str = int(m.group(1)), m.group(2)
            y = int(yy_str)
            if y < 100: y = 2000 + y
            candidates.append((q, y))
    if candidates:
        if filename_year is not None:
            same_year = [c for c in candidates if c[1] == filename_year]
            if same_year: return (filename_year, same_year[0][0])
        q, y = max(candidates, key=lambda t: t[1])
        return (y, q)
    m = _FY_PAT_2.search(head)
    if m: return (int(m.group(1)), None)
    return (None, None)

_KEY_TABLE_SPECS = [
    (re.compile(r"\bnet\s*interest\s*margin\b|\bnim\b", re.I), "NIM table"),
    (re.compile(r"\b(total|operating)\s+income\b", re.I), "Total/Operating income"),
    (re.compile(r"\b(operating|staff|other)?\s*expenses\b|\bopex\b|\bcosts?\b", re.I), "Opex table"),
    (re.compile(r"cost\s*[/\-\–_]?\s*to?\s*income(\s*ratio)?|cost\s*/\s*income|\bcti\b", re.I), "CTI table"),
]

def extract_key_tables_from_page(text: str) -> List[Tuple[str, str]]:
    if not text: return []
    text = re.sub(r"\s+", " ", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out: List[Tuple[str, str]] = []
    for i, ln in enumerate(lines):
        for pat, label in _KEY_TABLE_SPECS:
            if pat.search(ln):
                start, end = max(0, i - 8), min(len(lines), i + DEFAULT_WINDOW_LINES)
                out.append((label, "\n".join(lines[start:end]))); break
    return out

_TABULAR_EXTS = {'.csv', '.xls', '.xlsx'}
def _is_pdf(path: str) -> bool: return str(path).lower().endswith('.pdf')
def _is_tabular(path: str) -> bool: return any(str(path).lower().endswith(ext) for ext in _TABULAR_EXTS)

def infer_period_from_filename(fname: str) -> Tuple[Optional[int], Optional[int]]:
    base = fname.upper()
    m = _Q_PAT.search(base)
    if m:
        q, yy = int(m.group(1)), int(m.group(2))
        return (2000 + yy if yy < 100 else yy, q)
    m = _YEAR_PAT.search(base)
    if m: return (int(m.group(1)), None)
    m = _FY_PAT.search(base)
    if m: return (int(m.group(1)), None)
    return (None, None)

def _split_text(text: str) -> List[str]:
    text = (text or "").strip()
    if not text: return []
    chunk_size, overlap = 1800, 320
    out, i, n = [], 0, len(text)
    while i < n:
        j = min(n, i + chunk_size)
        out.append(text[i:j])
        if j == n: break
        i = max(i + chunk_size - overlap, j)
    return out

def extract_pdf_pages(path: str) -> List[Tuple[int, str]]:
    if not _HAVE_PDF: raise RuntimeError("pypdf not installed. pip install pypdf")
    reader = PdfReader(path)
    return [(i, p.extract_text() or "") for i, p in enumerate(reader.pages, 1)]

def _df_to_blocks(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty: return []
    df = df.dropna(axis=1, how='all').astype(str)
    return [df.iloc[i:i+40].to_csv(index=False) for i in range(0, len(df), 40)]

def extract_tabular_chunks(path: str) -> List[Tuple[str, Optional[str]]]:
    base = os.path.basename(path)
    try:
        lower = path.lower()
        if lower.endswith('.csv'):
            df = pd.read_csv(path, low_memory=False, dtype=object)
            print(f"          → CSV parsed: shape={df.shape}")
            return [(block, 'CSV') for block in _df_to_blocks(df)]
        engine = 'openpyxl' if lower.endswith('.xlsx') else ('xlrd' if lower.endswith('.xls') else None)
        xl = pd.ExcelFile(path, engine=engine)
        out = []
        for sheet in xl.sheet_names:
            df = xl.parse(sheet, dtype=object)
            print(f"          → Excel sheet parsed '{sheet}': shape={df.shape}")
            for block in _df_to_blocks(df): out.append((block, sheet))
        return out
    except Exception as e:
        print(f"          → ⚠️ WARNING: Parse failed for {base}: {e}")
        return []

def pick_provider() -> Tuple[Any, str]:  # Simplified EmbeddingProvider
    from sentence_transformers import SentenceTransformer
    model_name = "sentence-transformers/all-MiniLM-L12-v2"
    return SentenceTransformer(model_name), model_name

def walk_all_docs(root: str) -> List[str]:
    paths = []
    for p in pathlib.Path(root).rglob("*"):
        if p.is_file() and (_is_pdf(str(p)) or _is_tabular(str(p))):
            paths.append(str(p))
    return sorted(paths)

def build_kb() -> Dict[str, Any]:
    docs = walk_all_docs(DATA_DIR)
    print(f"[Stage1] Scanning folder: {DATA_DIR} → found {len(docs)} document(s)")
    if not docs: raise SystemExit(f"No documents found under {DATA_DIR}.")

    rows, texts = [], []
    for path in docs:
        fname = os.path.basename(path)
        print(f"[Stage1] Processing: {fname}")
        year, quarter = infer_period_from_filename(fname)
        print(f"          → Period (filename): year={year or 'NULL'}, quarter={quarter or 'NULL'}")
        
        if _is_pdf(path):
            pages = extract_pdf_pages(path)
            print(f"          → Pages detected: {len(pages)}")
            for page_num, page_text in pages:
                if not page_text.strip(): continue
                for chunk_text in _split_text(page_text):
                    rows.append({"doc_id": str(uuid.uuid4()), "file": fname, "page": page_num, "year": year, "quarter": quarter, "section_hint": None})
                    texts.append(chunk_text)
            
            for page_num, page_text in pages:
                final_year, final_quarter = year, quarter
                page_year, page_quarter = infer_period_from_text(page_text, filename_year=year)
                if final_quarter is None and page_quarter is not None:
                    if page_year == final_year: final_quarter = page_quarter
                    elif final_year is None: final_year, final_quarter = page_year, page_quarter
                
                for label, block in extract_key_tables_from_page(page_text):
                    rows.append({"doc_id": str(uuid.uuid4()), "file": fname, "page": page_num, "year": final_year, "quarter": final_quarter, "section_hint": label})
                    texts.append(block)

        elif _is_tabular(path):
            blocks = extract_tabular_chunks(path)
            if not blocks:
                print(f"          → WARNING: No content extracted from table: {fname}")
            else:
                print(f"          → Table blocks: {len(blocks)}")
            for block_text, sheet in blocks:
                section_hint = sheet_section_label(sheet) or f"table:{sheet}"
                rows.append({"doc_id": str(uuid.uuid4()), "file": fname, "page": 1, "year": year, "quarter": quarter, "section_hint": section_hint})
                texts.append(block_text)
        print(f"[Stage1] Done: {fname}")

    print(f"[Stage1] Total raw chunks prepared: {len(texts)}")
    kb = pd.DataFrame(rows)

    # --- Ingestion Reconciliation Report ---
    print("\n" + "-"*50)
    print("[Stage1] Final Ingestion Reconciliation Report")
    print("-"*50)
    discovered_files = {os.path.basename(p) for p in docs}
    indexed_files = set(kb['file'].unique()) if not kb.empty else set()
    missing_files = discovered_files - indexed_files
    print(f"  - Documents Discovered: {len(discovered_files)}")
    print(f"  - Documents Indexed:    {len(indexed_files)}")
    print(f"  - Unindexed / Empty:    {len(missing_files)}")
    if missing_files:
        print("\n  [ATTENTION] The following files were NOT indexed (likely empty or parse failure):")
        for fname in sorted(list(missing_files)): print(f"    - {fname}")
    else:
        print("\n  ✅ All discovered documents were successfully indexed.")
    print("-"*50)
    
    # --- Per-File Period Tagging Verification ---
    print("\n" + "-"*50)
    print("[Stage1] Per-File Period Tagging Verification Report")
    print("-"*50)
    if not kb.empty:
        for fname in sorted(list(indexed_files)):
            year_fn, quarter_fn = infer_period_from_filename(fname)
            expected_str = f"Y={year_fn or 'N/A'}, Q={quarter_fn or 'N/A'}"
            file_df = kb[kb['file'] == fname]
            stored_periods = {(int(y) if pd.notna(y) else None, int(q) if pd.notna(q) else None)
                              for y, q in file_df[['year', 'quarter']].drop_duplicates().to_numpy()}
            stored_str = "; ".join([f"Y={p[0] or 'N/A'}, Q={p[1] or 'N/A'}" for p in stored_periods])
            status = ""
            if len(stored_periods) > 1:
                status = "⚠️ INCONSISTENT (Multiple periods tagged for one file)"
            elif len(stored_periods) == 1:
                y_s, q_s = list(stored_periods)[0]
                if y_s == year_fn and q_s == quarter_fn: status = "✅ OK"
                elif y_s == year_fn and quarter_fn is None and q_s is not None: status = "✅ OK (ENHANCED)"
                else: status = "⚠️ MISMATCH (Stored period conflicts with filename)"
            print(f"📄 File: {fname}\n   - Expected: {expected_str}\n   - Stored:   {stored_str}\n   - Status:   {status}\n" + "-"*25)
    print("-"*50 + "\n")
    
    if kb.empty: raise SystemExit("No data was indexed. Halting before embedding.")

    provider, provider_name = pick_provider()
    # The provider is now the all-mpnet-base-v2 model
    print(f"[Stage1] Embedding with model: {provider_name}")
    vecs = provider.encode(texts, normalize_embeddings=True).astype(np.float32)
    dim = provider.get_sentence_embedding_dimension()
    print(f"[Stage1] Embedded {vecs.shape[0]} chunks (dim={dim})")

    if not _HAVE_FAISS: raise SystemExit("faiss not installed. pip install faiss-cpu")
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    print(f"[Stage1] FAISS index size: {index.ntotal}")

    kb_path, text_path, index_path, meta_path = [os.path.join(OUT_DIR, f) for f in ["kb_chunks.parquet", "kb_texts.npy", "kb_index.faiss", "kb_meta.json"]]
    kb.to_parquet(kb_path, engine='pyarrow', index=False)
    np.save(text_path, np.array(texts, dtype=object))
    faiss.write_index(index, index_path)
    # Also update the meta file to reflect the new model if necessary
    with open(meta_path, "w") as f: json.dump({"embedding_provider": f"st:{provider_name}", "dim": dim}, f)

    print(f"Saved KB: {len(kb)} rows → {kb_path}")
    return {"kb": kb_path, "texts": text_path, "index": index_path, "meta": meta_path}

if __name__ == "__main__":
    build_kb()