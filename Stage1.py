"""
Stage1.py — Ingestion Pipeline

Builds a Knowledge Base (KB) + Vector Store with metadata.
Outputs:
  - data/kb_chunks.parquet      # canonical KB with metadata per chunk
  - data/kb_texts.npy           # chunk texts (parallel array)
  - data/kb_index.faiss         # FAISS index of embeddings
  - data/kb_meta.json           # small meta: embedding dim, model, version

Environment (optional):
  OPENAI_API_KEY    — for text-embedding-3-large or 3-small
  GEMINI_API_KEY    — for gemini-embedding text-002 (if you prefer)

You can also use local SentenceTransformers if installed.
"""
from __future__ import annotations
import os, re, json, math, uuid, pathlib, warnings
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Iterable, Tuple

import pandas as pd
import numpy as np

# --- optional deps ---
try:
    import faiss  # type: ignore
    _HAVE_FAISS = True
except Exception:
    _HAVE_FAISS = False

try:
    from rank_bm25 import BM25Okapi  # lightweight BM25 for hybrid
    _HAVE_BM25 = True
except Exception:
    _HAVE_BM25 = False

# PDF text extraction (pypdf) — optional
try:
    from pypdf import PdfReader  # minimal + reliable
    _HAVE_PDF = True
except Exception:
    _HAVE_PDF = False

# Embeddings backends (we'll load lazily in Provider)


DATA_DIR = os.environ.get("AGENT_CFO_DATA_DIR", "All")
OUT_DIR = os.environ.get("AGENT_CFO_OUT_DIR", "data")
EMBED_BACKEND = os.environ.get("AGENT_CFO_EMBED_BACKEND", "auto")  # 'auto', 'openai', 'gemini', 'st'
CHUNK_TOKENS = 450  # ~sentence-y chunks; we chunk by chars but aim for this size
CHUNK_OVERLAP = 80

pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# -----------------------------
# Utilities
# -----------------------------

_YEAR_PAT = re.compile(r"\b(20\d{2})\b")
_Q_PAT = re.compile(r"([1-4])Q(\d{2})", re.I)  # e.g., 3Q24 (relaxed, allows underscores etc.)
_FY_PAT = re.compile(r"\bFY\s?(20\d{2})\b", re.I)

# Additional period patterns found in page headers
_QY_PAT_1 = re.compile(r"\b([1-4])\s*Q\s*(20\d{2}|\d{2})\b", re.I)   # e.g., 1 Q 2025, 2Q24
_QY_PAT_2 = re.compile(r"\bQ\s*([1-4])\s*(20\d{2}|\d{2})\b", re.I)     # e.g., Q3 2024
_QY_PAT_3 = re.compile(r"\b([1-4])Q\s*(20\d{2}|\d{2})\b", re.I)        # e.g., 3Q 2024
_FY_PAT_2 = re.compile(r"\bF[Yy]\s*(20\d{2})\b")


def infer_period_from_text(text: str) -> Tuple[Optional[int], Optional[int]]:
    """Try to infer (year, quarter) from page text (headers/footers).
    Rules:
    - Prefer explicit quarter-year patterns (1Q25, Q3 2024, 3Q 2024).
    - Accept FY headers (FY2024) as (2024, None).
    - Ignore lone years to avoid picking up copyright years (e.g., © 2023).
    """
    if not text:
        return (None, None)
    s = text[:500]  # scan a bit more of the header area
    # 1) Explicit quarter-year first
    for pat in (_QY_PAT_1, _QY_PAT_2, _QY_PAT_3):
        m = pat.search(s)
        if m:
            q = int(m.group(1))
            yy = int(m.group(2))
            y = 2000 + yy if yy < 100 else yy
            return (y, q)
    # 2) FY header
    m = _FY_PAT_2.search(s)
    if m:
        return (int(m.group(1)), None)
    # 3) Ignore bare years (too noisy: copyright, footers, etc.)
    return (None, None)
# -----------------------------
# Lightweight table extractor (keywords windows)
# -----------------------------

_KEY_TABLE_SPECS = [
    (re.compile(r"net\s+interest\s+margin|\bnim\b", re.I), "NIM table"),
    (re.compile(r"operating\s+expenses|\bopex\b|staff\s+costs", re.I), "Opex table"),
    (re.compile(r"cost[- ]?to[- ]?income|\bcti\b|efficiency\s+ratio", re.I), "CTI table"),
]

def extract_key_tables_from_page(text: str, window_lines: int = 18) -> List[Tuple[str, str]]:
    """Find small windows around key table keywords and return blocks.
    Returns list of (section_hint, block_text).
    """
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out: List[Tuple[str, str]] = []
    for i, ln in enumerate(lines):
        for pat, label in _KEY_TABLE_SPECS:
            if pat.search(ln):
                start = max(0, i - 2)
                end = min(len(lines), i + window_lines)
                block = "\n".join(lines[start:end])
                out.append((label, block))
                break
    return out

SECTION_LABELS = {
    r"key ratios|highlights|summary": "highlights/summary",
    r"net interest margin|nim\b": "Net interest margin (NIM)",
    r"cost[- ]?to[- ]?income|cti|efficiency ratio": "Cost-to-income (CTI)",
    r"operating expenses|opex|expenses": "Operating expenses (Opex)",
    r"income statement|statement of (comprehensive )?income": "Income statement",
    r"balance sheet|statement of financial position": "Balance sheet",
    r"management discussion|md&a": "MD&A",
}

_TABULAR_EXTS = {'.csv', '.xls', '.xlsx'}

def _is_pdf(path: str) -> bool:
    return str(path).lower().endswith('.pdf')

def _is_tabular(path: str) -> bool:
    return any(str(path).lower().endswith(ext) for ext in _TABULAR_EXTS)


def infer_period_from_filename(fname: str) -> Tuple[Optional[int], Optional[int]]:
    """Infer (year, quarter) from common file naming conventions.
    Examples: DBS_3Q24_CFO_Presentation.pdf -> (2024, 3)
              dbs-annual-report-2023.pdf    -> (2023, None)
    """
    base = fname.upper()
    m = _Q_PAT.search(base)
    if m:
        q = int(m.group(1))
        yy = int(m.group(2))
        year = 2000 + yy if yy < 100 else yy
        return (year, q)
    m = _YEAR_PAT.search(base)
    if m:
        return (int(m.group(1)), None)
    m = _FY_PAT.search(base)
    if m:
        return (int(m.group(1)), None)
    return (None, None)


def clean_section_hint(text: str) -> Optional[str]:
    # naive regex scan to tag common sections; optional
    for pat, label in SECTION_LABELS.items():
        if re.search(pat, text, flags=re.IGNORECASE):
            return label
    return None


# -----------------------------
# Chunking
# -----------------------------

def _split_text(text: str, chunk_size_chars: int = 1800, overlap_chars: int = 320) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_size_chars)
        out.append(text[i:j])
        if j == n:
            break
        i = max(i + chunk_size_chars - overlap_chars, j)  # ensure progress
    return out


# -----------------------------
# PDF parsing
# -----------------------------

def extract_pdf_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """Return list of (page_number_1based, text)."""
    if not _HAVE_PDF:
        raise RuntimeError("pypdf not installed. pip install pypdf")
    reader = PdfReader(pdf_path)
    out = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        out.append((i, txt))
    return out


# -----------------------------
# Tabular (CSV/Excel) parsing
# -----------------------------

def _df_to_blocks(df: pd.DataFrame, rows_per_block: int = 40) -> List[str]:
    """Split a DataFrame into row blocks and render each as a compact CSV string.
    Keeps headers on each block for standalone readability.
    """
    if df is None or df.empty:
        return []
    # Drop all-empty columns
    df = df.dropna(axis=1, how='all')
    # Convert everything to string to prevent pyarrow dtype issues downstream
    df = df.astype(str)
    blocks = []
    n = len(df)
    for i in range(0, n, rows_per_block):
        part = df.iloc[i:i+rows_per_block]
        csv_str = part.to_csv(index=False)
        blocks.append(csv_str)
    return blocks


def extract_tabular_chunks(path: str) -> List[Tuple[str, Optional[str]]]:
    """Return a list of (block_text, sheet_name) for CSV/Excel files.
    For CSV → one sheet named 'CSV'. For Excel → one per sheet.
    """
    out: List[Tuple[str, Optional[str]]] = []
    lower = path.lower()
    try:
        if lower.endswith('.csv'):
            df = pd.read_csv(path, low_memory=False)
            for block in _df_to_blocks(df):
                out.append((block, 'CSV'))
        else:
            # Excel: iterate sheets safely
            xl = pd.ExcelFile(path)
            for sheet in xl.sheet_names:
                try:
                    df = xl.parse(sheet)
                except Exception:
                    continue
                for block in _df_to_blocks(df):
                    out.append((block, sheet))
    except Exception:
        # If any parsing error, skip gracefully
        return []
    return out


# -----------------------------
# Embedding providers
# -----------------------------
class EmbeddingProvider:
    name: str = ""
    dim: int = 0
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class OpenAIProvider(EmbeddingProvider):
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI  # requires OPENAI_API_KEY
        self.client = OpenAI()
        self.model = model
        # dims: 3-small=1536, 3-large=3072
        self.dim = 1536 if "small" in model else 3072
        self.name = f"openai:{model}"
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        resp = self.client.embeddings.create(model=self.model, input=texts)
        vecs = [d.embedding for d in resp.data]
        return np.asarray(vecs, dtype=np.float32)


class STProvider(EmbeddingProvider):
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional
        self.model_name = model
        self.model = SentenceTransformer(model)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.name = f"st:{model}"
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)


def pick_provider(backend: str = EMBED_BACKEND) -> EmbeddingProvider:
    """Pick embedding provider based on argument or environment variable.
    backend can be 'auto', 'openai', 'gemini', or 'st'.
    Auto-detect priority: OpenAI → Gemini → SentenceTransformers."""
    backend = (backend or 'auto').lower()

    # --- Explicit backend ---
    if backend == 'openai':
        return OpenAIProvider('text-embedding-3-small')
    elif backend == 'st' or backend == 'sentence-transformers':
        return STProvider('sentence-transformers/all-MiniLM-L6-v2')
    elif backend == 'gemini':
        try:
            from google import generativeai as genai
            key = os.environ.get('GEMINI_API_KEY')
            if not key:
                raise RuntimeError('GEMINI_API_KEY not set')
            genai.configure(api_key=key)
            class GeminiProvider(EmbeddingProvider):
                def __init__(self):
                    self.name = 'gemini:embedding-001'
                    self.dim = 0  # default size unknown initially
                def embed_batch(self, texts: List[str]) -> np.ndarray:
                    vecs = []
                    for t in texts:
                        resp = genai.embed_content(model='models/embedding-001', content=t)
                        emb = resp.get('embedding') if isinstance(resp, dict) else getattr(resp, 'embedding', None)
                        if emb is None:
                            raise RuntimeError('Gemini embed_content returned no embedding')
                        vecs.append(emb)
                    arr = np.asarray(vecs, dtype=np.float32)
                    if self.dim == 0 and arr.size:
                        self.dim = int(arr.shape[1])
                    return arr
            return GeminiProvider()
        except Exception as e:
            warnings.warn(f'Gemini provider init failed: {e}')

    # --- Auto detection ---
    if os.environ.get('OPENAI_API_KEY'):
        try:
            return OpenAIProvider('text-embedding-3-small')
        except Exception as e:
            warnings.warn(f'OpenAI provider init failed: {e}')
    if os.environ.get('GEMINI_API_KEY'):
        try:
            from google import generativeai as genai
            genai.configure(api_key=os.environ['GEMINI_API_KEY'])
            class GeminiProvider(EmbeddingProvider):
                def __init__(self):
                    self.name = 'gemini:embedding-001'
                    self.dim = 0
                def embed_batch(self, texts: List[str]) -> np.ndarray:
                    vecs = []
                    for t in texts:
                        resp = genai.embed_content(model='models/embedding-001', content=t)
                        emb = resp.get('embedding') if isinstance(resp, dict) else getattr(resp, 'embedding', None)
                        if emb is None:
                            raise RuntimeError('Gemini embed_content returned no embedding')
                        vecs.append(emb)
                    arr = np.asarray(vecs, dtype=np.float32)
                    if self.dim == 0 and arr.size:
                        self.dim = int(arr.shape[1])
                    return arr
            return GeminiProvider()
        except Exception as e:
            warnings.warn(f'Gemini provider init failed: {e}')
    try:
        return STProvider('sentence-transformers/all-MiniLM-L6-v2')
    except Exception as e:
        raise SystemExit(f'No embedding backend available. Install sentence-transformers or set an API key. {e}')


# -----------------------------
# Safe Parquet save with dtype sanitization
# -----------------------------

def _sanitize_and_save_parquet(df: pd.DataFrame, path: str) -> None:
    """Sanitize dtypes and save to Parquet, with fallbacks.
    - Forces primitive/nullable dtypes that are parquet-friendly
    - Tries pyarrow → fastparquet → CSV fallback
    """
    d = df.copy()
    # Standardize dtypes
    if 'doc_id' in d:
        d['doc_id'] = d['doc_id'].astype('string')
    if 'file' in d:
        d['file'] = d['file'].astype('string')
    if 'section_hint' in d:
        d['section_hint'] = d['section_hint'].astype('string')
    if 'page' in d:
        d['page'] = pd.to_numeric(d['page'], errors='coerce').fillna(0).astype('int32')
    if 'year' in d:
        # nullable small int for compactness
        d['year'] = pd.to_numeric(d['year'], errors='coerce').astype('Int16')
    if 'quarter' in d:
        d['quarter'] = pd.to_numeric(d['quarter'], errors='coerce').astype('Int8')

    # Try engines in order
    errors = []
    for engine in ('pyarrow', 'fastparquet'):
        try:
            d.to_parquet(path, engine=engine, index=False)
            return
        except Exception as e:
            errors.append(f"{engine}: {e}")
    # Final CSV fallback
    csv_path = os.path.splitext(path)[0] + '.csv'
    d.to_csv(csv_path, index=False)
    raise RuntimeError(
        "Failed to save Parquet with both pyarrow and fastparquet. "
        f"Wrote CSV fallback at {csv_path}. Errors: {' | '.join(errors)}"
    )


# -----------------------------
# Main ingest
# -----------------------------
@dataclass
class Chunk:
    doc_id: str
    file: str
    page: int
    year: Optional[int]
    quarter: Optional[int]
    section_hint: Optional[str]
    text: str


def walk_pdfs(root: str) -> List[str]:
    # Kept for backward compatibility (returns only PDFs)
    files = []
    for p in pathlib.Path(root).rglob("*.pdf"):
        files.append(str(p))
    return sorted(files)


def walk_all_docs(root: str) -> List[str]:
    """Return PDFs + CSV + Excel paths under root."""
    paths: List[str] = []
    for p in pathlib.Path(root).rglob("*"):
        if not p.is_file():
            continue
        s = str(p)
        if _is_pdf(s) or _is_tabular(s):
            paths.append(s)
    return sorted(paths)


def build_kb() -> Dict[str, Any]:
    docs = walk_all_docs(DATA_DIR)
    print(f"[Stage1] Scanning folder: {DATA_DIR} → found {len(docs)} document(s)")
    if not docs:
        raise SystemExit(f"No PDFs, CSVs or Excels found under {DATA_DIR}. Place files there.")

    rows: List[Dict[str, Any]] = []
    texts: List[str] = []

    for path in docs:
        fname = os.path.basename(path)
        print(f"[Stage1] Processing: {fname}")
        year, quarter = infer_period_from_filename(fname)
        if _is_pdf(path):
            pages = extract_pdf_pages(path)
            print(f"          → Pages detected: {len(pages)}")
            for page_num, page_text in pages:
                if not page_text.strip():
                    continue
                section_hint = clean_section_hint(page_text[:500])
                for chunk_text in _split_text(page_text):
                    doc_id = str(uuid.uuid4())
                    rows.append({
                        "doc_id": doc_id,
                        "file": fname,
                        "page": page_num,
                        "year": year,
                        "quarter": quarter,
                        "section_hint": section_hint,
                    })
                    texts.append(chunk_text)
            # Second pass: infer period from page header text if missing, and extract key tables
            # Re-iterate pages to attach refined (year, quarter) per page and table windows
            for page_num, page_text in pages:
                if not page_text.strip():
                    continue
                # Infer per-page period (only trust explicit QY or FY)
                y2, q2 = infer_period_from_text(page_text)
                # Start from filename-derived period
                y_eff, q_eff = year, quarter
                # If we detected a quarter-year on the page, use both
                if q2 is not None:
                    q_eff = q2
                    if y2 is not None:
                        y_eff = y2
                else:
                    # No quarter found on page; only allow FY to override year
                    if y2 is not None and q_eff is None:
                        # Only replace year if we don't already have a quarter from filename
                        y_eff = y2
                # Extract small windows for key tables (NIM/Opex/CTI)
                for label, block in extract_key_tables_from_page(page_text):
                    doc_id = str(uuid.uuid4())
                    rows.append({
                        "doc_id": doc_id,
                        "file": fname,
                        "page": page_num,
                        "year": y_eff,
                        "quarter": q_eff,
                        "section_hint": label,
                    })
                    texts.append(block)
        elif _is_tabular(path):
            blocks = extract_tabular_chunks(path)
            print(f"          → Table blocks: {len(blocks)}")
            # Use page=1 for tabular sources; include sheet name in section_hint
            for block_text, sheet in blocks:
                hint_from_name = clean_section_hint(fname) or "table"
                section_hint = f"{hint_from_name} / {sheet}" if sheet else hint_from_name
                doc_id = str(uuid.uuid4())
                rows.append({
                    "doc_id": doc_id,
                    "file": fname,
                    "page": 1,
                    "year": year,
                    "quarter": quarter,
                    "section_hint": section_hint,
                })
                texts.append(block_text)
        else:
            print(f"          → Skipped (unsupported type)")
        print(f"[Stage1] Done: {fname}")

    print(f"[Stage1] Total raw chunks prepared: {len(texts)}")

    kb = pd.DataFrame(rows)
    print(f"[Stage1] Metadata rows: {len(kb)}")

    texts_np = np.array(texts, dtype=object)

    # embed
    provider = pick_provider(EMBED_BACKEND)
    print(f"[Stage1] Embedding provider selected: {getattr(provider, 'name', type(provider).__name__)} (backend={EMBED_BACKEND})")
    try:
        vecs = provider.embed_batch(list(texts_np))
    except Exception as e:
        warn_msg = str(e)
        print(f"[Stage1] ⚠️ Provider failed: {getattr(provider, 'name', type(provider).__name__)} → {warn_msg}")
        print("[Stage1] → Falling back to SentenceTransformers (all-MiniLM-L6-v2)...")
        fallback = STProvider('sentence-transformers/all-MiniLM-L6-v2')
        provider = fallback
        vecs = provider.embed_batch(list(texts_np))
    print(f"[Stage1] Embedded {vecs.shape[0]} chunks (dim={vecs.shape[1]})")

    if not _HAVE_FAISS:
        raise SystemExit("faiss is not installed. pip install faiss-cpu")

    # build index (L2 on normalized vectors works as cosine)
    index = faiss.IndexFlatIP(vecs.shape[1])
    # ensure normalized
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs_norm = (vecs / norms).astype(np.float32)
    index.add(vecs_norm)
    print(f"[Stage1] FAISS index size: {index.ntotal}")

    # save artifacts
    kb_path = os.path.join(OUT_DIR, "kb_chunks.parquet")
    text_path = os.path.join(OUT_DIR, "kb_texts.npy")
    index_path = os.path.join(OUT_DIR, "kb_index.faiss")
    meta_path = os.path.join(OUT_DIR, "kb_meta.json")

    # Save KB with robust parquet saver
    _sanitize_and_save_parquet(kb, kb_path)
    np.save(text_path, texts_np)
    faiss.write_index(index, index_path)
    with open(meta_path, "w") as f:
        json.dump({"embedding_provider": provider.name, "dim": int(vecs.shape[1])}, f)

    print(f"Saved KB rows: {len(kb)} → {kb_path}")
    print(f"Saved texts:    {texts_np.shape} → {text_path}")
    print(f"Saved index:    {index.ntotal} vecs → {index_path}")
    print(f"Saved meta:     {meta_path}")

    # --- Post-build coverage report ---
    try:
        qm = (~kb['quarter'].isna()).mean()
        ym = (~kb['year'].isna()).mean()
        print(f"[Stage1] Coverage → year filled: {ym:.1%}, quarter filled: {qm:.1%}")
        # spot-check mismatches between filename and stored metadata
        import re
        pat = re.compile(r"([1-4])Q(\d{2})", re.I)
        mismatches = 0
        for i,r in kb.iterrows():
            m = pat.search(str(r['file']))
            if not m:
                continue
            qf = int(m.group(1)); yf = 2000 + int(m.group(2))
            y_ok = (pd.isna(r['year'])) or (int(r['year']) == yf)
            q_ok = (pd.isna(r['quarter'])) or (int(r['quarter']) == qf)
            if not (y_ok and q_ok):
                mismatches += 1
                if mismatches <= 5:
                    print(f"  ↳ mismatch: {r['file']} p.{r['page']} stored=({r['year']},{r['quarter']}) expected=({yf},{qf})")
        if mismatches:
            print(f"[Stage1] Mismatch count (sampled): {mismatches}")
    except Exception as _:
        pass

    return {"kb": kb_path, "texts": text_path, "index": index_path, "meta": meta_path}


if __name__ == "__main__":
    build_kb()
