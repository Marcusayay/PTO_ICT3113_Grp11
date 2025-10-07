_Q_PAT_FN = re.compile(r"([1-4])Q(\d{2})", re.I)

def _infer_yq_from_filename(fname: str) -> tuple[Optional[int], Optional[int]]:
    if not fname:
        return (None, None)
    s = str(fname).upper()
    m = _Q_PAT_FN.search(s)
    if m:
        q = int(m.group(1)); yy = int(m.group(2)); y = 2000 + yy
        return (y, q)
    m = re.search(r"(20\d{2})", s)
    if m:
        return (int(m.group(1)), None)
    return (None, None)
"""
Stage2.py — Baseline Retrieval + Generation (RAG)

Consumes Stage1 artifacts:
  data/kb_chunks.parquet
  data/kb_texts.npy
  data/kb_index.faiss

Retrieval:
  - Hybrid (Vector + BM25 if available)
  - Period-aware filter for phrases like "last N years/quarters"
Generation:
  - One LLM call (Gemini/OpenAI placeholder); returns answer + citations
"""
from __future__ import annotations
import os, re, json, math
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

# Timing / logging (simple)
import time, contextlib

@contextlib.contextmanager
def timeblock(row: dict, key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        row[key] = round((time.perf_counter() - t0) * 1000.0, 2)

class _Instr:
    def __init__(self):
        self.rows = []
    def log(self, row):
        self.rows.append(row)
    def df(self):
        cols = ['Query','T_retrieve','T_rerank','T_reason','T_generate','T_total','Tokens','Tools']
        df = pd.DataFrame(self.rows)
        for c in cols:
            if c not in df:
                df[c] = None
        return df[cols]

instr = _Instr()


VERBOSE = bool(int(os.environ.get("AGENT_CFO_VERBOSE", "1")))  # default ON; set 0 to silence

# --- Hardcoded LLM selection (instead of environment variables) ---
LLM_BACKEND = "gemini"  # choose from "gemini" or "openai"
GEMINI_MODEL_NAME = "models/gemini-2.5-flash"
OPENAI_MODEL_NAME = "gpt-4o-mini"

# --- Retrieval toggles ---
USE_VECTOR = True   # set False to force BM25-only retrieval

# --- Lazy, notebook-friendly globals (set by init_stage2) ---
OUT_DIR = None
KB_PARQUET = None
KB_TEXTS = None
KB_INDEX = None
KB_META = None

kb: Optional[pd.DataFrame] = None
texts: Optional[np.ndarray] = None
index = None
bm25 = None
_HAVE_FAISS = False
_HAVE_BM25 = False
_INITIALIZED = False

class _EmbedLoader:
    def __init__(self):
        self.impl = None
        self.dim = None
        self.name = None
        if KB_META and os.path.exists(KB_META):
            with open(KB_META) as f:
                meta = json.load(f)
                self.name = meta.get("embedding_provider")
                self.dim = meta.get("dim")
    def embed(self, texts: List[str]) -> np.ndarray:
        if self.impl is None:
            preferred = (self.name or '').lower()
            # 1) If KB was built with Sentence-Transformers
            if 'sentence-transformers' in preferred or preferred.startswith('st'):
                from sentence_transformers import SentenceTransformer
                model = "sentence-transformers/all-MiniLM-L6-v2"
                st = SentenceTransformer(model)
                self.impl = ("st", model)
                self.dim = st.get_sentence_embedding_dimension()
                def _fn(batch):
                    vecs = st.encode(batch, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
                    return vecs.astype(np.float32)
                self.fn = _fn
            # 2) If KB was built with OpenAI
            elif preferred.startswith('openai'):
                from openai import OpenAI
                if not os.environ.get("OPENAI_API_KEY"):
                    raise RuntimeError("KB was built with OpenAI embeddings but OPENAI_API_KEY is not set.")
                self.client = OpenAI()
                model = "text-embedding-3-small"
                self.impl = ("openai", model)
                self.dim = 1536
                def _fn(batch):
                    resp = self.client.embeddings.create(model=model, input=batch)
                    vecs = [d.embedding for d in resp.data]
                    return np.asarray(vecs, dtype=np.float32)
                self.fn = _fn
            # 3) If KB was built with Gemini
            elif preferred.startswith('gemini'):
                try:
                    from google import generativeai as genai
                except Exception as e:
                    raise RuntimeError("KB was built with Gemini embeddings but google-generativeai is not installed. `pip install google-generativeai`.") from e
                if not os.environ.get("GEMINI_API_KEY"):
                    raise RuntimeError("KB was built with Gemini embeddings but GEMINI_API_KEY is not set.")
                genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
                self.impl = ("gemini", "models/embedding-001")
                self.dim = 768 if (self.dim is None) else self.dim
                def _fn(batch):
                    vecs = []
                    for t in batch:
                        resp = genai.embed_content(model='models/embedding-001', content=t)
                        emb = resp.get('embedding') if isinstance(resp, dict) else getattr(resp, 'embedding', None)
                        if emb is None:
                            raise RuntimeError('Gemini embed_content returned no embedding')
                        vecs.append(emb)
                    return np.asarray(vecs, dtype=np.float32)
                self.fn = _fn
            # 4) Fallback auto-detect (prefer ST so it works offline)
            else:
                if os.environ.get("OPENAI_API_KEY"):
                    from openai import OpenAI
                    self.client = OpenAI()
                    model = "text-embedding-3-small"
                    self.impl = ("openai", model)
                    self.dim = 1536
                    def _fn(batch):
                        resp = self.client.embeddings.create(model=model, input=batch)
                        vecs = [d.embedding for d in resp.data]
                        return np.asarray(vecs, dtype=np.float32)
                    self.fn = _fn
                elif os.environ.get("GEMINI_API_KEY"):
                    from google import generativeai as genai
                    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
                    self.impl = ("gemini", "models/embedding-001")
                    self.dim = 768 if (self.dim is None) else self.dim
                    def _fn(batch):
                        vecs = []
                        for t in batch:
                            resp = genai.embed_content(model='models/embedding-001', content=t)
                            emb = resp.get('embedding') if isinstance(resp, dict) else getattr(resp, 'embedding', None)
                            if emb is None:
                                raise RuntimeError('Gemini embed_content returned no embedding')
                            vecs.append(emb)
                        return np.asarray(vecs, dtype=np.float32)
                    self.fn = _fn
                else:
                    from sentence_transformers import SentenceTransformer
                    model = "sentence-transformers/all-MiniLM-L6-v2"
                    st = SentenceTransformer(model)
                    self.impl = ("st", model)
                    self.dim = st.get_sentence_embedding_dimension()
                    def _fn(batch):
                        vecs = st.encode(batch, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
                        return vecs.astype(np.float32)
                    self.fn = _fn
        return self.fn(texts)

EMB = None  # will be initialized inside init_stage2() after KB_META is known

def init_stage2(out_dir: str = "data") -> None:
    """Initialize Stage 2 in a Jupyter-friendly way.
    Loads KB artifacts, FAISS, and BM25. Call this once per notebook kernel.
    """
    import os
    global OUT_DIR, KB_PARQUET, KB_TEXTS, KB_INDEX, KB_META
    global kb, texts, index, bm25, _HAVE_FAISS, _HAVE_BM25, _INITIALIZED

    OUT_DIR = out_dir
    KB_PARQUET = os.path.join(OUT_DIR, "kb_chunks.parquet")
    KB_TEXTS   = os.path.join(OUT_DIR, "kb_texts.npy")
    KB_INDEX   = os.path.join(OUT_DIR, "kb_index.faiss")
    KB_META    = os.path.join(OUT_DIR, "kb_meta.json")

    if VERBOSE:
        print(f"[Stage2] init → OUT_DIR={OUT_DIR}")

    if not (os.path.exists(KB_PARQUET) and os.path.exists(KB_TEXTS) and os.path.exists(KB_INDEX)):
        raise RuntimeError(f"KB artifacts not found under '{OUT_DIR}'. Run Stage1.build_kb() first.")

    # Load KB tables
    kb = _load_kb_table(KB_PARQUET)
    texts = np.load(KB_TEXTS, allow_pickle=True)

    # (Optional but helpful) Print embedding provider from KB meta if available
    if KB_META and os.path.exists(KB_META):
        try:
            meta = json.load(open(KB_META))
            if VERBOSE:
                print(f"[Stage2] KB embedding provider={meta.get('embedding_provider')} dim={meta.get('dim')}")
        except Exception:
            pass

    if VERBOSE:
        print(f"[Stage2] KB rows={len(kb)}, texts={len(texts)}")

    # FAISS
    try:
        import faiss  # type: ignore
        _HAVE_FAISS = True
        idx = faiss.read_index(KB_INDEX)
    except Exception as e:
        _HAVE_FAISS = False
        idx = None
    globals()['index'] = idx

    if VERBOSE:
        print(f"[Stage2] FAISS loaded={bool(idx)}")

    # BM25 (optional)
    try:
        from rank_bm25 import BM25Okapi
        tokenized = [str(t).lower().split() for t in texts]
        bm25 = BM25Okapi(tokenized)
        _HAVE_BM25 = True
    except Exception:
        bm25 = None
        _HAVE_BM25 = False
    globals()['bm25'] = bm25

    if VERBOSE:
        print(f"[Stage2] BM25 enabled={_HAVE_BM25}")

    # Initialize query embedder **after** KB_META is known so it matches the store
    globals()['EMB'] = _EmbedLoader()
    if VERBOSE:
        try:
            impl = getattr(EMB, 'impl', None)
            print(f"[Stage2] Query embedder ready: {impl if impl else 'lazy-init'}")
        except Exception:
            pass

    # Mark initialized
    _INITIALIZED = True

def _ensure_init():
    if not globals().get('_INITIALIZED', False):
        raise RuntimeError("Stage2 is not initialized. Call init_stage2(out_dir='data') first in your notebook.")

# -----------------------------
# Robust KB loader (parquet → fastparquet → csv)
# -----------------------------

def _load_kb_table(parquet_path: str) -> pd.DataFrame:
    """Load the KB table with fallbacks.
    1) pandas.read_parquet (default engine)
    2) pandas.read_parquet(engine='fastparquet')
    3) CSV fallback at same basename (kb_chunks.csv)
    """
    try:
        return pd.read_parquet(parquet_path)
    except Exception as e1:
        try:
            return pd.read_parquet(parquet_path, engine='fastparquet')
        except Exception as e2:
            csv_path = os.path.splitext(parquet_path)[0] + '.csv'
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Ensure required columns exist
                for c in ['doc_id','file','page','year','quarter','section_hint']:
                    if c not in df.columns:
                        df[c] = np.nan
                # Coerce numeric cols
                if 'page' in df: df['page'] = pd.to_numeric(df['page'], errors='coerce').fillna(0).astype(int)
                if 'year' in df: df['year'] = pd.to_numeric(df['year'], errors='coerce')
                if 'quarter' in df: df['quarter'] = pd.to_numeric(df['quarter'], errors='coerce')
                return df
            raise RuntimeError(
                "Failed to read KB Parquet with both engines and no CSV fallback. "
                f"Errors: pyarrow={e1} | fastparquet={e2}"
            )

# -----------------------------
# Helper: period filters
# -----------------------------

def _detect_last_n_years(q: str) -> Optional[int]:
    ql = q.lower()
    for pat in ["last three years", "last 3 years", "past three years", "past 3 years"]:
        if pat in ql:
            return 3
    return None

def _detect_last_n_quarters(q: str) -> Optional[int]:
    ql = q.lower()
    for pat in ["last five quarters", "last 5 quarters", "past five quarters", "past 5 quarters"]:
        if pat in ql:
            return 5
    return None


def _period_filter(hits: List[Dict[str, Any]], want_years: Optional[int], want_quarters: Optional[int]) -> List[Dict[str, Any]]:
    if not hits:
        return hits
    df = pd.DataFrame(hits)
    if want_quarters:
        df = df.sort_values(["year", "quarter"], ascending=[False, False])
        df = df[df["quarter"].notna()]
        seen = set(); keep_idx = []
        for i, r in df.iterrows():
            key = (int(r.year), int(r.quarter))
            if key in seen: continue
            keep_idx.append(i); seen.add(key)
            if len(keep_idx) >= want_quarters: break
        if VERBOSE:
            print(f"[Stage2] period filter (quarters) → kept={[(int(hits[i]['year']), int(hits[i]['quarter'])) for i in keep_idx]}")
        return [hits[i] for i in keep_idx] if keep_idx else hits
    if want_years:
        df = df.sort_values(["year"], ascending=[False])
        df = df[df["year"].notna()]
        seen = set(); keep_idx = []
        for i, r in df.iterrows():
            y = int(r.year)
            if y in seen: continue
            keep_idx.append(i); seen.add(y)
            if len(keep_idx) >= want_years: break
        if VERBOSE:
            print(f"[Stage2] period filter (years) → kept={[(int(hits[i]['year'])) for i in keep_idx]}")
        return [hits[i] for i in keep_idx] if keep_idx else hits
    return hits

# -----------------------------
# Hybrid retrieval
# -----------------------------

def hybrid_search(query: str, top_k=12, alpha=0.6) -> List[Dict[str, Any]]:
    _ensure_init()
    """Return list of hit dicts with metadata.
    alpha weights vector vs BM25: score = alpha*vec + (1-alpha)*bm25
    """
    row = {"Query": query, "Tools": ["retriever"]}
    with timeblock(row, "T_total"):
        with timeblock(row, "T_retrieve"):
            vec_scores = None
            if USE_VECTOR and _HAVE_FAISS and index is not None and EMB is not None:
                try:
                    qv = EMB.embed([query])
                    # Validate dimensionality against KB meta if available
                    try:
                        meta_dim = int(EMB.dim) if EMB.dim is not None else None
                    except Exception:
                        meta_dim = None
                    if meta_dim is not None and qv.shape[1] != meta_dim:
                        raise RuntimeError(f"Embedding dimension mismatch: query={qv.shape[1]} vs KB={meta_dim}. Rebuild Stage1 with the same provider or align Stage2 to use the same embedding backend.")
                    qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)
                    sims, ids = index.search(qv.astype(np.float32), top_k)
                    vec_scores = {int(ix): float(s) for ix, s in zip(ids[0], sims[0]) if ix != -1}
                except Exception as e:
                    if VERBOSE:
                        print(f"[Stage2] Vector search disabled for this query → {type(e).__name__}: {e}")
                    vec_scores = None  # continue with BM25-only
            bm25_scores = None
            if _HAVE_BM25 and bm25 is not None:
                scores = bm25.get_scores(query.lower().split())
                top_idx = np.argsort(scores)[-top_k:][::-1]
                bm25_scores = {int(i): float(scores[i]) for i in top_idx}
        with timeblock(row, "T_rerank"):
            fused = {}
            if vec_scores:
                for i,s in vec_scores.items():
                    fused[i] = fused.get(i, 0.0) + alpha*s
            if bm25_scores:
                m = max(bm25_scores.values()) or 1.0
                for i,s in bm25_scores.items():
                    fused[i] = fused.get(i, 0.0) + (1-alpha)*(s/m)
            if not fused:
                hits = []
            else:
                top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
                hits = []
                for i,score in top:
                    meta = kb.iloc[i]
                    y = int(meta.year) if not pd.isna(meta.year) else None
                    q = int(meta.quarter) if not pd.isna(meta.quarter) else None
                    if (y is None) or (q is None):
                        y2, q2 = _infer_yq_from_filename(meta.file)
                        if y is None:
                            y = y2
                        if q is None:
                            q = q2
                    hits.append({
                        "doc_id": meta.doc_id,
                        "file": meta.file,
                        "page": int(meta.page),
                        "year": y,
                        "quarter": q,
                        "section_hint": meta.section_hint if isinstance(meta.section_hint, str) else None,
                        "preview": str(texts[i])[:800],
                        "score": float(score),
                    })
    instr.log(row)
    if VERBOSE:
        kept = [(h.get('year'), h.get('quarter'), h.get('file')) for h in hits[:5]]
        print(f"[Stage2] retrieved top={len(hits)} sample={kept}")
    return hits


def format_citation(hit: dict) -> str:
    parts = [hit.get("file","?")]
    if hit.get("year"):
        if hit.get("quarter"):
            parts.append(f"{hit['quarter']}Q{str(hit['year'])[2:]}")
        else:
            parts.append(str(hit["year"]))
    parts.append(f"p.{hit.get('page','?')}")
    sec = hit.get("section_hint")
    if sec:
        parts.append(sec)
    return " — ".join(parts)


def _context_from_hits(hits: List[Dict[str,Any]], top_ctx=3, max_chars=1200) -> str:
    _ensure_init()
    blocks = []
    for h in hits[:top_ctx]:
        text = str(texts[kb.index[kb.doc_id == h["doc_id"]][0]]) if (kb.doc_id == h["doc_id"]).any() else h.get("preview","")
        if len(text) > max_chars:
            text = text[:max_chars] + " ..."
        blocks.append(f"[{format_citation(h)}]\n{text}")
    return "\n\n".join(blocks)

# -----------------------------
# LLM call helper
# -----------------------------

def _call_llm(prompt: str) -> str:
    backend = LLM_BACKEND.lower()
    if backend == "gemini":
        try:
            from google import generativeai as genai
        except Exception as e:
            raise RuntimeError("Selected backend 'gemini' but google-generativeai is not installed. `pip install google-generativeai`.") from e
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Selected backend 'gemini' but GEMINI_API_KEY is not set.")
        model_name = GEMINI_MODEL_NAME
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            text = getattr(resp, 'text', None) if resp is not None else None
            if not text:
                text = str(resp)
            if VERBOSE:
                print(f"[Stage2] LLM=Gemini ({model_name})")
            return text
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}") from e
    elif backend == "openai":
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("Selected backend 'openai' but the OpenAI SDK is not installed. `pip install openai`.") from e
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Selected backend 'openai' but OPENAI_API_KEY is not set.")
        try:
            client = OpenAI()
            model = OPENAI_MODEL_NAME
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":"You are Agent CFO."},{"role":"user","content": prompt}],
                temperature=0.2,
            )
            text = resp.choices[0].message.content
            if VERBOSE:
                print(f"[Stage2] LLM=OpenAI ({model})")
            return text
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {e}") from e
    else:
        raise RuntimeError("Invalid LLM_BACKEND setting; choose 'gemini' or 'openai'.")

# -----------------------------
# Generation (one call)
# -----------------------------

def answer_with_llm(query: str, top_k_retrieval=12, top_ctx=3) -> Dict[str, Any]:
    _ensure_init()
    want_years = _detect_last_n_years(query)
    want_quarters = _detect_last_n_quarters(query)

    hits = hybrid_search(query, top_k=top_k_retrieval, alpha=0.6)
    hits = _period_filter(hits, want_years, want_quarters)

    context = _context_from_hits(hits, top_ctx=top_ctx)

    system_task = (
        "You are Agent CFO. Answer the user's finance/operations question using ONLY the provided context. "
        "When you state any figures, also provide citations in the format: "
        "[Report, Year/Quarter, p.X, Section/Table]. Keep the answer concise and factual."
    )
    user_prompt = (
        f"Question:\n{query}\n\n"
        f"Context passages (use for citations):\n{context}\n\n"
        "Instructions:\n"
        "1) If a value cannot be supported by the context, say so.\n"
        "2) Include citations inline like: (DBS 3Q24 CFO Presentation — p.14 — Cost/Income table).\n"
        "3) End with a short one-line takeaway."
    )
    prompt = f"{system_task}\n\n{user_prompt}"

    row = {"Query": f"[generate] {query}", "Tools": ["retriever","generator"], "Tokens": 0}

    # Placeholder for your LLM call; swap in Gemini/OpenAI
    with timeblock(row, "T_total"), timeblock(row, "T_generate"):
        text = _call_llm(prompt)
        row["Tokens"] = int(len(prompt)//4)

    instr.log(row)

    explicit_citations = "\n".join(f"- {format_citation(h)}" for h in hits[:top_ctx])
    final_answer = text.strip() + "\n\nCitations:\n" + explicit_citations

    return {"answer": final_answer, "hits": hits[:top_ctx], "raw_model_text": text}

def get_logs() -> pd.DataFrame:
    """Return the instrumentation DataFrame for display in notebooks."""
    return instr.df()

def is_initialized() -> bool:
    return bool(globals().get('_INITIALIZED', False))

# Benchmark queries as required
BENCHMARK_QUERIES = [
    "Report the Gross Margin (or Net Interest Margin, if a bank) over the last 5 quarters, with values.",
    "Show Operating Expenses for the last 3 fiscal years, year-on-year comparison.",
    "Calculate the Operating Efficiency Ratio (Opex ÷ Operating Income) for the last 3 fiscal years, showing the working.",
]


def run_benchmark(top_k_retrieval=12, top_ctx=3) -> List[Dict[str, Any]]:
    out = []
    for q in BENCHMARK_QUERIES:
        out.append({"query": q, **answer_with_llm(q, top_k_retrieval=top_k_retrieval, top_ctx=top_ctx)})
    return out


if __name__ == "__main__":
    od = os.environ.get("AGENT_CFO_OUT_DIR", "data")
    init_stage2(od)
    if VERBOSE:
        print("[Stage2] Ready. Use answer_with_llm(query) to generate.")
    if os.environ.get("RUN_DEMO", "0") == "1":
        for r in run_benchmark():
            print("\nQ:", r["query"], "\n")
            print(r["answer"])