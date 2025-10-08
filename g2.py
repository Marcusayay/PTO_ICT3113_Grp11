"""
Stage2.py — Baseline Retrieval + Generation (RAG) & Agentic Reasoning

Consumes Stage1 artifacts. Provides two main functions:
1. answer_with_llm: A simple, single-call RAG pipeline.
2. answer_with_agent: An advanced, multi-step agentic pipeline with tool use.
"""
from __future__ import annotations
import os, re, json, math, traceback
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


VERBOSE = bool(int(os.environ.get("AGENT_CFO_VERBOSE", "1")))

# --- Hardcoded LLM selection (instead of environment variables) ---
LLM_BACKEND = "gemini"
GEMINI_MODEL_NAME = "models/gemini-2.5-flash"
OPENAI_MODEL_NAME = "gpt-4o-mini"

# --- Query-aware preferences and numeric helpers ---
QUERY_HINTS = {
    "nim": { "prefer_sections": ["Net interest margin (NIM)", "NIM table", "highlights/summary"]},
    "opex": {"prefer_sections": ["Operating expenses (Opex)", "Expenses", "Staff expenses", "Operating costs", "Income statement", "MD&A", "highlights/summary"]},
    "cti": {"prefer_sections": ["Cost-to-income (CTI)", "Income statement", "highlights/summary"]},
    "oer": {"prefer_sections": ["Operating expenses (Opex)", "Total/Operating income", "Income statement", "highlights/summary"]},
}

def _numeric_score(s: str) -> float:
    if not s: return 0.0
    return min(0.35, 0.05 * max(0, len(re.findall(r"\d[\d,\.]*", s))-1))

# --- Retrieval toggles ---
USE_VECTOR = True
def _classify_query(q: str) -> Optional[str]:
    ql = q.lower()
    if "nim" in ql or "net interest margin" in ql: return "nim"
    if "opex" in ql or "operating expense" in ql or re.search(r"\bexpenses\b", ql): return "opex"
    if re.search(r"\bcti\b|cost[\s\-_\/]*to?\s*[\s\-_\/]*income|efficiency\s*ratio", ql): return "cti"
    # Operating Efficiency Ratio (OER): explicit phrase, acronym, or division symbol context
    if re.search(r"\boperating\s+efficiency\s+ratio\b|\boer\b", ql) or ("÷" in ql and "operating" in ql and "income" in ql):
        return "oer"
    return None

# --- Lazy, notebook-friendly globals (set by init_stage2) ---
kb: Optional[pd.DataFrame] = None
texts: Optional[np.ndarray] = None
index, bm25, EMB = None, None, None
_HAVE_FAISS, _HAVE_BM25, _INITIALIZED = False, False, False

class _EmbedLoader:
    def __init__(self):
        self.impl, self.dim, self.name, self.fn = None, None, None, None
        meta_path = os.path.join(os.environ.get("AGENT_CFO_OUT_DIR", "data"), "kb_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.name = json.load(f).get("embedding_provider")
    def embed(self, texts: List[str]) -> np.ndarray:
        if self.impl is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                st = SentenceTransformer(model_name)
                self.impl, self.dim = ("st", model_name), st.get_sentence_embedding_dimension()
                self.fn = lambda b: st.encode(b, normalize_embeddings=True).astype(np.float32)
            except ImportError:
                raise RuntimeError("Default embedding provider not found. Please `pip install sentence-transformers`.")
        return self.fn(texts)

def init_stage2(out_dir: str = "data") -> None:
    global kb, texts, index, bm25, _HAVE_FAISS, _HAVE_BM25, _INITIALIZED, EMB
    os.environ["AGENT_CFO_OUT_DIR"] = out_dir
    paths = [os.path.join(out_dir, f) for f in ["kb_chunks.parquet", "kb_texts.npy", "kb_index.faiss"]]
    if not all(os.path.exists(p) for p in paths):
        raise RuntimeError(f"KB artifacts not found in '{out_dir}'. Run Stage1 first.")
    kb, texts = pd.read_parquet(paths[0]), np.load(paths[1], allow_pickle=True)
    try:
        import faiss
        _HAVE_FAISS, index = True, faiss.read_index(paths[2])
    except ImportError: _HAVE_FAISS, index = False, None
    try:
        from rank_bm25 import BM25Okapi
        _HAVE_BM25, bm25 = True, BM25Okapi([str(t).lower().split() for t in texts])
    except ImportError: _HAVE_BM25, bm25 = False, None
    EMB = _EmbedLoader()
    _INITIALIZED = True
    if VERBOSE: print(f"[Stage2] Initialized successfully from '{out_dir}'.")

def _ensure_init():
    if not _INITIALIZED: raise RuntimeError("Stage2 not initialized. Call init_stage2() first.")

def _infer_yq_from_filename(fname: str) -> tuple[Optional[int], Optional[int]]:
    if not fname: return (None, None)
    s = str(fname).upper()
    m = re.search(r"([1-4])Q(\d{2})", s, re.I)
    if m:
        q, yy = int(m.group(1)), int(m.group(2))
        return (2000 + yy if yy < 100 else yy, q)
    m = re.search(r"(20\d{2})", s)
    if m: return (int(m.group(1)), None)
    return (None, None)

def _detect_last_n_years(q: str) -> Optional[int]:
    ql = q.lower()
    # explicit three/3 + optional 'fiscal'
    if re.search(r"last\s+(three|3)\s+(fiscal\s+)?years?", ql):
        return 3
    # generic integer before (fiscal) years
    m = re.search(r"last\s+(\d+)\s+(fiscal\s+)?years?", ql)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def _detect_last_n_quarters(q: str) -> Optional[int]:
    if re.search(r"last (five|5) quarters", q, re.I): return 5
    return None

def _period_filter(hits: List[Dict[str, Any]], want_years: Optional[int], want_quarters: Optional[int]) -> List[Dict[str, Any]]:
    if not hits or (want_years is None and want_quarters is None): return hits
    df = pd.DataFrame(hits)
    if want_quarters:
        df = df.sort_values(["year", "quarter"], ascending=False).dropna(subset=["year", "quarter"])
        keep_idx = df.drop_duplicates(subset=["year", "quarter"]).index[:want_quarters]
        return [hits[i] for i in keep_idx]
    if want_years:
        df = df.sort_values("year", ascending=False).dropna(subset=["year"])
        keep_idx = df.drop_duplicates(subset=["year"]).index[:want_years]
        return [hits[i] for i in keep_idx]
    return hits

def hybrid_search(query: str, top_k=12, alpha=0.6) -> List[Dict[str, Any]]:
    _ensure_init()
    vec_scores, bm25_scores = {}, {}
    if USE_VECTOR and _HAVE_FAISS and index and EMB:
        qv = EMB.embed([query])
        qv /= np.linalg.norm(qv, axis=1, keepdims=True)
        sims, ids = index.search(qv.astype(np.float32), top_k * 2)
        vec_scores = {int(i): float(s) for i, s in zip(ids[0], sims[0]) if i != -1}
    if _HAVE_BM25 and bm25:
        scores = bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[-top_k*2:]
        bm25_scores = {int(i): float(scores[i]) for i in top_idx}
    
    fused = {k: (alpha * vec_scores.get(k, 0)) + ((1 - alpha) * (bm25_scores.get(k, 0) / (max(bm25_scores.values()) or 1.0))) for k in set(vec_scores) | set(bm25_scores)}
    
    qtype = _classify_query(query)
    want_years, want_quarters = _detect_last_n_years(query), _detect_last_n_quarters(query)
    latest_year = kb['year'].max()
    if want_years and not kb[kb['quarter'].isna()].empty: latest_year = kb[kb['quarter'].isna()]['year'].max()

    # NEW: favor explicit periods mentioned in the query (e.g., "4Q24", "FY2024")
    desired_periods = _desired_periods_from_query(query)  # list of (year, quarter) where quarter=None means annual
    desired_set = set(desired_periods) if desired_periods else set()

    for i in fused:
        meta = kb.iloc[i]
        boost = _numeric_score(str(texts[i])[:800])
        # Strongly boost exact period matches; de-boost non-matches when a period is explicitly requested
        hit_y = int(meta.year) if pd.notna(meta.year) else None
        hit_q = int(meta.quarter) if pd.notna(meta.quarter) else None
        if desired_set:
            if (hit_y, hit_q) in desired_set:
                boost += 1.2
            else:
                boost -= 2.0 # MODIFIED: Increased penalty for period mismatch

        # Additional anchor-aware boosts/penalties
        sec_low = str(meta.section_hint or "").lower()
        file_low = str(meta.file or "").lower()

        # If querying CTI, strongly prefer CTI/highlights and penalize NIM pages
        if qtype == "cti":
            if ("cti" in sec_low) or ("cost-to-income" in sec_low) or re.search(r"cost\s*[/\-\–_]?\s*to\s*income", sec_low):
                boost += 0.6
            if "nim" in sec_low:
                boost -= 0.8
            # Ask for Highlights or supplement explicitly → boost sheets
            if ("highlights" in query.lower()) and (("highlights" in sec_low) or ("highlights" in file_low)):
                boost += 0.5
            if ("suppl" in query.lower() or "2q24_suppl" in query.lower()) and ("suppl" in file_low):
                boost += 0.4

        # Mild preference to Excel/tabular supplements for ratio % that come from highlights tables
        if qtype in ("cti", "nim", "opex") and file_low.endswith((".xls", ".xlsx")):
            boost += 0.15

        if qtype and isinstance(meta.section_hint, str) and meta.section_hint in QUERY_HINTS[qtype]["prefer_sections"]: boost += 0.25
        if (want_years or want_quarters) and pd.notna(meta.year):
            year_diff = latest_year - meta.year
            if year_diff == 0: boost += 1.0
            elif year_diff <= 2: boost += 0.6
            elif year_diff <= 4: boost += 0.25
        is_annual = pd.isna(meta.quarter)
        if want_years and is_annual: boost += 0.3
        if want_quarters and not is_annual: boost += 0.3
        # Prefer PDFs slightly for chart-derived % metrics; prefer tables slightly for sums.
        ext = str(kb.iloc[i].file).lower().rsplit(".", 1)[-1]
        # If the query mentions a particular file or the 'Highlights' tab, boost matching hits
        qlow = query.lower()
        hit_file = str(kb.iloc[i].file).lower()
        hit_section = str(kb.iloc[i].section_hint or "").lower()
        # Keep generic boosts (lower than the CTI-specific ones above)
        if "2q24_suppl" in qlow and "2q24_suppl" in hit_file:
            boost += 0.3
        if "highlights" in qlow and ("highlights" in hit_section or "highlights" in hit_file):
            boost += 0.25
        if ext in ("xls", "xlsx"):
            # Prefer sheets for YoY/aggregations
            if re.search(r"\byoy\b|year[- ]?on[- ]?year|total\b|sum\b|\blast\s+\d+\s+years", query, re.I):
                boost += 0.15
            # For NIM specifically, sheets often have the % cleanly; small positive nudge
            if re.search(r"\bnim\b|net\s*interest\s*margin", query, re.I):
                boost += 0.10
            # For generic %/ratio (CTI, other ratios) give sheets a mild *positive* nudge because Excel “Highlights” often holds clean decimals.
            if re.search(r"\bcti\b|cost\s*/\s*income|efficiency\s*ratio|(?:^| )ratio\b|%", query, re.I):
                boost += 0.05
        else:
            # small preference for PDFs when question is for reported % (NIM/CTI/ratio)
            if re.search(r"\bnim\b|net\s*interest\s*margin|cti|cost\s*/\s*income|ratio|%", query, re.I):
                boost += 0.1
        fused[i] += boost
        
    hits = [{"doc_id": kb.iloc[i].doc_id, "file": kb.iloc[i].file, "page": int(kb.iloc[i].page), "year": int(kb.iloc[i].year) if pd.notna(kb.iloc[i].year) else None, "quarter": int(kb.iloc[i].quarter) if pd.notna(kb.iloc[i].quarter) else None, "section_hint": kb.iloc[i].section_hint, "score": float(score)} for i, score in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]]
    return hits

def format_citation(hit: dict) -> str:
    parts = [hit.get("file", "?")]
    y = hit.get("year")
    q = hit.get("quarter")
    if y is not None and q is not None:
        parts.append(f"{int(q)}Q{str(int(y))[-2:]}")
    elif y is not None:
        parts.append(str(int(y)))
    if hit.get("page") is not None:
        parts.append(f"p.{int(hit['page'])}")
    if hit.get("section_hint"):
        parts.append(hit["section_hint"])
    return ", ".join(parts)

def _context_from_hits(hits: List[Dict[str, Any]], top_ctx=3) -> str:
    return "\n\n".join([f"[{format_citation(h)}]\n{texts[kb.index[kb.doc_id == h['doc_id']][0]][:1200]}" for h in hits[:top_ctx]])

def _call_llm(prompt: str, dry_run: bool = False) -> str:
    """
    Calls the selected LLM API.
    MODIFIED: Now accepts a 'dry_run' boolean toggle.
    """
    if dry_run:
        print("\n" + "="*25 + " DRY RUN: PROMPT PREVIEW " + "="*25)
        print(prompt)
        print("="*70)
        if "Return ONLY a valid JSON object" in prompt:
            return '{"plan": [{"tool": "dry_run_tool", "parameters": {"status": "Dry run mode enabled"}}]}'
        else:
            return "This is a dry run. The API was not called."

    backend = LLM_BACKEND.lower()
    try:
        if backend == "gemini":
            from google import generativeai as genai
            if not os.environ.get("GEMINI_API_KEY"): raise ValueError("GEMINI_API_KEY not set.")
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            return model.generate_content(prompt).text
        elif backend == "openai":
            from openai import OpenAI
            if not os.environ.get("OPENAI_API_KEY"): raise ValueError("OPENAI_API_KEY not set.")
            client = OpenAI()
            resp = client.chat.completions.create(model=OPENAI_MODEL_NAME, messages=[{"role":"user","content": prompt}], temperature=0.1)
            return resp.choices[0].message.content
        else: raise ValueError(f"Invalid LLM_BACKEND: {backend}")
    except Exception as e:
        return f"LLM Generation Failed: {e}"

def answer_with_llm(query: str, top_k_retrieval=12, top_ctx=3, dry_run: bool = False) -> Dict[str, Any]:
    _ensure_init()
    want_years, want_quarters = _detect_last_n_years(query), _detect_last_n_quarters(query)
    hits = hybrid_search(query, top_k=top_k_retrieval)
    hits = _period_filter(hits, want_years, want_quarters)
    context = _context_from_hits(hits, top_ctx=top_ctx)
    prompt = f"You are Agent CFO. Answer the question based ONLY on the provided context. Cite your sources inline. Question: {query}\n\nContext:\n{context}"
    answer = _call_llm(prompt, dry_run=dry_run)
    return {"answer": answer, "hits": hits[:top_ctx]}

def tool_calculator(expression: str) -> str:
    try:
        import re
        # Normalize: remove thousands separators, handle %, and simple units
        s = str(expression)
        # 1) remove thousands separators (1,234,567.89)
        s = re.sub(r'(?<=\d),(?=\d{3}\b)', '', s)
        # 2) turn percentages like 37% into (37/100)
        s = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'(\1/100)', s)
        # 3) currency symbols
        s = re.sub(r'(?i)[s]?\$\s*', '', s)
        # 4) units to scientific notation
        s = re.sub(r'(?i)\b(bn|billion|b)\b', 'e9', s)
        s = re.sub(r'(?i)\b(mn|million|m)\b', 'e6', s)
        # 4.5) stray trailing commas or semicolons
        s = re.sub(r'[,\;]\s*$', '', s)
        # 5) allowlist filter
        safe = re.sub(r'[^0-9eE\+\-*/(). ]', '', s)
        # remove any remaining commas inside numbers
        safe = re.sub(r'(?<=\d),(?=\d)', '', safe)
        result = eval(safe)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

def _desired_periods_from_query(query: str) -> list[tuple[int|None, int|None]]:
    """
    Parse explicit periods from query text, like '1Q25', '4Q24', or 'FY2024'.
    Returns list of (year, quarter) where quarter=None denotes annual.
    """
    out: list[tuple[int|None, int|None]] = []
    for m in re.finditer(r"\b([1-4])Q(\d{2})\b", query.upper()):
        q, yy = int(m.group(1)), int(m.group(2))
        out.append((2000 + yy, q))
    for m in re.finditer(r"\bFY\s?(20\d{2})\b", query.upper()):
        out.append((int(m.group(1)), None))
    return out

def tool_table_extraction(query: str) -> str:
    """
    Robust single-value extractor with anchor-aware windows and scoring.
    Prefers:
      - Percentages with decimals near 'NIM/Net interest margin' or 'Cost / income/CTI'
      - Monetary amounts next to 'Operating expenses/Total income' with units (S$m, bn/mn)
    Avoids:
      - Isolated chart ticks (e.g., 1, 4, 6)
      - Dashes ('-') or values without supporting context
    Returns: "Value: <clean>[%], Source: <citation>"
    """
    if VERBOSE: print(f"  [Tool Call: table_extraction] with query: '{query}'")
    hits = hybrid_search(query, top_k=6)
    if not hits:
        return "Error: No relevant data found."

    qtype = _classify_query(query) or ""
    desired_periods = _desired_periods_from_query(query)

    ql = query.lower()
    want_percent = bool(re.search(r"\b(cti|cost[\s\-_\/]*to?\s*income|margin|nim|ratio|%)\b", ql))
    want_opex    = bool(re.search(r"\b(opex|operating\s+expenses?)\b", ql))
    want_income  = bool(re.search(r"\b(total\s+(?:operating\s+)?income|operating\s+income|total\s+income)\b", ql))

    # Annual/quarter query detection
    is_annual_query = bool(re.search(r"\bfy\s?20\d{2}\b|last\s+\d+\s+(?:fiscal\s+)?years?", query, re.I))

    # Anchors
    anchors = [
        r"net\s*interest\s*margin|nim",
        r"cost\s*[/\-\–_]?\s*to?\s*income|cti|efficiency\s*ratio|operating\s+efficiency\s+ratio",
        r"\boperating\s+expenses?\b|\bopex\b|\bstaff\s+expenses?\b|\bother\s+expenses?\b|\bcosts?\b",
        r"\btotal\s+operating\s+income\b|\btotal\s+income\b|\boperating\s+income\b"
    ]
    anchor_pat = re.compile("|".join(anchors), re.I)

    # Numbers
    # Percent-without-symbols are only accepted near anchors (see logic below)
    pct_pat_strict = re.compile(r"\b(\d{1,2}\.\d{1,2})\s*%")     # 2.68%
    pct_pat_loose  = re.compile(r"\b(\d{1,2}(?:\.\d{1,2})?)\s*%") # 2.7% / 40%
    # NEW: percent-without-symbol candidates (used only near anchors)
    nim_pct_nosym  = re.compile(r"\b(\d\.\d{1,2})\b")             # 2.68
    # allow CTI like 38 or 38.1 without a % symbol
    cti_pct_nosym  = re.compile(r"\b([1-9]\d(?:\.\d{1,2})?)\b")
    money_pat = re.compile(r"([-\d]{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(S\$\s*)?(?:\((?:S\$\s*)?m\)|\bmn\b|\bmillion\b|\bm\b|\((?:S\$\s*)?bn\)|\bbn\b|\bbillion\b|\bb\b)?", re.I)
    unit_pat  = re.compile(r"\((?:S\$\s*)?m\)|\bmn\b|\bmillion\b|\bm\b|\((?:S\$\s*)?bn\)|\bbn\b|\bbillion\b|\bb\b|\(S\$m\)", re.I)
    sgdm_hint = re.compile(r"\(S\$\s*m\)|S\$m|S\$\s*m", re.I)
    yoy_guard = re.compile(r"\b(yoy|qoq|vs)\b", re.I)

    def _section_score(hit: dict) -> float:
        sec = (hit.get("section_hint") or "").lower()
        score = 0.0
        if "nim" in sec: score += 1.0
        if "cti" in sec or "cost-to-income" in sec: score += 1.0
        if "opex" in sec or "expenses" in sec: score += 0.8
        if "income" in sec: score += 0.5
        return score

    def clean_amount(num_str: str) -> str:
        return (num_str or "").strip().replace(",", "")

    def with_unit_to_scientific(window: str, raw: str, want_opex: bool, want_income: bool) -> str:
        # Proximity-aware unit detection: only trust a unit if it appears near the number token
        wlow = window.lower()
        try:
            pos = wlow.find(str(raw).lower())
        except Exception:
            pos = -1

        def nearest_pos(tokens: list[str]) -> int:
            best = 10**9
            for t in tokens:
                j = wlow.find(t)
                if j != -1 and pos != -1:
                    best = min(best, abs(j - pos))
            return best

        bn_tokens = ["(s$bn)", " s$bn", " bn", "billion", " b)"]  # simple set; spacing handles common OCR
        m_tokens  = ["(s$m)", " s$m", " mn", "million", " m)"]

        dist_bn = nearest_pos(bn_tokens)
        dist_m  = nearest_pos(m_tokens)

        # If both are present, prefer the closer one; if tie or neither close, prefer S$m for Opex/Income
        near_thresh = 18  # characters
        if dist_bn < dist_m and dist_bn <= near_thresh:
            return f"{raw}e9"
        if dist_m <= near_thresh or sgdm_hint.search(window):
            return f"{raw}e6"

        # If both tokens appear somewhere in the window but not near, prefer S$m (common slide header)
        if re.search(r"\(s\$\s*bn\)|\bbn\b|\bbillion\b", wlow) and re.search(r"\(s\$\s*m\)|\bmn\b|\bmillion\b|\bm\b", wlow):
            return f"{raw}e6"

        # Default scaling: for Opex/Income queries, assume S$m
        if want_opex or want_income:
            return f"{raw}e6"

        # Otherwise, leave as-is
        return raw

    candidates: list[tuple[float, str, str]] = []  # (score, value_repr, citation)

    for hit in hits:
        full_text = str(texts[kb.index[kb.doc_id == hit["doc_id"]][0]])
        flat = " ".join(full_text.split())

        # Widen scan: up to 6 anchors and a wider window—OCR often separates the number from the label
        windows = []
        for m in list(anchor_pat.finditer(flat))[-6:]:
            start = max(0, m.start() - 500)
            end   = min(len(flat), m.end() + 500)
            windows.append(flat[start:end])
        if not windows:
            windows = [flat]

        base = 0.4 + _section_score(hit)

        # Add preference to annual docs for annual queries, quarterly for quarter queries
        if is_annual_query and pd.isna(hit.get("quarter")):
            base += 0.5
        if (not is_annual_query) and (hit.get("quarter") is not None):
            base += 0.2
            
        # If explicit periods were requested and this hit doesn't match, penalize
        if desired_periods and (hit.get("year"), hit.get("quarter")) not in desired_periods:
            base -= 0.4 # MODIFIED: Added penalty for period mismatch within the tool

        for w in windows:
            # Strip S$m and similar tokens before matching %
            w = re.sub(r'\(S\$m\)|S\$m', '', w)
            # Percent path
            if want_percent:
                def _pct_ok(v: float) -> bool:
                    if qtype == "nim":
                        return 0.5 <= v <= 5.0   # typical NIM range
                    if qtype == "cti":
                        return 15.0 <= v <= 80.0 # typical CTI range (broad)
                    return 0.01 <= v <= 100.0

                # MODIFIED: Stricter guards to prevent pulling CTI data for NIM queries and vice-versa
                if qtype == "cti":
                    if re.search(r"(margin|nim)", w, re.I) and not re.search(r"(cost\s*/\s*income|cti|efficiency)", w, re.I):
                        continue
                if qtype == "nim":
                    if re.search(r"(cost\s*/\s*income|cti|efficiency)", w, re.I) and not re.search(r"(margin|nim)", w, re.I):
                        continue

                # strict decimals first (avoid chart tick integers)
                for m in pct_pat_strict.finditer(w):
                    val = float(m.group(1))
                    if not _pct_ok(val):
                        continue
                    if qtype == "cti" and not re.search(r"(cost\s*/\s*income|cti|efficiency)", w, re.I):
                        continue
                    if qtype == "nim" and not re.search(r"(margin|nim)", w, re.I):
                        continue
                    s = base + 1.2
                    if re.search(r"margin|nim|cost\s*/\s*income|cti|efficiency", w, re.I): s += 0.6
                    candidates.append((s, f"Value: {val}%, Source: {format_citation(hit)}", format_citation(hit)))

                # then loose (allow integers but heavily penalize)
                for m in pct_pat_loose.finditer(w):
                    val_str = m.group(1)
                    if yoy_guard.search(w) and float(val_str) < 100:
                        continue
                    if re.search(rf"Value:\s*{re.escape(val_str)}%", " ".join(c[1] for c in candidates)):
                        continue
                    try: val = float(val_str)
                    except: continue
                    if not _pct_ok(val): continue
                    if qtype == "cti" and not re.search(r"(cost\s*/\s*income|cti|efficiency)", w, re.I): continue
                    if qtype == "nim" and not re.search(r"(margin|nim)", w, re.I): continue
                    s = base + (0.05 if "." not in val_str else 0.6)
                    if re.search(r"margin|nim|cost\s*/\s*income|cti|efficiency", w, re.I): s += 0.2
                    candidates.append((s, f"Value: {val}%, Source: {format_citation(hit)}", format_citation(hit)))

                # symbol-less % candidates near anchors
                if re.search(r"(margin|nim)", w, re.I):
                    for m in nim_pct_nosym.finditer(w):
                        val = float(m.group(1))
                        if 0.5 <= val <= 5.0 and not yoy_guard.search(w):
                            s = base + 0.9
                            candidates.append((s, f"Value: {val}%, Source: {format_citation(hit)}", format_citation(hit)))
                if qtype == "cti" and re.search(r"(cost\s*/\s*income|cti|efficiency)", w, re.I):
                    for m in cti_pct_nosym.finditer(w):
                        val = float(m.group(1))
                        if 15.0 <= val <= 80.0 and not yoy_guard.search(w):
                            s = base + 0.9
                            candidates.append((s, f"Value: {val}%, Source: {format_citation(hit)}", format_citation(hit)))

            # Monetary path
            if want_opex or want_income or not want_percent:
                if want_opex and not re.search(r"\boperating\s+expenses?\b|\bopex\b|\bstaff\s+expenses?\b|\bother\s+expenses?\b|\bcosts?\b", w, re.I): continue
                if want_income and not re.search(r"\btotal\s+operating\s+income\b|\btotal\s+income\b|\boperating\s+income\b", w, re.I): continue
                if (want_opex or want_income) and (str(hit.get("section_hint") or "").lower().startswith("nim")): continue
                # MODIFIED: Stricter guard for CTI queries
                if qtype == "cti" and re.search(r"(margin|nim)", w, re.I): continue
                if want_income and re.search(r"margin|nim", w, re.I): continue

                for m in money_pat.finditer(w):
                    raw = clean_amount(m.group(1))
                    if not raw or raw in ("-", "–"): continue
                    tail = w[w.find(raw) + len(raw): w.find(raw) + len(raw) + 3]
                    if "%" in tail: continue
                    if "." not in raw and len(raw) <= 2 and not unit_pat.search(w): continue
                    has_unit = bool(unit_pat.search(w) or sgdm_hint.search(w))
                    if not has_unit and not (want_opex or want_income): continue
                    
                    num = with_unit_to_scientific(w, raw, want_opex, want_income)
                    try: val = float(num.replace('e9','e9').replace('e6','e6'))
                    except Exception: val = None
                    
                    # MODIFIED: Plausibility gates for monetary values
                    too_huge_without_bn = (val is not None and val > 80e9 and not re.search(r"\b(s\$\s*bn|bn|billion)\b", w, re.I))
                    if too_huge_without_bn: continue
                    
                    if yoy_guard.search(w):
                        try:
                            if val is not None and val < 100 and not unit_pat.search(w): continue
                        except Exception: pass
                        
                    if is_annual_query:
                        if want_income and (val is None or val < 1000e6): continue
                        if want_opex and (val is None or val < 200e6): continue

                    s = base + 0.9
                    if re.search(r"\boperating\s+expenses\b|\bopex\b", w, re.I): s += 0.6
                    if re.search(r"\btotal\s+operating\s+income\b|\btotal\s+income\b|\boperating\s+income\b", w, re.I): s += 0.5
                    candidates.append((s, f"Value: {num}, Source: {format_citation(hit)}", format_citation(hit)))

    if not candidates:
        return "Error: No plausible value found in documents."

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_candidate = candidates[0][1] # Return the highest scored candidate
    return best_candidate

def tool_multi_document_compare(topic: str, files: list[str]) -> str:
    if VERBOSE: print(f"  [Tool Call: multi_document_compare] for topic '{topic}' in files: {files}")
    results = []
    for file_name in files:
        hits = hybrid_search(f"In {file_name}, find info on: {topic}", top_k=1)
        if hits:
            top_hit = hits[0]
            full_text = texts[kb.index[kb.doc_id == top_hit["doc_id"]][0]]
            results.append(f"From {file_name}:\n{full_text}\nSource: {format_citation(top_hit)}")
        else: results.append(f"From {file_name}: No data found.")
    return "\n---\n".join(results)

def _compile_or_repair_plan(query: str, plan: list[dict]) -> list[dict]:
    """
    Ensure every tool step has required parameters.
    If the LLM omitted parameters, synthesize a deterministic plan based on the query.
    Returns a fixed plan.
    """
    def _has_params(step: dict) -> bool:
        params = step.get("parameters")
        if not isinstance(params, dict): return False
        return any(v not in (None, "", []) for v in params.values())

    if plan and all(_has_params(s) for s in plan):
        return plan

    qtype = _classify_query(query) or ""
    if not qtype and ("÷" in query and re.search(r"operating", query, re.I) and re.search(r"income", query, re.I)):
        qtype = "oer"
    want_years  = _detect_last_n_years(query)
    want_quarts = _detect_last_n_quarters(query)

    df = kb.copy()
    df["y"] = pd.to_numeric(df["year"], errors="coerce")
    df["q"] = pd.to_numeric(df["quarter"], errors="coerce")

    steps: list[dict] = []

    if qtype == "nim":
        n = want_quarts or 5
        qdf = df.dropna(subset=["y","q"]).sort_values(["y","q"], ascending=[False, False])
        periods = qdf[["y","q"]].drop_duplicates().head(n).to_records(index=False)
        for y, q in periods:
            y, q = int(y), int(q)
            label = f"{q}Q{str(y)[-2:]}"
            steps.append({ "step": f"Extract NIM for {label}", "tool": "table_extraction", "parameters": {"query": f"Net interest margin (%) for {label}"}, "store_as": f"nim_{y}_{q}"})
        return steps

    if qtype == "opex":
        n = want_years or 3
        ydf = df[df["q"].isna()].dropna(subset=["y"]).sort_values("y", ascending=False)
        if ydf.empty: ydf = df.dropna(subset=["y"]).sort_values("y", ascending=False)
        years = [int(y) for y in ydf["y"].drop_duplicates().head(n)]
        for y in years:
            steps.append({ "step": f"Extract Operating expenses for FY{y}", "tool": "table_extraction", "parameters": {"query": f"Operating expenses (total) for fiscal year {y}"}, "store_as": f"opex_fy{y}"})
        if len(years) >= 2:
            y0, y1 = years[0], years[1]
            steps.append({ "step": f"Compute YoY % change in Opex FY{y0} vs FY{y1}", "tool": "calculator", "parameters": {"expression": f"(( ${{opex_fy{y0}}} - ${{opex_fy{y1}}} ) / ${{opex_fy{y1}}}) * 100"}, "store_as": f"opex_yoy_{y0}_{y1}"})
        if len(years) >= 3:
            y1, y2 = years[1], years[2]
            steps.append({ "step": f"Compute YoY % change in Opex FY{y1} vs FY{y2}", "tool": "calculator", "parameters": {"expression": f"(( ${{opex_fy{y1}}} - ${{opex_fy{y2}}} ) / ${{opex_fy{y2}}}) * 100"}, "store_as": f"opex_yoy_{y1}_{y2}"})
        latest = years[0] if years else None
        if latest:
            steps.append({ "step": f"Compare MD&A Opex drivers for FY{latest}", "tool": "multi_document_compare", "parameters": {"topic": f"Operating expense drivers FY{latest}", "files": ["dbs-annual-report-2024.pdf", "4Q24_CFO_presentation.pdf", "4Q24_performance_summary.pdf"]}, "store_as": "opex_drivers_fylatest"})
        return steps

    # MODIFIED: Corrected the deterministic plan for Operating Efficiency Ratio
    if qtype == "oer":
        n = want_years or 3
        ydf = df[df["q"].isna()].dropna(subset=["y"]).sort_values("y", ascending=False)
        if ydf.empty: ydf = df.dropna(subset=["y"]).sort_values("y", ascending=False)
        years = [int(y) for y in ydf["y"].drop_duplicates().head(n)]
        for y in years:
            steps.append({ "step": f"Extract Opex for FY{y}", "tool": "table_extraction", "parameters": {"query": f"Operating expenses (total) for fiscal year {y}"}, "store_as": f"opex_fy{y}"})
            steps.append({ "step": f"Extract Operating income for FY{y}", "tool": "table_extraction", "parameters": {"query": f"Operating income for fiscal year {y}"}, "store_as": f"opinc_fy{y}"})
            steps.append({ "step": f"Compute Operating Efficiency Ratio (Opex / Operating Income) for FY{y}", "tool": "calculator", "parameters": {"expression": f"(${{opex_fy{y}}} / ${{opinc_fy{y}}}) * 100"}, "store_as": f"oer_fy{y}"})
        return steps

    if qtype == "cti":
        n = want_years or 3
        ydf = df[df["q"].isna()].dropna(subset=["y"]).sort_values("y", ascending=False)
        if ydf.empty: ydf = df.dropna(subset=["y"]).sort_values("y", ascending=False)
        years = [int(y) for y in ydf["y"].drop_duplicates().head(n)]
        for y in years:
            steps.append({ "step": f"Extract CTI (reported %) for FY{y}", "tool": "table_extraction", "parameters": {"query": f"Cost / income (%) for fiscal year {y}"}, "store_as": f"cti_fy{y}"})
            steps.append({ "step": f"Extract Opex for FY{y}", "tool": "table_extraction", "parameters": {"query": f"Operating expenses (total) for fiscal year {y}"}, "store_as": f"opex_fy{y}"})
            steps.append({ "step": f"Extract Total/Operating income for FY{y}", "tool": "table_extraction", "parameters": {"query": f"Total income (or Operating income) for fiscal year {y}"}, "store_as": f"income_fy{y}"})
            steps.append({ "step": f"Compute CTI for FY{y} if not reported", "tool": "calculator", "parameters": {"expression": f"${{opex_fy{y}}} / ${{income_fy{y}}}"}, "store_as": f"cti_calc_fy{y}"})
        return steps

    steps.append({ "step": "Extract a directly relevant figure", "tool": "table_extraction", "parameters": {"query": query}, "store_as": "value_1"})
    return steps

def answer_with_agent(query: str, dry_run: bool = False) -> Dict[str, Any]:
    _ensure_init()
    row = {"Query": f"[agent] {query}"}
    execution_log = []
    
    with timeblock(row, "T_total"), timeblock(row, "T_reason"):
        # == STEP 1: PLANNING ==
        planning_prompt = f"""You are a financial analyst agent. Create a JSON plan to answer the user's query.

Tools Available:
- `table_extraction(query: str)`: Finds a single reported data point (e.g., a percentage or a monetary value) from slides/annuals/supplements.
- `calculator(expression: str)`: Calculates a math expression using numbers you already extracted.
- `multi_document_compare(topic: str, files: list[str])`: Pulls comparable snippets from multiple files.

Planning Rules:
1) **Prefer reported metrics over recomputing from components.** For NIM and CTI, extract the **reported percentage** (e.g., "Net interest margin (%)" or "Cost / income (%)") from CFO deck, performance summary, or the Excel supplement. For **Operating Efficiency Ratio (Opex ÷ Operating Income)** there may not be a reported field; plan to compute it from Opex and Operating Income if needed.
2) When the request is for the **last N quarters/years**, plan steps that **directly extract those N reported values** (e.g., 1Q25, 4Q24, 3Q24...) instead of deriving them.
3) Use `calculator` only for simple arithmetic (e.g., YoY %, CTI if you have Opex and Total/Operating Income). Never pass text with units/commas/% into the calculator—use only clean numeric placeholders you previously extracted.
4) Always include `"store_as"` for every extraction step. Use short keys like `nim_1q25`, `cti_fy2024`, `opex_fy2023`, `income_fy2023`, etc.
5) If the query asks for drivers/MD&amp;A points, add one step to extract or quote the relevant lines (you may use `table_extraction` for MD&amp;A text).

User Query: "{query}"
Return ONLY a valid JSON object with a "plan" key."""
        if VERBOSE: print("[Agent] Step 1: Generating execution plan...")
        
        plan_response = _call_llm(planning_prompt)
        plan = None
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', plan_response, re.DOTALL)
            plan_str = json_match.group(1) if json_match else plan_response
            plan = json.loads(plan_str)["plan"]
            execution_log.append({"step": "Planning", "plan": plan})
            if VERBOSE: print("[Agent] Plan generated successfully.")
        except (json.JSONDecodeError, KeyError) as e:
            error_msg = f"Failed to parse a valid plan from LLM response.\nError: {e}\nLLM Response:\n---\n{plan_response}\n---"
            return {"answer": error_msg, "hits": [], "execution_log": execution_log}
        
        if dry_run:
            answer = f"DRY RUN MODE: The agent generated the following plan and stopped before execution.\n\n{json.dumps(plan, indent=2)}"
            return {"answer": answer, "hits": [], "execution_log": execution_log}

        # == STEP 2: ACTING (Live Mode Only) ==
        if VERBOSE: print("[Agent] Step 2: Executing plan...")
        tool_mapping = {"calculator": tool_calculator, "table_extraction": tool_table_extraction, "multi_document_compare": tool_multi_document_compare}
        execution_state = {}

        repaired_plan = _compile_or_repair_plan(query, plan)
        if repaired_plan != plan:
            execution_log.append({"step": "PlanRepair", "note": "LLM plan lacked parameters; synthesized deterministic plan.", "repaired_plan": repaired_plan})
        plan = repaired_plan

        for i, step in enumerate(plan):
            tool, params, store_as = step.get("tool"), step.get("parameters", {}), step.get("store_as")

            if tool == "table_extraction" and not params.get("query"): params["query"] = query
            if tool == "calculator" and not params.get("expression"):
                execution_log.append({"step": f"Execution {i+1}", "tool_call": f"{tool}({params})", "error": "Missing 'expression' parameter"})
                continue

            for p_name, p_value in params.items():
                if isinstance(p_value, str):
                    for var_name, var_value in execution_state.items():
                        p_value = p_value.replace(f"${{{var_name}}}", str(var_value))
                params[p_name] = p_value

            if tool in tool_mapping:
                try:
                    result = tool_mapping[tool](**params)
                    execution_log.append({"step": f"Execution {i+1}", "tool_call": f"{tool}({params})", "result": result})
                    if store_as:
                        cap = None
                        m_val = re.search(r'Value:\s*([-\d.,]+)\s*(%|e9|e6|bn|billion|b|mn|million|m)?', result, re.I)
                        if m_val:
                            raw = m_val.group(1).replace(',', '')
                            unit = (m_val.group(2) or '').lower()
                            if unit == '%': cap = f"({raw}/100)"
                            elif unit in ('bn', 'billion', 'b', 'e9'): cap = f"{raw}e9"
                            elif unit in ('mn', 'million', 'm', 'e6'): cap = f"{raw}e6"
                            else: cap = raw
                        else:
                            m_any = re.search(r'([-\d]+(?:\.\d+)?)', result)
                            if m_any: cap = m_any.group(1)
                        if store_as and cap is not None:
                            execution_state[store_as] = cap
                except Exception as e:
                    execution_log.append({"step": f"Execution {i+1}", "tool_call": f"{tool}({params})", "error": str(e)})
            else:
                execution_log.append({"step": f"Execution {i+1}", "error": f"Tool '{tool}' not found."})
        if VERBOSE: print("[Agent] Plan execution complete.")

        # == STEP 3: SYNTHESIS (Live Mode Only) ==
        if VERBOSE: print("[Agent] Step 3: Synthesizing final answer...")
        synthesis_prompt = f"""You are Agent CFO. Provide a final answer to the user's query based ONLY on the provided Tool Execution Log.
User Query: "{query}"
Tool Execution Log:
{json.dumps(execution_log, indent=2)}
Final Answer:"""
        final_answer = _call_llm(synthesis_prompt)
        
    row["Tools"] = json.dumps([step.get("tool_call") for step in execution_log if "Execution" in step.get("step", "")])
    instr.log(row)
    return {"answer": final_answer, "hits": [], "execution_log": execution_log}

def get_logs() -> pd.DataFrame:
    return instr.df()

def is_initialized() -> bool:
    return _INITIALIZED

if __name__ == "__main__":
    init_stage2()
    if VERBOSE: print("[Stage2] Ready. Use answer_with_llm() or answer_with_agent().")