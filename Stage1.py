import os
from typing import List, Dict, Any
import numpy as np

# --- helpers: year/quarter parsing and hit filtering for recency-aware retrieval ---
import re

SECTION_LABELS = {
    "key ratios|highlights|summary": "highlights/summary",
    "net interest margin|nim\\b": "Net interest margin (NIM)",
    "cost[- ]?to[- ]?income|cti|efficiency ratio": "Cost-to-income (CTI)",
    "operating expenses|opex|expenses": "Operating expenses (Opex)",
    "income statement|statement of (comprehensive )?income": "Income statement",
    "balance sheet|statement of financial position": "Balance sheet",
    "management discussion|md&a": "MD&amp;A",
}

def _infer_year_from_filename(fname: str):
    m = re.search(r'(20\\d{2})', fname)
    return int(m.group(1)) if m else None

def _is_quarterly(fname: str):
    return bool(re.search(r'\\b[1-4]Q\\d{2}\\b', fname.upper()))

def _prefer_quarterly(hits, must_contain_any=('CFO','CEO','performance','summary')):
    out = [h for h in hits if _is_quarterly(h.get('file',''))]
    if must_contain_any:
        out2 = [h for h in out if any(s.lower() in h.get('file','').lower() for s in must_contain_any)]
        if out2:
            return out2
    return out or hits

def _keep_recent_years(hits, n_years=3):
    pairs = []
    for h in hits:
        y = _infer_year_from_filename(h.get('file','')) or 0
        pairs.append((y, h))
    if not pairs:
        return hits
    pairs.sort(key=lambda t: t[0], reverse=True)
    # collect top N distinct years
    years = []
    for y,_ in pairs:
        if y and y not in years:
            years.append(y)
        if len(years) >= n_years:
            break
    if not years:
        return hits
    filtered = [h for (y,h) in pairs if y in years]
    return filtered if filtered else hits

def _clean_section_hint(h):
    raw = (h.get('section_hint') or '').strip()
    if not raw:
        return raw
    for pat, label in SECTION_LABELS.items():
        if re.search(pat, raw, flags=re.IGNORECASE):
            return label
    # fallback: strip regex noise
    return re.sub(r'\\|', '/', raw)


def format_citation(hit: dict) -> str:
    parts = [hit["file"]]
    if hit.get("year_qtr"): parts.append(hit["year_qtr"])
    parts.append(f"p.{hit['page']}")
    sec = _clean_section_hint(hit)
    if sec: parts.append(sec)
    return " — ".join(parts)


def answer_with_gemini(query: str, top_k_retrieval=16, top_ctx=3, model_name="gemini-2.5-flash-latest") -> Dict[str,Any]:
    hits = retrieve_then_rerank(query, top_k=top_k_retrieval, alpha=0.6)
    ql = (query or "").lower()
    # If the user asks for "last three years", keep only most recent 3 distinct years
    if ("last three years" in ql) or ("last 3 years" in ql) or ("past three years" in ql):
        hits = _keep_recent_years(hits, n_years=3)
    # If the user asks for "last five quarters", prefer quarterly decks/summaries
    if ("last five quarters" in ql) or ("last 5 quarters" in ql):
        hits = _prefer_quarterly(hits)
    # ... rest of function ...