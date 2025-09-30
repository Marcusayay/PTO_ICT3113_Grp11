# === STAGE 2: BASELINE RETRIEVAL & ONE LLM CALL (NO CACHING) ===
# - Single-pass retrieve (hybrid BM25+TF-IDF) -> light MMR rerank
# - Exactly ONE Gemini call per query (no caching)
# - Returns answer + guaranteed explicit citations

%pip install --upgrade --force-reinstall google-generativeai


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
    m = re.search(r'(20\d{2})', fname)
    return int(m.group(1)) if m else None

def _is_quarterly(fname: str):
    return bool(re.search(r'\b[1-4]Q\d{2}\b', fname.upper()))

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

# -- tiny MMR-like reranker (NumPy 2.0-safe) --
def mmr_rerank(hits: List[Dict[str,Any]], lambda_mult=0.7, top_k=5) -> List[Dict[str,Any]]:
    if not hits:
        return []
    rel = np.array([h['score'] for h in hits], dtype=float)
    if np.ptp(rel) > 1e-9:
        rel = (rel - rel.min()) / (rel.max() - rel.min())
    sim = np.zeros((len(hits), len(hits)))
    for i,a in enumerate(hits):
        for j,b in enumerate(hits):
            sim[i,j] = 1.0 if (a['file']==b['file'] and a['page']==b['page']) else (0.3 if a['file']==b['file'] else 0.0)
    picked, out = set(), []
    while len(out) < min(top_k, len(hits)):
        scores = []
        for i in range(len(hits)):
            if i in picked:
                scores.append(-1e9); continue
            red = 0.0 if not picked else max(sim[i,j] for j in picked)
            scores.append(lambda_mult*rel[i] - (1-lambda_mult)*red)
        i_best = int(np.argmax(scores))
        picked.add(i_best); out.append(hits[i_best])
    for r,h in enumerate(out,1):
        h['rank'] = r
    return out

def retrieve_then_rerank(query: str, top_k=8, alpha=0.6):
    row = {"Query": query, "Tools": ["retriever"], "CacheHits": 0, "Tokens": 0}
    with timeblock(row, "T_total"):
        with timeblock(row, "T_retrieve"):
            raw_hits = hybrid_search(query, top_k=top_k, alpha=alpha)
        with timeblock(row, "T_rerank"):
            hits = mmr_rerank(raw_hits, lambda_mult=0.7, top_k=min(5, top_k))
    instr.log(row)
    return hits

# -- citation helpers --
def format_citation(hit: dict) -> str:
    parts = [hit["file"]]
    if hit.get("year_qtr"): parts.append(hit["year_qtr"])
    parts.append(f"p.{hit['page']}")
    sec = _clean_section_hint(hit)
    if sec: parts.append(sec)
    return " — ".join(parts)

# Map doc_id -> full chunk text to give richer context to the LLM (not just preview)
_doc_text_map = None
def _build_doc_text_map():
    global _doc_text_map
    if _doc_text_map is None:
        _doc_text_map = {c.doc_id: c.text for c in chunks}
    return _doc_text_map

def _context_from_hits(hits: List[Dict[str,Any]], max_chars_per_chunk=1200, top_ctx=3) -> str:
    m = _build_doc_text_map()
    ctx_blocks = []
    for h in hits[:top_ctx]:
        text = m.get(h["doc_id"], h.get("preview","")) or ""
        text = text.strip().replace("\u0000"," ")
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + " ..."
        ctx_blocks.append(
            f"[{format_citation(h)}]\n{text}"
        )
    return "\n\n".join(ctx_blocks)

# -- ONE Gemini call (no caching) --
def answer_with_gemini(query: str, top_k_retrieval=16, top_ctx=3, model_name="gemini-2.5-flash-latest") -> Dict[str,Any]:
    # 1) Retrieve + rerank
    hits = retrieve_then_rerank(query, top_k=top_k_retrieval, alpha=0.6)

    ql = (query or "").lower()
    # If the user asks for "last three years", keep only most recent 3 distinct years
    if ("last three years" in ql) or ("last 3 years" in ql) or ("past three years" in ql):
        hits = _keep_recent_years(hits, n_years=3)
    # If the user asks for "last five quarters", prefer quarterly decks/summaries
    if ("last five quarters" in ql) or ("last 5 quarters" in ql):
        hits = _prefer_quarterly(hits)

    # 2) Prepare prompt with explicit instruction to cite
    context = _context_from_hits(hits, max_chars_per_chunk=1200, top_ctx=top_ctx)
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

    # 3) Call Gemini ONCE
    row = {"Query": f"[generate] {query}", "Tools": ["generator"], "CacheHits": 0, "Tokens": 0}
    try:
        import google.generativeai as genai
    except Exception as e:
        raise SystemExit("google-generativeai package not installed. Run: pip install google-generativeai") from e

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY. Set os.environ['GEMINI_API_KEY'] = '...'.")
    genai.configure(api_key=api_key)

    with timeblock(row, "T_total"):
        with timeblock(row, "T_reason"):
            # (Place for any pre-LLM reasoning like light parsing if needed)
            pass
        with timeblock(row, "T_generate"):
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", "") or ""
            # Try to record token usage if available; else estimate
            try:
                usage = resp.usage_metadata
                row["Tokens"] = int((usage.prompt_token_count or 0) + (usage.candidates_token_count or 0))
            except Exception:
                # naive estimate: 4 chars ≈ 1 token
                row["Tokens"] = int(len(prompt)//4 + len(text)//4)

    instr.log(row)

    # 4) Guarantee citations are present (append explicit list of top contexts)
    explicit_citations = "\n".join(f"- {format_citation(h)}" for h in hits[:top_ctx])
    final_answer = text.strip()
    if not final_answer:
        final_answer = "No answer generated."
    final_answer += "\n\nCitations:\n" + explicit_citations

    return {
        "answer": final_answer,
        "hits": hits[:top_ctx],
        "raw_model_text": text
    }

# --- quick demo calls (ONE LLM CALL EACH; no caching) ---
demo_queries = [
    "Net Interest Margin (NIM) trend over the last five quarters; provide the values and 1–2 lines of explanation.",
    "Operating expenses YoY for the last three years; list the top three drivers from MD&A.",
    "Cost-to-Income ratio for the last three years; show your working and implications."
]
for q in demo_queries:
    out = answer_with_gemini(q, top_k_retrieval=10, top_ctx=3)
    print("\nQ:", q, "\n")
    print(out["answer"])