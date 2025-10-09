# """
# Stage2.py — DEFINITIVE FINAL VERSION
# """
# from __future__ import annotations
# import os, re, json, math, traceback
# from typing import List, Dict, Any, Optional

# import numpy as np
# import pandas as pd
# import time, contextlib

# # --- Logging Setup ---
# @contextlib.contextmanager
# def timeblock(row: dict, key: str):
#     t0 = time.perf_counter()
#     try:
#         yield
#     finally:
#         row[key] = round((time.perf_counter() - t0) * 1000.0, 2)

# class _Instr:
#     def __init__(self):
#         self.rows = []
#     def log(self, row):
#         self.rows.append(row)
#     def df(self):
#         cols = ['Query','T_retrieve','T_rerank','T_reason','T_generate','T_total','Tokens','Tools']
#         df = pd.DataFrame(self.rows)
#         for c in cols:
#             if c not in df:
#                 df[c] = None
#         return df[cols]

# instr = _Instr()

# # --- Configuration ---
# VERBOSE = bool(int(os.environ.get("AGENT_CFO_VERBOSE", "1")))
# LLM_BACKEND = "gemini"
# GEMINI_MODEL_NAME = "models/gemini-2.5-flash"

# # --- Global Variables ---
# kb: Optional[pd.DataFrame] = None
# texts: Optional[np.ndarray] = None
# index, bm25, EMB = None, None, None
# _HAVE_FAISS, _HAVE_BM25, _INITIALIZED = False, False, False

# # --- Core Logic Functions ---
# def _classify_query(q: str) -> Optional[str]:
#     ql = q.lower()
#     if re.search(r"\boperating\s+efficiency\s+ratio\b|\boer\b", ql) or ("÷" in ql and "operating" in ql and "income" in ql):
#         return "oer"
#     if "nim" in ql or "net interest margin" in ql: 
#         return "nim"
#     if "opex" in ql or "operating expense" in ql or re.search(r"\bexpenses\b", ql): 
#         return "opex"
#     if re.search(r"\b(total\s+income|operating\s+income)\b", ql):
#         return "income"
#     if re.search(r"\bcti\b|cost[\s\-_\/]*to?\s*[\s\-_\/]*income", ql): 
#         return "cti"
#     return None

# class _EmbedLoader:
#     def __init__(self):
#         self.impl, self.dim, self.name, self.fn = None, None, None, None
#     def embed(self, texts: List[str]) -> np.ndarray:
#         if self.impl is None:
#             try:
#                 from sentence_transformers import SentenceTransformer
#                 model_name = "sentence-transformers/all-MiniLM-L6-v2"
#                 st = SentenceTransformer(model_name)
#                 self.impl, self.dim = ("st", model_name), st.get_sentence_embedding_dimension()
#                 self.fn = lambda b: st.encode(b, normalize_embeddings=True).astype(np.float32)
#             except ImportError: raise RuntimeError("sentence-transformers not installed.")
#         return self.fn(texts)

# def init_stage2(out_dir: str = "data"):
#     global kb, texts, index, bm25, _HAVE_FAISS, _HAVE_BM25, _INITIALIZED, EMB
#     os.environ["AGENT_CFO_OUT_DIR"] = out_dir
#     paths = [os.path.join(out_dir, f) for f in ["kb_chunks.parquet", "kb_texts.npy", "kb_index.faiss"]]
#     if not all(os.path.exists(p) for p in paths): raise RuntimeError(f"KB artifacts not found in '{out_dir}'.")
#     kb, texts = pd.read_parquet(paths[0]), np.load(paths[1], allow_pickle=True)
#     try:
#         import faiss
#         _HAVE_FAISS, index = True, faiss.read_index(paths[2])
#     except ImportError: _HAVE_FAISS, index = False, None
#     try:
#         from rank_bm25 import BM25Okapi
#         _HAVE_BM25, bm25 = True, BM25Okapi([str(t).lower().split() for t in texts])
#     except ImportError: _HAVE_BM25, bm25 = False, None
#     EMB = _EmbedLoader()
#     _INITIALIZED = True
#     if VERBOSE: print(f"[Stage2] Initialized successfully from '{out_dir}'.")

# def _ensure_init():
#     if not _INITIALIZED: raise RuntimeError("Stage2 not initialized. Call init_stage2() first.")

# def _detect_last_n_years(q: str) -> Optional[int]:
#     m = re.search(r"last\s+(\d+|three|five)\s+(fiscal\s+)?years?", q, re.I)
#     if m:
#         try:
#             val = m.group(1).lower();
#             if val == 'three': return 3
#             if val == 'five': return 5
#             return int(val)
#         except: return None
#     return None

# def _detect_last_n_quarters(q: str) -> Optional[int]:
#     m = re.search(r"last\s+(\d+|five)\s+quarters", q, re.I)
#     if m:
#         try:
#             val = m.group(1).lower();
#             if val == 'five': return 5
#             return int(val)
#         except: return None
#     return None

# def hybrid_search(query: str, top_k=12, alpha=0.6) -> List[Dict[str, Any]]:
#     _ensure_init()
#     vec_scores, bm25_scores = {}, {}
#     if _HAVE_FAISS and index and EMB:
#         qv = EMB.embed([query]); qv /= np.linalg.norm(qv, axis=1, keepdims=True)
#         sims, ids = index.search(qv.astype(np.float32), top_k * 4)
#         vec_scores = {int(i): float(s) for i, s in zip(ids[0], sims[0]) if i != -1}
#     if _HAVE_BM25 and bm25:
#         scores = bm25.get_scores(query.lower().split())
#         top_idx = np.argsort(scores)[-top_k*4:]
#         bm25_scores = {int(i): float(scores[i]) for i in top_idx}
    
#     fused = {k: (alpha * vec_scores.get(k, 0)) + ((1 - alpha) * (bm25_scores.get(k, 0) / (max(bm25_scores.values()) or 1.0))) for k in set(vec_scores) | set(bm25_scores)}
    
#     is_annual_query = bool(re.search(r"\bfy\b|fiscal\s+year|last\s+\d+\s+years", query, re.I))
#     year_match = re.search(r'\b(20\d{2})\b', query)
#     desired_year = int(year_match.group(1)) if year_match else None

#     qtype = _classify_query(query)
#     for i in fused:
#         meta = kb.iloc[i]
#         boost = 0.0
#         text_l = str(texts[i]).lower()
#         # --- Extended domain-aware features ---
#         file_l = str(meta.file).lower()
#         section_l = (str(meta.section_hint).lower() if isinstance(meta.section_hint, str) else "")
#         mentions_nim = ("net interest margin" in text_l) or re.search(r"\bnim\b", text_l)
#         mentions_percent_nim = bool(re.search(r"net\s+interest\s+margin[^%]{0,200}%|([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:p|pts|percentage\s*points)?", text_l, flags=re.I))
#         mentions_expenses = ("operating expenses" in text_l) or re.search(r"\bexpenses\b", text_l)
#         has_money_units = bool(re.search(r"\(\$?\s*m\)|s\$\s*m|\(\$m\)|\bmillion\b|\bmn\b|\bbn\b|\bbillion\b", text_l, flags=re.I))
#         is_tableish = section_l.startswith("table_p")
#         is_vision = "vision_summary" in section_l
#         is_quarterly_doc = pd.notna(meta.quarter)
#         is_press_or_trading = bool(re.search(r"press[_\s-]?statement|trading[_\s-]?update", file_l))
#         is_corp_gov = "corporate governance" in text_l or "board of directors" in text_l
#         is_cfo_or_perf = bool(re.search(r"cfo[_\s-]?presentation|performance[_\s-]?summary", file_l))

#         # Year/annual vs quarterly alignment
#         if desired_year and pd.notna(meta.year):
#             if int(meta.year) == desired_year:
#                 boost += 5.0
#             else:
#                 boost -= 5.0

#         is_annual_doc = pd.isna(meta.quarter)
#         if is_annual_query:
#             boost += 5.0 if is_annual_doc else -5.0
#         else:
#             boost += 2.0 if not is_annual_doc else 0.0

#         # --- Domain-aware boosts ---
#         if qtype == "nim":
#             # Prefer quarterly docs and chunks explicitly mentioning NIM with a %
#             if is_quarterly_doc:
#                 boost += 4.0
#             if mentions_nim:
#                 boost += 4.0
#             if mentions_nim and mentions_percent_nim:
#                 boost += 6.0
#             # Strongly favour structured sources
#             if is_tableish and mentions_nim:
#                 boost += 5.0
#             if is_vision and (mentions_nim or "net interest margin" in text_l):
#                 boost += 5.0
#             # Penalise generic prose that often lacks explicit % values
#             if is_press_or_trading and not mentions_percent_nim:
#                 boost -= 10.0

#         if qtype == "opex" or qtype == "oer" or qtype == "cti":
#             # Prefer chunks that talk about (operating) expenses with monetary units
#             if mentions_expenses and has_money_units:
#                 boost += 6.0
#             # Extra rewards for structured/table/vision sources
#             if is_tableish and mentions_expenses:
#                 boost += 3.0
#             if is_vision and mentions_expenses:
#                 boost += 4.0
#             # Vision summary pages tend to have "For FYXXXX, Opex were NNNN million."
#             if is_vision and (mentions_expenses):
#                 boost += 5.0
#             # For Opex/CTI/OER annual asks, prefer annual docs
#             if is_annual_query and is_annual_doc:
#                 boost += 3.0
                
#         if qtype == "income":
#             if "total income" in text_l:
#                 boost += 6.0
#             if is_tableish:
#                 boost += 3.0
#             if is_vision:
#                 boost += 4.0
#             if is_annual_query and is_annual_doc:
#                 boost += 3.0

#         # Global penalties for off-topic governance prose
#         if is_corp_gov:
#             boost -= 8.0
#         # Light reward for CFO/performance decks (usually contain crisp metrics)
#         if is_cfo_or_perf:
#             boost += 2.0

#         fused[i] += boost
        
#     hits = [{"doc_id": kb.iloc[i].doc_id, "file": kb.iloc[i].file, "page": int(kb.iloc[i].page), "year": int(kb.iloc[i].year) if pd.notna(kb.iloc[i].year) else None, "quarter": int(kb.iloc[i].quarter) if pd.notna(kb.iloc[i].quarter) else None, "section_hint": kb.iloc[i].section_hint, "score": float(score)} for i, score in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]]
#     return hits

# def format_citation(hit: dict) -> str:
#     parts = [hit.get("file", "?")]
#     y = hit.get("year"); q = hit.get("quarter")
#     if y is not None and q is not None: parts.append(f"{int(q)}Q{str(int(y))[-2:]}")
#     elif y is not None: parts.append(str(int(y)))
#     if hit.get("page") is not None: parts.append(f"p.{int(hit['page'])}")
#     return ", ".join(parts)

# def _call_llm(prompt: str, dry_run: bool = False) -> str:
#     if dry_run:
#         return '{"plan": []}'
#     try:
#         from google import generativeai as genai
#         genai.configure(api_key=os.environ['GEMINI_API_KEY'])
#         model = genai.GenerativeModel(GEMINI_MODEL_NAME)
#         safety_settings = [
#             {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
#             {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
#             {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
#             {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
#         ]
#         return model.generate_content(prompt, safety_settings=safety_settings).text
#     except Exception as e:
#         return f"LLM Generation Failed: {e}"

# def tool_calculator(expression: str) -> str:
#     try:
#         s = str(expression)

#         # Guard: unresolved placeholders like ${var}
#         placeholders = re.findall(r"\$\{([^}]+)\}", s)
#         if placeholders:
#             return f"Error: unresolved placeholders: {', '.join(placeholders)}"

#         # Normalizations
#         s = re.sub(r'(?<=\d),(?=\d{3}\b)', '', s)               # 12,345 -> 12345
#         s = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'(\1/100)', s)       # 12% -> (12/100)
#         s = re.sub(r'(?i)[s]?\$\s*', '', s)                      # S$ / $ -> strip
#         s = re.sub(r'(?i)\b(bn|billion|b)\b', 'e9', s)           # bn -> e9
#         s = re.sub(r'(?i)\b(mn|million|m)\b', 'e6', s)           # mn -> e6

#         # Safety: allow only digits, + - * / ( ) . e E and spaces
#         safe = re.sub(r'[^0-9eE\+\-*/(). ]', '', s)

#         result = eval(safe)
#         return f"Result: {result}"
#     except Exception as e:
#         return f"Error: {e}"

# def _desired_periods_from_query(query: str) -> list[tuple[int|None, int|None]]:
#     out = []
#     # Quarters like 1Q25
#     for m in re.finditer(r"\b([1-4])Q(\d{2})\b", query, re.I):
#         out.append((2000 + int(m.group(2)), int(m.group(1))))

#     # FY2024 / FY 2024
#     for m in re.finditer(r"\bFY\s?(20\d{2})\b", query, re.I):
#         out.append((int(m.group(1)), None))

#     # "fiscal year 2024"
#     for m in re.finditer(r"\bfiscal\s+year\s+(20\d{2})\b", query, re.I):
#         out.append((int(m.group(1)), None))

#     # bare year (only if nothing else found)
#     if not out:
#         m = re.search(r"\b(20\d{2})\b", query)
#         if m:
#             out.append((int(m.group(1)), None))

#     return out

# def tool_table_extraction(query: str) -> str:
#     """
#     Finds a single reported data point from the knowledge base using hybrid search,
#     then extracts and cleans the most likely numerical value from the retrieved text.

#     Improvements vs. previous version:
#       • Robust row-to-text mapping using positional index (not label).
#       • Query-aware extraction (Opex → 'million' values; NIM → percentages).
#       • Period-aware filtering (prefer sentences containing requested FY/quarter).
#       • Avoids 4-digit years being misread as values.
#       • Falls back through multiple heuristics and multiple hits if needed.
#     """
#     if VERBOSE:
#         print(f"  [Tool Call: table_extraction] with query: '{query}'")

#     hits = hybrid_search(query, top_k=12)
#     # --- Vision-first rescue: ensure year-matched vision_summary candidates are in the pool ---
#     try:
#         desired_periods = _desired_periods_from_query(query)
#         desired_years = [y for (y, q) in desired_periods if y]
#         sh_series = kb["section_hint"].astype(str).str.contains("vision_summary", case=False, na=False)
#         mask = sh_series
#         if desired_years:
#             mask = mask & kb["year"].isin(desired_years)
#         vis_idxs = np.flatnonzero(mask.to_numpy())
#         base_score = (min([float(h.get("score") or 0.0) for h in hits]) - 1.0) if hits else 0.0
#         extra_hits = []
#         for idx in vis_idxs[:6]:
#             row = kb.iloc[idx]
#             extra_hits.append({
#                 "doc_id": row.doc_id,
#                 "file": row.file,
#                 "page": int(row.page) if pd.notna(row.page) else None,
#                 "year": int(row.year) if pd.notna(row.year) else None,
#                 "quarter": int(row.quarter) if pd.notna(row.quarter) else None,
#                 "section_hint": row.section_hint,
#                 "score": base_score
#             })
#         if extra_hits:
#             hits = hits + extra_hits

#         # Deduplicate by doc_id
#         seen = set()
#         deduped = []
#         for h in hits:
#             did = h.get("doc_id")
#             if did in seen:
#                 continue
#             seen.add(did)
#             deduped.append(h)
#         hits = deduped

#         # --- Priority ordering of hits: vision first, then tables, then others
#         vision_hits = [h for h in hits if "vision_summary" in str(h.get("section_hint") or "").lower()]
#         table_hits  = [h for h in hits if str(h.get("section_hint") or "").lower().startswith("table_p")]
#         other_hits  = [h for h in hits if h not in vision_hits and h not in table_hits]
#     except Exception:
#         # Fail open; rely on original hits if rescue logic errors out
#         vision_hits, table_hits, other_hits = [], [], []
#         pass

#     if not hits:
#         return "Error: No relevant documents found."

#     # Helper: map a doc_id to the correct position in `texts` using a boolean mask.
#     def _pos_of_docid(did: str) -> Optional[int]:
#         mask = (kb["doc_id"] == did).to_numpy()
#         idxs = np.flatnonzero(mask)
#         return int(idxs[0]) if idxs.size else None

#     # Helper: safer float parsing (strip commas etc.)
#     def _clean_number(s: str) -> Optional[str]:
#         t = s.strip()
#         t = re.sub(r"[,\s]", "", t)
#         # Reject years (e.g., 2024) and obviously huge integers without unit context
#         if re.fullmatch(r"\d{4}", t):
#             return None
#         try:
#             float(t)
#             return t
#         except Exception:
#             return None

#     # Helper: plausibility check for NIM
#     def _plausible_nim_value(x: float) -> bool:
#         # DBS group NIM is realistically ~0.5%–3.5%
#         try:
#             return 0.5 <= float(x) <= 3.5
#         except Exception:
#             return False
        
#     # Helper: choose the best number from text given the query intent
#     def _extract_value(text: str, query: str) -> Optional[str]:
#         ql = query.lower()
#         is_nim = ("nim" in ql) or ("net interest margin" in ql)
#         is_opex = ("opex" in ql) or ("operating expense" in ql) or re.search(r"\bexpenses\b", ql)
#         is_income = re.search(r"\b(total\s+income|operating\s+income)\b", ql) is not None
#         # Detect if this is an annual ask (not a specific quarter)
#         annual_ask = not re.search(r"\b[1-4]Q\d{2}\b", query, re.I)

#         # If the query mentions a specific period, try to narrow the search window.
#         desired_periods = _desired_periods_from_query(query)
#         windows = []
#         if desired_periods:
#             for (yy, qq) in desired_periods:
#                 if yy and qq:
#                     tag = fr"{qq}q{str(yy)[-2:]}"
#                 elif yy:
#                     tag = fr"fy{yy}"
#                 else:
#                     tag = None
#                 if tag:
#                     m = re.search(tag, text, flags=re.I)
#                     if m:
#                         # take a sentence-sized window around the tag
#                         start = max(0, text.rfind(".", 0, m.start()))
#                         end = text.find(".", m.end())
#                         if end == -1:
#                             end = len(text)
#                         windows.append(text[start:end])
#         if not windows:
#             # fallback: whole text
#             windows = [text]

#         # Query-aware patterns
#         # 1) NIM → percentages, prioritizing text near "net interest margin"
#         if is_nim:
#             # 1) Strongly anchored: look for "…margin was/to/at/of N.NN%"
#             for win in windows:
#                 m = re.search(
#                     r"net\s+interest\s+margin[^%]{0,120}?(?:was|to|at|of)\s*([0-9]+(?:\.[0-9]+)?)\s*%",
#                     win, flags=re.I | re.S
#                 )
#                 if m:
#                     v = m.group(1)
#                     if _plausible_nim_value(v):
#                         return _clean_number(v)

#             # 2) Vision-summary phrasing: "Group/Commercial Book Net Interest Margin was 2.13%."
#             for win in windows:
#                 m = re.search(
#                     r"(?:group|commercial(?:\s*book)?)\s*net\s+interest\s+margin.*?(?:was|to|at|of)\s*([0-9]+(?:\.[0-9]+)?)\s*%",
#                     win, flags=re.I | re.S
#                 )
#                 if m:
#                     v = m.group(1)
#                     if _plausible_nim_value(v):
#                         return _clean_number(v)

#             # 3) Anchored fallback: only if NIM is explicitly mentioned; pick the nearest plausible %
#             for win in windows:
#                 m_phrase = re.search(r"net\s+interest\s+margin|\bnim\b", win, flags=re.I)
#                 if not m_phrase:
#                     continue
#                 best = None
#                 best_dist = 1e9
#                 for p in re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*%", win):
#                     try:
#                         val = float(p.group(1))
#                     except Exception:
#                         continue
#                     if not _plausible_nim_value(val):
#                         continue
#                     dist = abs(p.start() - m_phrase.start())
#                     if dist < best_dist:
#                         best_dist = dist
#                         best = p.group(1)
#                 if best:
#                     return _clean_number(best)

#             # Do NOT fall back to non-% numbers for NIM; better to return None than a wrong value
#             return None

#         # 2) Opex / Operating Expenses → numbers followed by a 'million/bn' unit
#         if is_opex:
#             # --- FAST PATH (Vision summary exact sentence for annual Opex) ---
#             # Prefer the Vision-summary wording:
#             # "For FY2024, total Operating Expenses (Opex) were 8895 million."
#             try:
#                 desired_periods_fp = _desired_periods_from_query(query)
#             except Exception:
#                 desired_periods_fp = []
#             target_years_fp = [yy for (yy, qq) in desired_periods_fp if yy and (qq is None)]
#             if target_years_fp:
#                 for yy in target_years_fp:
#                     m_fp = re.search(
#                         rf"For\s*FY{yy}\s*,?\s*total\s+Operating\s+Expenses\s*\(Opex\)\s*were\s*([0-9][\d,]*(?:\.[0-9]+)?)\s*(million|mn|m|bn|billion)\b",
#                         text,
#                         flags=re.I
#                     )
#                     if m_fp:
#                         val_fp = _clean_number(m_fp.group(1)) or None
#                         unit_fp = (m_fp.group(2) or "").lower()
#                         if val_fp:
#                             try:
#                                 v_fp = float(val_fp)
#                                 if unit_fp in ("bn", "billion", "b"):
#                                     v_fp *= 1000.0
#                                 # Annual Opex sanity range in $m for DBS scale
#                                 if 2000.0 <= v_fp <= 15000.0:
#                                     return ("%g" % v_fp)
#                             except Exception:
#                                 pass
#             # Vision-summary phrasing: "For FY2024, total Operating Expenses (Opex) were 8895 million."
#             for win in windows:
#                 m = re.search(
#                     r"operating\s+expenses.*?(?:were|:)?\s*([0-9][\d,]*(?:\.[0-9]+)?)\s*(million|mn|m|bn|billion)\b",
#                     win,
#                     flags=re.I | re.S,
#                 )
#                 if m:
#                     val = _clean_number(m.group(1))
#                     unit = (m.group(2) or "").lower()
#                     if val:
#                         try:
#                             v = float(val)
#                             # Normalise units to millions
#                             if unit in ("bn", "billion", "b"):
#                                 v *= 1000.0
#                             # Annual asks must be a sensible magnitude in $m (reject too-small or absurdly large)
#                             if annual_ask and unit in ("million", "mn", "m", "bn", "billion", "b") and not (2000 <= v <= 15000):
#                                 val = None
#                             else:
#                                 val = ("%g" % v)
#                         except Exception:
#                             pass
#                     if val:
#                         return val

#             # Generic '... expenses ... 8,895 million' even without "operating"
#             for win in windows:
#                 m = re.search(
#                     r"\bexpenses\b.*?(?:were|:)?\s*([0-9][\d,]*(?:\.[0-9]+)?)\s*(million|mn|m|bn|billion)\b",
#                     win,
#                     flags=re.I | re.S,
#                 )
#                 if m:
#                     val = _clean_number(m.group(1))
#                     unit = (m.group(2) or "").lower()
#                     if val:
#                         try:
#                             v = float(val)
#                             if unit in ("bn", "billion", "b"):
#                                 v *= 1000.0
#                             # Annual asks must be a sensible magnitude in $m (reject too-small or absurdly large)
#                             if annual_ask and unit in ("million", "mn", "m", "bn", "billion", "b") and not (2000 <= v <= 15000):
#                                 val = None
#                             else:
#                                 val = ("%g" % v)
#                         except Exception:
#                             pass
#                     if val:
#                         return val

#             # Table/markdown style: headers carry units like "($m)" or "S$ m", and the value is a 4+ digit number
#             for win in windows:
#                 # e.g., "| Operating expenses | 8,895 |" or "Operating expenses 8,895"
#                 m = re.search(
#                     r"(?:operating\s+expenses|^\s*\|\s*operating\s+expenses.*?)\D([0-9][\d,]{3,})\b",
#                     win, flags=re.I | re.S | re.M
#                 )
#                 if m:
#                     val = _clean_number(m.group(1))
#                     if val:
#                         return val
#             # If the surrounding text mentions monetary units like '($m)' or 'S$ m', prefer 4+ digit numbers anywhere in the window
#             for win in windows:
#                 if re.search(r"\(\$?\s*m\)|s\$\s*m|\(\$m\)|\(\$ million\)", win, flags=re.I):
#                     m = re.search(r"\b([0-9][\d,]{3,})\b", win)
#                     if m:
#                         val = _clean_number(m.group(1))
#                         if val:
#                             return val

#             # As a last resort, only if the window itself mentions expenses/opex AND a money unit cue is present.
#             # This avoids accidentally picking unrelated large numbers from generic prose (e.g., CFO narrative pages).
#             for win in windows:
#                 if re.search(r"\b(operating\s+)?expenses?\b|\bopex\b", win, flags=re.I):
#                     # Require a nearby money unit cue to reduce false positives.
#                     if not re.search(r"\(\$?\s*m\)|s\$\s*m|\(\$m\)|\bmillion\b|\bmn\b|\bbn\b|\bbillion\b", win, flags=re.I):
#                         continue
#                     m = re.search(r"\b([0-9][\d,]{3,})\b", win)
#                     if m:
#                         val = _clean_number(m.group(1))
#                         if val:
#                             return val
                        
#         # 3) Total/Operating Income → require the phrase and a plausible 4+ digit value
#         if is_income:
#             # Prefer explicit "Total income ... NNNN"
#             for win in windows:
#                 if re.search(r"\btotal\s+income\b", win, flags=re.I):
#                     m = re.search(r"\btotal\s+income\b[^0-9]{0,60}([0-9][\d,]{3,})", win, flags=re.I)
#                     if m:
#                         val = _clean_number(m.group(1))
#                         if val:
#                             try:
#                                 v = float(val)
#                                 if 1000.0 <= v <= 50000.0:  # DBS scale in $m
#                                     return val
#                             except Exception:
#                                 pass
#             # Vision-summary phrasing: "... Total income was 22297."
#             for win in windows:
#                 m = re.search(r"\btotal\s+income\b\s*(?:was|:)?\s*([0-9][\d,]{3,})", win, flags=re.I)
#                 if m:
#                     val = _clean_number(m.group(1))
#                     if val:
#                         try:
#                             v = float(val)
#                             if 1000.0 <= v <= 50000.0:
#                                 return val
#                         except Exception:
#                             pass
#             # Markdown/table row style
#             for win in windows:
#                 m = re.search(r"(?:^\s*\|\s*)?total\s+income(?:\s*\|)?\s*([0-9][\d,]{3,})\b", win, flags=re.I | re.M)
#                 if m:
#                     val = _clean_number(m.group(1))
#                     if val:
#                         return val
#             # If the window says "$m" / "In $ millions", allow a nearby 4+ digit number
#             for win in windows:
#                 if re.search(r"\(\$?\s*m\)|in\s*\$?\s*millions", win, flags=re.I):
#                     m = re.search(r"\b([0-9][\d,]{3,})\b", win)
#                     if m:
#                         val = _clean_number(m.group(1))
#                         if val:
#                             try:
#                                 v = float(val)
#                                 if 1000.0 <= v <= 50000.0:
#                                     return val
#                             except Exception:
#                                 pass
#             # Avoid grabbing random numbers (like '31' from dates)
#             return None

#         # 4) Generic fallback: only for non-domain queries. For NIM/Opex, avoid bogus picks.
#         if not (is_nim or is_opex or is_income):
#             for win in windows:
#                 m = re.search(r"(-?\$?S?\s*[0-9][\d,]*(?:\.[0-9]+)?)", win)
#                 if m:
#                     val = re.sub(r"[S$\s]", "", m.group(1))
#                     val = _clean_number(val)
#                     if val:
#                         return val

#         return None

#     # --- Hard preference for Vision hits when Opex asks for a specific FY ---
#     try:
#         ql_pref = query.lower()
#         is_opex_pref = ("opex" in ql_pref) or ("operating expense" in ql_pref) or re.search(r"\bexpenses\b", ql_pref)
#         desired_periods_pref = _desired_periods_from_query(query)
#         explicit_fy_years = [yy for (yy, qq) in desired_periods_pref if yy and (qq is None)]
#         if is_opex_pref and explicit_fy_years:
#             yy = explicit_fy_years[0]
#             vision_for_year = [h for h in hits if "vision_summary" in str(h.get("section_hint") or "").lower() and h.get("year") == yy]
#             if vision_for_year:
#                 # Put those Vision hits first to be tried before any prose/table chunks
#                 rest = [h for h in hits if h not in vision_for_year]
#                 hits = vision_for_year + rest
#     except Exception:
#         pass

#     # Local rerank of hits to prefer structured/vision chunks for domain queries
#     ql = query.lower()
#     is_nim = ("nim" in ql) or ("net interest margin" in ql)
#     is_opex = ("opex" in ql) or ("operating expense" in ql) or re.search(r"\bexpenses\b", ql)
#     is_income = re.search(r"\b(total\s+income|operating\s+income)\b", ql) is not None

#     def _local_hit_score(h: dict) -> float:
#         sh = str(h.get("section_hint") or "").lower()
#         file_l = str(h.get("file") or "").lower()
#         s = 0.0

#         # Pull the actual text for content checks
#         pos = _pos_of_docid(h.get("doc_id", ""))
#         text_l = str(texts[pos]).lower() if pos is not None else ""

#         mentions_nim = ("net interest margin" in text_l) or re.search(r"\bnim\b", text_l) is not None
#         mentions_expenses = ("operating expenses" in text_l) or re.search(r"\bexpenses\b", text_l) is not None
#         mentions_total_income = re.search(r"\btotal\s+income\b", text_l) is not None
#         has_money_units = re.search(r"\(\$?\s*m\)|s\$\s*m|\(\$m\)|\bmillion\b|\bmn\b|\bbn\b|\bbillion\b", text_l, flags=re.I) is not None
#         mentions_percent = "%" in text_l

#         if "vision_summary" in sh:
#             s += 500.0
#         if sh.startswith("table_p"):
#             s += 30.0

#         # For NIM, demand the NIM phrase be present; otherwise heavily penalize
#         if is_nim:
#             if h.get("quarter") is not None:
#                 s += 20.0
#             if mentions_nim:
#                 s += 20.0
#                 if mentions_percent:
#                     s += 10.0
#             else:
#                 s -= 80.0  # do not allow non-NIM tables to outrank true NIM chunks

#         # For Opex-like asks, require expenses to be mentioned; favor money units
#         if is_opex:
#             if mentions_expenses:
#                 s += 20.0
#                 if has_money_units:
#                     s += 8.0
#             else:
#                 s -= 60.0  # push away tables/pages without expenses language
#             # Prefer structured sources over plain prose when scores tie
#             if sh == "prose":
#                 s -= 5.0
                
#         if is_income:
#             if "vision_summary" in sh:
#                 s += 60.0
#             if sh.startswith("table_p"):
#                 s += 25.0
#             if mentions_total_income:
#                 s += 20.0
#             else:
#                 s -= 40.0
#             if re.search(r"\(\$?\s*m\)|in\s*\$?\s*millions", text_l, flags=re.I):
#                 s += 6.0

#         # Deprioritize press/trading noise for numeric extractions
#         if re.search(r"press[_\s-]?statement|trading[_\s-]?update", file_l):
#             s -= 30.0

#         # fall back to hybrid score to break ties
#         s += float(h.get("score") or 0.0) * 0.01
#         return s

#     if is_nim or is_opex:
#         # Order: vision → tables → other, each block locally reranked
#         hits = (
#             sorted(vision_hits, key=_local_hit_score, reverse=True) +
#             sorted(table_hits,  key=_local_hit_score, reverse=True) +
#             sorted(other_hits,  key=_local_hit_score, reverse=True)
#         )

#     # Snapshot of the current hit ordering (useful for debugging/reuse in nested helpers)
#     _hits_snapshot = hits[:]

#     # Try the top-k hits in order until we successfully extract a plausible value
#     last_citation = None
#     for hit in hits:
#         pos = _pos_of_docid(hit["doc_id"])
#         if pos is None:
#             continue

#         text_content = str(texts[pos])
#         citation = f"Source: {format_citation(hit)}"
#         last_citation = citation

#         value = _extract_value(text_content, query)
#         if value is not None:
#             return f"Value: {value}, {citation}"

#     # If we got here, extraction failed for all hits
#     return f"Error: No numerical value found in the relevant document chunk. {last_citation or ''}"
  
# def tool_nim_series(last_n: int = 5, variant: str = "group") -> str:
#     """
#     Extract the last N quarters of Net Interest Margin (Group or Commercial Book).
#     Retrieval: FAISS (semantic) + BM25 (keyword) hybrid via hybrid_search().
#     Parsing priority: Vision summaries (nim_analysis-style lines), then structured tables,
#     then generic 'quarter → %' mentions anchored to NIM.
#     """
#     # --- 1) Gather a broader candidate pool (multiple queries) ---
#     queries = [
#         "Net interest margin (%)",
#         "NIM (%)",
#         "Group Net Interest Margin quarterly",
#         "Commercial book Net Interest Margin (%)",
#         "Net interest margin group commercial"
#     ]
#     hits: List[Dict[str, Any]] = []
#     seen_doc_ids = set()
#     for q in queries:
#         for h in hybrid_search(q, top_k=40):
#             did = h.get("doc_id")
#             if did not in seen_doc_ids:
#                 seen_doc_ids.add(did)
#                 hits.append(h)

#     # Always include any vision_summary chunks (often hold clean 'For 2Q24, Group NIM was 2.13%' lines)
#     try:
#         sh_series = kb["section_hint"].astype(str).str.contains("vision_summary", case=False, na=False)
#         vis_idxs = np.flatnonzero(sh_series.to_numpy())
#         base_score = (min([float(h.get("score") or 0.0) for h in hits]) - 1.0) if hits else 0.0
#         for idx in vis_idxs[:20]:
#             row = kb.iloc[idx]
#             did = row.doc_id
#             if did in seen_doc_ids:
#                 continue
#             seen_doc_ids.add(did)
#             hits.append({
#                 "doc_id": row.doc_id,
#                 "file": row.file,
#                 "page": int(row.page) if pd.notna(row.page) else None,
#                 "year": int(row.year) if pd.notna(row.year) else None,
#                 "quarter": int(row.quarter) if pd.notna(row.quarter) else None,
#                 "section_hint": row.section_hint,
#                 "score": base_score
#             })
#     except Exception:
#         pass

#     # --- Helper: fetch raw text for a hit ---
#     def _pos_of_docid(did: str) -> Optional[int]:
#         mask = (kb["doc_id"] == did).to_numpy()
#         idxs = np.flatnonzero(mask)
#         return int(idxs[0]) if idxs.size else None

#     # --- Helper: plausibility filter for NIM values (in %) ---
#     def _nim_ok(x: float) -> bool:
#         try:
#             xf = float(x)
#         except Exception:
#             return False
#         return 0.5 <= xf <= 3.5

#     # --- 2) Parse points: map ("2Q25","group|commercial") → value ---
#     from typing import Tuple
#     points: Dict[Tuple[str, str], float] = {}

#     # Order candidates: vision → tables → other
#     vision_hits = [h for h in hits if "vision_summary" in str(h.get("section_hint") or "").lower()]
#     table_hits  = [h for h in hits if str(h.get("section_hint") or "").lower().startswith("table_p")]
#     other_hits  = [h for h in hits if h not in vision_hits and h not in table_hits]
#     ordered = vision_hits + table_hits + other_hits

#     # --- 3) Parsing routines ---
#     re_qtr  = re.compile(r"\b([1-4]Q\d{2})\b", flags=re.I)
#     re_pct  = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")
#     re_num  = re.compile(r"([0-9]+(?:\.[0-9]+)?)")  # for tables where % sign is omitted
#     re_nim_phrase = re.compile(r"net\s*interest\s*margin|\bnim\b", flags=re.I)

#     def _maybe_add(qlabel: str, who: str, val: float):
#         who_norm = "commercial" if "commercial" in who.lower() else "group"
#         key = (qlabel.upper(), who_norm)
#         if _nim_ok(val) and key not in points:
#             points[key] = float(val)

#     for h in ordered:
#         pos = _pos_of_docid(h.get("doc_id", ""))
#         if pos is None:
#             continue
#         text = str(texts[pos])

#         # Skip chunks that don't obviously mention NIM to avoid 5% from unrelated places
#         if not re_nim_phrase.search(text):
#             continue

#         # (A) Vision-style lines from g1.format_vision_json_to_text
#         for m in re.finditer(
#             r"For\s+([1-4]Q\d{2}),\s+the\s+(Group|Commercial(?:\s*book)?)\s+Net\s+Interest\s+Margin.*?([0-9]+(?:\.[0-9]+)?)\s*%",
#             text, flags=re.I
#         ):
#             qlabel, who, val = m.group(1), m.group(2), float(m.group(3))
#             _maybe_add(qlabel, who, val)

#         # (B) Markdown table row like: "| Net interest margin (%) | 2Q25 | 1Q25 | ...\n| ... | 2.61 | 2.70 | ..."
#         lines = text.splitlines()
#         header_quarters: Optional[List[str]] = None
#         for li, line in enumerate(lines):
#             # Update current header_quarters if this line looks like a quarter header row
#             q_in_line = re_qtr.findall(line.upper())
#             if len(q_in_line) >= 2:
#                 header_quarters = q_in_line

#             if re.search(r"net\s*interest\s*margin|\bnim\b", line, flags=re.I):
#                 # 1) Same-line values (e.g., '| Net interest margin (%) | 2.61 | 2.70 | ...')
#                 vals_inline = [float(x) for x in re_num.findall(line) if _nim_ok(x)]
#                 if header_quarters and len(vals_inline) >= len(header_quarters):
#                     for ql, v in zip(header_quarters, vals_inline[:len(header_quarters)]):
#                         _maybe_add(ql, "group", float(v))

#                 # 2) Next-line values (common in markdown tables: headers then a metrics row on the next line)
#                 if li + 1 < len(lines):
#                     nxt = lines[li + 1]
#                     vals_next = [float(x) for x in re_num.findall(nxt) if _nim_ok(x)]
#                     if header_quarters and len(vals_next) >= len(header_quarters):
#                         for ql, v in zip(header_quarters, vals_next[:len(header_quarters)]):
#                             _maybe_add(ql, "group", float(v))

#         # (C) Generic anchored fallback:
#         # For each quarter mention, search a short window to the right for a plausible % or number.
#         # Expand the window to 160 chars to capture "… 2Q25 … NIM … 2.61%".
#         for m in re.finditer(r"([1-4]Q\d{2})", text, flags=re.I):
#             span_end = min(len(text), m.end() + 160)
#             window = text[m.start():span_end]
#             if not re_nim_phrase.search(window):
#                 continue
#             m_pct = re_pct.search(window)
#             if m_pct:
#                 val = float(m_pct.group(1))
#                 if _nim_ok(val):
#                     _maybe_add(m.group(1), "group", val)
#                     continue
#             # If % sign omitted in tables, allow a plain number in plausible range
#             m_num = re_num.search(window)
#             if m_num:
#                 try:
#                     val = float(m_num.group(1))
#                 except Exception:
#                     val = None
#                 if val is not None and _nim_ok(val):
#                     _maybe_add(m.group(1), "group", val)

#     # --- 4) Keep only the requested variant & take most recent N points ---
#     series = []
#     for (qlabel, who), val in points.items():
#         if (variant == "group" and who == "group") or (variant != "group" and who != "group"):
#             qnum = int(qlabel[0])
#             yy = int(qlabel[2:])
#             year = 2000 + yy
#             series.append((year, qnum, qlabel.upper(), float(val)))

#     if not series:
#         return "Error: No NIM values found."

#     series.sort(key=lambda t: (t[0], t[1]), reverse=True)
#     take = max(1, int(last_n or 5))
#     series = series[:take]

#     formatted = ", ".join(f"{ql}: {v:.2f}%" for (_, _, ql, v) in series)
#     who_title = "Group" if variant == "group" else "Commercial Book"
#     return f"NIM ({who_title}) last {len(series)} quarters → {formatted}"
  
# def tool_multi_document_compare(topic: str, files: list[str]) -> str:
#     results = []
#     for file_name in files:
#         hits = hybrid_search(f"{topic} in file {file_name}", top_k=2)
#         file_hits = [h for h in hits if h.get('file') == file_name]
#         if file_hits:
#             top_hit = file_hits[0]
#             citation = format_citation(top_hit)
#             text_content = texts[kb.index[kb['doc_id'] == top_hit['doc_id']][0]]
#             results.append(f"Source: [{citation}]\nContent: {text_content[:800]}")
#         else:
#             results.append(f"Source: {file_name}\nContent: No relevant information found.")
#     return "\n---\n".join(results)

# def _compile_or_repair_plan(query: str, plan: list[dict]) -> list[dict]:
#     def _has_params(step: dict) -> bool:
#         params = step.get("parameters")
#         return isinstance(params, dict) and any(v not in (None, "", []) for v in params.values())

#     if plan and all(_has_params(s) for s in plan):
#         return plan

#     qtype = _classify_query(query)
#     want_years  = _detect_last_n_years(query)
#     want_quarts = _detect_last_n_quarters(query)
    
#     df = kb.copy()
#     df["y"] = pd.to_numeric(df["year"], errors="coerce")
#     df["q"] = pd.to_numeric(df["quarter"], errors="coerce")
#     steps: list[dict] = []

#     if qtype == "nim":
#         n = want_quarts or 5
#         steps.append({
#             "step": f"Extract last {n} quarters of NIM (group)",
#             "tool": "nim_series",
#             "parameters": {"last_n": n, "variant": "group"},
#             "store_as": f"nim_series_last_{n}"
#         })
#         return steps

#     if qtype == "opex":
#         # If the user asked for a specific fiscal year (e.g., "FY2024" or "fiscal year 2024"),
#         # do a single extraction for that year and STOP. Do not add YoY steps.
#         periods = _desired_periods_from_query(query)
#         explicit_fy = [y for (y, q) in periods if y and (q is None)]
#         if explicit_fy:
#             y = int(explicit_fy[0])
#             steps.append({
#                 "step": f"Extract Opex for FY{y}",
#                 "tool": "table_extraction",
#                 "parameters": {"query": f"Operating expenses for fiscal year {y}"},
#                 "store_as": f"opex_fy{y}"
#             })
#             return steps

#         # Otherwise, assume a multi‑year ask. Default to the last 3 fiscal years and include a YoY calc.
#         n = want_years or 3
#         df_local = kb.copy()
#         df_local["y"] = pd.to_numeric(df_local["year"], errors="coerce")
#         df_local["q"] = pd.to_numeric(df_local["quarter"], errors="coerce")

#         ydf = df_local[df_local["q"].isna()].dropna(subset=["y"]).sort_values("y", ascending=False)
#         if ydf.empty:
#             ydf = df_local.dropna(subset=["y"]).sort_values("y", ascending=False)

#         years = [int(y) for y in ydf["y"].drop_duplicates().head(n)]
#         for y in years:
#             steps.append({
#                 "step": f"Extract Opex for FY{y}",
#                 "tool": "table_extraction",
#                 "parameters": {"query": f"Operating expenses for fiscal year {y}"},
#                 "store_as": f"opex_fy{y}"
#             })
#         if len(years) >= 2:
#             y0, y1 = years[0], years[1]
#             steps.append({
#                 "step": f"Compute YoY % change FY{y0} vs FY{y1}",
#                 "tool": "calculator",
#                 "parameters": {"expression": f"((${{opex_fy{y0}}} - ${{opex_fy{y1}}}) / ${{opex_fy{y1}}}) * 100"},
#                 "store_as": f"opex_yoy_{y0}_{y1}"
#             })
#         return steps
    
#     if qtype == "oer":
#         n = want_years or 3
#         ydf = df[df["q"].isna()].dropna(subset=["y"]).sort_values("y", ascending=False)
#         if ydf.empty: ydf = df.dropna(subset=["y"]).sort_values("y", ascending=False)
#         years = [int(y) for y in ydf["y"].drop_duplicates().head(n)]
#         for y in years:
#             steps.append({ "step": f"Extract Opex for FY{y}", "tool": "table_extraction", "parameters": {"query": f"Operating expenses for fiscal year {y}"}, "store_as": f"opex_fy{y}"})
#             steps.append({ "step": f"Extract Operating Income for FY{y}", "tool": "table_extraction", "parameters": {"query": f"Total income for fiscal year {y}"}, "store_as": f"income_fy{y}"})
#             steps.append({ "step": f"Compute OER for FY{y}", "tool": "calculator", "parameters": {"expression": f"(${{opex_fy{y}}} / ${{income_fy{y}}}) * 100"}, "store_as": f"oer_fy{y}"})
#         return steps
    
#     return [{"step": "Extract relevant figure", "tool": "table_extraction", "parameters": {"query": query}, "store_as": "value_1"}]

# def answer_with_agent(query: str, dry_run: bool = False) -> Dict[str, Any]:
#     _ensure_init()
#     execution_log = []
    
#     planning_prompt = f"""You are a financial analyst agent. Create a JSON plan to answer the user's query.
# Tools Available:
# - `table_extraction(query: str)`: Finds a single reported data point.
# - `calculator(expression: str)`: Calculates a math expression.
# User Query: "{query}"
# Return ONLY a valid JSON object with a "plan" key."""
#     if VERBOSE: print("[Agent] Step 1: Generating execution plan...")
    
#     plan_response = _call_llm(planning_prompt, dry_run)
#     plan = []
    
#     if dry_run:
#         plan = _compile_or_repair_plan(query, [])
#         answer = f"DRY RUN MODE: The agent generated the following plan and stopped before execution.\n\n{json.dumps(plan, indent=2)}"
#         return {"answer": answer, "hits": [], "execution_log": [{"step": "Planning", "plan": plan}]}

#     try:
#         json_match = re.search(r'```json\s*(\{.*?\})\s*```', plan_response, re.DOTALL)
#         plan_str = json_match.group(1) if json_match else plan_response
#         plan = json.loads(plan_str)["plan"]
#         execution_log.append({"step": "Planning", "plan": plan})
#         if VERBOSE: print("[Agent] Plan generated successfully.")
#     except Exception:
#         if VERBOSE: print("[Agent] LLM failed to generate valid plan. Using deterministic repair.")
#         plan = []

#     plan = _compile_or_repair_plan(query, plan)
#     if not execution_log or "repaired_plan" not in execution_log[0]:
#         execution_log.insert(0, {"step": "PlanRepair", "repaired_plan": plan})
    
#     if VERBOSE: print("[Agent] Step 2: Executing plan...")
#     tool_mapping = {
#         "calculator": tool_calculator,
#         "table_extraction": tool_table_extraction,
#         "multi_document_compare": tool_multi_document_compare,
#         "nim_series": tool_nim_series
#     }
#     execution_state = {}
    
#     for i, step in enumerate(plan):
#         tool = step.get("tool")
#         params = step.get("parameters", {}).copy() # Use copy to avoid modifying plan dict
#         store_as = step.get("store_as")

#         for p_name, p_value in params.items():
#             if isinstance(p_value, str):
#                 for var_name, var_value in execution_state.items():
#                     p_value = p_value.replace(f"${{{var_name}}}", str(var_value))
#             params[p_name] = p_value
        
#         try:
#             if tool not in tool_mapping:
#                 raise ValueError(f"Tool '{tool}' not found.")
            
#             result = tool_mapping[tool](**params)
#             execution_log.append({"step": f"Execution {i+1}", "tool_call": f"{tool}({params})", "result": result})
            
#             if store_as:
#                 val_for_state = result # Default to full result
#                 m_calc = re.search(r'Result:\s*([-\d\.]+e?[-\d]*)', result, re.I)
#                 if m_calc: val_for_state = m_calc.group(1)
                
#                 m_val = re.search(r'Value:\s*([^,]+)', result, re.I)
#                 if m_val: val_for_state = m_val.group(1).strip()

#                 execution_state[store_as] = val_for_state

#         except Exception as e:
#             execution_log.append({"step": f"Execution {i+1}", "tool_call": f"{tool}({params})", "error": traceback.format_exc()})

#     if VERBOSE: print("[Agent] Step 3: Synthesizing final answer...")
#     synthesis_prompt = f"""You are Agent CFO. Provide a final answer to the user's query based ONLY on the provided Tool Execution Log.
# User Query: "{query}"
# Tool Execution Log:
# {json.dumps(execution_log, indent=2)}
# Final Answer:"""
#     final_answer = _call_llm(synthesis_prompt)
    
#     return {"answer": final_answer, "hits": [], "execution_log": execution_log}

# def get_logs():
#     return instr.df()

# if __name__ == "__main__":
#     init_stage2()
#     print("[Stage2] Ready. Use answer_with_llm() or answer_with_agent().")


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
    Finds a single reported data point from the knowledge base using hybrid search,
    then extracts and cleans the most likely numerical value from the retrieved text.

    (This is the robust version from g2.py, now enhanced with the superior NIM-parsing logic
    from g2x.py to correctly handle decimals.)
    """
    if VERBOSE:
        print(f"  [Tool Call: table_extraction] with query: '{query}'")

    hits = hybrid_search(query, top_k=12)
    # Vision-first rescue logic...
    try:
        desired_periods = _desired_periods_from_query(query)
        desired_years = [y for (y, q) in desired_periods if y]
        sh_series = kb["section_hint"].astype(str).str.contains("vision_summary", case=False, na=False)
        mask = sh_series
        if desired_years:
            mask = mask & kb["year"].isin(desired_years)
        vis_idxs = np.flatnonzero(mask.to_numpy())
        base_score = (min([float(h.get("score") or 0.0) for h in hits]) - 1.0) if hits else 0.0
        extra_hits = [kb.iloc[idx] for idx in vis_idxs[:6]]
        if extra_hits:
            # Reconstruct hit dicts for consistency
            reconstructed_hits = []
            for _, row in pd.DataFrame(extra_hits).iterrows():
                 reconstructed_hits.append({
                    "doc_id": row.doc_id, "file": row.file, 
                    "page": int(row.page) if pd.notna(row.page) else None,
                    "year": int(row.year) if pd.notna(row.year) else None,
                    "quarter": int(row.quarter) if pd.notna(row.quarter) else None,
                    "section_hint": row.section_hint, "score": base_score
                })
            hits.extend(reconstructed_hits)
            
        # Deduplicate by doc_id
        seen = set()
        deduped = []
        for h in hits:
            did = h.get("doc_id")
            if did not in seen:
                seen.add(did)
                deduped.append(h)
        hits = deduped
    except Exception:
        pass

    if not hits:
        return "Error: No relevant documents found."

    def _pos_of_docid(did: str) -> Optional[int]:
        mask = (kb["doc_id"] == did).to_numpy()
        idxs = np.flatnonzero(mask)
        return int(idxs[0]) if idxs.size else None

    def _clean_number(s: str) -> Optional[str]:
        t = (s or "").strip().replace(",", "")
        if re.fullmatch(r"\d{4}", t): return None
        try:
            float(t)
            return t
        except Exception: return None

    def _plausible_nim_value(x: float) -> bool:
        try:
            return 0.5 <= float(x) <= 5.0
        except Exception: return False
        
    def _extract_value(text: str, query: str) -> Optional[str]:
        ql = query.lower()
        is_nim = ("nim" in ql) or ("net interest margin" in ql)
        is_opex = ("opex" in ql) or ("operating expense" in ql) or re.search(r"\bexpenses\b", ql)
        is_income = re.search(r"\b(total\s+income|operating\s+income)\b", ql) is not None
        
        # --- START MERGED NIM LOGIC ---
        if is_nim:
            # Pattern to find numbers like 2.13, 1.45 etc.
            nim_pattern = r"([0-9]\.[0-9]{1,2})"
            
            # 1. Prioritize text with explicit NIM mentions and a plausible decimal number
            for m in re.finditer(r"(net\s+interest\s+margin|\bnim\b).{0,80}" + nim_pattern, text, re.I | re.S):
                val_str = m.group(2)
                if _plausible_nim_value(val_str):
                    return val_str

            # 2. Vision summary sentences
            for m in re.finditer(r"margin\s*(?:was|to|at|of)?\s*" + nim_pattern, text, re.I):
                val_str = m.group(1)
                if _plausible_nim_value(val_str):
                    return val_str

            # 3. Fallback for any plausible decimal number if NIM is mentioned anywhere
            if re.search(r"net\s+interest\s+margin|\bnim\b", text, re.I):
                for m in re.finditer(nim_pattern, text):
                    val_str = m.group(1)
                    if _plausible_nim_value(val_str):
                        return val_str
            return None # Avoid returning incorrect integers for NIM
        # --- END MERGED NIM LOGIC ---
        
        if is_opex:
            # (Keeping the robust Opex logic from g2.py)
            try:
                desired_periods_fp = _desired_periods_from_query(query)
            except Exception: desired_periods_fp = []
            target_years_fp = [yy for (yy, qq) in desired_periods_fp if yy and (qq is None)]
            if target_years_fp:
                for yy in target_years_fp:
                    m_fp = re.search(rf"For\s*FY{yy}\s*,?\s*total\s+Operating\s+Expenses\s*\(Opex\)\s*were\s*([0-9][\d,]*(?:\.[0-9]+)?)\s*(million|mn|m|bn|billion)\b", text, re.I)
                    if m_fp:
                        val_fp = _clean_number(m_fp.group(1)) or None
                        unit_fp = (m_fp.group(2) or "").lower()
                        if val_fp:
                            v_fp = float(val_fp)
                            if unit_fp in ("bn", "billion", "b"): v_fp *= 1000.0
                            if 2000.0 <= v_fp <= 15000.0: return ("%g" % v_fp)

            m = re.search(r"operating\s+expenses.*?(?:were|:)?\s*([0-9][\d,]*(?:\.[0-9]+)?)\s*(million|mn|m|bn|billion)\b", text, re.I | re.S)
            if m:
                val = _clean_number(m.group(1))
                if val: return ("%g" % (float(val) * 1000.0 if (m.group(2) or "").lower() in ("bn", "billion", "b") else float(val)))
        
        if is_income:
            m = re.search(r"\btotal\s+income\b\s*(?:was|:)?\s*([0-9][\d,]{3,})", text, re.I)
            if m:
                val = _clean_number(m.group(1))
                if val and (1000.0 <= float(val) <= 50000.0): return val
        return None

    last_citation = "No documents found."
    for hit in hits:
        pos = _pos_of_docid(hit["doc_id"])
        if pos is None: continue
        text_content = str(texts[pos])
        citation = f"Source: {format_citation(hit)}"
        last_citation = citation
        value = _extract_value(text_content, query)
        if value is not None:
            # Append '%' for NIM results for clarity in the log
            unit = "%" if (("nim" in query.lower()) or ("net interest margin" in query.lower())) else ""
            return f"Value: {value}{unit}, {citation}"

    return f"Error: No numerical value found. Last attempted source: {last_citation}"
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
    
    qtype = _classify_query(query)
    # For standardized queries, ALWAYS use the deterministic plan. Ignore the LLM's output.
    if qtype in ["nim", "opex", "oer", "cti"]:
        
        want_years  = _detect_last_n_years(query)
        want_quarts = _detect_last_n_quarters(query)
        df = kb.copy()
        df["y"] = pd.to_numeric(df["year"], errors="coerce")
        df["q"] = pd.to_numeric(df["quarter"], errors="coerce")
        steps: list[dict] = []

        if qtype == "nim":
            n = want_quarts or 5
            steps.append({
                "step": f"Extract last {n} quarters of NIM (group)",
                "tool": "nim_series",
                "parameters": {"last_n": n, "variant": "group"},
                "store_as": f"nim_series_last_{n}"
            })
            return steps

        if qtype == "opex":
            n = want_years or 3
            ydf = df[df["q"].isna()].dropna(subset=["y"]).sort_values("y", ascending=False)
            if ydf.empty: ydf = df.dropna(subset=["y"]).sort_values("y", ascending=False)
            years = [int(y) for y in ydf["y"].drop_duplicates().head(n)]
            for y in years:
                steps.append({ "step": f"Extract Opex for FY{y}", "tool": "table_extraction", "parameters": {"query": f"Operating expenses for fiscal year {y}"}, "store_as": f"opex_fy{y}"})
            if len(years) >= 2:
                y0, y1 = years[0], years[1]
                steps.append({ "step": f"Compute YoY % change FY{y0} vs FY{y1}", "tool": "calculator", "parameters": {"expression": f"((${{opex_fy{y0}}} - ${{opex_fy{y1}}}) / ${{opex_fy{y1}}}) * 100"}, "store_as": f"opex_yoy_{y0}_{y1}"})
            return steps
        
        if qtype == "oer":
            n = want_years or 3
            ydf = df[df["q"].isna()].dropna(subset=["y"]).sort_values("y", ascending=False)
            if ydf.empty: ydf = df.dropna(subset=["y"]).sort_values("y", ascending=False)
            years = [int(y) for y in ydf["y"].drop_duplicates().head(n)]
            for y in years:
                steps.append({ "step": f"Extract Opex for FY{y}", "tool": "table_extraction", "parameters": {"query": f"Operating expenses for fiscal year {y}"}, "store_as": f"opex_fy{y}"})
                steps.append({ "step": f"Extract Operating Income for FY{y}", "tool": "table_extraction", "parameters": {"query": f"Total income for fiscal year {y}"}, "store_as": f"income_fy{y}"})
                steps.append({ "step": f"Compute OER for FY{y}", "tool": "calculator", "parameters": {"expression": f"(${{opex_fy{y}}} / ${{income_fy{y}}}) * 100"}, "store_as": f"oer_fy{y}"})
            return steps
            
    # Fallback for non-standardized queries: trust the LLM if the plan looks valid.
    def _has_params(step: dict) -> bool:
        params = step.get("parameters")
        return isinstance(params, dict) and any(v not in (None, "", []) for v in params.values())

    if plan and all(_has_params(s) for s in plan):
        return plan
    
    # Final fallback if LLM plan is invalid and it's not a standard query type
    return [{"step": "Extract relevant figure", "tool": "table_extraction", "parameters": {"query": query}, "store_as": "value_1"}]

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