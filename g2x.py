#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
g2x.py — Agentic RAG with tools on top of data_marker/ (FAISS + Marker outputs)
       - BM25, Reciprocal Rank Fusion, and Cross-Encoder Reranking

Artifacts required in ./data_marker:
  - kb_index.faiss
  - kb_index_meta.json
  - kb_texts.npy
  - kb_chunks.parquet
  - kb_tables.parquet        (recommended for table tools)
  - kb_outline.parquet       (optional, for section hints)

Tools exposed:
  1) CalculatorTool           -> safe arithmetic, deltas, YoY
  2) TableExtractionTool      -> pull metric rows; extract {year -> value}
  3) MultiDocCompareTool      -> compare a metric across multiple docs
Also:
  - Vector search (FAISS) for grounding

Agent runtime: Plan -> Act -> Observe -> (optional) Refine -> Final
"""

from pathlib import Path
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI

import re, json, math, ast
import numpy as np
import pandas as pd
import faiss
import os

# ----------------------------- LLM (single-call baseline) -----------------------------

def _make_llm_client():
    """Minimal provider selection for LLM"""
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        return ("groq", client, model)
    
    gem_key = os.environ.get("GEMINI_API_KEY")
    if gem_key:
        return ("gemini", None, os.getenv("GEMINI_MODEL_NAME", "models/gemini-2.5-flash"))
    
    raise RuntimeError("No LLM credentials found. Set GROQ_API_KEY or GEMINI_API_KEY.")

def _llm_provider_info() -> str:
    try:
        prov, _, model = _make_llm_client()
        return f"{prov}:{model}"
    except Exception as e:
        return f"unconfigured ({e})"

def _llm_single_call(prompt: str, system: str = "You are a precise finance analyst.") -> str:
    prov, client, model = _make_llm_client()
    print(f"[LLM] provider={prov} model={model}")
    if prov == "groq":
        try:
            chat = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            return chat.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM error: {e}"
    
    try:
        from google import generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model_obj = genai.GenerativeModel(model)
        out = model_obj.generate_content(prompt)
        return getattr(out, "text", "") or "LLM returned empty response."
    except Exception as e:
        return f"LLM error (Gemini): {e}"


def _page_or_none(x):
    try:
        import math
        import pandas as pd
        if x is None:
            return None
        if (hasattr(pd, 'isna') and pd.isna(x)) or (isinstance(x, float) and math.isnan(x)):
            return None
        return int(x)
    except Exception:
        return None


# ----------------------------- KB loader with BM25 + Reranker -----------------------------

class KBEnv:
    def __init__(self, base="./data_marker", enable_bm25=True, enable_reranker=True):
        self.base = Path(base)
        self.faiss_path = self.base / "kb_index.faiss"
        self.meta_path = self.base / "kb_index_meta.json"
        self.texts_path = self.base / "kb_texts.npy"
        self.chunks_path = self.base / "kb_chunks.parquet"
        self.tables_path = self.base / "kb_tables.parquet"
        self.outline_path = self.base / "kb_outline.parquet"

        if not self.faiss_path.exists():
            raise FileNotFoundError(self.faiss_path)
        if not self.meta_path.exists():
            raise FileNotFoundError(self.meta_path)
        if not self.texts_path.exists():
            raise FileNotFoundError(self.texts_path)
        if not self.chunks_path.exists():
            raise FileNotFoundError(self.chunks_path)

        self.texts: List[str] = np.load(self.texts_path, allow_pickle=True).tolist()
        self.meta_df: pd.DataFrame = pd.read_parquet(self.chunks_path)
        
        if 'page' in self.meta_df.columns:
            self.meta_df['page'] = pd.to_numeric(self.meta_df['page'], errors='coerce').astype('Int64')
            
        if len(self.texts) != len(self.meta_df):
            raise ValueError(f"texts ({len(self.texts)}) and meta ({len(self.meta_df)}) mismatch")

        self.tables_df: Optional[pd.DataFrame] = (
            pd.read_parquet(self.tables_path) if self.tables_path.exists() else None
        )
        self.outline_df: Optional[pd.DataFrame] = (
            pd.read_parquet(self.outline_path) if self.outline_path.exists() else None
        )

        # FAISS index
        self.index = faiss.read_index(str(self.faiss_path))
        idx_meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.model_name = idx_meta.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.embed_dim = int(idx_meta.get("dim", 384))
        self.model = SentenceTransformer(self.model_name)

        # ========== NEW: BM25 Index ==========
        self.bm25 = None
        if enable_bm25:
            # print("[BM25] Building BM25 index...")
            tokenized_corpus = [text.lower().split() for text in self.texts]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"[BM25] ✓ Indexed {len(self.texts)} documents")
        elif enable_bm25:
            print("[BM25] ✗ rank_bm25 not installed, skipping BM25")

        # ========== NEW: Reranker ==========
        self.reranker = None
        if enable_reranker:
            # print("[Reranker] Loading cross-encoder...")
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("[Reranker] ✓ Loaded cross-encoder/ms-marco-MiniLM-L-6-v2")
        elif enable_reranker:
            print("[Reranker] ✗ CrossEncoder unavailable")

    def _embed(self, texts: List[str]) -> np.ndarray:
        v = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(v, dtype="float32")

    # ========== NEW: Hybrid Search with BM25 + Vector + RRF ==========
    def search(
        self, 
        query: str, 
        k: int = 12,
        alpha: float = 0.6,  # Weight for vector vs BM25 (0.0=pure BM25, 1.0=pure vector)
        rerank_top_k: int = None  # Rerank top candidates (default: 2*k)
    ) -> pd.DataFrame:
        """
        Hybrid search with BM25 + Vector + optional RRF + optional Reranking
        
        Pipeline:
        1. BM25 search → get scores
        2. Vector search → get scores
        3. Fusion: RRF (reciprocal rank) or weighted score fusion
        4. Rerank: Cross-encoder on top candidates
        5. Return top-k
        """
        if rerank_top_k is None:
            rerank_top_k = k * 2  # Get 2x candidates for reranking

        # ========== Step 1: Vector Search ==========
        qv = self._embed([query])
        vec_scores, vec_idxs = self.index.search(qv, min(rerank_top_k * 2, len(self.texts)))
        vec_idxs, vec_scores = vec_idxs[0], vec_scores[0]
        
        # Filter valid indices
        vec_results = {int(i): float(s) for i, s in zip(vec_idxs, vec_scores) if i >= 0 and i < len(self.texts)}

        # ========== Step 2: BM25 Search ==========
        bm25_results = {}
        if self.bm25 is not None:
            query_tokens = query.lower().split()
            bm25_scores = self.bm25.get_scores(query_tokens)
            
            # Normalize BM25 scores to [0, 1]
            max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1.0
            if max_bm25 > 0:
                bm25_scores = bm25_scores / max_bm25
            
            # Get top candidates
            top_bm25_idx = np.argsort(bm25_scores)[-rerank_top_k * 2:][::-1]
            bm25_results = {int(i): float(bm25_scores[i]) for i in top_bm25_idx if bm25_scores[i] > 0}

        # ========== Step 3: Fusion (RRF or Weighted Score) ==========
        all_indices = set(vec_results.keys()) | set(bm25_results.keys())
        
        if self.bm25 is not None:
            # Reciprocal Rank Fusion
            vec_ranks = {idx: rank for rank, idx in enumerate(sorted(vec_results, key=vec_results.get, reverse=True), 1)}
            bm25_ranks = {idx: rank for rank, idx in enumerate(sorted(bm25_results, key=bm25_results.get, reverse=True), 1)}
            
            k_rrf = 60  # RRF constant
            fused_scores = {}
            for idx in all_indices:
                vec_rank = vec_ranks.get(idx, len(self.texts))
                bm25_rank = bm25_ranks.get(idx, len(self.texts))
                fused_scores[idx] = (1 / (k_rrf + vec_rank)) + (1 / (k_rrf + bm25_rank))
            
            print(f"[Search] RRF fusion: {len(all_indices)} candidates")
        else:
            # Weighted score fusion (fallback if BM25 disabled or RRF=False)
            fused_scores = {}
            for idx in all_indices:
                vec_score = vec_results.get(idx, 0.0)
                bm25_score = bm25_results.get(idx, 0.0)
                fused_scores[idx] = alpha * vec_score + (1 - alpha) * bm25_score
            
            print(f"[Search] Weighted fusion (α={alpha}): {len(all_indices)} candidates")

        # Sort by fused score
        sorted_indices = sorted(fused_scores.keys(), key=fused_scores.get, reverse=True)[:rerank_top_k]

        # ========== Step 4: Reranking (Optional) ==========
        if self.reranker is not None and len(sorted_indices) > k:
            print(f"[Rerank] Reranking top-{len(sorted_indices)} candidates...")
            
            # Prepare query-document pairs
            pairs = [[query, self.texts[idx]] for idx in sorted_indices]
            
            # Get rerank scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Update fused scores with rerank scores
            for idx, score in zip(sorted_indices, rerank_scores):
                fused_scores[idx] = float(score)
            
            # Re-sort by rerank scores
            sorted_indices = sorted(sorted_indices, key=fused_scores.get, reverse=True)
            
            print(f"[Rerank] ✓ Reranked to top-{k}")

        # ========== Step 5: Build Results DataFrame ==========
        final_indices = sorted_indices[:k]
        rows = []
        for rank, idx in enumerate(final_indices, start=1):
            md = self.meta_df.iloc[idx]
            item = {
                "rank": rank,
                "score": fused_scores[idx],
                "text": self.texts[idx],
                "doc": md.get("doc"),
                "path": md.get("path"),
                "modality": md.get("modality"),
                "chunk": int(md.get("chunk", 0)),
                "page": _page_or_none(md.get("page")),
            }
            
            # Section hint
            if self.outline_df is not None:
                toc = self.outline_df[self.outline_df["doc_name"] == item["doc"]]
                if not toc.empty:
                    item["section_hint"] = toc.iloc[0]["title"]
            
            rows.append(item)
        
        return pd.DataFrame(rows)
    
def baseline_answer_one_call(
    kb: KBEnv,
    query: str,
    k_ctx: int = 8,
    table_rows: Optional[List[Dict[str, Any]]] = None
) -> dict:
    """
    Baseline (Stage 4) requirements:
      - Naive chunking (we use existing kb_texts)
      - Single-pass vector search (FAISS only)
      - One LLM call, no caching
    """
    # 1) Retrieve top-k chunks
    ctx_df = kb.search(query, k=k_ctx)
    if ctx_df is None or ctx_df.empty:
        answer = "I couldn't find any relevant context in the KB for this query."
        print(answer)
        return {"answer": answer, "contexts": []}

    # 2) Build context and simple citations
    ctx_lines = []
    for _, row in ctx_df.iterrows():
        text = str(row["text"]).replace("\\n", " ").strip()
        if len(text) > 800:
            text = text[:800] + "..."
        ctx_lines.append(f"- {text}")

    # We will build citations later; prefer table-row provenance if provided
    cits = []

    # Build citations: prefer structured table rows with pages
    if table_rows:
        for r in table_rows[:5]:
            doc = str(r.get("doc") or "")
            page = r.get("page")
            if page is not None:
                cits.append(f"{doc}, page {int(page)}")
            else:
                cits.append(f"{doc}, table {r.get('table_id')} row {r.get('row_id')} (no page)")
    else:
        for _, row in ctx_df.iterrows():
            doc = str(row.get("doc") or "")
            mod = str(row.get("modality") or "")
            page = row.get("page")
            if page is not None:
                cits.append(f"{doc}, page {page}")
            else:
                ch = int(row.get("chunk") or 0)
                if mod in ("md", "table_row"):
                    cits.append(f"{doc}, chunk {ch} (no page; {mod})")
                else:
                    cits.append(f"{doc}, chunk {ch} (no page)")

    # Optional: include structured table rows so the LLM doesn't deny available data
    table_lines = []
    if table_rows:
        table_lines.append("STRUCTURED TABLE ROWS (authoritative):")
        for r in table_rows[:6]:
            ser_q = r.get("series_q") or {}
            ser_y = r.get("series") or {}
            if ser_q:
                def _qkey(k: str):
                    m = re.match(r"([1-4])Q(20\\d{2})$", k)
                    return (int(m.group(2)), int(m.group(1))) if m else (0, 0)
                qkeys = sorted(ser_q.keys(), key=_qkey)[-5:]
                table_lines.append(f"- {r.get('doc')} | {r.get('label')} | " + ", ".join(f"{k}: {ser_q[k]}" for k in qkeys))
            elif ser_y:
                ys = sorted(ser_y.keys())[-3:]
                table_lines.append(f"- {r.get('doc')} | {r.get('label')} | " + ", ".join(f"{y}: {ser_y[y]}" for y in ys))

    # 3) Compose strict prompt
    if table_lines:
        # When we have structured rows, exclude noisy text snippets to avoid conflicting numbers.
        prompt = (
            "USER QUESTION:\n"
            f"{query}\n\n"
            + "\n".join(table_lines) + "\n\n"
            "INSTRUCTIONS:\n"
            "- Use ONLY the numbers in STRUCTURED TABLE ROWS for calculations and final values.\n"
            "- If the task asks for 'Operating Income' but only 'Total income' is present, use 'Total income' as the denominator.\n"
            "- Do NOT refuse or say 'data missing' if the required numbers appear in the structured rows provided.\n"
            "- If a requested period is not present in these rows, say so explicitly (do NOT infer from narrative text).\n"
            "- Return a concise answer, then a small table if applicable.\n"
        )
    else:
        prompt = (
            "USER QUESTION:\n"
            f"{query}\n\n"
            "CONTEXT (verbatim snippets from the reports):\n"
            + "\n".join(ctx_lines) +
            "\n\nINSTRUCTIONS:\n"
            "- Use ONLY facts present in the CONTEXT; do not invent numbers. If values are not present, explicitly state which ones are missing.\n"
            "- If the exact values for the requested periods are not present, say so explicitly.\n"
            "- Return a concise answer, then a small table if applicable, then a 'Citations' bullet list with 2â€“5 items.\n"
        )

    # 4) One LLM call
    print(f"[LLM] single-call baseline using {_llm_provider_info()}")
    answer = _llm_single_call(prompt)

    # 5) Print nicely in notebooks
    print("""\nBASELINE (Single LLM Call)\n--------------------------------""")
    print(answer)
    print("\nCitations:")
    for c in cits[:5]:
        print(f"- {c}")

    return {"answer": answer, "contexts": ctx_df.head(5)}
    

# ----------------------------- Tool: Calculator -----------------------------

class CalculatorTool:
    """
    Safe arithmetic eval (supports +,-,*,/,**, parentheses) and helpers for deltas/YoY.
    """

    ALLOWED = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Load,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
        ast.Mod, ast.FloorDiv, ast.Constant, ast.Call, ast.Name
    }
    SAFE_FUNCS = {"round": round, "abs": abs}

    @classmethod
    def safe_eval(cls, expr: str) -> float:
        node = ast.parse(expr, mode="eval")
        for n in ast.walk(node):
            if type(n) not in cls.ALLOWED:
                raise ValueError(f"Disallowed expression: {type(n).__name__}")
            if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name) and n.func.id in cls.SAFE_FUNCS):
                raise ValueError("Only round(...) and abs(...) calls are allowed")
        code = compile(node, "<expr>", "eval")
        return float(eval(code, {"__builtins__": {}}, cls.SAFE_FUNCS))

    @staticmethod
    def delta(a: float, b: float) -> float:
        return float(a) - float(b)

    @staticmethod
    def yoy(a: float, b: float) -> Optional[float]:
        b = float(b)
        if b == 0: return None
        return (float(a) - b) / b * 100.0


# ----------------------------- Tool: Table Extraction -----------------------------

class TableExtractionTool:
    """
    Look up a metric row in kb_tables.parquet and extract {year -> value_num}.
    Heuristic: find any row where any cell (value_str) contains the metric term,
    then collect all cells in that row whose column is a 4-digit year.
    """

    # --- normalization helpers & synonyms (for robust matching) ---
    @staticmethod
    def _norm(s: str) -> str:
        """Lowercase, replace '&' with 'and', strip punctuation, collapse spaces."""
        if s is None:
            return ""
        s = str(s).lower()
        s = s.replace("&", " and ")
        s = re.sub(r"[^a-z0-9 ]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # Expanded metric synonyms
    SYNONYMS = {
        # NIM
        "nim": ["net interest margin", "nim", "net interest margin group", "nim group"],
        "net interest margin": ["net interest margin", "nim", "net interest margin group", "nim group"],
        # Gross margin (treat as NIM for banks)
        "gross margin": ["net interest margin", "nim", "net interest margin group", "nim group", "gross margin"],
        # Opex
        "operating expenses and income": [
            "operating expenses and income",
            "operating expenses",
            "total expenses",
            "expenses",
        ],
        "operating expenses": [
            "operating expenses",
            "total expenses",
            "expenses",
            "opex",
        ],
        "total expenses": [
            "total expenses",
            "expenses",
            "operating expenses",
            "opex",
        ],
        # Income
        "operating income": [
            "operating income",
            "total operating income",
            "total income",
            "income",
        ],
        "total income": [
            "total income",
            "operating income",
            "total operating income",
            "income",
        ],
    }

    def __init__(self, tables_df: Optional[pd.DataFrame]):
        self.df = tables_df

    @staticmethod
    def _is_year(col: str) -> bool:
        return bool(re.fullmatch(r"\d{4}", str(col).strip()))

    @staticmethod
    def _parse_quarter_token(col: str):
        """
        Parse common quarter column labels like '1Q24', '1Q 2024', 'Q1 2024', '4QFY24'.
        Returns a tuple (year:int, quarter:int, display:str) or None if not a quarter.
        """
        s = str(col).strip()
        # 1) Compact form like '1Q24' or '4Q2024'
        m = re.search(r'(?i)\b([1-4])\s*q\s*((?:20)?\d{2})\b', s)
        if not m:
            # 2) 'Q1 2024' or 'Q3 FY24'
            m = re.search(r'(?i)\bq\s*([1-4])\s*(?:fy)?\s*((?:20)?\d{2})\b', s)
        if not m:
            # 3) '([1-4])Q((?:20)?\d{2})' without space
            m = re.search(r'(?i)\b([1-4])q((?:20)?\d{2})\b', s)
        if not m:
            return None
        q = int(m.group(1))
        ytxt = m.group(2)
        y = int(ytxt)
        if y < 100:  # normalize '24' -> 2024
            y += 2000
        display = f"{q}Q{y}"
        return (y, q, display)

    @staticmethod
    def _is_quarter(col: str) -> bool:
        return TableExtractionTool._parse_quarter_token(col) is not None

    def get_metric_rows(self, metric: str, doc: Optional[str] = None, limit: int = 5):
        if self.df is None or self.df.empty:
            return []
        base_df = self.df

        # Build normalized copies for robust matching
        df = base_df.assign(
            _val_norm=base_df["value_str"].astype(str).map(self._norm),
            _col_norm=base_df["column"].astype(str).map(self._norm),
        )

        metric_norm = self._norm(metric)
        cand_terms = self.SYNONYMS.get(metric_norm, [metric_norm])

        mask = pd.Series(False, index=df.index)
        for term in cand_terms:
            term_norm = self._norm(term)
            mask = mask | df["_val_norm"].str.contains(term_norm, na=False) | df["_col_norm"].str.contains(term_norm, na=False)

        if doc:
            mask = mask & (df["doc_name"] == doc)

        if not mask.any():
            return []

        # --- ORIENTATION A: metric appears as a COLUMN header; quarters are in ROW label cells ---
        results: List[Dict[str, Any]] = []
        table_keys = (
            df.loc[mask, ["doc_name", "table_id"]]
              .drop_duplicates()
              .itertuples(index=False, name=None)
        )
        for (d, t) in table_keys:
            tbl = base_df[(base_df["doc_name"] == d) & (base_df["table_id"] == t)].copy()
            if tbl.empty:
                continue
            # normalized copies to detect metric column(s)
            tbln = tbl.assign(
                _val_norm=tbl["value_str"].astype(str).map(self._norm),
                _col_norm=tbl["column"].astype(str).map(self._norm),
            )
            # columns whose header contains the metric term
            metric_cols = sorted(tbln.loc[tbln["_col_norm"].str.contains(metric_norm, na=False), "column"].unique().tolist())
            if metric_cols:
                mcol = str(metric_cols[0])
                # build series_q by iterating all rows in the table and picking the metric cell + a quarter label cell
                series_q: Dict[str, float] = {}
                series_y: Dict[int, float] = {}
                series_pct: Dict[int, float] = {}
                pages_seen: list[int] = []
                for rid in sorted(tbl["row_id"].unique()):
                    row_cells = tbl[tbl["row_id"] == rid]
                    # collect page numbers for this row (if available)
                    try:
                        pser = row_cells.get("page")
                        if pser is not None:
                            pages_seen += [int(p) for p in pser.dropna().astype(int).tolist()]
                    except Exception:
                        pass
                    # find the cell for the metric column in this row
                    mcell = row_cells[row_cells["column"].astype(str) == mcol]
                    if mcell.empty:
                        continue
                    val = mcell.iloc[0].get("value_num")
                    # also try to pick YoY % values when the metric column header is a YoY column
                    # e.g., column header contains 'yoy' or '%'
                    for _, rc in row_cells.iterrows():
                        ctext = str(rc.get("column") or "")
                        if re.search(r"(?i)yoy|%", ctext):
                            try:
                                ylab = (rc.get("value_str") or "").strip()
                                if self._is_year(ylab):
                                    vnum = rc.get("value_num")
                                    if pd.notna(vnum):
                                        series_pct[int(ylab)] = float(vnum)
                            except Exception:
                                pass
                    # find a row label that looks like a quarter or a year in any non-year/quarter column
                    label_text = None
                    for _, rc in row_cells.iterrows():
                        vstr = (rc.get("value_str") or "").strip()
                        if not vstr:
                            continue
                        # prefer quarter tokens
                        qtok = self._parse_quarter_token(vstr)
                        if qtok:
                            disp = qtok[2]
                            label_text = disp
                            break
                        # else maybe pure year row label like "2024"
                        if self._is_year(vstr):
                            label_text = vstr
                            break
                    if pd.notna(val) and label_text:
                        # decide if it's quarter or year
                        qtok2 = self._parse_quarter_token(label_text)
                        if qtok2:
                            series_q[qtok2[2]] = float(val)
                        elif self._is_year(label_text):
                            try:
                                series_y[int(label_text)] = float(val)
                            except Exception:
                                pass
                page_val = None
                if pages_seen:
                    try:
                        page_val = max(set(pages_seen), key=pages_seen.count)
                    except Exception:
                        page_val = pages_seen[-1]
                if series_q or series_y:
                    # label: use the metric column header text
                    label = str(mcol)
                    results.append({
                        "doc": d,
                        "table_id": int(t),
                        "row_id": -1,  # synthetic aggregation over rows
                        "label": label,
                        "series": series_y,
                        "series_q": series_q,
                        "series_pct": series_pct,
                        "page": page_val,
                    })

        # stop early if we already found enough good quarter rows
        if results and len(results) >= limit:
            # rank quarter-first
            def _rank_q(r):
                sq = r.get("series_q", {}) or {}
                def _qkey(k: str):
                    m = re.match(r"([1-4])Q(20\\d{2})$", k)
                    if m:
                        return (int(m.group(2)), int(m.group(1)))
                    return (0, 0)
                if sq:
                    qkeys = sorted(sq.keys(), key=_qkey)
                    latest_qy, latest_q = _qkey(qkeys[-1]) if qkeys else (0, 0)
                    return ( -len(sq), -latest_qy, -latest_q, 0, 0 )
                years = sorted((results[0].get("series") or {}).keys())
                latest_y = years[-1] if years else 0
                return ( 0, 0, 0, -len(years), -latest_y )
            results.sort(key=_rank_q)
            return results[:limit]

        # --- ORIENTATION B (fallback): metric appears as a ROW label; years/quarters are COLUMNS ---
        key_cols = ["doc_name", "table_id", "row_id"]
        row_keys = (
            df.loc[mask, key_cols]
              .drop_duplicates()
              .itertuples(index=False, name=None)
        )

        for (d, t, r) in row_keys:
            # Load the FULL row from the base dataframe (not the masked slice)
            row_cells = base_df[(base_df["doc_name"] == d) & (base_df["table_id"] == t) & (base_df["row_id"] == r)]
            if row_cells.empty:
                continue

            # choose a representative page for this row
            page_val = None
            try:
                pser = row_cells.get("page")
                if pser is not None:
                    vals = [int(p) for p in pser.dropna().astype(int).tolist()]
                    if vals:
                        page_val = max(set(vals), key=vals.count)
            except Exception:
                pass

            # Determine label
            label = None
            rc_norm = row_cells.assign(
                _val_norm=row_cells["value_str"].astype(str).map(self._norm),
                _col_norm=row_cells["column"].astype(str).map(self._norm),
            )
            metric_hits = rc_norm[~rc_norm["column"].astype(str).map(self._is_year) & rc_norm["_val_norm"].str.contains(metric_norm, na=False)]
            if not metric_hits.empty:
                label = (metric_hits.iloc[0]["value_str"] or "").strip()
            if not label:
                non_year = row_cells[~row_cells["column"].astype(str).map(self._is_year)]
                if not non_year.empty:
                    label = (non_year.iloc[0]["value_str"] or "").strip() or str(non_year.iloc[0]["column"])
            if not label:
                label = f"row {int(r)}"

            # Build year and quarter series from ALL cells in this row
            series: Dict[int, float] = {}
            series_q: Dict[str, float] = {}
            for _, cell in row_cells.iterrows():
                col = str(cell["column"]).strip()
                val = cell.get("value_num")
                if pd.isna(val):
                    continue
                if self._is_year(col):
                    try:
                        y = int(col); series[y] = float(val); continue
                    except Exception:
                        pass
                qtok = self._parse_quarter_token(col)
                if qtok:
                    series_q[qtok[2]] = float(val)

            results.append({
                "doc": d,
                "table_id": int(t),
                "row_id": int(r),
                "label": label,
                "series": series,
                "series_q": series_q,
                "page": page_val
            })

        # Rank results: quarters first by count/recency, then years
        def _row_rank(r):
            sq = r.get("series_q", {}) or {}
            def _qkey(k: str):
                m = re.match(r"([1-4])Q(20\\d{2})$", k)
                if m:
                    return (int(m.group(2)), int(m.group(1)))
                return (0, 0)
            if sq:
                qkeys = sorted(sq.keys(), key=_qkey)
                latest_qy, latest_q = _qkey(qkeys[-1]) if qkeys else (0, 0)
                return ( -len(sq), -latest_qy, -latest_q, 0, 0 )
            years = sorted(r["series"].keys())
            latest_y = years[-1] if years else 0
            return ( 0, 0, 0, -len(years), -latest_y )

        results.sort(key=_row_rank)
        return results[:limit]

    @staticmethod
    def last_n_years(series: Dict[int, float], n: int = 3) -> List[Tuple[int, float]]:
        ys = sorted(series.keys())
        return [(y, series[y]) for y in ys[-n:]]


#
# ----------------------------- Tool: Text Extraction (fallback for quarters) -----------------------------
class TextExtractionTool:
    """
    Regex-based fallback when Marker tables don't carry the quarter series.
    Currently focuses on percentage metrics like Net Interest Margin (NIM).
    It scans the KB text chunks and tries to pair quarter tokens with the nearest % value.
    """
    QPAT = re.compile(r"(?i)(?:\b([1-4])\s*q\s*((?:20)?\d{2})\b|\bq\s*([1-4])\s*((?:20)?\d{2})\b|\b([1-4])q((?:20)?\d{2})\b)")
    PCT = re.compile(r"(?i)(\d{1,2}(?:\.\d{1,2})?)\s*%")

    def __init__(self, kb: 'KBEnv'):
        self.kb = kb

    @staticmethod
    def _norm(s: str) -> str:
        return TableExtractionTool._norm(s)

    @staticmethod
    def _mk_qdisp(q: int, y: int) -> str:
        if y < 100: y += 2000
        return f"{q}Q{y}"

    def extract_quarter_pct(self, metric: str, top_k_text: int = 200) -> Dict[str, float]:
        metric_n = self._norm(metric)
        hits = self.kb.search(metric, k=top_k_text)
        if hits is None or hits.empty:
            return {}
        series_q: Dict[str, float] = {}
        for _, row in hits.iterrows():
            txt = str(row["text"])
            # Quick filter: only consider chunks that mention the metric name
            if metric_n not in self._norm(txt):
                continue
            # Find all quarter tokens in this chunk
            quarts = []
            for m in self.QPAT.finditer(txt):
                # groups: (q1,y1) or (q2,y2) or (q3,y3)
                if m.group(1):   q, y = int(m.group(1)), int(m.group(2))
                elif m.group(3): q, y = int(m.group(3)), int(m.group(4))
                else:            q, y = int(m.group(5)), int(m.group(6))
                if y < 100: y += 2000
                quarts.append((q, y, m.start(), m.end()))
            if not quarts:
                continue
            # Find % values; take the nearest % to each quarter mention
            pcts = [(pm.group(1), pm.start(), pm.end()) for pm in self.PCT.finditer(txt)]
            if not pcts:
                continue
            MAX_CHARS = 48  # require proximity
            for (q, y, qs, qe) in quarts:
                best = None; best_d = 1e9
                for (val, ps, pe) in pcts:
                    d = min(abs(ps - qe), abs(pe - qs))
                    if d < best_d and d <= MAX_CHARS:
                        try:
                            num = float(val)
                        except Exception:
                            continue
                        # sanity for NIM-like percentages
                        if 0.0 <= num <= 6.0:
                            best_d = d; best = num
                if best is not None:
                    disp = self._mk_qdisp(q, y)
                    series_q[disp] = float(best)
        return series_q

# ----------------------------- Tool: Multi-Doc Compare -----------------------------

class MultiDocCompareTool:
    """
    Compare the same metric across multiple docs by pulling each doc's row
    and extracting aligned year/value pairs.
    """

    def __init__(self, table_tool: TableExtractionTool):
        self.table_tool = table_tool

    def compare(self, metric: str, years: Optional[List[int]] = None, top_docs: int = 6):
        # get top rows across all docs
        rows = self.table_tool.get_metric_rows(metric, limit=50)
        if not rows:
            return []
        # take first occurrence per doc
        seen = set()
        picked = []
        for r in rows:
            if r["doc"] in seen: 
                continue
            seen.add(r["doc"])
            picked.append(r)
            if len(picked) >= top_docs:
                break
        # align years
        if years is None:
            all_years = set()
            for r in picked:
                all_years.update(r["series"].keys())
            years = sorted(all_years)[-3:]  # default: last 3 years available
        out = []
        for r in picked:
            values = {y: r["series"].get(y) for y in years}
            out.append({"doc": r["doc"], "label": r["label"], "years": years, "values": values})
        return out


# ----------------------------- Agent: plan → act → observe -----------------------------

@dataclass
class AgentResult:
    plan: List[str]
    actions: List[str]
    observations: List[str]
    final: Dict[str, Any]

class Agent:
    """
    Very small rule-based planner:
      - If query has 'compare', 'vs', 'across docs' → MultiDocCompareTool
      - Else try TableExtractionTool for a metric row
      - If calculation phrasing (yoy, growth, %), compute deltas with CalculatorTool
      - Always fetch top-k vector contexts for grounding
    """

    def __init__(self, kb: KBEnv):
        self.kb = kb
        self.calc = CalculatorTool()
        self.table = TableExtractionTool(kb.tables_df)
        self.compare_tool = MultiDocCompareTool(self.table)
        self.text_tool = TextExtractionTool(kb)

    @staticmethod
    def _extract_metric(query: str) -> Optional[str]:
        # naive metric detection: quoted phrase or capitalized words
        quoted = re.findall(r'"([^"]+)"', query)
        if quoted:
            return quoted[0]
        # common finance metrics heuristics
        candidates = [
            r"net interest margin", r"nim", r"gross margin",
            r"operating expenses(?: &| and)?(?: income)?",
            r"operating income", r"operating profit",
            r"total income", r"cost-to-income", r"allowances", r"profit before tax",
        ]
        ql = query.lower()
        for pat in candidates:
            m = re.search(pat, ql)
            if m:
                return m.group(0)
        # fallback: capitalized phrase
        m2 = re.findall(r'\b([A-Z][A-Za-z&% ]{3,})\b', query)
        return m2[0] if m2 else None

    @staticmethod
    def _want_compare(query: str) -> bool:
        return bool(re.search(r"\b(compare|vs\.?|versus|across docs?|between)\b", query, re.I))

    @staticmethod
    def _want_yoy(query: str) -> bool:
        return bool(re.search(r"\b(yoy|year[- ]over[- ]year|growth|change|%|delta)\b", query, re.I))

    @staticmethod
    def _want_quarters(query: str) -> bool:
        return bool(re.search(r"\bquarter|quarters|\bq[1-4]\b", query, re.I))

    @staticmethod
    def _extract_years(query: str) -> List[int]:
        years = [int(y) for y in re.findall(r"\b(20\d{2})\b", query)]
        # de-dup and sort
        return sorted(set(years))

    def run(self, query: str, k_ctx: int = 6) -> AgentResult:
        plan, actions, observations = [], [], []
        final: Dict[str, Any] = {}

        plan.append("1) Ground the question with vector search for context.")
        ctx_df = self.kb.search(query, k=k_ctx)
        observations.append(f"Vector contexts: {len(ctx_df)} found.")
        final["contexts"] = ctx_df

        metric = self._extract_metric(query)
        years = self._extract_years(query)

        if self._want_compare(query):
            plan.append("2) Compare the metric across multiple documents via table extraction.")
            if not metric:
                metric = "net interest margin"  # default guess
                observations.append("No explicit metric found; defaulting to 'net interest margin'.")
            actions.append(f"MultiDocCompareTool.compare(metric='{metric}', years={years or 'last3'})")
            compare_rows = self.compare_tool.compare(metric, years=years or None)
            observations.append(f"Compare results: {len(compare_rows)} docs.")
            final["compare"] = compare_rows
        else:
            plan.append("2) Extract the metric row from tables for the requested (or last 3) years.")
            if not metric:
                metric = "net interest margin"
                observations.append("No explicit metric found; defaulting to 'net interest margin'.")
            actions.append(f"TableExtractionTool.get_metric_rows(metric='{metric}', limit=5)")
            # Prefer quarters strictly when requested; otherwise fallback to any rows
            rows = self.table.get_metric_rows(metric, limit=50)  # fetch more candidates for better recall
            observations.append(f"Table rows matched: {len(rows)}")

            prefer_quarters = self._want_quarters(query)
            rows_q = [r for r in rows if r.get("series_q") and len(r.get("series_q") or {}) > 0]

            if prefer_quarters:
                if rows_q:
                    observations.append("User requested quarters; prioritizing rows with quarter columns.")
                    final["table_rows"] = rows_q[:5]
                else:
                    # Fallback: try text extraction for quarter-form percentages (e.g., NIM)
                    series_q_txt = self.text_tool.extract_quarter_pct(metric, top_k_text=200)
                    if series_q_txt:
                        observations.append("Quarter tables missing; recovered quarter % series from text.")
                        final["table_rows"] = [{
                            "doc": "(text_fallback)",
                            "table_id": -1,
                            "row_id": -1,
                            "label": metric,
                            "series": {},
                            "series_q": series_q_txt,
                        }]
                    else:
                        observations.append("User requested quarters but none found in indexed tables.")
                        final["table_rows"] = []
                        final["notice"] = "No quarterly data found for the requested metric in the indexed tables."
            else:
                final["table_rows"] = rows[:5]
                if rows_q:
                    observations.append("Quarterly data available; showing last 5 quarters where present.")

            if self._want_yoy(query) and (final.get("table_rows") and len(final["table_rows"]) > 0):
                plan.append("3) Compute YoY or growth using CalculatorTool on extracted series.")
                # pick the first row’s series
                series = final["table_rows"][0]["series"]
                ys = years if years else sorted(series.keys())[-2:]  # last 2 years if none given
                calc_out = []
                if len(ys) >= 2:
                    for i in range(1, len(ys)):
                        y0, y1 = ys[i-1], ys[i]
                        a, b = series.get(y1), series.get(y0)
                        if a is not None and b is not None:
                            yoy = self.calc.yoy(a, b)
                            calc_out.append({"from": y0, "to": y1, "value_from": b, "value_to": a, "yoy_pct": None if yoy is None else round(yoy, 2)})
                actions.append(f"CalculatorTool.yoy on years={ys}")
                observations.append(f"Computed {len(calc_out)} YoY deltas.")
                final["calc"] = calc_out

        final["plan"] = plan
        final["actions"] = actions
        final["observations"] = observations
        return AgentResult(plan, actions, observations, final)


# ----------------------------- Pretty print helpers -----------------------------

def _fmt_series(series: Dict[int, float], n: int = 3) -> str:
    if not series: return "—"
    ys = sorted(series.keys())[-n:]
    return ", ".join(f"{y}: {series[y]}" for y in ys)

def show_agent_result(res: AgentResult, show_ctx: int = 3):
    print("PLAN:")
    for step in res.plan:
        print("  -", step)
    print("\nACTIONS:")
    for a in res.actions:
        print("  -", a)
    print("\nOBSERVATIONS:")
    for o in res.observations:
        print("  -", o)

    fin = res.final

    # TABLE ROWS block
    if not fin.get("table_rows"):
        msg = fin.get("notice") or "No matching table rows were found for your request."
        print(f"\n⚠️ {msg}")
    elif "table_rows" in fin and fin["table_rows"]:
        print("\nTABLE ROWS (first few):")
        shown = 0
        for r in fin["table_rows"]:
            if shown >= 3:
                break
            sq = (r.get("series_q") or {})
            if sq:
                # sort quarters chronologically by (year, quarter)
                def _qkey(k):
                    m = re.match(r"([1-4])Q(20\\d{2})$", k)
                    if m:
                        return (int(m.group(2)), int(m.group(1)))
                    return (0, 0)
                qkeys = sorted(sq.keys(), key=_qkey)
                last5 = qkeys[-5:]
                ser = ", ".join(f"{k}: {sq[k]}" for k in last5)
                print(f"  doc={r['doc']} | label={r['label']} | quarters(last5)={ser}")
                shown += 1
            else:
                ys = sorted(r["series"].keys())
                ser = ", ".join(f"{y}: {r['series'][y]}" for y in ys[-3:]) if ys else "—"
                print(f"  doc={r['doc']} | label={r['label']} | years(last3)={ser}")
                shown += 1
    if "compare" in fin and fin["compare"]:
        print("\nCOMPARE (first few):")
        for r in fin["compare"][:3]:
            row = ", ".join(f"{y}: {r['values'].get(y)}" for y in r["years"])
            print(f"  doc={r['doc']} | label={r['label']} | {row}")
    if "calc" in fin and fin["calc"]:
        print("\nCALC (YoY):")
        for c in fin["calc"]:
            print(f"  {c['from']}→{c['to']}: {c['value_from']} → {c['value_to']} | YoY={c['yoy_pct']}%")

    # Contexts
    ctx = fin.get("contexts")
    if ctx is not None and not ctx.empty:
        print("\nCONTEXTS:")
        for _, row in ctx.head(show_ctx).iterrows():
            t = str(row["text"]).replace("\n", " ")
            if len(t) > 240: t = t[:237] + "..."
            hint = f" — {row.get('section_hint')}" if "section_hint" in row else ""
            print(f"  [{row['rank']}] {row['doc']} | {row['modality']}{hint}")
            print("     ", t)


# ----------------------------- CLI / Notebook ------------------------------------

# ----------------------------- Notebook Runtime ------------------------------------

# This section is safe for direct use inside a Jupyter/Colab/VSCode notebook cell.
# It avoids argparse/sys parsing and simply runs a default demo or accepts a variable `query`.

# Example usage in a notebook:
# from g2x import KBEnv, Agent, show_agent_result
# kb = KBEnv(base="./data_marker")
# agent = Agent(kb)
# res = agent.run("Compare Net Interest Margin across docs for 2022–2024")
# show_agent_result(res)

if __name__ == "__main__" or "__file__" not in globals():
    kb = KBEnv(base="./data_marker")
    agent = Agent(kb)

    try:
        query = globals().get("query", None)
    except Exception:
        query = None

    if not query:
        query = "What is the Net Interest Margin over the last 5 quarters?"
        print("ℹ️ Running notebook demo query:")
        print(f"   → {query}\n")

    # BASELINE execution (single LLM, no caching)
    out = baseline_answer_one_call(kb, query, k_ctx=8)