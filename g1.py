# g1_final_v2.py (Per-PDF JSON Outputs, ArrowKeyError Fix)
from __future__ import annotations
import os
import re
import json
import uuid
import pathlib
import statistics
import io
import time
from typing import List, Dict, Any, Optional, Tuple

# Main Libraries
import pandas as pd
import numpy as np
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image

# Gemini Vision API
import google.generativeai as genai

# ML/Vector Imports
try:
    import faiss
    _HAVE_FAISS = True
except ImportError:
    _HAVE_FAISS = False
from sentence_transformers import SentenceTransformer

# --- 1. Configuration ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DATA_DIR = os.environ.get("AGENT_CFO_DATA_DIR", "All")
OUT_DIR = os.environ.get("AGENT_CFO_OUT_DIR", "data")
CACHE_DIR = os.path.join(OUT_DIR, "vision_cache")

# --- Vision API Toggle ---
USE_VISION_MODEL = False

pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

SEARCH_TERMS = {
    "Expenses Chart": ["(E) Expenses", "Excludes one-time items", "Cost / income (%)"],
    "Five-Year Summary": ["Financial statements", "DBS Group Holdings and its Subsidiaries"],
    "NIM Chart": ["Net interest margin (%)", "Group", "Commercial book"]
}

# --- 2. All Helper Functions (pdfplumber, Heuristics, Gemini) ---

# --- Patterns and Basic Helpers ---
QTR_PAT = re.compile(r"^(?:[1-4]Q|[12]H)\d{2}$", re.IGNORECASE)
NUM_PAT = re.compile(r"^\s*(-?\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?\s*%?\s*$")
MONTH_PAT = re.compile(r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s?\d{2}$", re.IGNORECASE)

CATEGORY_LABELS = [
    "Investment banking", "Wealth management", "Loan-related", "Cards", "Transaction services",
    "Markets trading", "Other non-interest income", "Net fee income", "Commercial book",
    "Others", "CBG / WM", "Other IBG", "Trade", "FD and others", "FCY Casa", "SGD Casa",
    "Net interest income", "Non-interest income",
]

def is_period_label(t: str) -> bool:
    if not t: return False
    tt = t.strip()
    return bool(QTR_PAT.match(tt) or MONTH_PAT.match(tt))

def word_cx(w):
    x0, x1 = w.get("x0"), w.get("x1")
    return (x0 + x1) / 2.0 if x0 is not None and x1 is not None else None

def to_float(s):
    txt = (s or "").strip()
    if re.match(r"^\(\s*[\d,]+(?:\.\d+)?\s*\)$", txt):
        inner = txt.strip("()").replace(",", "")
        return -float(inner)
    m = NUM_PAT.match(txt)
    if not m: return None
    whole = m.group(1).replace(",", "")
    frac  = m.group(2)
    return float(f"{whole}.{frac}" if frac else whole)

# --- Heuristic Title & Legend Detection ---
# def detect_metric_title_from_words(words, page_w, page_h):
#     if not words: return None
#     lines = {}
#     for w in words:
#         t = (w.get("text") or "").strip()
#         if not t: continue
#         top, bottom = w.get("top"), w.get("bottom")
#         if top is None or bottom is None: continue
#         yb = round(top / 3.0)
#         lines.setdefault(yb, []).append(w)
#     candidates = []
#     for yb, ws in lines.items():
#         tokens = [(ww.get("text") or "").strip() for ww in ws if (ww.get("text") or "").strip()]
#         if not tokens: continue
#         text_join = " ".join(tokens)
#         avg_y = sum((ww.get("top", 0.0) + ww.get("bottom", 0.0)) / 2.0 for ww in ws) / len(ws)
#         avg_size = sum((ww.get("size") or 0.0) for ww in ws) / len(ws)
#         x0s = [ww.get("x0") for ww in ws if ww.get("x0") is not None]
#         x1s = [ww.get("x1") for ww in ws if ww.get("x1") is not None]
#         if not x0s or not x1s: continue
#         min_x0, max_x1 = min(x0s), max(x1s)
#         span = max_x1 - min_x0
#         if avg_y > page_h * 0.28 or len(text_join) < 12 or min_x0 < page_w * 0.12 or span < page_w * 0.45: continue
#         digits = sum(c.isdigit() for c in text_join); letters = sum(c.isalpha() for c in text_join)
#         if digits > letters: continue
#         score = (avg_size * 2.0) + (span / page_w) + (1.0 - (avg_y / page_h))
#         candidates.append((score, text_join))
#     if not candidates: return None
#     return sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
def detect_metric_title_from_words(words, page_w, page_h):
    if not words:
        return None

    # 1) bucket words into rough lines
    lines = {}
    for w in words:
        t = (w.get("text") or "").strip()
        if not t:
            continue
        top, bottom = w.get("top"), w.get("bottom")
        if top is None or bottom is None:
            continue
        yb = round(top / 3.0)  # original bucketing
        (lines.setdefault(yb, [])).append(w)

    # 2) compute features per line and score headline-like candidates
    cand = []
    for yb, ws in lines.items():
        toks = [(ww.get("text") or "").strip() for ww in ws if (ww.get("text") or "").strip()]
        if not toks:
            continue
        text_join = " ".join(toks)
        avg_y = sum((ww.get("top", 0.0) + ww.get("bottom", 0.0)) / 2.0 for ww in ws) / len(ws)
        avg_size = sum((ww.get("size") or 0.0) for ww in ws) / len(ws)
        x0s = [ww.get("x0") for ww in ws if ww.get("x0") is not None]
        x1s = [ww.get("x1") for ww in ws if ww.get("x1") is not None]
        if not x0s or not x1s:
            continue
        min_x0, max_x1 = min(x0s), max(x1s)
        span = max_x1 - min_x0

        # original gates, but only for the *first* headline row
        if avg_y > page_h * 0.28:
            continue
        if len(text_join) < 12:
            continue
        if min_x0 < page_w * 0.12:
            continue
        if span < page_w * 0.45:
            continue
        digits = sum(c.isdigit() for c in text_join)
        letters = sum(c.isalpha() for c in text_join)
        if digits > letters:
            continue

        score = (avg_size * 2.0) + (span / page_w) + (1.0 - (avg_y / page_h))
        cand.append({
            "score": score,
            "text": text_join,
            "avg_y": avg_y,
            "min_x0": min_x0,
            "span": span,
            "yb": yb
        })

    if not cand:
        return None

    # 3) pick the best first line
    best = max(cand, key=lambda c: c["score"])

    # 4) try to append the next headline row: look for the nearest line just below,
    #    roughly left-aligned, and still near the top of the page.
    #    We search across original buckets for robustness.
    FOLLOW_DY_MAX = 0.08 * page_h     # vertical gap tolerance (~8% page height)
    LEFT_TOL = 0.06 * page_w          # left alignment tolerance
    TOP_BAND = 0.35 * page_h          # only stitch while still in top band

    # build simple list of (text, avg_y, min_x0, span) for all lines
    line_infos = []
    for yb, ws in lines.items():
        toks = [(ww.get("text") or "").strip() for ww in ws if (ww.get("text") or "").strip()]
        if not toks:
            continue
        text_join = " ".join(toks)
        avg_y = sum((ww.get("top", 0.0) + ww.get("bottom", 0.0)) / 2.0 for ww in ws) / len(ws)
        x0s = [ww.get("x0") for ww in ws if ww.get("x0") is not None]
        x1s = [ww.get("x1") for ww in ws if ww.get("x1") is not None]
        if not x0s or not x1s:
            continue
        min_x0, max_x1 = min(x0s), max(x1s)
        span = max_x1 - min_x0
        line_infos.append({"text": text_join, "avg_y": avg_y, "min_x0": min_x0, "span": span})

    # find the single nearest line below the best line
    followers = [
        li for li in line_infos
        if (li["avg_y"] > best["avg_y"]) and
           (li["avg_y"] - best["avg_y"] <= FOLLOW_DY_MAX) and
           (abs(li["min_x0"] - best["min_x0"]) <= LEFT_TOL) and
           (li["avg_y"] <= TOP_BAND)
    ]
    follower = None
    if followers:
        # choose the closest in y
        follower = min(followers, key=lambda li: li["avg_y"] - best["avg_y"])

    title = best["text"]
    if follower:
        # do not enforce span/min_x0 gates for follower; it’s allowed to be short
        title = f"{title} {follower['text']}".strip()

    return title


def find_category_bands(words):
    bands = {}
    if not words: return bands
    lines, line_infos = {}, []
    for w in words:
        if not (w.get("text") or "").strip() or w.get("top") is None: continue
        lines.setdefault(round(w.get("top") / 3.0), []).append(w)
    for yb, ws in lines.items():
        txt = " ".join((ww.get("text") or "").strip().lower() for ww in ws if (ww.get("text") or "").strip())
        if not txt: continue
        avg_y = sum((ww.get("top", 0.0) + ww.get("bottom", 0.0)) / 2.0 for ww in ws) / len(ws)
        min_x0 = min((ww.get("x0") for ww in ws if ww.get("x0") is not None), default=1e9)
        line_infos.append({"txt": txt, "avg_y": avg_y, "min_x0": min_x0})
    for label in CATEGORY_LABELS:
        tokens = [tok for tok in label.lower().split() if tok]
        best = None
        for li in line_infos:
            if li["min_x0"] > 420.0 or not all(tok in li["txt"] for tok in tokens): continue
            score = (li["min_x0"], li["avg_y"])
            if best is None or score < best[0]: best = (score, li)
        if best: bands[label] = best[1]["avg_y"]
    return bands

# --- Heuristic Data Binding Logic ---
def kmeans_1d(vals, iters=10):
    if not vals: return None, []
    vals_sorted = sorted(vals)
    if len(vals_sorted) >= 4:
        q1, q3 = statistics.quantiles(vals_sorted, n=4)[0], statistics.quantiles(vals_sorted, n=4)[-1]
    else: q1, q3 = min(vals_sorted), max(vals_sorted)
    centers=[q1, q3]
    for _ in range(iters):
        A, B = [], []
        for v in vals_sorted: (A if abs(v-centers[0])<=abs(v-centers[1]) else B).append(v)
        if A: centers[0]=sum(A)/len(A)
        if B: centers[1]=sum(B)/len(B)
    assigns=[0 if abs(v-centers[0])<=abs(v-centers[1]) else 1 for v in vals_sorted]
    idx_map = {v_i:i for i, v_i in enumerate(vals_sorted)}
    return centers, [assigns[idx_map[v]] for v in vals]

def looks_like_nim_value(n):
    txt, v = (n.get("text") or "").strip(), n.get("_num")
    return (v is not None) and (0.5 <= v <= 5.0) and ("." in txt or "%" in txt)

def bind_line_like(quarters, numbers, page_h):
    if not quarters or not numbers: return {}
    quarters = sorted(quarters, key=lambda w: w["_cx"])
    dxs=[quarters[i+1]["_cx"]-quarters[i]["_cx"] for i in range(len(quarters)-1)]
    X_TOL = max(30.0, (statistics.median(dxs) if dxs else 80.0)*0.45)
    tops=[n.get("top") for n in numbers if n.get("top") is not None]
    centers, assigns = kmeans_1d(tops) if tops else (None, [])
    if not centers: return {}
    upper_idx, lower_idx = (0,1) if centers[0] <= centers[1] else (1,0)
    band_upper, band_lower = [], []
    for i, n in enumerate(numbers):
        if n.get("top") is None: continue
        (band_upper if assigns[i]==upper_idx else band_lower).append(n)
    
    def pick_nearest_above(qw, pool):
        qx, q_top = qw["_cx"], qw.get("top", page_h)
        cand = []
        for n in pool:
            nx, nbot = word_cx(n), n.get("bottom")
            if nx is not None and nbot is not None and abs(nx-qx) <= X_TOL and 6.0 <= q_top - nbot:
                cand.append((q_top - nbot, n))
        return sorted(cand, key=lambda x:x[0])[0][1] if cand else None
    
    out={}
    for qw in quarters:
        up = pick_nearest_above(qw, band_upper)
        lo = pick_nearest_above(qw, band_lower)
        if up and lo:
            out[qw.get("text")] = {"group_nim": lo["_num"], "commercial_nim": up["_num"]}
    return out

def bind_line_like_adaptive(quarters, numbers, page_h):
    """
    Adaptive binder for NIM charts:
      - dynamic X_TOL from quarter spacing
      - split numbers into (upper/lower) bands via 1D k-means on 'top'
      - pick nearest-above in a reasonable vertical window
    Returns the same dict shape as bind_line_like().
    """
    if not quarters or not numbers:
        return {}

    # keep only plausible NIM tokens
    nums = [n for n in numbers if looks_like_nim_value(n)]
    if not nums:
        return {}

    # sort quarters and derive adaptive X window
    quarters_sorted = sorted(quarters, key=lambda w: w["_cx"])
    dxs = [quarters_sorted[i+1]["_cx"] - quarters_sorted[i]["_cx"] for i in range(len(quarters_sorted)-1)]
    typical_dx = statistics.median(dxs) if dxs else 80.0
    X_TOL = max(0.40 * typical_dx, 30.0)

    # 1-D k-means on 'top' positions to split into two horizontal bands
    num_tops = [n.get("top") for n in nums if n.get("top") is not None]
    if not num_tops:
        return {}
    centers, assigns = kmeans_1d(num_tops, iters=12)
    if not centers:
        mid = statistics.median(num_tops)
        centers = [mid - 60.0, mid + 60.0]
        assigns = [0 if y <= mid else 1 for y in num_tops]

    # which cluster is 'upper' (smaller top) vs 'lower'
    upper_idx, lower_idx = (0, 1) if centers[0] <= centers[1] else (1, 0)

    band_upper, band_lower = [], []
    for n, idx in zip(nums, assigns):
        (band_upper if idx == upper_idx else band_lower).append(n)

    # vertical window params
    all_tops = [n.get("top") for n in nums if n.get("top") is not None]
    global_min_top = min(all_tops) if all_tops else 0.0
    Y_MIN_GAP = 6.0
    def _max_above(q_top):  # allow a reasonable search span above label
        return max(200.0, (q_top - global_min_top) * 1.10)

    def pick_nearest_above(qw, pool):
        qx = qw["_cx"]; q_top = qw.get("top", page_h)
        cand = []
        for nw in pool:
            nx = nw.get("_cx"); nbot = nw.get("bottom")
            if nx is None or nbot is None:
                continue
            if abs(nx - qx) <= X_TOL:
                dy = q_top - nbot
                if dy >= Y_MIN_GAP and dy <= _max_above(q_top):
                    cand.append((dy, nw))
        cand.sort(key=lambda x: x[0])
        return cand[0][1] if cand else None

    out = {}
    for qw in quarters_sorted:
        up = pick_nearest_above(qw, band_upper)
        lo = pick_nearest_above(qw, band_lower)

        # fallback: closest-in-X if strict "above" fails
        if up is None and band_upper:
            up = min(band_upper, key=lambda n: abs(n.get("_cx", 0.0) - qw["_cx"]))
        if lo is None and band_lower:
            lo = min(band_lower, key=lambda n: abs(n.get("_cx", 0.0) - qw["_cx"]))

        if up and lo:
            out[qw.get("text")] = {"group_nim": lo["_num"], "commercial_nim": up["_num"]}
    return out


def bind_stacked_bar_like(quarters, numbers, words, page_h):
    if not quarters or not numbers: return {}
    cat_bands = find_category_bands(words)
    if not cat_bands: return {}
    quarters = sorted(quarters, key=lambda w: w["_cx"])
    dxs = [quarters[i+1]["_cx"] - quarters[i]["_cx"] for i in range(len(quarters)-1)]
    X_TOL = max(36.0, (statistics.median(dxs) if dxs else 80.0) * 0.45)
    nums = [n for n in numbers if n.get("top") is not None and n.get("bottom") is not None]
    if not nums: return {}

    def ny(n): return (n.get("top", 0.0) + n.get("bottom", 0.0)) / 2.0
    band_y_values = list(cat_bands.values())
    highest_band, lowest_band = min(band_y_values), max(band_y_values)
    total_above, total_below = highest_band - 18.0, lowest_band + 48.0
    out = {}
    for qw in quarters:
        qx, row = qw["_cx"], {}
        for label, y_band in cat_bands.items():
            best, best_score = None, 1e9
            for n in nums:
                nx = word_cx(n)
                if nx is not None and abs(nx - qx) <= X_TOL and abs(ny(n) - y_band) <= 26.0:
                    score = abs(ny(n) - y_band) + 0.01 * abs(nx - qx)
                    if score < best_score: best, best_score = n, score
            if best: row[label] = best.get("_num")
        
        candidates = [n for n in nums if word_cx(n) is not None and abs(word_cx(n) - qx) <= X_TOL and (n.get("top", page_h) <= total_above or n.get("top", page_h) >= total_below)]
        if candidates:
            total_val = max(candidates, key=lambda n: n.get("_num", -1e9)).get("_num")
            if total_val is not None: row["Total"] = total_val
        if row: out[qw.get("text")] = row
    return out

# --- Main Heuristic Consolidator ---
def consolidate_metrics_from_page(pg: dict) -> dict:
    page_no, page_h, page_w = pg.get("page_number"), pg.get("height", 540.0), pg.get("width", 960.0)
    text, words = pg.get("text", ""), pg.get("words", [])
    
    quarters, numbers, plain = [], [], []
    for w in words:
        t = (w.get("text") or "").strip()
        if not t: continue
        cx = word_cx(w)
        if cx is None: continue
        if is_period_label(t):
            quarters.append({**w, "_cx":cx})
        else:
            val=to_float(t)
            if val is not None: numbers.append({**w, "_cx":cx, "_num":val})
            else: plain.append(w)

    metric_title = (detect_metric_title_from_words(words, page_w, page_h) or (text.splitlines()[0] if text else "")).strip()
    mt_low = metric_title.lower()
    # Title OR legend mention (handles hybrid slides)
    is_nim_page = ("net interest margin" in mt_low) or any(
        "net interest margin" in (w.get("text") or "").lower()
        for w in words
    )
    looks_like_chart = bool(quarters) and bool(numbers)

    # Priority 1: NIM page (line chart)
    if is_nim_page and looks_like_chart:
        # Focus on the top band where the NIM polylines live to avoid stacked-bar numbers.
        # TOP_BAND_1 = 0.35  # first, tighter band
        # TOP_BAND_2 = 0.42  # second, slightly looser band (retry)
        TOP_BAND_1 = 0.50   # many DBS NIM labels sit ~45–52% page height
        TOP_BAND_2 = 0.58   # retry band

        def _nim_filter(nums, top_ratio):
            return [
                n for n in nums
                if looks_like_nim_value(n) and (n.get("top", page_h) <= page_h * top_ratio)
            ]

        # Try tight band first
        nim_numbers = _nim_filter(numbers, TOP_BAND_1)
        result = bind_line_like(quarters, nim_numbers, page_h)

        # If nothing bound (e.g., slightly lower labels), retry with looser band
        if not result:
            nim_numbers = _nim_filter(numbers, TOP_BAND_2)
            result = bind_line_like(quarters, nim_numbers, page_h)
            
        if not result:
            # Final attempt: adaptive binder without hard top-band gating
            result = bind_line_like_adaptive(quarters, numbers, page_h)

        if result:
            return {
                "page": page_no,
                "metric": metric_title,
                "chart_type": "line-like",
                "extracted": result
            }

    # Priority 2: Stacked Bar Chart (if category bands are found)
    if looks_like_chart:
        stacked_result = bind_stacked_bar_like(quarters, numbers, words, page_h)
        if stacked_result: return {"page": page_no, "metric": metric_title, "chart_type": "stacked-bar", "extracted": stacked_result}
    
    # Fallback to text or empty
    return {"page": page_no, "metric": metric_title, "chart_type": "text-or-table", "extracted": {}}

# --- Formatting for Indexing ---
def format_vision_json_to_text(data: dict) -> str:
    facts = []
    if "expenses_analysis" in data:
        for year, value in data["expenses_analysis"].get("yearly_total_expenses", {}).items():
            facts.append(f"Source: Expenses Chart (vision_analysis). For FY{year}, yearly_total_expenses (Opex) was {value} million.")
    if "five_year_summary" in data:
        for metric, year_data in data["five_year_summary"].items():
            for year, value in year_data.items():
                facts.append(f"Source: Five-Year Summary Table (vision_analysis). Metric '{metric}' for FY{year} is {value}.")
    if "nim_analysis" in data:
        for quarter, values in data["nim_analysis"].items():
            if "group_nim" in values: facts.append(f"Source: NIM Chart (vision_analysis). For {quarter}, the Group Net Interest Margin was {values['group_nim']}%.")
            if "commercial_nim" in values: facts.append(f"Source: NIM Chart (vision_analysis). For {quarter}, the Commercial Book Net Interest Margin was {values['commercial_nim']}%.")
    return "\n".join(facts)

def format_heuristic_json_to_text(data: dict) -> str:
    facts = []
    metric, chart_type, extracted = data.get("metric"), data.get("chart_type"), data.get("extracted", {})
    if not extracted: return ""
    
    if chart_type == "line-like": # NIM
        for period, values in extracted.items():
            if "group_nim" in values: facts.append(f"Source: {metric} (heuristic_parser). For {period}, Group NIM was {values['group_nim']}.")
            if "commercial_nim" in values: facts.append(f"Source: {metric} (heuristic_parser). For {period}, Commercial Book NIM was {values['commercial_nim']}.")
    
    elif chart_type == "stacked-bar":
        for period, categories in extracted.items():
            total = categories.pop("Total", None)
            if total is not None: facts.append(f"Source: {metric} (heuristic_parser). For {period}, the Total was {total}.")
            for cat, val in categories.items():
                facts.append(f"Source: {metric} (heuristic_parser). For {period}, the value for '{cat}' was {val}.")

    return "\n".join(facts)

def table_to_markdown(headers, rows, max_rows=50):
    hdr = "|" + "|".join(h or "" for h in headers) + "|\n"
    sep = "|" + "|".join("---" for _ in headers) + "|\n"
    lines = ["|" + "|".join(str(c or "") for c in r) + "|\n" for r in rows[:max_rows]]
    return hdr + sep + "".join(lines)
    
# --- General Helpers ---
def infer_period_from_filename(fname: str) -> Tuple[Optional[int], Optional[int]]:
    m = re.search(r"([1-4])Q(\d{2})", fname, re.I); 
    if m: return (2000 + int(m.group(2)), int(m.group(1)))
    m = re.search(r"\b(20\d{2})\b", fname, re.I); 
    if m: return (int(m.group(1)), None)
    return (None, None)

def find_key_pages(pdf_path: str) -> Dict[str, List[int]]:
    found_pages = {}
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text_l = page.get_text("text").lower()
                for desc, kws in SEARCH_TERMS.items():
                    if all(k.lower() in text_l for k in kws):
                        found_pages.setdefault(desc, []).append(page_num)
    except Exception as e:
        print(f"      ⚠️  Could not scout pages in {os.path.basename(pdf_path)}: {e}")
    return found_pages

# --- 3. Main Processing Functions ---
def process_pdf(path: str, fname: str, year: Optional[int], quarter: Optional[int], vision_model: genai.GenerativeModel, key_pages: Dict[str, List[int]]) -> Tuple[List, List, List]:
    chunks, pdf_scan_pages, pdf_metrics_pages = [], [], []
    all_key_page_numbers = [p for pages in key_pages.values() for p in pages]
    vision_prompt = """Analyze the attached image, which is a page from a financial report. Identify if it's an "Expenses" chart, "Five-year summary" table, or "Net Interest Margin" chart.
1. For "Expenses" Chart: Extract yearly/quarterly total expenses into a JSON with a key "expenses_analysis".
2. For "Five-year summary" Table: Extract "Total income", "Net profit", "Cost-to-income ratio (%)", and "Net interest margin (%)" for all years under the key "five_year_summary".
3. For "Net Interest Margin (%)" Chart: Extract "Group" and "Commercial book" NIM for all quarters under the key "nim_analysis".
If none are found, return {}.
    """
    
    with pdfplumber.open(path) as pdf_plumber, fitz.open(path) as doc_fitz:
        for idx, page_pl in enumerate(pdf_plumber.pages, start=1):
            page_fitz = doc_fitz[idx-1]
            row_template = {"doc_id": None, "file": fname, "page": idx, "year": year, "quarter": quarter}
            
            # --- Perform full page scan for JSON output ---
            page_scan_data = { "page_number": idx, "width": page_pl.width, "height": page_pl.height, "text": page_pl.extract_text() or "", "words": page_pl.extract_words() or [] }
            pdf_scan_pages.append(page_scan_data)
            
            # --- ALWAYS run local heuristic parser on EVERY page ---
            heuristic_metrics = consolidate_metrics_from_page(page_scan_data)
            pdf_metrics_pages.append(heuristic_metrics)
            heuristic_text = format_heuristic_json_to_text(heuristic_metrics)
            if heuristic_text:
                row = {**row_template, "doc_id": str(uuid.uuid4()), "section_hint": f"heuristic_summary_p{idx}"}
                chunks.append((row, heuristic_text))

            # --- Handle Key Pages (Vision API or Cache) ---
            if idx in all_key_page_numbers:
                cache_filename = f"{fname.replace('.pdf', '')}__page_{idx}.json"
                cache_filepath = os.path.join(CACHE_DIR, cache_filename)
                parsed_json = None
                
                if os.path.exists(cache_filepath):
                    with open(cache_filepath, 'r') as f: parsed_json = json.load(f)
                    print(f"      -> CACHE HIT for key page {idx}.")
                elif USE_VISION_MODEL:
                    print(f"      -> CACHE MISS. Calling Vision API for key page {idx}...")
                    try:
                        pix = page_fitz.get_pixmap(dpi=200)
                        image = Image.open(io.BytesIO(pix.tobytes("png")))
                        response = vision_model.generate_content([vision_prompt, image])
                        time.sleep(2) # Rate limiting
                        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                        if json_match:
                            parsed_json = json.loads(json_match.group(0))
                            with open(cache_filepath, 'w') as f: json.dump(parsed_json, f, indent=2)
                        else: print(f"          ⚠️  Vision model did not return valid JSON for page {idx}.")
                    except Exception as e: print(f"          ⚠️  Vision API call failed for page {idx}: {e}")

                if parsed_json:
                    vision_text = format_vision_json_to_text(parsed_json)
                    if vision_text:
                        row = {**row_template, "doc_id": str(uuid.uuid4()), "section_hint": f"vision_summary_p{idx}"}
                        chunks.append((row, vision_text))

            # --- ALWAYS add local prose + basic tables for ALL pages ---
            plain_text = page_fitz.get_text("text")
            if plain_text and plain_text.strip():
                row = {**row_template, "doc_id": str(uuid.uuid4()), "section_hint": "prose"}
                chunks.append((row, plain_text))
            
            # Extract basic tables using pdfplumber's default find_tables
            try:
                for i, table in enumerate(page_pl.find_tables()):
                    table_data = table.extract()
                    if table_data and len(table_data) > 1:
                        headers = [str(h or "") for h in table_data[0]]
                        rows = [[str(c or "") for c in r] for r in table_data[1:]]
                        table_md = table_to_markdown(headers, rows)
                        if table_md:
                            row = {**row_template, "doc_id": str(uuid.uuid4()), "section_hint": f"table_p{idx}_{i+1}"}
                            chunks.append((row, table_md))
            except Exception: pass
            
    return chunks, pdf_scan_pages, pdf_metrics_pages

def build_kb():
    # --- Gemini Setup ---
    vision_model = None
    if USE_VISION_MODEL:
        if 'GEMINI_API_KEY' not in os.environ: raise SystemExit("❌ ERROR: GEMINI_API_KEY not set.")
        try:
            genai.configure(api_key=os.environ['GEMINI_API_KEY'])
            vision_model = genai.GenerativeModel('gemini-2.5-flash')
        except Exception as e:
            raise SystemExit(f"❌ ERROR: Could not configure Gemini. Details: {e}")
    else:
        print("\n[Stage1] USE_VISION_MODEL is False. Skipping Gemini setup and API calls.")

    # --- Document Discovery & Scouting ---
    pdf_docs = sorted([str(p) for p in pathlib.Path(DATA_DIR).rglob("*.pdf")])
    print("[Stage1] Scouting Pass to find key pages...")
    all_key_pages = {os.path.basename(p): find_key_pages(p) for p in pdf_docs}
    
    # --- Extraction Pass ---
    all_rows, all_texts = [], []

    for path in pdf_docs:
        fname = os.path.basename(path)
        print(f"\n[Stage1] Processing: {fname}")
        year, quarter = infer_period_from_filename(fname)
        key_pages_for_file = all_key_pages.get(fname, {})
        
        doc_chunks, scan_pages, metrics_pages = process_pdf(path, fname, year, quarter, vision_model, key_pages_for_file)
        
        # --- MODIFIED: Save JSON outputs for this specific PDF ---
        base_name = os.path.splitext(fname)[0]
        scan_out_path = os.path.join(OUT_DIR, f"{base_name}_scan.json")
        metrics_out_path = os.path.join(OUT_DIR, f"{base_name}_metrics.json")

        scan_doc = {"source": fname, "pages": scan_pages}
        with open(scan_out_path, "w", encoding="utf-8") as f:
            json.dump(scan_doc, f, ensure_ascii=False, indent=2)
        
        metrics_doc = {"source": fname, "pages": metrics_pages}
        with open(metrics_out_path, "w", encoding="utf-8") as f:
            json.dump(metrics_doc, f, indent=2)
        print(f"      → Saved verification JSONs: {os.path.basename(scan_out_path)}, {os.path.basename(metrics_out_path)}")

        if doc_chunks:
            print(f"      → Extracted {len(doc_chunks)} chunks from {fname}.")
            for row, text in doc_chunks: all_rows.append(row); all_texts.append(text)
        else:
            print(f"      ⚠️ WARNING: No content extracted from {fname}.")

    if not all_texts: raise SystemExit("No data was indexed.")
    
    # --- Finalization Pass (Embedding, Indexing, Saving) ---
    print(f"\n[Stage1] Total chunks to be indexed: {len(all_texts)}")
    kb = pd.DataFrame(all_rows)

    # --- NEW: Fix for ArrowKeyError ---
    # Explicitly cast dtypes to be pyarrow-friendly before saving
    try:
        kb['year'] = kb['year'].astype('Int64')
        kb['quarter'] = kb['quarter'].astype('Int64')
        kb['page'] = kb['page'].astype('Int64')
        for col in ['doc_id', 'file', 'section_hint']:
            if col in kb.columns:
                kb[col] = kb[col].astype(str)
    except Exception as e:
        print(f"      ⚠️  Could not cast DataFrame types cleanly: {e}")


    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    provider = SentenceTransformer(model_name)
    vecs = provider.encode(all_texts, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    dim = provider.get_sentence_embedding_dimension()

    kb_path = os.path.join(OUT_DIR, "kb_chunks.parquet")
    text_path = os.path.join(OUT_DIR, "kb_texts.npy")
    index_path = os.path.join(OUT_DIR, "kb_index.faiss")
    meta_path = os.path.join(OUT_DIR, "kb_meta.json")
    
    kb.to_parquet(kb_path, engine="pyarrow", index=False)
    np.save(text_path, np.array(all_texts, dtype=object))
    
    if _HAVE_FAISS:
        index = faiss.IndexFlatIP(dim); index.add(vecs); faiss.write_index(index, index_path)
        index_meta = {"index": "faiss_ip", "dim": dim, "embedding_provider": f"st:{model_name}"}
    else:
        np.save(os.path.join(OUT_DIR, "kb_vectors.npy"), vecs)
        index_meta = {"index": "naive_ip_numpy", "dim": dim, "embedding_provider": f"st:{model_name}"}
    with open(meta_path, "w") as f: json.dump(index_meta, f)

    print(f"\n[Stage1] Successfully saved all artifacts to '{OUT_DIR}'")

if __name__ == "__main__":
    build_kb()