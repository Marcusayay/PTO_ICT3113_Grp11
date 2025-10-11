# === ALL-PAGES EXTRACTOR (pdfplumber only; no OCR/vision) ===
# - Scans all pages: text, words (with x/y), table candidates (two strategies)
# - Binds chart-like content:
#     * "Net interest margin (%)" → line-like (quarters → {group_nim, commercial_nim})
#     * Bar-like metrics (e.g., "Net interest income (S$m)") → {quarter → value}
# - Outputs:
#     1) out/pdfplumber_scan_all.json        (raw per-page index)
#     2) out/metrics_all_pages.json          (clean per-page metrics)

import json, re, csv, os, statistics
from pathlib import Path
import pdfplumber

# --------- CONFIG ---------
TARGET_PDF = "All/2Q25_CFO_presentation.pdf"
OUT_SCAN_JSON = "out/pdfplumber_scan_all.json"
OUT_METRICS_JSON = "out/metrics_all_pages.json"
DUMP_TABLES_DIR = "out/tables_by_page"   # set to None to disable CSV dumps

PREVIEW_FIRST_N_PAGES = 0                # set >0 to print quick previews
# --------------------------

# Known stacked-bar categories often used in DBS decks
CATEGORY_LABELS = [
    # Fee income page (e.g., page 9)
    "Investment banking",
    "Wealth management",
    "Loan-related",
    "Cards",
    "Transaction services",
    # Commercial book non-interest income page (e.g., page 11)
    "Markets trading",
    "Other non-interest income",
    "Net fee income",
    "Commercial book",
    # Loans page (e.g., page 7)
    "Others",
    "CBG / WM",
    "Other IBG",
    "Trade",
    # Deposits page (e.g., page 8)
    "FD and others",
    "FCY Casa",
    "SGD Casa",
    # Two-band stacks on WM page (e.g., page 10)
    "Net interest income",
    "Non-interest income",
]

# ---- Table extraction helpers ----
def extract_tables_with_settings(page, setting_name, table_settings):
    results = []
    try:
        found = page.find_tables(table_settings=table_settings)
    except Exception as e:
        return [{"setting": setting_name, "error": f"find_tables error: {e}", "rows": [], "headers": [], "bbox": None}]

    for t in found:
        try:
            data = t.extract(x_tolerance=2, y_tolerance=2)
        except Exception as e:
            results.append({"setting": setting_name, "error": f"extract error: {e}", "rows": [], "headers": [], "bbox": getattr(t, "bbox", None)})
            continue

        if not data or len(data) < 2 or not any(data[0]):
            results.append({"setting": setting_name, "warning": "empty_or_headerless_table", "rows": [], "headers": [], "bbox": getattr(t, "bbox", None)})
            continue

        header_row = ["" if h is None else str(h).strip() for h in data[0]]
        body_rows = [[("" if c is None else str(c)) for c in row] for row in data[1:]]
        # If header row looks numeric-heavy, fall back to generic headers
        if sum(bool(re.search(r"\d", h or "")) for h in header_row) > len(header_row) // 2:
            header_row = [f"col_{i+1}" for i in range(len(header_row))]

        results.append({
            "setting": setting_name,
            "bbox": getattr(t, "bbox", None),
            "headers": header_row,
            "rows": body_rows,
        })
    return results

# ---- Scan ALL pages into a single JSON ----
pdf_path = Path(TARGET_PDF)
if not pdf_path.exists():
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

doc = {"source": str(pdf_path), "pages": []}

with pdfplumber.open(str(pdf_path)) as pdf:
    for idx, page in enumerate(pdf.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            text, text_error = "", f"text error: {e}"
        else:
            text_error = None

        try:
            words = page.extract_words() or []
        except Exception as e:
            words, words_error = [], f"words error: {e}"
        else:
            words_error = None

        # Two table strategies
        settings_A = dict(vertical_strategy="lines", horizontal_strategy="lines",
                          snap_tolerance=3, join_tolerance=3, edge_min_length=15,
                          intersection_tolerance=3)
        settings_B = dict(vertical_strategy="text", horizontal_strategy="text",
                          text_tolerance=2, snap_tolerance=3, join_tolerance=3,
                          intersection_tolerance=3)

        tables_A = extract_tables_with_settings(page, "A_lines", settings_A)
        tables_B = extract_tables_with_settings(page, "B_text", settings_B)

        page_entry = {
            "page_number": idx,
            "width": page.width,
            "height": page.height,
            "text_error": text_error,
            "words_error": words_error,
            "text": text,
            "words": [
                {
                    "text": w.get("text", ""),
                    "x0": w.get("x0"),
                    "top": w.get("top"),
                    "x1": w.get("x1"),
                    "bottom": w.get("bottom"),
                    "upright": w.get("upright"),
                    "direction": w.get("direction"),
                    "fontname": w.get("fontname"),
                    "size": w.get("size"),
                }
                for w in words
            ],
            "tables": tables_A + tables_B,
        }
        doc["pages"].append(page_entry)

        # Optional: dump tables per page
        if DUMP_TABLES_DIR:
            outdir = Path(DUMP_TABLES_DIR) / f"page_{idx:02d}"
            outdir.mkdir(parents=True, exist_ok=True)
            for i, t in enumerate(page_entry["tables"], start=1):
                if not t.get("rows"):
                    continue
                csv_path = outdir / f"table-{i}_{t['setting']}.csv"
                with csv_path.open("w", newline="", encoding="utf-8") as cf:
                    writer = csv.writer(cf)
                    writer.writerow(t.get("headers", []))
                    writer.writerows(t.get("rows", []))

# Save scan JSON
Path(OUT_SCAN_JSON).parent.mkdir(parents=True, exist_ok=True)
with open(OUT_SCAN_JSON, "w", encoding="utf-8") as f:
    json.dump(doc, f, ensure_ascii=False, indent=2)
print(f"✅ Scanned all pages → {OUT_SCAN_JSON}")

if PREVIEW_FIRST_N_PAGES > 0:
    for p in doc["pages"][:PREVIEW_FIRST_N_PAGES]:
        print(f"\n=== Page {p['page_number']} preview ===")
        print((p["text"] or "")[:400], "..." if len(p["text"] or "") > 400 else "")

# =========================
#   METRICS CONSOLIDATION
# =========================

QTR_PAT     = re.compile(r"^(?:[1-4]Q|[12]H)\d{2}$", re.IGNORECASE)  # 2Q24, 1Q25, 1H25, 2H24
NUM_PAT     = re.compile(r"^\s*(-?\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?\s*%?\s*$")

# --- Month/period pattern and utility ---
MONTH_PAT  = re.compile(r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s?\d{2}$", re.IGNORECASE)

def is_period_label(t: str) -> bool:
    if not t:
        return False
    tt = t.strip()
    return bool(QTR_PAT.match(tt) or MONTH_PAT.match(tt))

def word_cx(w): 
    x0, x1 = w.get("x0"), w.get("x1")
    return (x0 + x1) / 2.0 if x0 is not None and x1 is not None else None

def to_float(s):
    txt = (s or "").strip()
    # handle (11) style negatives
    if re.match(r"^\(\s*[\d,]+(?:\.\d+)?\s*\)$", txt):
        inner = txt.strip("()").replace(",", "")
        return -float(inner)
    m = NUM_PAT.match(txt)
    if not m:
        return None
    whole = m.group(1).replace(",", "")
    frac  = m.group(2)
    return float(f"{whole}.{frac}" if frac else whole)


def split_words(words):
    quarters=[]; numbers=[]; plain=[]
    for w in words:
        t=(w.get("text") or "").strip()
        if not t: continue
        if is_period_label(t):
            cx=word_cx(w);
            if cx is not None: quarters.append({**w,"_cx":cx})
        else:
            val=to_float(t)
            if val is not None:
                cx=word_cx(w);
                if cx is not None: numbers.append({**w,"_cx":cx,"_num":val})
            else:
                plain.append(w)
    return quarters, numbers, plain

# --- Helper: find category label bands for stacked-bar charts
def find_category_bands(words):
    """
    Return dict of {label: y_center} for known stacked-bar categories by matching
    left-side text labels. More robust by grouping words into lines and requiring
    all tokens of the label to appear on the same line. Wider left-margin tolerance.
    """
    bands = {}
    if not words:
        return bands

    # Group words into 'lines' by quantized y (top)
    lines = {}  # key: y_bucket -> list[word]
    for w in words:
        t = (w.get("text") or "").strip()
        if not t:
            continue
        top = w.get("top")
        if top is None:
            continue
        yb = round(top / 3.0)  # bucket size ~3pt
        lines.setdefault(yb, []).append(w)

    # For each line, build a lowercase string and compute average y and min x0
    line_infos = []
    for yb, ws in lines.items():
        txt = " ".join((ww.get("text") or "").strip().lower() for ww in ws if (ww.get("text") or "").strip())
        if not txt:
            continue
        avg_y = sum((ww.get("top", 0.0) + ww.get("bottom", 0.0)) / 2.0 for ww in ws) / len(ws)
        min_x0 = min((ww.get("x0") for ww in ws if ww.get("x0") is not None), default=1e9)
        line_infos.append({"yb": yb, "txt": txt, "avg_y": avg_y, "min_x0": min_x0})

    # Wider left margin tolerance: labels can sit up to ~420pt from left
    LEFT_X_MAX = 420.0

    for label in CATEGORY_LABELS:
        tokens = [tok for tok in label.lower().split() if tok]
        # find a line on the left that contains ALL tokens (in any order)
        best = None
        for li in line_infos:
            if li["min_x0"] is None or li["min_x0"] > LEFT_X_MAX:
                continue
            if all(tok in li["txt"] for tok in tokens):
                # prefer the left-most, then highest on page
                score = (li["min_x0"], li["avg_y"])
                if best is None or score < best[0]:
                    best = (score, li)
        if best:
            bands[label] = best[1]["avg_y"]

    # --- Auto-legend fallback (no whitelist) ---
    # If we found too few bands (e.g., new slide layouts), infer labels from left-side lines.
    if len(bands) < 2:
        BLOCK_TOKENS = {"yoy", "(%)", "%", "(s$", "s$m", "$", "bn", "aum", "earning assets"}
        auto_candidates = []
        for li in line_infos:
            if li["min_x0"] is None or li["min_x0"] > LEFT_X_MAX:
                continue
            txt = li["txt"]
            letters = sum(ch.isalpha() for ch in txt)
            digits  = sum(ch.isdigit() for ch in txt)
            # accept lines that look like category phrases (more letters than digits, not unit lines)
            if letters <= digits:
                continue
            if any(bt in txt for bt in BLOCK_TOKENS):
                continue
            # allow short all-caps labels like 'GP' / 'SP'
            if len(txt) < 5 and not (txt.isupper() and 2 <= len(txt) <= 3 and txt.isalpha()):
                continue
            # prefer multi-word phrases
            word_count = len([t for t in txt.split() if t])
            if word_count < 1:
                continue
            # keep as candidate
            auto_candidates.append(li)

        # Sort by being leftmost then by top position; take up to 6
        auto_candidates.sort(key=lambda x: (x["min_x0"], x["avg_y"]))
        for li in auto_candidates[:6]:
            # Normalise label: title-case but keep slashes/hyphens as-is
            label_txt = " ".join(w.capitalize() if w.isalpha() else w for w in li["txt"].split())
            # Only add if not already present
            if label_txt not in bands:
                bands[label_txt] = li["avg_y"]

    return bands

# --- Heuristic: derive slide title from word positions/font sizes ---
def detect_metric_title_from_words(words, page_w, page_h):
    """
    Heuristic: slide titles are large-font, top-centered lines spanning wide width.
    - Group words into lines (by y bucket)
    - Filter to top ~28% of the page, long-ish text, wide span, not left-legend
    - Score by font size, span width, and proximity to the top
    Returns the best-matching line text, or None.
    """
    if not words:
        return None

    # Group words into 'lines' by quantized y (top)
    lines = {}
    for w in words:
        t = (w.get("text") or "").strip()
        if not t:
            continue
        top = w.get("top"); bottom = w.get("bottom")
        if top is None or bottom is None:
            continue
        yb = round(top / 3.0)  # ~3pt bucket
        lines.setdefault(yb, []).append(w)

    candidates = []
    for yb, ws in lines.items():
        # Build text and features for this visual line
        tokens = [(ww.get("text") or "").strip() for ww in ws if (ww.get("text") or "").strip()]
        if not tokens:
            continue
        text_join = " ".join(tokens)
        low = text_join.lower()
        avg_y = sum((ww.get("top", 0.0) + ww.get("bottom", 0.0)) / 2.0 for ww in ws) / len(ws)
        avg_size = sum((ww.get("size") or 0.0) for ww in ws) / len(ws)
        x0s = [ww.get("x0") for ww in ws if ww.get("x0") is not None]
        x1s = [ww.get("x1") for ww in ws if ww.get("x1") is not None]
        if not x0s or not x1s:
            continue
        min_x0 = min(x0s); max_x1 = max(x1s)
        span = max_x1 - min_x0

        # Basic filters
        if avg_y > page_h * 0.28:      # too low on the page to be the title
            continue
        if len(text_join) < 12:        # very short lines are unlikely to be the title
            continue
        if min_x0 < page_w * 0.12:     # exclude left legend/axis area
            continue
        if span < page_w * 0.45:       # title usually spans a good width
            continue
        # Avoid picking lines that are mostly numbers/units
        digits = sum(ch.isdigit() for ch in text_join)
        letters = sum(ch.isalpha() for ch in text_join)
        if digits > letters:
            continue

        # Score: large font, wide span, close to the top
        score = (avg_size * 2.0) + (span / page_w) + (1.0 - (avg_y / page_h))
        candidates.append((score, text_join))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def detect_metric_title(page_text):
    lines = [ln.strip() for ln in (page_text or "").splitlines() if ln.strip()]
    if not lines:
        return "metric"
    # 1) Prioritize exact match anywhere
    for ln in lines:
        if "net interest margin" in ln.lower():
            return ln
    # 2) Early strong hints (top lines)
    for ln in lines[:12]:
        low = ln.lower()
        if "margin" in low or "%" in low or "net interest" in low or "allowances" in low:
            return ln
    # 3) Fallback
    return lines[0]

def guess_legend_labels(plain_words):
    text = " ".join((w.get("text") or "") for w in plain_words).lower()
    labels=[]
    if "commercial book" in text: labels.append("Commercial book")
    if "group" in text: labels.append("Group")
    return labels or ["Series A","Series B"]

def looks_like_nim_value(n):
    """
    Heuristic filter for NIM (%): keep plausible percentage-like values only.
    - numeric value between ~0.5 and 5.0
    - and the source text had a decimal point or a percent sign
    """
    txt = (n.get("text") or "").strip()
    v = n.get("_num")
    has_decimal = "." in txt
    has_pct = "%" in txt
    return (v is not None) and (0.5 <= v <= 5.0) and (has_decimal or has_pct)

def kmeans_1d(vals, iters=10):
    if not vals: return None, []
    vals_sorted = sorted(vals)
    # init using quartiles if possible
    if len(vals_sorted) >= 4:
        q1 = statistics.quantiles(vals_sorted, n=4)[0]
        q3 = statistics.quantiles(vals_sorted, n=4)[-1]
    else:
        q1, q3 = min(vals_sorted), max(vals_sorted)
    centers=[q1, q3]
    for _ in range(iters):
        A,B=[],[]
        for v in vals_sorted:
            (A if abs(v-centers[0])<=abs(v-centers[1]) else B).append(v)
        if A: centers[0]=sum(A)/len(A)
        if B: centers[1]=sum(B)/len(B)
    assigns=[0 if abs(v-centers[0])<=abs(v-centers[1]) else 1 for v in vals_sorted]
    # Map assignments back to original order
    idx_map = {v_i:i for i,v_i in enumerate(vals_sorted)}
    return centers, [assigns[idx_map[v]] for v in [w for w in vals]]

def bind_line_like(quarters, numbers, page_h):
    if not quarters or not numbers: return {}
    quarters = sorted(quarters, key=lambda w: w["_cx"])
    # X window from quarter spacing
    dxs=[quarters[i+1]["_cx"]-quarters[i]["_cx"] for i in range(len(quarters)-1)]
    X_TOL = max(30.0, (statistics.median(dxs) if dxs else 80.0)*0.45)

    tops=[n.get("top") for n in numbers if n.get("top") is not None]
    centers, assigns = kmeans_1d(tops) if tops else (None, [])
    if not centers:
        mid = statistics.median(tops) if tops else page_h/2
        centers=[mid-60, mid+60]
        assigns=[0 if y<=mid else 1 for y in tops]

    # decide which center is upper/lower
    upper_idx, lower_idx = (0,1) if centers[0] <= centers[1] else (1,0)
    band_upper=[]; band_lower=[]
    # assign numbers to bands in original order of 'numbers'
    j=0
    for n in numbers:
        if n.get("top") is None: continue
        idx = assigns[j]; j+=1
        (band_upper if idx==upper_idx else band_lower).append(n)

    global_min_top = min(tops) if tops else 0.0
    q_top0 = quarters[0].get("top", page_h)
    global_max_above = max(220.0, (q_top0 - global_min_top)*1.15)
    Y_MIN_GAP = 6.0

    def pick_nearest_above(qw, pool):
        qx=qw["_cx"]; q_top=qw.get("top", page_h)
        cand=[]
        for n in pool:
            nx=word_cx(n); nbot=n.get("bottom")
            if nx is None or nbot is None: continue
            if abs(nx-qx) <= X_TOL:
                dy = q_top - nbot
                if dy >= Y_MIN_GAP and dy <= global_max_above:
                    cand.append((dy,n))
        cand.sort(key=lambda x:x[0])
        return cand[0][1] if cand else None

    out={}
    for qw in quarters:
        up = pick_nearest_above(qw, band_upper)
        lo = pick_nearest_above(qw, band_lower)
        if up is None and band_upper: up=min(band_upper, key=lambda n: abs(word_cx(n)-qw["_cx"]))
        if lo is None and band_lower: lo=min(band_lower, key=lambda n: abs(word_cx(n)-qw["_cx"]))
        if up and lo:
            out[qw.get("text")] = {"group_nim": lo["_num"], "commercial_nim": up["_num"]}
    return out


def bind_bar_like(quarters, numbers, page_h):
    if not quarters or not numbers: return {}
    quarters = sorted(quarters, key=lambda w: w["_cx"])
    dxs=[quarters[i+1]["_cx"]-quarters[i]["_cx"] for i in range(len(quarters)-1)]
    X_TOL = max(30.0, (statistics.median(dxs) if dxs else 80.0)*0.40)
    Y_MIN_GAP = 4.0
    global_max_above = page_h

    def pick_nearest_above(qw, pool):
        qx=qw["_cx"]; q_top=qw.get("top", page_h)
        cand=[]
        for n in pool:
            nx=word_cx(n); nbot=n.get("bottom")
            if nx is None or nbot is None: continue
            if abs(nx-qx) <= X_TOL:
                dy = q_top - nbot
                if dy >= Y_MIN_GAP and dy <= global_max_above:
                    cand.append((dy,n))
        cand.sort(key=lambda x:x[0])
        return cand[0][1] if cand else None

    out={}
    for qw in quarters:
        n = pick_nearest_above(qw, numbers)
        if n:
            out[qw.get("text")] = n["_num"]
    return out

# --- Stacked bar binder
def bind_stacked_bar_like(quarters, numbers, words, page_h):
    """
    Heuristic for stacked bar charts:
    - Detect left-side category labels (CATEGORY_LABELS) to form horizontal bands (y positions).
    - For each quarter (x), pick the nearest number within each category band (y proximity).
    - Detect a 'Total' as the largest number either clearly ABOVE the bands or clearly BELOW them.
    Output:
        { "2Q25": {"Investment banking": 31, "Wealth management": 649, ... , "Total": 1395}, ... }
    """
    if not quarters or not numbers:
        return {}

    cat_bands = find_category_bands(words)
    if not cat_bands:
        return {}

    quarters = sorted(quarters, key=lambda w: w["_cx"])
    dxs = [quarters[i+1]["_cx"] - quarters[i]["_cx"] for i in range(len(quarters)-1)]
    X_TOL = max(36.0, (statistics.median(dxs) if dxs else 80.0) * 0.45)

    nums = [n for n in numbers if n.get("top") is not None and n.get("bottom") is not None]
    if not nums:
        return {}

    # y centers for numbers
    def ny(n): return (n.get("top", 0.0) + n.get("bottom", 0.0)) / 2.0

    # Band Y tolerance for category assignment
    BAND_Y_TOL = 26.0

    # Band centers
    band_y_values = list(cat_bands.values())
    highest_band_center = min(band_y_values)   # smaller y = higher on page
    lowest_band_center  = max(band_y_values)

    # Define cutoffs for "Total" zones
    total_above_cutoff = highest_band_center - 18.0   # numbers above all bands
    total_below_cutoff = lowest_band_center + 48.0    # numbers below all bands

    out = {}
    for qw in quarters:
        qx = qw["_cx"]
        quarter_key = qw.get("text")
        row = {}

        # Per-category pick by vertical proximity to label band
        for label, y_band in cat_bands.items():
            best = None
            best_score = 1e9
            for n in nums:
                nx = word_cx(n)
                if nx is None or abs(nx - qx) > X_TOL:
                    continue
                dy = abs(ny(n) - y_band)
                if dy <= BAND_Y_TOL:
                    score = dy + 0.01 * abs(nx - qx)
                    if score < best_score:
                        best_score = score
                        best = n
            if best is not None:
                row[label] = best.get("_num")

        # Detect Total above-or-below bands within the same X window
        candidates = []
        for n in nums:
            nx = word_cx(n)
            if nx is None or abs(nx - qx) > X_TOL:
                continue
            t = n.get("top", page_h)
            if t <= total_above_cutoff or t >= total_below_cutoff:
                candidates.append(n)

        if candidates:
            total_val = max(candidates, key=lambda n: n.get("_num", float("-inf"))).get("_num")
            if total_val is not None:
                row["Total"] = total_val

        if row:
            out[quarter_key] = row

    return out


def infer_period_headers_from_words(words, max_labels=12):
    """
    Scan page words to infer column headers that look like period labels
    (1H24, 2Q24, Mar 25, etc.). Returns a left-to-right ordered list.
    """
    labels = []
    seen = set()
    for w in words or []:
        t = (w.get("text") or "").strip()
        if not t:
            continue
        if is_period_label(t):
            cx = word_cx(w)
            if cx is None:
                continue
            key = (t.upper(), round(cx, 1))
            if key in seen:
                continue
            seen.add(key)
            labels.append((cx, t))
    labels.sort(key=lambda x: x[0])
    # Keep order, but dedupe by text (some decks repeat the label above/below)
    ordered = []
    seen_txt = set()
    for _, t in labels:
        tu = t.upper()
        if tu in seen_txt:
            continue
        seen_txt.add(tu)
        ordered.append(t)
        if len(ordered) >= max_labels:
            break
    return ordered

def clean_numeric_cell(s):
    """Convert '(11)' -> -11, '1,234' -> 1234.0, leave text as-is."""
    v = to_float(s)
    return v if v is not None else s


def clean_table_object(raw_table, page_words):
    """
    Given a raw {headers, rows} from pdfplumber, remove spacer rows,
    collapse numeric strings, and upgrade headers using inferred period labels
    from page words when helpful.
    """
    headers = list(raw_table.get("headers") or [])
    rows    = list(raw_table.get("rows") or [])

    # Drop spacer rows (all empty)
    def is_spacer(row):
        return not any((c or "").strip() for c in row)
    rows = [r for r in rows if not is_spacer(r)]

    # If the header cells are mostly generic / numeric, try inferring
    mostly_generic = (not headers) or all(h.strip().lower().startswith("col_") or not h.strip() for h in headers)
    if mostly_generic:
        inferred = infer_period_headers_from_words(page_words)
        # If the table width matches 1 label column + inferred periods, upgrade headers
        if inferred and len(inferred) + 1 == (len(rows[0]) if rows else 0):
            headers = ["Metric"] + inferred

    # Normalise cell values (numbers -> float, keep strings otherwise)
    norm_rows = []
    for r in rows:
        norm_rows.append([clean_numeric_cell(c) for c in r])

    return {"headers": headers, "rows": norm_rows}

# --- Semantic table builder: period columns, left labels as rows, fill with nearest numbers ---

def cluster_by_y(words, bucket=3.0):
    lines = {}
    for w in words or []:
        t = (w.get("text") or "").strip()
        if not t:
            continue
        top = w.get("top")
        if top is None:
            continue
        yb = round(top / bucket)
        lines.setdefault(yb, []).append(w)
    # produce line objects: text, avg_y, min_x0, max_x1
    out = []
    for yb, ws in lines.items():
        txt = " ".join((ww.get("text") or "").strip() for ww in ws if (ww.get("text") or "").strip())
        if not txt:
            continue
        avg_y = sum((ww.get("top", 0.0) + ww.get("bottom", 0.0)) / 2.0 for ww in ws) / len(ws)
        x0s = [ww.get("x0") for ww in ws if ww.get("x0") is not None]
        x1s = [ww.get("x1") for ww in ws if ww.get("x1") is not None]
        if not x0s or not x1s:
            continue
        out.append({
            "txt": txt,
            "avg_y": avg_y,
            "min_x0": min(x0s),
            "max_x1": max(x1s),
            "words": ws,
        })
    out.sort(key=lambda r: r["avg_y"])  # top->bottom
    return out


def infer_left_labels(words, page_w, max_rows=20):
    """
    Auto-detect left-label rows for table-like slides:
    pick text lines in the left ~40% region that are letter-dominant and not unit lines.
    """
    lines = cluster_by_y(words)
    LEFT_MAX_X = page_w * 0.45
    BLOCK = {"(s$m)", "(s$)", "s$m", "(%)", "%", "($m)", "(bn)", "aum"}
    rows = []
    for li in lines:
        if li["min_x0"] > LEFT_MAX_X:
            continue
        txt_low = li["txt"].lower()
        letters = sum(ch.isalpha() for ch in txt_low)
        digits = sum(ch.isdigit() for ch in txt_low)
        if letters <= digits:
            continue
        if any(b in txt_low for b in BLOCK):
            continue
        rows.append({"label": " ".join(li["txt"].split()), "y": li["avg_y"]})
        if len(rows) >= max_rows:
            break
    # de-duplicate labels with very close y (merge)
    dedup = []
    for r in rows:
        if dedup and abs(dedup[-1]["y"] - r["y"]) < 10.0:
            continue
        dedup.append(r)
    return dedup


def build_semantic_table_from_words(words, page_w, require_min_rows=3, require_min_cols=3):
    """
    Build a table purely from words:
    - columns: inferred period headers (left->right) using x-centers
    - rows: inferred left labels (top->bottom)
    - cells: nearest numeric to the row y and column x
    """
    # 1) Build ordered period headers as [(cx, "1H24"), ...]
    period_points = []
    seen = set()
    for w in words or []:
        t = (w.get("text") or "").strip()
        if not t:
            continue
        if is_period_label(t):
            cx = word_cx(w)
            if cx is None:
                continue
            key = (t.upper(), round(cx, 1))
            if key in seen:
                continue
            seen.add(key)
            period_points.append((cx, t))
    period_points.sort(key=lambda x: x[0])

    # Deduplicate by label text while preserving left->right order
    ordered = []
    used_lbls = set()
    for cx, lbl in period_points:
        ul = lbl.upper()
        if ul in used_lbls:
            continue
        used_lbls.add(ul)
        ordered.append((cx, lbl))

    if len(ordered) < require_min_cols:
        return None

    # 2) Infer left labels (row names)
    rows = infer_left_labels(words, page_w)
    if len(rows) < require_min_rows:
        return None

    # 3) Collect numeric candidates with coordinates
    nums = []
    for w in words or []:
        v = to_float(w.get("text"))
        if v is not None:
            nums.append({
                "x": word_cx(w),
                "y": (w.get("top", 0.0) + w.get("bottom", 0.0)) / 2.0,
                "v": v,
            })
    if not nums:
        return None

    # 4) Fill matrix: nearest number by (|dy| + 0.02*|dx|)
    def pick_value(y_row, x_col):
        best = None
        best_score = 1e9
        for n in nums:
            if n["x"] is None:
                continue
            dy = abs(n["y"] - y_row)
            dx = abs(n["x"] - x_col)
            score = dy + 0.02 * dx
            if score < best_score:
                best_score = score
                best = n
        return best["v"] if best else None

    headers = ["Metric"] + [t for _, t in ordered]
    matrix = []
    for r in rows:
        row_vals = [r["label"]]
        for cx, _t in ordered:
            row_vals.append(pick_value(r["y"], cx))
        matrix.append(row_vals)

    return {"headers": headers, "rows": matrix}

def pick_biggest_table(tables):
    def table_size(t):
        rows = t.get("rows") or []
        cols = len(t.get("headers") or [])
        return (len(rows) * max(cols, 1))
    return max(tables, key=table_size) if tables else None

def looks_like_big_table(pg):
    tables = pg.get("tables") or []
    biggest = pick_biggest_table(tables)
    if not biggest:
        return None
    rows = biggest.get("rows") or []
    cols = len(biggest.get("headers") or [])
    # heuristic threshold for "big": many rows/cols (tuned for DBS deck tables)
    if len(rows) >= 5 and cols >= 3:
        return biggest
    return None

def consolidate_metrics(scanned_doc):
    all_out = {"source": scanned_doc.get("source"), "pages": []}
    for pg in scanned_doc.get("pages", []):
        page_no = pg.get("page_number")
        page_h = pg.get("height", 540.0)
        text = pg.get("text", "")
        words = pg.get("words", [])
        quarters, numbers, plain = split_words(words)

        # Prefer a title inferred from word positions/font sizes; fall back to text-only
        metric_from_words = detect_metric_title_from_words(words, pg.get("width", 960.0), page_h)
        metric_title = (metric_from_words or detect_metric_title(text)).strip()
        mt_low = metric_title.lower()
        is_percentage = ("net interest margin" in mt_low) or ("margin" in mt_low and "%" in mt_low)
        looks_like_chart = bool(quarters) and bool(numbers)

        # --- Prefer stacked-bar FIRST on non-NIM pages when left legend bands are visible ---
        if not is_percentage and looks_like_chart:
            cat_bands = find_category_bands(words)
            if len(cat_bands) >= 1:
                stacked_try = bind_stacked_bar_like(quarters, numbers, words, page_h)
                if stacked_try:
                    all_out["pages"].append({
                        "page": page_no,
                        "metric": metric_title,
                        "chart_type": "stacked-bar",
                        "extracted": stacked_try
                    })
                    continue

        # --- Semantic table from words (no whitelists) for period-vs-metric layouts ---
        if not is_percentage:
            semantic_tbl = build_semantic_table_from_words(words, pg.get("width", 960.0))
            if semantic_tbl:
                all_out["pages"].append({
                    "page": page_no,
                    "metric": metric_title,
                    "chart_type": "table",
                    "extracted": semantic_tbl
                })
                continue
# --- Table fallback: if no chart extracted, try the largest detected table ---

        # --- Prefer a big detected table for non-NIM pages (e.g., SP detail tables like page 17) ---
        if not is_percentage:
            biggest = looks_like_big_table(pg)
            if biggest:
                cleaned = clean_table_object({"headers": biggest.get("headers") or [], "rows": biggest.get("rows") or []}, words)
                all_out["pages"].append({
                    "page": page_no,
                    "metric": metric_title,
                    "chart_type": "table",
                    "extracted": cleaned
                })
                continue

        result={}
        chart_type="text-or-table"
        if looks_like_chart and is_percentage:
            chart_type="line-like"
            numbers_for_line = [n for n in numbers if looks_like_nim_value(n)]
            result = bind_line_like(quarters, numbers_for_line, page_h)
        elif looks_like_chart:
            # Try stacked-bar first if we can see any known category labels
            cat_bands = find_category_bands(words)
            if len(cat_bands) >= 1:
                chart_type = "stacked-bar"
                result = bind_stacked_bar_like(quarters, numbers, words, page_h)
                # fall back to simple bar if nothing extracted
                if not result:
                    chart_type = "bar-like"
                    result = bind_bar_like(quarters, numbers, page_h)
            else:
                chart_type = "bar-like"
                result = bind_bar_like(quarters, numbers, page_h)

        # Optional: rename keys if legend clearly detected
        if chart_type == "line-like" and result:
            legend = guess_legend_labels(plain)
            if legend == ["Commercial book","Group"]:
                # already named in bind_line_like as group_nim/commercial_nim
                pass

        # --- Table fallback: if no chart extracted, try the largest detected table ---
        if (not result) and (pg.get("tables")):
            tables = pg.get("tables") or []
            biggest = pick_biggest_table(tables)
            if biggest and (biggest.get("rows")):
                chart_type = "table"
                result = clean_table_object({"headers": biggest.get("headers") or [], "rows": biggest.get("rows") or []}, words)

        all_out["pages"].append({
            "page": page_no,
            "metric": metric_title,
            "chart_type": chart_type,
            "extracted": result
        })
    return all_out

metrics = consolidate_metrics(doc)
Path(OUT_METRICS_JSON).parent.mkdir(parents=True, exist_ok=True)
with open(OUT_METRICS_JSON, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"✅ Wrote metrics → {OUT_METRICS_JSON}")

# === Print a concise summary of detected metrics for ALL pages ===
print("\n=== Page metrics summary ===")
print(f"Source: {metrics.get('source')}")
pages_list = metrics.get("pages", [])
print(f"Total pages indexed: {len(pages_list)}\n")

# === Detailed extracts (only pages with extracted data) ===
print("\n=== Detailed extracts (pages with extracted data) ===")
for p in pages_list:
    extracted = p.get("extracted")
    if not extracted:
        continue
    page = p.get("page")
    metric = (p.get("metric") or "").strip()
    ctype = p.get("chart_type")
    print(f"\n[Page {page}] {metric}  ({ctype})")
    # pretty-print dicts or small tables
    if isinstance(extracted, dict) and "headers" in extracted and "rows" in extracted:
        headers = extracted.get("headers") or []
        rows = extracted.get("rows") or []
        preview = rows[:8]
        print("headers:", headers)
        for r in preview:
            print(r)
        if len(rows) > len(preview):
            print(f"... (+{len(rows)-len(preview)} more rows)")
    else:
        print(json.dumps(extracted, indent=2))

# === Quick confirm for NIM pages (line-like only) ===
for p in pages_list:
    metric_text = (p.get("metric") or "").lower()
    if p.get("chart_type") == "line-like" and "net interest margin" in metric_text:
        print(f"\n[NIM] page {p.get('page')}")
        print(json.dumps(p.get("extracted", {}), indent=2))

