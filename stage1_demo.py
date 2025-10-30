# 1. Install the marker library
# This command should be run in your terminal or a Colab cell:
# !pip install marker-pdf -q

# 2. Import necessary components
import subprocess
import shutil
from pathlib import Path
import sys
import hashlib
import re
import cv2
import numpy as np
import pandas as pd


def md5sum(file_path: Path, chunk_size: int = 8192) -> str:
    """Return the hex md5 of a file."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

# === OCR & extraction helpers ===
NUM_PAT = re.compile(r"^[+-]?\d{1,4}(?:[.,]\d+)?%?$")
NIM_KEYWORDS = ["net interest margin", "nim"]

QUARTER_PAT = re.compile(r"\b([1-4Iil|])\s*[QO0]\s*([0-9O]{2,4})\b", re.IGNORECASE)
# Simpler decade-only pattern for quarters, e.g., 2Q24, 1Q25
QUARTER_SIMPLE_PAT = re.compile(r"\b([1-4])Q(2\d)\b", re.IGNORECASE)  # e.g., 2Q24, 1Q25

# --- OCR character normalization for quarter tokens (common OCR mistakes) ---
_CHAR_FIX = str.maketrans({
    "O":"0","o":"0",
    "S":"5","s":"5",
    "I":"1","l":"1","|":"1","!":"1",
    "D":"0",
    "B":"3","8":"3",
    "Z":"2","z":"2"
})
def normalize_token(t: str) -> str:
    t = (t or "").strip()
    return t.translate(_CHAR_FIX).replace(" ", "")

# --- Helper: detect quarter tokens from nearby Markdown file ---
def detect_qlabels_from_md(dest_dir: Path, image_name: str) -> list[str]:
    """
    Scan the figure's markdown file for quarter tokens (e.g., 2Q24, 1Q2025).
    Returns tokens in document order (deduped).
    """
    try:
        md_file = dest_dir / f"{dest_dir.name}.md"
        if not md_file.exists():
            cand = list(dest_dir.glob("*.md"))
            if not cand:
                return []
            md_file = cand[0]
        text = md_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    # Collect all quarter tokens across the document
    tokens = []
    for m in QUARTER_PAT.finditer(text):
        q = f"{m.group(1)}Q{m.group(2)[-2:]}"
        tokens.append(q)
    # Deduplicate preserving order
    seen = set()
    ordered = []
    for q in tokens:
        if q not in seen:
            seen.add(q)
            ordered.append(q)
    return ordered

def load_image(path):
    p = Path(path)
    im = cv2.imread(str(p))
    if im is None:
        raise RuntimeError(f"cv2.imread() failed: {p}")
    return im

def preprocess(img_bgr):
    scale = 2.0
    img = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 8)
    return img, gray, thr, scale

def norm_num(s):
    s = s.replace(",", "").strip()
    pct = s.endswith("%")
    if pct:
        s = s[:-1]
    try:
        return float(s), pct
    except:
        return None, pct

def extract_numbers(ocr_results):
    rows = []
    for r in ocr_results or []:
        txt = str(r.get("text","")).strip()
        if NUM_PAT.match(txt):
            val, is_pct = norm_num(txt)
            if val is None:
                continue
            x1,y1,x2,y2 = r["bbox"]
            rows.append({
                "raw": txt, "value": val, "is_pct": is_pct, "conf": r.get("conf", None),
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "cx": int((x1+x2)/2), "cy": int((y1+y2)/2)
            })
    df = pd.DataFrame(rows).sort_values(["cy","cx"]).reset_index(drop=True)
    if "is_pct" not in df.columns and not df.empty:
        df["is_pct"] = df["raw"].astype(str).str.endswith("%")
    return df

def kmeans_1d(values, k=2, iters=20):
    values = np.asarray(values, dtype=float).reshape(-1,1)
    centers = np.array([values.min(), values.max()]).reshape(k,1)
    for _ in range(iters):
        d = ((values - centers.T)**2)
        labels = d.argmin(axis=1)
        new_centers = np.array([values[labels==i].mean() if np.any(labels==i) else centers[i] for i in range(k)]).reshape(k,1)
        if np.allclose(new_centers, centers, atol=1e-3):
            break
        centers = new_centers
    return labels, centers.flatten()

def run_easyocr(img_rgb):
    import easyocr
    global _EASY_OCR_READER
    try:
        _EASY_OCR_READER
    except NameError:
        _EASY_OCR_READER = None
    if _EASY_OCR_READER is None:
        _EASY_OCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    results = _EASY_OCR_READER.readtext(img_rgb, detail=1, paragraph=False)
    out = []
    for quad, text, conf in results:
        (x1,y1),(x2,y2),(x3,y3),(x4,y4) = quad
        out.append({"bbox": (int(x1),int(y1),int(x3),int(y3)), "text": str(text), "conf": float(conf)})
    return out

# --- Focused bottom-axis quarter detection using EasyOCR (robust to OCR confusions) ---
def detect_quarters_easyocr(img_bgr):
    """
    Use EasyOCR to read quarter labels along the bottom axis.
    Returns a list of (x_global, 'nQyy') sorted left→right, with half-year tokens removed.
    """
    H, W = img_bgr.shape[:2]
    y0 = int(H * 0.66)  # bottom ~34%
    crop = img_bgr[y0:H, 0:W]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 8)
    # kernel = np.ones((3,3), np.uint8)
    # thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    up = cv2.resize(thr, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    img_rgb = cv2.cvtColor(up, cv2.COLOR_GRAY2RGB)
    ocr = run_easyocr(img_rgb)
    # PASS 1 — direct regex on normalized tokens
    tokens = []
    for r in ocr or []:
        raw = str(r.get("text","")).strip()
        x1,y1,x2,y2 = r["bbox"]
        cx_local = (x1 + x2) // 2
        cx_global = int(cx_local / 3.0)  # undo scaling
        tokens.append({"x": cx_global, "raw": raw, "norm": normalize_token(raw)})
    def _is_half_token(t: str) -> bool:
        t = (t or "").lower().replace(" ", "")
        return ("9m" in t) or ("1h" in t) or ("h1" in t) or ("h2" in t) or ("2h" in t)
    quarters = []
    for t in tokens:
        if _is_half_token(t["norm"]):
            continue
        m = QUARTER_PAT.search(t["norm"])
        if m:
            q = f"{m.group(1)}Q{m.group(2)[-2:]}"
            q = normalize_token(q)
            quarters.append((t["x"], q))
    # PASS 2 — stitch split tokens if too few quarters were found
    if len(quarters) < 4 and tokens:
        pieces = sorted(tokens, key=lambda d: d["x"])
        digits_1to4 = [p for p in pieces if p["norm"] in ("1","2","3","4")]
        q_only      = [p for p in pieces if p["norm"].upper() == "Q"]
        q_with_year = [p for p in pieces if re.fullmatch(r"Q[0-9O]{2,4}", p["norm"], flags=re.I)]
        years_2d    = [p for p in pieces if re.fullmatch(r"[0-9O]{2,4}", p["norm"])]
        def near(a, b, tol=70):
            return abs(a["x"] - b["x"]) <= tol
        for d in digits_1to4:
            # digit + Qyy
            candidates = [q for q in q_with_year if near(d, q)]
            if candidates:
                qtok = min(candidates, key=lambda q: abs(q["x"]-d["x"]))
                qyy = normalize_token(qtok["norm"])[1:]
                quarters.append(((d["x"]+qtok["x"])//2, f"{d['norm']}Q{qyy[-2:]}"))
                continue
            # digit + Q + yy
            qs = [q for q in q_only if near(d, q)]
            ys = [y for y in years_2d if near(d, y, tol=120)]
            if qs and ys:
                qtok = min(qs, key=lambda q: abs(q["x"]-d["x"]))
                ytok = min(ys, key=lambda y: abs(y["x"]-qtok["x"]))
                yy = normalize_token(ytok["norm"])
                quarters.append(((d["x"]+ytok["x"])//2, f"{d['norm']}Q{yy[-2:]}"))
                continue
    if not quarters:
        return []
    quarters.sort(key=lambda t: t[0])
    deduped, last_x = [], -10**9
    for x,q in quarters:
        if abs(x - last_x) <= 22:
            continue
        deduped.append((x,q))
        last_x = x
    return deduped

# NIM value band (pct) and geometry heuristics for verification
NIM_MIN, NIM_MAX = 1.3, 3.2
TOP_FRACTION = 0.65     # widen band: NIM labels often sit higher than 45%
RIGHT_HALF_ONLY = True  # NIM values appear on right panel in these deck

def is_strict_nim_image(img_path: Path) -> tuple[bool, str]:
    """
    Heuristic re-check:
      1) Title/text contains NIM keywords (coarse gate)
      2) Percent tokens mostly within NIM_MIN..NIM_MAX
      3) Tokens located in the top region (and right half, if enabled)
    Returns (ok, reason)
    """
    try:
        img_bgr = load_image(img_path)
        H, W = img_bgr.shape[:2]
        # 1) quick-text gate (soft): don't return yet; allow numeric signature to validate
        kw_ok = is_relevant_image(img_path, NIM_KEYWORDS)
        # 2) numeric gate on enhanced image
        img_up, gray, thr, scale = preprocess(img_bgr)
        img_rgb = cv2.cvtColor(thr, cv2.COLOR_GRAY2RGB)
        ocr = run_easyocr(img_rgb)
        # --- Semantic gate: accept classic NIM slides based on stable labels ---
        text_lower = " ".join(str(r.get("text", "")).lower() for r in ocr or [])
        has_nim = "net interest margin" in text_lower
        has_cb  = "commercial book" in text_lower
        has_grp = "group" in text_lower
        if has_nim and (has_cb or has_grp):
            which = [w for w, ok in (("nim", has_nim), ("cb", has_cb), ("grp", has_grp)) if ok]
            return (True, f"ok_semantic({'+' .join(which)})")
        df = extract_numbers(ocr)
        if df.empty:
            return (False, "no_numbers")
        # geometry filters (apply before value checks)
        top_cut = int(img_up.shape[0] * 0.62)
        cond_geom = (df["cy"] < top_cut)
        if RIGHT_HALF_ONLY:
            cond_geom &= (df["cx"] > (img_up.shape[1] // 2))

        # 2a) Preferred path: explicit percentage tokens
        df_pct = df[(df["is_pct"] == True) & cond_geom].copy()
        if not df_pct.empty:
            in_band = df_pct["value"].between(NIM_MIN, NIM_MAX)
            ratio = float(in_band.sum()) / float(len(df_pct))
            if ratio >= 0.6:
                return (True, "ok")
            else:
                return (False, f"non_nim_values_out_of_band({ratio:.2f})")

        # 2b) Fallback: some decks omit the % sign near the series values.
        # Accept plain numbers in the NIM range if units are explicit or implied, or if numeric signature is strong.
        title_text = text_lower  # already computed above
        has_units_pct = "(%)" in title_text or "margin (%)" in title_text or has_nim
        df_nums = df[(df["is_pct"] == False) & cond_geom].copy()
        if not df_nums.empty:
            in_band = df_nums["value"].between(NIM_MIN, NIM_MAX)
            ratio = float(in_band.sum()) / float(len(df_nums))
            # Case A: explicit or implied units in title → accept when enough in-band hits
            if has_units_pct and ratio >= 0.6 and in_band.sum() >= 3:
                return (True, "ok_no_percent_signs")
            # Case B: title OCR may have missed units; if the quick keyword gate succeeded, accept with a stricter ratio
            if kw_ok and ratio >= 0.7 and in_band.sum() >= 3:
                return (True, "ok_numeric_signature")
            # Case C: strong structural evidence (quarters on bottom) + numeric signature in band
            q_xy_fallback = detect_quarters_easyocr(img_bgr)
            if len(q_xy_fallback) >= 4 and ratio >= 0.6 and in_band.sum() >= 3:
                return (True, "ok_structural_numeric_signature")

        # Final decision: if numeric signature still failed, report clearer reason
        if not kw_ok:
            return (False, "irrelevant_non_nim")
        else:
            return (False, "no_percentages_or_units")
    except Exception as e:
        return (False, f"exception:{e}")


# --- Helper: detect and order quarter labels from OCR ---
def detect_qlabels(ocr_results, img_width: int) -> list[str]:
    """
    Extract quarter tokens like 1Q25, 2Q2025 from OCR and return them left→right.
    We keep only tokens on the right half (where the series values live in your layout).
    """
    qtokens = []
    mid_x = img_width // 2
    for r in ocr_results or []:
        txt = str(r.get("text","")).strip()
        m = QUARTER_PAT.search(txt)
        if not m:
            continue
        x1,y1,x2,y2 = r["bbox"]
        cx = (x1 + x2) // 2
        if cx <= mid_x:
            continue  # ignore left panel quarters/titles
        q = f"{m.group(1)}Q{m.group(2)[-2:]}"  # normalize to 1Q25 style
        qtokens.append((cx, q))
    # sort by visual x-position and deduplicate by both text and proximity (ignore near-duplicates)
    qtokens.sort(key=lambda x: x[0])
    # Deduplicate by both text and proximity (ignore near-duplicates)
    ordered = []
    last_x = -9999
    last_q = None
    for x, q in qtokens:
        if last_q == q and abs(x - last_x) < 30:
            continue
        ordered.append(q)
        last_x, last_q = x, q
    return ordered

# === Focused bottom-of-chart scan for small quarter labels ===
def detect_qlabels_bottom(img_bgr) -> list[str]:
    """
    Focused pass: crop the bottom ~30% (where quarter labels usually sit),
    enhance contrast, OCR, and extract quarter tokens left→right.
    """
    try:
        H, W = img_bgr.shape[:2]
        y0 = int(H * 0.60)  # bottom 40%
        crop = img_bgr[y0:H, 0:W]
        # Enhance: grayscale -> bilateral -> CLAHE -> adaptive threshold
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 8)
        # Morphological close to strengthen thin glyphs
        kernel = np.ones((3,3), np.uint8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Upscale for small text
        up = cv2.resize(thr, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        img_rgb = cv2.cvtColor(up, cv2.COLOR_GRAY2RGB)
        ocr = run_easyocr(img_rgb)
        # Map bboxes back to global coords: decide single-panel vs split-panel
        mid_x = W // 2
        left_quarters, right_quarters = [], []
        left_tokens_text, right_tokens_text = [], []
        for r in ocr or []:
            raw = str(r.get("text", "")).strip()
            x1,y1,x2,y2 = r["bbox"]
            cx_local = (x1 + x2) // 2
            cx_global = int(cx_local / 2.5)  # undo scale

            if cx_global <= mid_x:
                left_tokens_text.append(raw.lower())
            else:
                right_tokens_text.append(raw.lower())

            m = QUARTER_PAT.search(raw)
            if not m:
                continue
            q = f"{m.group(1)}Q{m.group(2)[-2:]}"
            if cx_global <= mid_x:
                left_quarters.append((cx_global, q))
            else:
                right_quarters.append((cx_global, q))

        def has_halfyear_or_9m(tokens: list[str]) -> bool:
            s = " ".join(tokens)
            return ("9m" in s) or ("1h" in s) or ("h1" in s) or ("h2" in s) or ("2h" in s)

        left_has_h = has_halfyear_or_9m(left_tokens_text)
        # Panel selection logic: prefer both halves unless left clearly half-year and right has ≥3 quarters
        if (not left_has_h) and (len(left_quarters) + len(right_quarters) >= 2):
            # Likely single panel or weak OCR on one side → use both halves
            qtokens = left_quarters + right_quarters
        elif len(right_quarters) >= 3:
            # Strong right panel signal → use right only
            qtokens = right_quarters
        else:
            # Fallback: use everything we found
            qtokens = left_quarters + right_quarters

        # Sort and dedupe close neighbors (≤18 px)
        qtokens.sort(key=lambda t: t[0])
        deduped = []
        last_x = -10**9
        for x, q in qtokens:
            if abs(x - last_x) <= 18:
                continue
            deduped.append((x, q))
            last_x = x

        return [q for _, q in deduped]
    except Exception:
        return []

# --- Same as detect_qlabels_bottom, but returns (x, label) for alignment ---
def detect_qlabels_bottom_with_xy(img_bgr) -> list[tuple[int, str]]:
    try:
        H, W = img_bgr.shape[:2]
        y0 = int(H * 0.60)
        crop = img_bgr[y0:H, 0:W]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 8)
        kernel = np.ones((3,3), np.uint8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
        up = cv2.resize(thr, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        img_rgb = cv2.cvtColor(up, cv2.COLOR_GRAY2RGB)
        ocr = run_easyocr(img_rgb)

        mid_x = W // 2
        left_quarters, right_quarters = [], []
        left_tokens_text = []
        for r in ocr or []:
            raw = str(r.get("text", "")).strip()
            x1,y1,x2,y2 = r["bbox"]
            cx_local = (x1 + x2) // 2
            cx_global = int(cx_local / 2.5)
            if cx_global <= mid_x:
                left_tokens_text.append(raw.lower())
            m = QUARTER_PAT.search(raw)
            if not m:
                continue
            q = f"{m.group(1)}Q{m.group(2)[-2:]}"
            if cx_global <= mid_x:
                left_quarters.append((cx_global, q))
            else:
                right_quarters.append((cx_global, q))

        def has_halfyear_or_9m(tokens: list[str]) -> bool:
            s = " ".join(tokens)
            return ("9m" in s) or ("1h" in s) or ("h1" in s) or ("h2" in s) or ("2h" in s)

        left_has_h = has_halfyear_or_9m(left_tokens_text)
        if (not left_has_h) and (len(left_quarters) + len(right_quarters) >= 2):
            # Likely single panel or weak OCR on one side → use both halves
            qtokens = left_quarters + right_quarters
        elif len(right_quarters) >= 3:
            # Strong right panel signal → use right only
            qtokens = right_quarters
        else:
            # Fallback: use everything we found
            qtokens = left_quarters + right_quarters

        qtokens.sort(key=lambda t: t[0])
        deduped = []
        last_x = -10**9
        for x, q in qtokens:
            if abs(x - last_x) <= 18:
                continue
            deduped.append((x, q))
            last_x = x
        return deduped
    except Exception:
        return []

# --- Merge two ordered quarter lists ---
def _merge_ordered(primary: list[str], secondary: list[str]) -> list[str]:
    """
    Merge two left→right sequences, keeping 'primary' order and filling with
    any unseen items from 'secondary' in their order.
    """
    out = list(primary)
    seen = set(primary)
    for q in secondary:
        if q not in seen:
            out.append(q)
            seen.add(q)
    return out

# --- Expand a quarter label like '2Q24' forward n quarters ---
def _expand_quarters(start_q: str, n: int) -> list[str]:
    """
    Given a label like '2Q24', produce a forward sequence of n quarters:
    2Q24, 3Q24, 4Q24, 1Q25, 2Q25, ...
    """
    m = QUARTER_PAT.match(start_q) or QUARTER_SIMPLE_PAT.match(start_q)
    if not m:
        return []
    q = int(m.group(1))
    yy = int(m.group(2)[-2:])
    seq = []
    for _ in range(n):
        seq.append(f"{q}Q{yy:02d}")
        q += 1
        if q == 5:
            q = 1
            yy = (yy + 1) % 100
    return seq

# --- Find a plausible anchor quarter like 2Q24 from OCR or markdown tokens ---
def _anchor_quarter_from_texts(ocr_results, md_tokens: list[str]) -> str | None:
    """
    Find any token like 1Q2x..4Q2x from OCR texts or markdown tokens.
    Returns the first plausible anchor (normalized to e.g. 2Q24) or None.
    """
    # prefer bottom/ocr-derived tokens first (already parsed in detect_qlabels_bottom)
    # fallback: scan all OCR texts with simple pattern
    for r in ocr_results or []:
        txt = str(r.get("text","")).strip()
        m = QUARTER_SIMPLE_PAT.search(txt)
        if m:
            return f"{m.group(1)}Q{m.group(2)}"
    # fallback to any markdown token that matches the decade pattern
    for t in md_tokens or []:
        m = QUARTER_SIMPLE_PAT.match(t)
        if m:
            return f"{m.group(1)}Q{m.group(2)}"
    return None

def extract_series_from_df(df, img_up, ocr_results=None, qlabels_hint=None):
    H, W = img_up.shape[:2]
    mid_x = W//2
    top_band_min = int(H * 0.38)
    top_band_max = int(H * 0.58)

    # Detect bottom quarter labels (with x) early to infer layout
    detected_q_bot_xy = detect_quarters_easyocr(img_up)
    left_count  = sum(1 for x, _ in detected_q_bot_xy if x <= mid_x)
    right_count = sum(1 for x, _ in detected_q_bot_xy if x >  mid_x)
    # Heuristic: if we see ≥4 quarter tokens spanning both halves, it's a single-panel timeline
    single_panel = (len(detected_q_bot_xy) >= 4 and left_count >= 1 and right_count >= 1)

    # Filter tokens: keep right-half only for split panels; keep all for single panels
    if single_panel:
        pct = df[(df.is_pct==True)].copy()
        nums = df[(df.is_pct==False)].copy()
    else:
        pct = df[(df.is_pct==True) & (df.cx > mid_x)].copy()
        nums = df[(df.is_pct==False) & (df.cx > mid_x)].copy()

    if pct.empty:
        # Fallback for charts that omit the '%' sign on the value dots.
        # Use a wider top band and avoid forcing right-half on single-panel timelines.
        approx_top = int(H * 0.60)
        if single_panel:
            cx_mask = (df.cx > 0)  # keep all x for single panel
        else:
            cx_mask = (df.cx > mid_x)
        cand_pct = df[cx_mask & df.value.between(NIM_MIN, NIM_MAX) & (df.cy < approx_top)].copy()
        if not cand_pct.empty:
            cand_pct["is_pct"] = True
            pct = cand_pct

    nim_df = pd.DataFrame()
    if not pct.empty:
        # Try to split into two horizontal series by Y even when we have only 3 quarters (→ 6 points)
        # Deduplicate by proximity on Y to stabilize clustering
        y_sorted = pct.sort_values("cy")["cy"].to_numpy()
        uniq_y = []
        last_y = -10**9
        for yy in y_sorted:
            if abs(yy - last_y) >= 6:  # 6px tolerance for duplicates
                uniq_y.append(yy)
                last_y = yy
        # Attempt k-means when we have at least 4 points total (≈ 2 series × 2 quarters)
        if pct.shape[0] >= 4 and len(uniq_y) >= 2:
            labels, centers = kmeans_1d(pct["cy"].values, k=2)
            pct["series"] = labels
            order = np.argsort(centers)  # top (commercial) should have smaller y
            remap = {order[0]: "Commercial NIM (%)", order[1]: "Group NIM (%)"}
            pct["series_name"] = pct["series"].map(remap)
            # Sanity: ensure both series have data; else collapse to one
            counts = pct["series_name"].value_counts()
            if any(counts.get(name, 0) == 0 for name in ["Commercial NIM (%)", "Group NIM (%)"]):
                pct["series_name"] = "NIM (%)"
        else:
            pct["series_name"] = "NIM (%)"

        # Reuse bottom-quarter labels captured above
        detected_q_bot = [q for _, q in detected_q_bot_xy]
        detected_q_ocr = detect_qlabels(ocr_results or [], W) if ocr_results is not None else []
        if len(detected_q_bot) > len(detected_q_ocr):
            detected_q = _merge_ordered(detected_q_bot, detected_q_ocr)
        else:
            detected_q = _merge_ordered(detected_q_ocr, detected_q_bot)
        rows = []
        for name, sub in pct.groupby("series_name"):
            # Sort left→right and collapse near-duplicates (same x within 12px)
            sub_sorted = sub.sort_values("cx")
            uniq_rows = []
            last_x = -10**9
            for r in sub_sorted.itertuples(index=False):
                if abs(r.cx - last_x) < 12:
                    continue
                uniq_rows.append(r)
                last_x = r.cx
            # Keep only the right-panel portion (already ensured by cx>mid_x earlier)
            pick = list(uniq_rows)[-5:]  # cap to 5 most recent positions, but may be <5
            n = len(pick)
            if n == 0:
                continue
            labels = []
            # Robust mapping: map each value x to its nearest bottom quarter label x (right panel).
            # Filter any accidental half-year tokens (1H/2H/H1/H2/9M) just in case OCR returns them.
            def _is_half_token(t: str) -> bool:
                t = (t or "").lower().replace(" ", "")
                return ("9m" in t) or ("1h" in t) or ("h1" in t) or ("h2" in t) or ("2h" in t) or ("h24" in t) or ("h23" in t)

            # detected_q_bot_xy already respects split vs single panel. Keep right-panel positions only here.
            q_xy = []
            for x, q in detected_q_bot_xy:
                if x <= mid_x:
                    continue
                if _is_half_token(q):
                    continue
                q_xy.append((x, q))

            if len(q_xy) < n:
                # Borrow from left panel if they look like quarters (and not half-year)
                for x, q in detected_q_bot_xy:
                    if x > mid_x:
                        continue
                    if _is_half_token(q):
                        continue
                    q_xy.append((x, q))

            if q_xy:
                q_xy.sort(key=lambda t: t[0])  # left→right
                # Map each picked value to nearest quarter label by x-position
                vx = [rr.cx for rr in pick]
                qx = [x for x, _ in q_xy]
                ql = [q for _, q in q_xy]
                mapped = []
                for x in vx:
                    j = int(np.argmin([abs(x - xx) for xx in qx])) if qx else -1
                    mapped.append(ql[j] if j >= 0 else None)
                labels = mapped
            else:
                detected_q_ocr = detect_qlabels(ocr_results or [], W) if ocr_results is not None else []
                if detected_q_ocr:
                    labels = detected_q_ocr[-n:] if len(detected_q_ocr) >= n else detected_q_ocr

            # If still short, use markdown tokens; else expand from an anchor like 2Q24
            if (not labels) or (len(labels) != n):
                if qlabels_hint:
                    labels = qlabels_hint[-n:] if len(qlabels_hint) >= n else qlabels_hint
            if (not labels) or (len(labels) != n):
                anchor = _anchor_quarter_from_texts(ocr_results, qlabels_hint)
                if anchor:
                    labels = _expand_quarters(anchor, n)
            if (not labels) or (len(labels) != n):
                labels = [f"{i+1}Q??" for i in range(n)]
            # Ensure left→right order for consistent mapping to labels
            pick = sorted(pick, key=lambda r: r.cx)
            labels = list(labels)[:n]
            for i, r in enumerate(pick):
                if i >= len(labels):
                    break
                rows.append({"Quarter": labels[i], "series": name, "value": r.value})
        if rows:
            nim_table = pd.DataFrame(rows)
            # Guard: drop rows with missing labels
            nim_table = nim_table.dropna(subset=["Quarter", "series"])  
            # If multiple detections map to the same (Quarter, series), average them
            if not nim_table.empty:
                dupe_mask = nim_table.duplicated(subset=["Quarter", "series"], keep=False)
                if dupe_mask.any():
                    # Aggregate duplicates by mean (stable for minor OCR jitter)
                    nim_table = nim_table.groupby(["Quarter", "series"], as_index=False)["value"].mean()
            nim_df = nim_table.pivot(index="Quarter", columns="series", values="value").reset_index()

    # NIM-only mode: skip NII extraction entirely
    nii_df = pd.DataFrame()

    def _sort_q(df_in):
        if df_in is None or df_in.empty or "Quarter" not in df_in.columns:
            return df_in
        # Try to sort by numeric (Q#, year) if labels are like 2Q24; else keep input order
        def _key(q):
            m = QUARTER_PAT.match(str(q))
            if not m:
                return (999, 999)
            qn = int(m.group(1))
            yr = int(m.group(2)[-2:])  # last two digits
            return (yr, qn)
        try:
            return df_in.assign(_k=df_in["Quarter"].map(_key)).sort_values("_k").drop(columns=["_k"]).reset_index(drop=True)
        except Exception:
            return df_in.reset_index(drop=True)

    return _sort_q(nim_df), _sort_q(nii_df)

def _extract_md_context(dest_dir: Path, image_name: str) -> dict:
    """
    Best-effort: read the <pdf_stem>.md in dest_dir, find the <image_name> reference,
    capture nearby headings and a neighbor paragraph to build context.
    """
    try:
        # Prefer "<pdf_stem>.md", else any .md
        md_file = dest_dir / f"{dest_dir.name}.md"
        if not md_file.exists():
            cands = list(dest_dir.glob("*.md"))
            if not cands:
                return {}
            md_file = cands[0]
        lines = md_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return {}

    # Find the image line
    idx = None
    for i, line in enumerate(lines):
        if image_name in line:
            idx = i
            break
    if idx is None:
        return {}

    # Walk upward to find up to two headings and a neighbor paragraph
    figure_title = None
    section_title = None
    neighbor_text = None

    # Find the closest preceding heading(s)
    for j in range(idx - 1, -1, -1):
        s = lines[j].strip()
        if not s:
            continue
        # markdown heading levels
        if s.startswith("#"):
            # Remove leading #'s and whitespace
            heading = s.lstrip("#").strip()
            if figure_title is None:
                figure_title = heading
            elif section_title is None:
                section_title = heading
                break

    # Find a non-empty paragraph between the image and last heading
    for j in range(idx - 1, -1, -1):
        s = lines[j].strip()
        if s and not s.startswith("#") and not s.startswith("![]("):
            neighbor_text = s
            break

    out = {}
    if figure_title: out["figure_title"] = figure_title
    if section_title: out["section_title"] = section_title
    if neighbor_text: out["neighbor_text"] = neighbor_text
    return out

def _parse_page_and_figure_from_name(image_name: str) -> dict:
    """
    Extract page/figure indices from names like '_page_0_Figure_2.jpeg'.
    """
    info = {}
    try:
        # Very loose parse
        if "_page_" in image_name:
            after = image_name.split("_page_", 1)[1]
            num = after.split("_", 1)[0]
            info["page"] = int(num) + 1  # 1-based for human readability
        if "Figure_" in image_name:
            after = image_name.split("Figure_", 1)[1]
            num = ""
            for ch in after:
                if ch.isdigit():
                    num += ch
                else:
                    break
            if num:
                info["figure_index"] = int(num)
    except Exception:
        pass
    return info

def is_relevant_image(img_path, keywords):
    """Robust relevance check for NIM slides.
    - Reuse the singleton EasyOCR reader (run_easyocr)
    - Accept split tokens like "Net" / "interest" / "margin" (not only the exact phrase)
    - Fallback: if we see ≥4 quarter labels on the bottom AND ≥3 top-band percent-like values in NIM range, treat as relevant.
    """
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return False

        # Pass A: OCR on lightly upscaled original
        view_a = cv2.resize(img, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
        ocr_a = run_easyocr(cv2.cvtColor(view_a, cv2.COLOR_BGR2RGB))
        tokens_a = [str(r.get("text","")).lower() for r in (ocr_a or [])]
        text_a = " ".join(tokens_a)

        # Quick phrase match (exact keywords like "net interest margin")
        if any(k in text_a for k in keywords):
            return True

        # Pass B: OCR on preprocessed thresholded view (more stable for thin fonts)
        _, _, thr, _ = preprocess(img)
        ocr_b = run_easyocr(cv2.cvtColor(thr, cv2.COLOR_GRAY2RGB))
        tokens_b = [str(r.get("text","")).lower() for r in (ocr_b or [])]
        text_b = " ".join(tokens_b)
        if any(k in text_b for k in keywords):
            return True

        # Token-level split-word check
        tokens = tokens_a + tokens_b
        has_net      = any("net" in t for t in tokens)
        has_interest = any("interest" in t for t in tokens)
        has_margin   = any("margin" in t for t in tokens or [])
        has_nim_abbr = any(re.search(r"\bnim\b", t) for t in tokens)
        has_cb       = any("commercial book" in t for t in tokens)
        has_grp      = any(re.search(r"\bgroup\b", t) for t in tokens)
        if (has_net and has_interest and has_margin) or has_nim_abbr:
            # Strengthen with context words if available
            if has_cb or has_grp:
                return True

        # Structural fallback: quarters + percent values in the NIM band
        q_xy = detect_quarters_easyocr(img)
        if len(q_xy) >= 4:
            # Look for ≥3 percent-ish values in the top band within NIM_MIN..NIM_MAX
            df = extract_numbers(ocr_b)
            if not df.empty:
                H, W = view_a.shape[:2]
                top_cut = int(H * 0.55)
                in_top = df["cy"] < top_cut
                in_band = df["value"].between(NIM_MIN, NIM_MAX)
                pctish = in_band  # allow numbers without % (the series sometimes omit it)
                if int((in_top & pctish).sum()) >= 3:
                    return True

        return False
    except Exception:
        return False


# =============== Pluggable OCR Extractor Framework ===============
class BaseChartExtractor:
    """
    Minimal interface for pluggable chart extractors.
    Implement `is_relevant` and `extract_table`, then call `handle_image(...)`.
    """
    name = "base"
    topic = "Generic Chart"
    units = None
    entity = None
    keywords = []

    def is_relevant(self, img_path: Path) -> bool:
        return is_relevant_image(img_path, self.keywords)

    def extract_table(self, img_path: Path, dest_dir: Path, pdf_name: str):
        """
        Return (df, context_dict) or (None, reason) on failure.
        context_dict will be merged into the _context object.
        """
        raise NotImplementedError

    def _build_context(self, pdf_name: str, img_path: Path, dest_dir: Path, extra: dict | None = None) -> dict:
        ctx = {
            "source_pdf": pdf_name,
            "image": img_path.name,
            "topic": self.topic,
        }
        if self.units:  ctx["units"]  = self.units
        if self.entity: ctx["entity"] = self.entity
        ctx.update(_parse_page_and_figure_from_name(img_path.name))
        md_ctx = _extract_md_context(dest_dir, img_path.name)
        if md_ctx: ctx.update(md_ctx)
        if extra:  ctx.update(extra)
        return ctx

    def _write_jsonl(self, out_path: Path, ctx: dict, df: pd.DataFrame):
        import json
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"_context": ctx}, ensure_ascii=False) + "\n")
            for rec in df.to_dict(orient="records"):
                rec_out = dict(rec)
                rec_out["_meta"] = {"source_pdf": ctx.get("source_pdf"), "image": ctx.get("image")}
                f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

    def handle_image(self, img_path: Path, dest_dir: Path, pdf_name: str, *, bypass_relevance: bool = False):
        if not bypass_relevance and not self.is_relevant(img_path):
            return False, "Not relevant"
        df, ctx_extra = self.extract_table(img_path, dest_dir, pdf_name)
        if df is None or df.empty:
            return False, ctx_extra if isinstance(ctx_extra, str) else "No data"
        # Build context and summary if possible
        ctx = self._build_context(pdf_name, img_path, dest_dir, extra=ctx_extra if isinstance(ctx_extra, dict) else {})
        try:
            cols = [c for c in df.columns if c != "Quarter"]
            if len(df) >= 2 and cols:
                def _pick_q(s):
                    return s if QUARTER_PAT.match(str(s) or "") else None
                _fq = str(df.iloc[0]["Quarter"])
                _lq = str(df.iloc[-1]["Quarter"])
                first_q = _pick_q(_fq) or (_fq if "??" not in _fq else "start")
                last_q  = _pick_q(_lq) or (_lq if "??" not in _lq else "end")
                pieces = []
                for col in cols[:2]:
                    a = df.iloc[0][col]
                    b = df.iloc[-1][col]
                    if pd.notna(a) and pd.notna(b):
                        suffix = "%" if "NIM" in col or ctx.get("units") == "percent" else ""
                        pieces.append(f"{col}: {a:.2f}{suffix} → {b:.2f}{suffix}")
                if pieces:
                    ctx["summary"] = f"Figure shows {', '.join(pieces)} from {first_q} to {last_q}."
        except Exception:
            pass
        out_path = img_path.with_suffix(f".{self.name}.jsonl")
        self._write_jsonl(out_path, ctx, df)
        return True, str(out_path)

class NIMExtractor(BaseChartExtractor):
    name = "nim"
    topic = "Net Interest Margin"
    units = "percent"
    entity = "DBS"
    keywords = NIM_KEYWORDS

    def extract_table(self, img_path: Path, dest_dir: Path, pdf_name: str):
        # Reuse the existing pipeline
        img_bgr = load_image(img_path)
        img_up, gray, thr, scale = preprocess(img_bgr)
        img_rgb = cv2.cvtColor(thr, cv2.COLOR_GRAY2RGB)
        ocr = run_easyocr(img_rgb)
        df_tokens = extract_numbers(ocr)
        if df_tokens.empty:
            return None, "No numeric tokens detected"
        md_q = detect_qlabels_from_md(dest_dir, img_path.name)
        nim_df, _nii_df = extract_series_from_df(df_tokens, img_up, ocr_results=ocr, qlabels_hint=md_q)
        if nim_df is None or nim_df.empty:
            return None, "No NIM table detected"
        return nim_df, {"topic": self.topic, "units": self.units, "entity": self.entity}

# Registry of extractors (add more later)
EXTRACTORS: list[BaseChartExtractor] = [
    NIMExtractor(),
]
# ============= End pluggable extractor framework =============

# === Single-image rebuild/verify mode (optional) ===
# Set single_image_mode=True and point single_image_path to a specific extracted image
# to run the two-stage gate + extraction just for that file, then exit.
single_image_mode = False
single_image_paths: list[Path] = [
   
]
# Optional singular fallback path (legacy): set to a string/Path if you want a single-image override
single_image_path = None

# Legacy fallback (ignored i
 # Toggle: if True → normal md5 skip; if False → always reprocess
md5_check = True

# 3. Define the path to the directory containing your PDF files
pdf_directory = Path("/Users/marcusfoo/Documents/GitHub/PTO_ICT3113_Grp1/Demo/")

# === Fast path: single image only ===
# === Fast path: single/multi-image only ===
if single_image_mode:
    paths: list[Path] = []
    if single_image_paths:
        paths = [Path(p) for p in single_image_paths if p is not None]
    elif single_image_path:
        paths = [Path(single_image_path)]

    if not paths:
        print("❌ single_image_mode=True but no paths were provided.")
        sys.exit(1)

    print("--- Multi-image mode ---")
    successes = 0
    for img_path in paths:
        if not img_path.exists():
            print(f"❌ Missing: {img_path}")
            continue

        dest_dir = img_path.parent
        pdf_name = f"{dest_dir.name}.pdf"
        print(f"\n🖼️  Image: {img_path.name}  |  PDF: {pdf_name}")

        # Quick quarter readout (EasyOCR-only, bottom axis)
        try:
            img_bgr_quarters = load_image(img_path)
            q_xy = detect_quarters_easyocr(img_bgr_quarters)
            if q_xy:
                print("   📎 Quarters (EasyOCR):", ", ".join([q for _,q in q_xy]))
            else:
                print("   📎 Quarters (EasyOCR): <none>")
        except Exception as _qe:
            print(f"   📎 Quarters (EasyOCR): error → {_qe}")

        any_hit = False

        for ex in EXTRACTORS:
            print(f"   · [{ex.name}] quick gate…", end=" ")
            if not ex.is_relevant(img_path):
                print("⏭️  Not relevant")
                continue
            print("✅ ok; strict gate…", end=" ")
            ok_strict, reason = is_strict_nim_image(img_path)
            if not ok_strict:
                print(f"⏭️  Failed strict ({reason})")
                continue
            print("✅ Strict OK — extracting…")

            # Extract directly so we can print the table; still write JSONL
            df, ctx_extra = ex.extract_table(img_path, dest_dir, pdf_name)
            if df is None or df.empty:
                print("   ⚠️ No data extracted.")
                continue

            any_hit = True
            successes += 1

            # Build context + summary and write JSONL
            ctx = ex._build_context(pdf_name, img_path, dest_dir, extra=ctx_extra if isinstance(ctx_extra, dict) else {})
            try:
                cols = [c for c in df.columns if c != "Quarter"]
                if len(df) >= 2 and cols:
                    def _pick_q(s):
                        return s if QUARTER_PAT.match(str(s) or "") else None
                    _fq = str(df.iloc[0]["Quarter"]); _lq = str(df.iloc[-1]["Quarter"])
                    first_q = _pick_q(_fq) or (_fq if "??" not in _fq else "start")
                    last_q  = _pick_q(_lq) or (_lq if "??" not in _lq else "end")
                    pieces = []
                    for col in cols[:2]:
                        a = df.iloc[0][col]; b = df.iloc[-1][col]
                        if pd.notna(a) and pd.notna(b):
                            suffix = "%" if "NIM" in col or ctx.get("units") == "percent" else ""
                            pieces.append(f"{col}: {a:.2f}{suffix} → {b:.2f}{suffix}")
                    if pieces:
                        ctx["summary"] = f"Figure shows {', '.join(pieces)} from {first_q} to {last_q}."
            except Exception:
                pass

            out_path = img_path.with_suffix(f".{ex.name}.jsonl")
            ex._write_jsonl(out_path, ctx, df)
            print(f"   💾 Saved JSONL → {out_path}")

            # Pretty-print the extracted table directly
            try:
                print("\n   📊 Extracted table:")
                print(df.to_string(index=False))
            except Exception:
                print(df)

        if not any_hit:
            print("   ⏭️  No matching extractors for this image.")

    print(f"\n✅ Done. Extracted from {successes} image(s).")
    # Prevent the pipeline (marker/md5) from running if notebook catches SystemExit
    globals()["_STOP_AFTER_SINGLE"] = True
    sys.exit(0)
    
# Check if the directory exists before proceeding
if not pdf_directory.is_dir():
    print(f"❌ ERROR: The directory was not found at '{pdf_directory}'.")
    sys.exit(1) # Exit the script if the directory doesn't exist

# 4. Check if the 'marker_single' command is available
if not shutil.which("marker_single"):
    print("❌ ERROR: The 'marker_single' command was not found.")
    print("Please ensure 'marker-pdf' is installed correctly in your environment's PATH.")
    sys.exit(1)

# Loop through every PDF file in the specified directory
for pdf_path in pdf_directory.glob("*.pdf"):
    print(f"--- Processing file: {pdf_path.name} ---")

    # 5. Let Marker create the <pdf_stem>/ subfolder automatically.
    # Point --output_dir to the *parent* folder so we don't end up with Demo PDF/Demo PDF/.
    output_parent = pdf_path.parent  # e.g., .../Demo/

    # Determine the destination folder Marker will create and a checksum sidecar file
    dest_dir = output_parent / pdf_path.stem
    checksum_file = dest_dir / ".marker_md5"

    # Compute the current md5 of the source PDF
    current_md5 = md5sum(pdf_path)

    # Define the expected main outputs (Marker uses the same stem)
    expected_md = dest_dir / f"{pdf_path.stem}.md"
    expected_json = dest_dir / f"{pdf_path.stem}.json"
    outputs_exist = expected_md.exists() and expected_json.exists()

    # md5 two-mode logic
    if md5_check:
        # Normal: skip if checksum matches and key outputs exist
        if dest_dir.is_dir() and checksum_file.exists() and outputs_exist:
            try:
                saved_md5 = checksum_file.read_text().strip()
            except Exception:
                saved_md5 = ""
            if saved_md5 == current_md5:
                print(f"⏭️  Skipping {pdf_path.name}: up-to-date (md5 match). → {dest_dir}")
                continue
            else:
                print(f"♻️  md5 mismatch → reprocessing {pdf_path.name}")
                print(f"    Cleaning old outputs in: {dest_dir}")
                try:
                    shutil.rmtree(dest_dir)
                except Exception as _e:
                    print(f"    ⚠️  Could not fully clean '{dest_dir}': {_e}")
        else:
            print("ℹ️  No prior checksum or outputs → processing normally.")
    else:
        # Force reprocess regardless of checksum
        print("⚙️  md5_check=False → forcing reprocess (marker + OCR).")
        if dest_dir.exists():
            print(f"    Cleaning existing folder: {dest_dir}")
            try:
                shutil.rmtree(dest_dir)
            except Exception as _e:
                print(f"    ⚠️  Could not fully clean '{dest_dir}': {_e}")

    try:
        # ======================================================================
        # 1. Run the CLI command to generate JSON output (with real-time output)
        # ======================================================================
        print(f"Running CLI command for JSON output on {pdf_path.name}...")
        json_command = [
            "marker_single",
            str(pdf_path),
            "--output_format", "json",
            "--output_dir", str(output_parent)
        ]
        # By removing 'capture_output', the subprocess will stream its output directly to the console in real-time.
        result_json = subprocess.run(json_command, check=True)
        print("✅ JSON file generated successfully by CLI.")


        # ======================================================================
        # 2. Run the CLI command to generate Markdown and Image output (with real-time output)
        # ======================================================================
        print(f"\nRunning CLI command for Markdown and Image output on {pdf_path.name}...")
        md_command = [
            "marker_single",
            str(pdf_path),
            # Default format is markdown, so we don't need to specify it
            "--output_dir", str(output_parent)
        ]
        result_md = subprocess.run(md_command, check=True)
        print("✅ Markdown file and images generated successfully by CLI.")

        print(f"\n✨ Files saved under '{output_parent / pdf_path.stem}'.")
        print("Note: Marker creates a subfolder named after the PDF automatically.")

        # === Post-processing: scan Marker images → filter relevant → save JSONL ===
        print("🔎 Scanning extracted images for relevant charts/plots…")
        img_exts = (".png", ".jpg", ".jpeg")
        img_files = [p for p in dest_dir.rglob("*") if p.suffix.lower() in img_exts]
        if not img_files:
            print("   🖼️  No images found in extracted folder.")
        for img_path in sorted(img_files):
            print(f"   • {img_path.name}")
            any_hit = False
            for ex in EXTRACTORS:
                # Stage 1: quick keyword/title skim
                print(f"      · [{ex.name}] quick gate…", end=" ")
                if not ex.is_relevant(img_path):
                    print("⏭️  Not relevant")
                    continue
                print("✅ ok; strict gate…", end=" ")

                # Stage 2: strict verifier (geometry + numeric band + semantic anchors)
                ok_strict, reason = is_strict_nim_image(img_path)
                if not ok_strict:
                    print(f"⏭️  Failed strict ({reason})")
                    continue

                any_hit = True
                print("✅ Strict OK — extracting…", end=" ")
                ok, msg = ex.handle_image(img_path, dest_dir, pdf_path.name, bypass_relevance=True)
                if ok:
                    print(f"💾 Saved → {msg}")
                else:
                    print(f"⚠️ Skipped ({msg})")
            if not any_hit:
                print("      ⏭️  No matching extractors for this image.")

        # After OCR completes, write/update checksum sidecar
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            checksum_file.write_text(current_md5)
            print(f"🧾 Recorded checksum in: {checksum_file}")
        except Exception as _e:
            print(f"⚠️  Failed to write checksum file at '{checksum_file}': {_e}")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ An error occurred while processing {pdf_path.name}.")
        print(f"Command: '{' '.join(e.cmd)}'")
        print(f"Return Code: {e.returncode}")
        print("Note: Outputs (if any) may be incomplete; checksum not updated.")
    except Exception as e:
        print(f"\nAn unexpected error occurred while processing {pdf_path.name}: {e}")
    
    print(f"--- Finished processing: {pdf_path.name} ---\n")

print("🎉 All PDF files in the directory have been processed.")
