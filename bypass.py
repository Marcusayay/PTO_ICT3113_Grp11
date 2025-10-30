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

def md5_of_folder_inputs(dest_dir: Path) -> str:
    """
    Compute an md5 hash over the extracted inputs in dest_dir.
    Includes only source artifacts likely produced by Marker (md/json/images),
    excludes derived *.jsonl outputs and .marker_md5.
    """
    h = hashlib.md5()
    include_exts = {".md", ".json", ".png", ".jpg", ".jpeg"}
    paths = []
    for p in dest_dir.rglob("*"):
        if p.is_file():
            if p.name == ".marker_md5":
                continue
            if p.suffix.lower() in include_exts:
                paths.append(p)
    # Stable order for deterministic hash
    for p in sorted(paths, key=lambda x: str(x.relative_to(dest_dir))):
        rel = str(p.relative_to(dest_dir)).encode("utf-8")
        try:
            with open(p, "rb") as f:
                data = f.read()
        except Exception:
            data = b""
        h.update(rel + b"\x00" + hashlib.md5(data).hexdigest().encode("utf-8"))
    return h.hexdigest()

# === OCR & extraction helpers (standalone in Untitled-1) ===
NUM_PAT = re.compile(r"^[+-]?\d{1,4}(?:[.,]\d+)?%?$")
# Strictly NIM-only keywords (exclude income/"nii" to avoid earnings slides)
NIM_KEYWORDS = ["net interest margin", "nim"]

QUARTER_PAT = re.compile(r"\b([1-4])Q(\d{2}|\d{4})\b", re.IGNORECASE)
# Simpler decade-only pattern for quarters, e.g., 2Q24, 1Q25
QUARTER_SIMPLE_PAT = re.compile(r"\b([1-4])Q(2\d)\b", re.IGNORECASE)  # e.g., 2Q24, 1Q25

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
    
    # 1. Create the DataFrame
    df = pd.DataFrame(rows)
    
    # 2. Check if it's empty BEFORE sorting
    if df.empty:
        return df  # Return the empty DataFrame immediately

    # 3. If not empty, proceed with sorting and processing
    df = df.sort_values(["cy","cx"]).reset_index(drop=True)

    if "is_pct" not in df.columns: # The 'and not df.empty' check is no longer needed
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
    rdr = easyocr.Reader(['en'], gpu=False, verbose=False)
    results = rdr.readtext(img_rgb, detail=1, paragraph=False)
    out = []
    for quad, text, conf in results:
        (x1,y1),(x2,y2),(x3,y3),(x4,y4) = quad
        out.append({"bbox": (int(x1),int(y1),int(x3),int(y3)), "text": str(text), "conf": float(conf)})
    return out


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
        y0 = int(H * 0.70)  # bottom 30%
        crop = img_bgr[y0:H, 0:W]
        # Enhance: grayscale -> bilateral -> CLAHE -> adaptive threshold
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 8)
        # Upscale for small text
        up = cv2.resize(thr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        img_rgb = cv2.cvtColor(up, cv2.COLOR_GRAY2RGB)
        ocr = run_easyocr(img_rgb)
        # Map bboxes back to global coords: for ordering we only need x
        qtokens = []
        for r in ocr or []:
            txt = str(r.get("text","")).strip()
            m = QUARTER_PAT.search(txt)
            if not m:
                continue
            x1,y1,x2,y2 = r["bbox"]
            # Convert x from upsampled-crop space back to original global space
            cx_local = (x1 + x2) // 2
            cx_global = int(cx_local / 2.0)  # undo scale
            q = f"{m.group(1)}Q{m.group(2)[-2:]}"
            qtokens.append((cx_global, q))
        qtokens.sort(key=lambda x: x[0])
        # Dedup by proximity and text
        ordered = []
        last_x = -9999
        last_q = None
        for x, q in qtokens:
            if last_q == q and abs(x - last_x) < 30:
                continue
            ordered.append(q)
            last_x, last_q = x, q
        return ordered
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

    pct = df[(df.is_pct==True) & (df.cx > mid_x)].copy()
    nums = df[(df.is_pct==False) & (df.cx > mid_x)].copy()

    if pct.empty:
        approx_top = int(H * 0.35)
        cand_pct = df[(df.cx > mid_x) & (df.value.between(1.3, 3.2)) & (df.cy < approx_top)].copy()
        if not cand_pct.empty:
            cand_pct["is_pct"] = True
            pct = cand_pct
            
    # --- Detect quarter labels once (OCR full + bottom crop), reusable below ---
    detected_q_ocr = detect_qlabels(ocr_results or [], W) if ocr_results is not None else []
    detected_q_bot = detect_qlabels_bottom(img_up)
    # Prefer the source with more tokens, but merge to keep any uniques
    if len(detected_q_bot) > len(detected_q_ocr):
        detected_q = _merge_ordered(detected_q_bot, detected_q_ocr)
    else:
        detected_q = _merge_ordered(detected_q_ocr, detected_q_bot)

    nim_df = pd.DataFrame()
    if not pct.empty:
        if pct.shape[0] >= 8:
            labels, centers = kmeans_1d(pct["cy"].values, k=2)
            pct["series"] = labels
            order = np.argsort(centers)
            remap = {order[0]: "Commercial NIM (%)", order[1]: "Group NIM (%)"}
            pct["series_name"] = pct["series"].map(remap)
        else:
            pct["series_name"] = "NIM (%)"

        rows = []
        for name, sub in pct.groupby("series_name"):
            pick = sub.sort_values("cx").tail(5).sort_values("cx").reset_index(drop=True)
            n = len(pick)
            # Choose quarter labels: detected OCR → markdown hint → anchor expansion → placeholders
            labels = []
            if detected_q:
                labels = detected_q[-n:] if len(detected_q) >= n else detected_q
            if (not labels or len(labels) != n) and qlabels_hint:
                labels = qlabels_hint[-n:] if len(qlabels_hint) >= n else qlabels_hint
            if not labels or len(labels) != n:
                anchor = _anchor_quarter_from_texts(ocr_results, qlabels_hint)
                if anchor:
                    labels = _expand_quarters(anchor, n)
            if not labels or len(labels) != n:
                labels = [f"{i+1}Q??" for i in range(n)]

            for i, r in enumerate(pick.itertuples(index=False)):
                rows.append({"Quarter": labels[i], "series": name, "value": r.value})

        if rows:
            nim_df = pd.DataFrame(rows)
            nim_df = nim_df.pivot_table(index="Quarter", columns="series", values="value", aggfunc="first").reset_index()
            # Make column order stable: Quarter first, then sorted series names
            cols = ["Quarter"] + sorted([c for c in nim_df.columns if c != "Quarter"])
            nim_df = nim_df[cols]

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
    """Quick OCR pass to check if an image is relevant by title text against given keywords."""
    try:
        import easyocr
        rdr = easyocr.Reader(['en'], gpu=False, verbose=False)
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        small = cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
        results = rdr.readtext(small, detail=0, paragraph=True)
        text = " ".join([t.lower() for t in results])
        return any(k in text for k in keywords)
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

    def handle_image(self, img_path: Path, dest_dir: Path, pdf_name: str):
        if not self.is_relevant(img_path):
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

# Toggle: if True → normal md5 skip; if False → always reprocess
md5_check = False


process_existing_only = True
# In process_existing_only mode, skip folders that already have a .marker_md5 (resume-friendly)
resume_existing_only_skip_md5 = True


# 3. Define the path to the directory containing your PDF files
pdf_directory = Path("/Users/marcusfoo/Documents/GitHub/PTO_ICT3113_Grp1/All/")

# Check if the directory exists before proceeding
if not pdf_directory.is_dir():
    print(f"❌ ERROR: The directory was not found at '{pdf_directory}'.")
    sys.exit(1) # Exit the script if the directory doesn't exist


# 4. Check if the 'marker_single' command is available (only when running Marker)
if not process_existing_only:
    if not shutil.which("marker_single"):
        print("❌ ERROR: The 'marker_single' command was not found.")
        print("Please ensure 'marker-pdf' is installed correctly in your environment's PATH, or set process_existing_only=True.")
        sys.exit(1)
        
if process_existing_only:
    print("🛠️  process_existing_only=True → Skipping Marker. Scanning existing extracted folders…")
    # Iterate subfolders under pdf_directory (each should be a per-PDF extraction folder)
    for dest_dir in sorted([p for p in pdf_directory.iterdir() if p.is_dir()]):
        pdf_name_guess = f"{dest_dir.name}.pdf"
        print(f"--- Processing extracted folder: {dest_dir} (as {pdf_name_guess}) ---")
        checksum_file = dest_dir / ".marker_md5"

        # Resume mode: if this folder already has a checksum, skip it
        if resume_existing_only_skip_md5 and checksum_file.exists():
            print(f"⏭️  Skipping (already processed): {dest_dir.name} [has .marker_md5]")
            continue

        # Scan images and run all registered extractors
        img_exts = (".png", ".jpg", ".jpeg")
        img_files = [p for p in dest_dir.rglob("*") if p.suffix.lower() in img_exts]
        if not img_files:
            print("   🖼️  No images found in extracted folder.")
        for img_path in sorted(img_files):
            print(f"   • {img_path.name}")
            any_hit = False
            for ex in EXTRACTORS:
                print(f"      · [{ex.name}] relevance check…", end=" ")
                if not ex.is_relevant(img_path):
                    print("⏭️  Not relevant")
                    continue
                any_hit = True
                print("✅ Relevant — extracting…", end=" ")
                ok, msg = ex.handle_image(img_path, dest_dir, pdf_name_guess)
                if ok:
                    print(f"💾 Saved → {msg}")
                else:
                    print(f"⚠️ Skipped ({msg})")
            if not any_hit:
                print("      ⏭️  No matching extractors for this image.")

        # After OCR completes for this folder, write/update checksum sidecar based on folder inputs
        try:
            checksum = md5_of_folder_inputs(dest_dir)
            checksum_file.write_text(checksum)
            print(f"🧾 Recorded folder checksum in: {checksum_file}")
        except Exception as _e:
            print(f"⚠️  Failed to write checksum file at '{checksum_file}': {_e}")

        print(f"--- Finished extracted folder: {dest_dir.name} ---\n")

    print("🎉 All extracted folders in the directory have been processed.")
    sys.exit(0)
    
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
                print(f"      · [{ex.name}] relevance check…", end=" ")
                if not ex.is_relevant(img_path):
                    print("⏭️  Not relevant")
                    continue
                any_hit = True
                print("✅ Relevant — extracting…", end=" ")
                ok, msg = ex.handle_image(img_path, dest_dir, pdf_path.name)
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
