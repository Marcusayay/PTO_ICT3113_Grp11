# --- One cell: run EasyOCR on your chart image (simplified) ---
# Works in a notebook. If you just installed packages, restart the kernel once if imports fail.

# --- (A) Installs ---
# Tip: if SSL cert errors happen on macOS, uncomment the certifi block below first.
# import certifi, os
# os.environ["SSL_CERT_FILE"] = certifi.where()
# os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# %pip -q install easyocr opencv-python-headless pillow matplotlib pandas numpy certifi

# --- (B) Config ---
IMG_PATHS = [
    "/Users/marcusfoo/Documents/GitHub/PTO_ICT3113_Grp1/All/1Q24_CFO_presentation/_page_4_Figure_1.jpeg",
    "/Users/marcusfoo/Documents/GitHub/PTO_ICT3113_Grp1/All/2Q24_CFO_presentation/_page_5_Figure_1.jpeg",
    "/Users/marcusfoo/Documents/GitHub/PTO_ICT3113_Grp1/All/3Q24_CFO_presentation/_page_7_Figure_1.jpeg",
    "/Users/marcusfoo/Documents/GitHub/PTO_ICT3113_Grp1/All/4Q24_CFO_presentation/_page_5_Figure_1.jpeg",
    "/Users/marcusfoo/Documents/GitHub/PTO_ICT3113_Grp1/All/1Q25_CFO_presentation/_page_4_Figure_1.jpeg",
    "/Users/marcusfoo/Documents/GitHub/PTO_ICT3113_Grp1/All/2Q25_CFO_presentation/_page_5_Figure_1.jpeg",
]

# --- (C) Imports & utils ---
import re, math, numpy as np, pandas as pd
from pathlib import Path
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display as _show

NUM_PAT = re.compile(r"^[+-]?\d{1,4}(?:[.,]\d+)?%?$")
# This is the FIX
QUARTER_PAT = re.compile(r"([1-4Iil|])\s*[QO0]\s*([0-9O]{2,4})", re.I)

_CHAR_FIX = str.maketrans({
    "O":"0","o":"0",
    "S":"5","s":"5",
    "I":"1","l":"1","|":"1","!":"1",
    "D":"0",
    "B":"3", "8":"3",  # Force B and 8 to become 3
    "Z":"2", "z":"2"   # Force Z to become 2
})

def normalize_token(t: str) -> str:
    t = (t or "").strip()
    return t.translate(_CHAR_FIX).replace(" ", "")

def load_image(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    im = cv2.imread(str(p))
    if im is None:
        raise RuntimeError("cv2.imread() returned None")
    return im

def preprocess(img_bgr):
    # Upscale + denoise + local contrast + threshold
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
    if pct: s = s[:-1]
    try:
        return float(s), pct
    except:
        return None, pct

def extract_numbers(ocr_results):
    rows = []
    for r in ocr_results or []:
        txt = str(r["text"]).strip()
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

def _is_half_token(t: str) -> bool:
    t = (t or "").lower().replace(" ", "")
    return ("9m" in t) or ("1h" in t) or ("h1" in t) or ("h2" in t) or ("2h" in t)

def overlay(img_bgr, df, title="Detections"):
    vis = img_bgr.copy()
    for _, r in (df if isinstance(df, pd.DataFrame) else pd.DataFrame()).iterrows():
        x1,y1,x2,y2 = int(r.x1),int(r.y1),int(r.x2),int(r.y2)
        color = (255,0,0) if bool(r.get("is_pct", False)) else (0,200,0)
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
        cv2.putText(vis, str(r.raw), (x1, max(12,y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
    plt.figure(figsize=(12,6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off"); plt.title(title); plt.show()

def kmeans_1d(values, k=2, iters=20):
    values = np.asarray(values, dtype=float).reshape(-1,1)
    centers = np.array([values.min(), values.max()]).reshape(k,1)
    for _ in range(iters):
        d = ((values - centers.T)**2)
        labels = d.argmin(axis=1)
        new_centers = np.array([values[labels==i].mean() if np.any(labels==i) else centers[i] for i in range(k)]).reshape(k,1)
        if np.allclose(new_centers, centers, atol=1e-3): break
        centers = new_centers
    return labels, centers.flatten()

# --- (D) Backend wrappers ---
_EASY_OCR_READER = None
def run_easyocr(img_rgb):
    import easyocr
    global _EASY_OCR_READER
    if _EASY_OCR_READER is None:
        _EASY_OCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    results = _EASY_OCR_READER.readtext(img_rgb, detail=1, paragraph=False)
    out = []
    for quad, text, conf in results:
        (x1,y1),(x2,y2),(x3,y3),(x4,y4) = quad
        out.append({"bbox": (int(x1),int(y1),int(x3),int(y3)), "text": str(text), "conf": float(conf)})
    return out

def detect_quarters_easyocr(img_bgr):
    """
    Use EasyOCR to read quarter labels along the bottom axis.
    Returns a list of (x_global, 'nQyy') sorted left→right, with half-year tokens removed.
    """
    H, W = img_bgr.shape[:2]
    y0 = int(H * 0.66)  # bottom 34% to include labels (changed from 0.60)
    crop = img_bgr[y0:H, 0:W]
    # preprocess: gray -> bilateral -> CLAHE -> adaptive thr -> morph close -> upsample
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 8)
    # kernel = np.ones((3,3), np.uint8)
    # thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    up = cv2.resize(thr, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)  # changed from 2.5
    img_rgb = cv2.cvtColor(up, cv2.COLOR_GRAY2RGB)
    ocr = run_easyocr(img_rgb)
    print("\n[DEBUG] Raw tokens from bottom axis:")
    for r in ocr or []:
        print(f"  - Raw: '{r.get('text','')}'  ->  Normalized: '{normalize_token(r.get('text',''))}'")
    # PASS 1 — direct regex on normalized tokens
    tokens = []
    for r in ocr or []:
        raw = str(r.get("text","")).strip()
        x1,y1,x2,y2 = r["bbox"]
        cx_local = (x1 + x2) // 2
        cx_global = int(cx_local / 3.0)  # undo scaling
        tokens.append({"x": cx_global, "raw": raw, "norm": normalize_token(raw)})

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
        # classify simple pieces
        pieces = sorted(tokens, key=lambda d: d["x"])
        digits_1to4 = [p for p in pieces if p["norm"] in ("1","2","3","4")]
        q_only      = [p for p in pieces if p["norm"].upper() == "Q"]
        q_with_year = [p for p in pieces if re.fullmatch(r"Q[0-9O]{2,4}", p["norm"], flags=re.I)]
        years_2d    = [p for p in pieces if re.fullmatch(r"[0-9O]{2,4}", p["norm"])]

        used = set()
        def near(a, b, tol=70):  # pixels in global coords
            return abs(a["x"] - b["x"]) <= tol

        # patterns: [digit][Q][year] or [digit][Qyy] or [digitQ][yy] or [digit][Qyy] (with OCR confusions)
        for d in digits_1to4:
            # digit + Qyy
            candidates = [q for q in q_with_year if near(d, q)]
            if candidates:
                qtok = min(candidates, key=lambda q: abs(q["x"]-d["x"]))
                qyy = normalize_token(qtok["norm"])[1:]
                quarters.append(( (d["x"]+qtok["x"])//2, f"{d['norm']}Q{qyy[-2:]}" ))
                used.add(id(d)); used.add(id(qtok)); continue
            # digit + Q + yy
            qs = [q for q in q_only if near(d, q)]
            ys = [y for y in years_2d if near(d, y, tol=120)]
            if qs and ys:
                qtok = min(qs, key=lambda q: abs(q["x"]-d["x"]))
                ytok = min(ys, key=lambda y: abs(y["x"]-qtok["x"]))
                yy = normalize_token(ytok["norm"])
                quarters.append(( (d["x"]+ytok["x"])//2, f"{d['norm']}Q{yy[-2:]}" ))
                used.add(id(d)); used.add(id(qtok)); used.add(id(ytok)); continue

    # PASS 3 — clean and dedupe results
    if not quarters:
        return []
    quarters.sort(key=lambda t: t[0])
    deduped = []
    last_x = -10**9
    for x,q in quarters:
        if abs(x - last_x) <= 22:
            continue
        deduped.append((x,q))
        last_x = x
    quarters = deduped

    return quarters

# --- (E) Extraction logic specific to your slide layout ---
# Removed entire extract_series_from_df function as per instructions

# --- (F) Run EasyOCR only ---
for path in IMG_PATHS:
    print("\n" + "="*88)
    print(f"Image: {path}")
    try:
        img_bgr = load_image(path)
        q_xy = detect_quarters_easyocr(img_bgr)
        if q_xy:
            quarters = [q for _, q in q_xy]
            xpos     = [int(x) for x, _ in q_xy]
            print("Detected quarters (EasyOCR, bottom axis):", ", ".join(quarters))
            print("x-pos:", xpos)
        else:
            print("Detected quarters (EasyOCR, bottom axis): <none>")
    except Exception as e:
        print(f"ERROR → {e}")
