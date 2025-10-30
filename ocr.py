
# --- One cell: test 3 programmatic backends on your chart image ---
# Works in a notebook. If you just installed packages, restart the kernel once if imports fail.

# --- (A) Installs ---
# Tip: if SSL cert errors happen on macOS, uncomment the certifi block below first.
# import certifi, os
# os.environ["SSL_CERT_FILE"] = certifi.where()
# os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# %pip -q install easyocr opencv-python-headless pytesseract pillow matplotlib pandas numpy paddleocr==2.7.0.3 certifi

# --- (B) Config ---
IMG_PATH = "/Users/marcusfoo/Documents/GitHub/PTO_ICT3113_Grp1/All/2Q25_CFO_presentation/_page_5_Figure_1.jpeg"
BACKENDS = ["easyocr", "tesseract", "paddleocr"]  # choose any subset of ["easyocr","tesseract","paddleocr"]

# --- (C) Imports & utils ---
import re, math, numpy as np, pandas as pd
from pathlib import Path
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display as _show

NUM_PAT = re.compile(r"^[+-]?\d{1,4}(?:[.,]\d+)?%?$")

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
def run_easyocr(img_rgb):
    import easyocr
    rdr = easyocr.Reader(['en'], gpu=False, verbose=False)
    results = rdr.readtext(img_rgb, detail=1, paragraph=False)
    out = []
    for quad, text, conf in results:
        (x1,y1),(x2,y2),(x3,y3),(x4,y4) = quad
        out.append({"bbox": (int(x1),int(y1),int(x3),int(y3)), "text": str(text), "conf": float(conf)})
    return out

def run_tesseract(img_rgb):
    import pytesseract
    _ = pytesseract.get_tesseract_version()  # raises if not installed
    cfg = r'--oem 3 --psm 6 -l eng -c tessedit_char_whitelist=0123456789.%,-'
    data = pytesseract.image_to_data(img_rgb, output_type=pytesseract.Output.DATAFRAME, config=cfg)
    data = data.dropna(subset=["text"])
    out = []
    for _, r in data.iterrows():
        x,y,w,h = int(r["left"]), int(r["top"]), int(r["width"]), int(r["height"])
        txt = str(r["text"]).strip()
        if not txt: 
            continue
        conf = float(r.get("conf", -1))
        out.append({"bbox": (x,y,x+w,y+h), "text": txt, "conf": conf})
    return out

def run_paddleocr(img_path):
    # PaddleOCR needs path (not array) for its high-level API
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    result = ocr.ocr(img_path, cls=True)
    out = []
    for block in result:
        for line in block:
            (x1,y1),(x2,y2),(x3,y3),(x4,y4) = line[0]
            text, conf = line[1][0], float(line[1][1])
            out.append({"bbox": (int(x1),int(y1),int(x3),int(y3)), "text": str(text), "conf": conf})
    return out

# --- (E) Extraction logic specific to your slide layout ---
def extract_series_from_df(df, img_up, backend_name):
    H, W = img_up.shape[:2]
    mid_x = W//2
    # Heuristic bands for right panel
    top_band_min = int(H * 0.38)
    top_band_max = int(H * 0.58)

    pct = df[(df.is_pct==True) & (df.cx > mid_x)].copy()
    nums = df[(df.is_pct==False) & (df.cx > mid_x)].copy()

    # Fallback to detect decimal NIM labels even when '%' is missed
    if pct.empty:
        approx_top = int(H * 0.35)
        cand_pct = df[(df.cx > mid_x) & (df.value.between(1.3, 3.2)) & (df.cy < approx_top)].copy()
        if not cand_pct.empty:
            cand_pct["is_pct"] = True
            pct = cand_pct

    # Split two NIM lines (Commercial vs Group) by vertical clustering (y)
    nim_df = pd.DataFrame()
    if not pct.empty:
        if pct.shape[0] >= 8:
            labels, centers = kmeans_1d(pct["cy"].values, k=2)
            pct["series"] = labels
            order = np.argsort(centers)
            remap = {order[0]:"Commercial NIM (%)", order[1]:"Group NIM (%)"}
            pct["series_name"] = pct["series"].map(remap)
        else:
            pct["series_name"] = "NIM (%)"

        qlabels = ["2Q24","3Q24","4Q24","1Q25","2Q25"]
        rows = []
        for name, sub in pct.groupby("series_name"):
            pick = sub.sort_values("cx").tail(5).sort_values("cx")
            for i, r in enumerate(pick.itertuples(index=False)):
                if i < len(qlabels):
                    rows.append({"Quarter": qlabels[i], "series": name, "value": r.value})
        if rows:
            nim_table = pd.DataFrame(rows)
            nim_df = nim_table.pivot(index="Quarter", columns="series", values="value").reset_index()

    # Net interest income bars (top-right values above beige bars)
    nii_df = pd.DataFrame()
    if not nums.empty:
        band = nums[(nums.value > 500) & (nums.value < 20000) & (nums.cy.between(top_band_min, top_band_max))]
        if band.shape[0] < 5:
            band = nums[(nums.value > 500) & (nums.value < 20000)]
        if not band.empty:
            pick = band.sort_values("cx").tail(5).sort_values("cx").reset_index(drop=True)
            nii_df = pd.DataFrame({
                "Quarter": ["2Q24","3Q24","4Q24","1Q25","2Q25"][:len(pick)],
                "Net interest income ($m)": pick["value"].tolist()
            })

    # Sort quarters chronologically
    def _sort_q(df_in):
        if df_in is None or df_in.empty or "Quarter" not in df_in.columns: return df_in
        order = pd.Categorical(df_in["Quarter"], ["2Q24","3Q24","4Q24","1Q25","2Q25"], ordered=True)
        return df_in.assign(Quarter=order).sort_values("Quarter").reset_index(drop=True)

    nim_df = _sort_q(nim_df)
    nii_df = _sort_q(nii_df)

    # Show per-backend results
    print(f"\n=== Backend: {backend_name} ===")
    if not nim_df.empty:
        print("Extracted NIM (Commercial vs Group):")
        _show(nim_df)
    else:
        print("No NIM table detected.")

    if not nii_df.empty:
        print("Extracted Net interest income ($m):")
        _show(nii_df)
    else:
        print("No Net interest income table detected.")

    # Overlay
    overlay(img_up, df, title=f"{backend_name}: detected tokens — red≈percent  green=number")

    return nim_df, nii_df

# --- (F) Run all backends ---
img_bgr = load_image(IMG_PATH)
img_up, gray, thr, scale = preprocess(img_bgr)
img_rgb = cv2.cvtColor(thr, cv2.COLOR_GRAY2RGB)

all_results = {}
for b in BACKENDS:
    try:
        if b == "easyocr":
            ocr = run_easyocr(img_rgb)
        elif b == "tesseract":
            ocr = run_tesseract(img_rgb)
        elif b == "paddleocr":
            ocr = run_paddleocr(IMG_PATH)  # takes file path
        else:
            print(f"Unknown backend: {b}"); continue

        df = extract_numbers(ocr)
        if df.empty:
            print(f"\n=== Backend: {b} ===\nNo numeric tokens detected.")
            continue

        nim_df, nii_df = extract_series_from_df(df, img_up, b)
        all_results[b] = {"nim": nim_df, "nii": nii_df}
    except Exception as e:
        print(f"\n=== Backend: {b} ===\nERROR → {e}")

# Optional: quick comparison of who found what
summary = []
for b, r in all_results.items():
    summary.append({
        "backend": b,
        "has_NIM": (r["nim"] is not None and not r["nim"].empty),
        "has_NII": (r["nii"] is not None and not r["nii"].empty),
        "NIM_cols": [] if r["nim"] is None or r["nim"].empty else [c for c in r["nim"].columns if c!="Quarter"]
    })
print("\n=== Backend coverage summary ===")
print(pd.DataFrame(summary))
