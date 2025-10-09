# g1.py (Final, Fully Automated & Integrated Version)
from __future__ import annotations
import os
import re
import json
import uuid
import pathlib
from typing import List, Dict, Any, Optional, Tuple
import time

# Main Libraries
import pandas as pd
import numpy as np
import camelot
import fitz  # PyMuPDF
from PIL import Image
import io

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

pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# This dictionary defines all the key pages we want to find automatically
SEARCH_TERMS = {
    "Expenses Chart": ["(E) Expenses", "Excludes one-time items", "Cost / income (%)"],  
    "Five-Year Summary": ["Financial statements", "DBS Group Holdings and its Subsidiaries"],
    "NIM Chart": ["Net interest margin (%)", "Group", "Commercial book"]
}

# --- 2. Helper Functions ---
def infer_period_from_filename(fname: str) -> Tuple[Optional[int], Optional[int]]:
    base = fname.upper()
    m = re.search(r"([1-4])Q(\d{2})", base, re.I); 
    if m: q, yy = int(m.group(1)), int(m.group(2)); return (2000 + yy if yy < 100 else yy, q)
    m = re.search(r"\b(20\d{2})\b", base); 
    if m: return (int(m.group(1)), None)
    return (None, None)

def find_key_pages(pdf_path: str) -> Dict[str, List[int]]:
    found_pages = {}
    print(f"      Scouting for key pages in '{os.path.basename(pdf_path)}'...")
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            for description, keywords in SEARCH_TERMS.items():
                if all(keyword in text for keyword in keywords):
                    found_pages.setdefault(description, []).append(page_num)
        doc.close()
    except Exception as e:
        print(f"      ⚠️  Could not scout pages in {os.path.basename(pdf_path)}: {e}")
    return found_pages

# def format_vision_json_to_text(data: dict) -> str:
#     facts = []
#     # NEW: More descriptive formatting for Expenses
#     if "expenses_analysis" in data:
#         analysis = data["expenses_analysis"]
#         if "yearly_total_expenses" in analysis:
#             for year, value in analysis["yearly_total_expenses"].items():
#                 # This new phrasing is much more unique and easier to find
#                 facts.append(f"From the Expenses Chart for FY{year}, the value for yearly total operating expenses (Opex) was {value} million.")

#     # NEW: More descriptive formatting for the Five-Year Summary
#     if "five_year_summary" in data:
#         summary = data["five_year_summary"]
#         for metric, year_data in summary.items():
#             for year, value in year_data.items():
#                 # Adding "From the five-year summary table" makes this distinct
#                 facts.append(f"From the five-year summary table for FY{year}, the value for '{metric}' was {value}.")
    
#     # NEW: More descriptive formatting for NIM
#     if "nim_analysis" in data:
#         analysis = data["nim_analysis"]
#         for quarter, values in analysis.items():
#             # Adding "From the NIM Chart" makes this distinct
#             if "group_nim" in values:
#                 facts.append(f"From the NIM Chart for {quarter}, the Group Net Interest Margin was {values['group_nim']}%.")
#             if "commercial_nim" in values:
#                 facts.append(f"From the NIM Chart for {quarter}, the Commercial Book Net Interest Margin was {values['commercial_nim']}%.")
    
#     return "\n".join(facts)


def format_vision_json_to_text(data: dict) -> str:
    facts = []
    # Definitive version with unique keywords to guide the retriever
    if "expenses_analysis" in data:
        analysis = data["expenses_analysis"]
        if "yearly_total_expenses" in analysis:
            for year, value in analysis["yearly_total_expenses"].items():
                facts.append(f"Source: Expenses Chart (expenses_analysis). For FY{year}, yearly_total_expenses (Opex) was {value} million.")

    if "five_year_summary" in data:
        summary = data["five_year_summary"]
        for metric, year_data in summary.items():
            for year, value in year_data.items():
                facts.append(f"Source: Five-Year Summary Table. Metric '{metric}' for FY{year} is {value}.")
    
    if "nim_analysis" in data:
        analysis = data["nim_analysis"]
        for quarter, values in analysis.items():
            if "group_nim" in values:
                facts.append(f"Source: NIM Chart (nim_analysis). For {quarter}, the Group Net Interest Margin was {values['group_nim']}%.")
            if "commercial_nim" in values:
                facts.append(f"Source: NIM Chart (nim_analysis). For {quarter}, the Commercial Book Net Interest Margin was {values['commercial_nim']}%.")
    
    return "\n".join(facts)


# --- 3. Main Processing Functions ---

def process_pdf(path: str, fname: str, year: Optional[int], quarter: Optional[int], vision_model: genai.GenerativeModel, key_pages: Dict[str, List[int]]) -> List[Tuple[Dict, str]]:
    chunks = []
    all_key_page_numbers = [p for pages in key_pages.values() for p in pages]
    
    vision_prompt = """
    Analyze the attached image, which is a full page from a financial report.
    Your task is to identify and extract data from ONE of three possible content types: an "Expenses" chart, a "Five-year summary" table, or a "Net Interest Margin" chart. Follow the instructions for the one you find.

    1. If you find an "Expenses" Chart: Extract yearly and quarterly total expenses. Return it in a JSON object under a main key "expenses_analysis".
    2. If you find a "Five-year summary" Table: Extract the values for "Total income", "Net profit", "Cost-to-income ratio (%)", and "Net interest margin (%)" for all years. Return this data in a JSON object under the key "five_year_summary".
    3. If you find a "Net Interest Margin (%)" Chart: Extract "Group" and "Commercial book" NIM values for all quarters. Return this data in a JSON object under the key "nim_analysis".

    If none of these specific items are on the page, return an empty JSON object {}.
    """

    doc = fitz.open(path)
    for page_num, page_fitz in enumerate(doc, start=1):
        row_template = {"doc_id": None, "file": fname, "page": page_num, "year": year, "quarter": quarter}

        if page_num in all_key_page_numbers:
            # --- This is a key page, use the powerful Gemini Vision model ---
            print(f"      -> Processing key page {page_num} with Vision...")
            cache_filename = f"{fname.replace('.pdf', '')}__page_{page_num}.json"
            cache_filepath = os.path.join(CACHE_DIR, cache_filename)
            
            parsed_json = None
            if os.path.exists(cache_filepath):
                with open(cache_filepath, 'r') as f: parsed_json = json.load(f)
                print(f"          CACHE HIT: Loaded vision data for page {page_num}.")
            else:
                print(f"          CACHE MISS: Calling Vision API for page {page_num}...")
                try:
                    pix = page_fitz.get_pixmap(dpi=200)
                    image = Image.open(io.BytesIO(pix.tobytes("png")))
                    response = vision_model.generate_content([vision_prompt, image])
                    time.sleep(6)
                    
                    json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                    if json_match:
                        parsed_json = json.loads(json_match.group(0))
                        with open(cache_filepath, 'w') as f: json.dump(parsed_json, f, indent=2)
                        print(f"          ✅ Successfully cached vision data for page {page_num}.")
                    else:
                        print(f"          ⚠️  Vision model did not return valid JSON for page {page_num}.")
                except Exception as e:
                    print(f"          ⚠️  Vision API call failed for page {page_num}: {e}")

            if parsed_json:
                vision_text = format_vision_json_to_text(parsed_json)
                if vision_text:
                    row = row_template.copy()
                    row["doc_id"] = str(uuid.uuid4())
                    row["section_hint"] = f"vision_summary_p{page_num}"
                    chunks.append((row, vision_text))
        else:
            # --- For all other "normal" pages, use the fast local extractors ---
            plain_text = page_fitz.get_text("text")
            if plain_text and plain_text.strip():
                row = row_template.copy()
                row["doc_id"] = str(uuid.uuid4())
                row["section_hint"] = "prose"
                chunks.append((row, plain_text))
            
            try:
                tables = camelot.read_pdf(path, pages=str(page_num), flavor='lattice', suppress_stdout=True)
                for i, table in enumerate(tables):
                    table_md = table.df.to_markdown(index=False)
                    if table_md:
                        row = row_template.copy()
                        row["doc_id"] = str(uuid.uuid4())
                        row["section_hint"] = f"table_p{page_num}_{i+1}"
                        chunks.append((row, table_md))
            except Exception:
                pass
            
    doc.close()
    return chunks

def build_kb():
    # --- Gemini Setup ---
    if 'GEMINI_API_KEY' not in os.environ: raise SystemExit("❌ ERROR: GEMINI_API_KEY not set.")
    try:
        genai.configure(api_key=os.environ['GEMINI_API_KEY'])
        vision_model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        raise SystemExit(f"❌ ERROR: Could not configure Gemini. Details: {e}")

    # --- Find all documents ---
    all_docs = sorted([str(p) for p in pathlib.Path(DATA_DIR).rglob("*") if p.is_file()])
    pdf_docs = [p for p in all_docs if p.lower().endswith(".pdf")]
    
    # --- SCOUTING PASS ---
    print("[Stage1] Starting Scouting Pass to find key pages...")
    all_key_pages = {}
    for path in pdf_docs:
        fname = os.path.basename(path)
        all_key_pages[fname] = find_key_pages(path)
    print("[Stage1] Scouting complete. Starting main ingestion process...")

    # --- EXTRACTION PASS ---
    all_rows, all_texts = [], []
    for path in all_docs:
        fname = os.path.basename(path)
        print(f"\n[Stage1] Processing: {fname}")
        year, quarter = infer_period_from_filename(fname)
        
        doc_chunks = []
        if path.lower().endswith(".pdf"):
            key_pages_for_file = all_key_pages.get(fname, {})
            doc_chunks = process_pdf(path, fname, year, quarter, vision_model, key_pages_for_file)
        # Add logic for tabular files (Excel/CSV) if needed
        # elif path.lower().endswith(('.xls', '.xlsx', '.csv')):
        #     doc_chunks = process_tabular(...) 

        if doc_chunks:
            print(f"      → Extracted {len(doc_chunks)} chunks from {fname}.")
            for row, text in doc_chunks: all_rows.append(row); all_texts.append(text)
        else:
            print(f"      ⚠️ WARNING: No content extracted from {fname}.")

    if not all_texts: raise SystemExit("No data was indexed.")
    
    # --- FINALIZATION PASS (Embedding, Indexing, Saving) ---
    print(f"\n[Stage1] Total chunks to be indexed: {len(all_texts)}")
    kb = pd.DataFrame(all_rows)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    provider = SentenceTransformer(model_name)
    print(f"[Stage1] Embedding {len(all_texts)} chunks...")
    vecs = provider.encode(all_texts, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    dim = provider.get_sentence_embedding_dimension()

    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    
    kb_path = os.path.join(OUT_DIR, "kb_chunks.parquet"); text_path = os.path.join(OUT_DIR, "kb_texts.npy")
    index_path = os.path.join(OUT_DIR, "kb_index.faiss"); meta_path = os.path.join(OUT_DIR, "kb_meta.json")

    kb.to_parquet(kb_path, engine='pyarrow', index=False)
    np.save(text_path, np.array(all_texts, dtype=object))
    faiss.write_index(index, index_path)
    with open(meta_path, "w") as f: json.dump({"embedding_provider": f"st:{model_name}", "dim": dim}, f)
    
    print(f"\n[Stage1] Successfully saved all artifacts to '{OUT_DIR}'")

if __name__ == "__main__":
    build_kb()