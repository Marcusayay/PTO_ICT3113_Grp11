import os
import numpy as np
import pandas as pd
import g2

# --- Configuration ---
# Ensure this points to the directory where your KB artifacts are saved
OUT_DIR = "data"

# --- Queries for Sanity Check ---
QUERIES = [
    "Net Interest Margin for the last 5 quarters",
    "Operating Expenses for the last 3 fiscal years",
    "Total Operating Income for the last 3 fiscal years"
]

def run_sanity_check():
    """
    Loads the knowledge base and runs predefined queries to verify data retrieval.
    """
    print("--- Starting Knowledge Base Sanity Check ---")

    # 1. --- Load Knowledge Base Artifacts ---
    kb_path = os.path.join(OUT_DIR, "kb_chunks.parquet")
    text_path = os.path.join(OUT_DIR, "kb_texts.npy")
    
    if not os.path.exists(kb_path) or not os.path.exists(text_path):
        print(f"❌ ERROR: Knowledge base files not found in '{OUT_DIR}'.")
        print("Please run the `build_kb()` function from g1.py first.")
        return

    print(f"✅ Successfully loaded KB artifacts from '{OUT_DIR}'.")
    kb_df = pd.read_parquet(kb_path)
    texts = np.load(text_path, allow_pickle=True)

    # 2. --- Initialize Stage2 KB (loads index, bm25, embedder)
    g2.init_stage2(out_dir=OUT_DIR)

    # 3. --- Run Queries ---
    for query in QUERIES:
        print("\n" + "="*80)
        print(f"🔍 EXECUTING QUERY: \"{query}\"")
        print("="*80)

        hits = g2.hybrid_search(query, top_k=3)
        for i, hit in enumerate(hits, start=1):
            # Map doc_id to index position in texts
            mask = (kb_df["doc_id"] == hit["doc_id"]).to_numpy()
            idxs = np.flatnonzero(mask)
            if idxs.size == 0:
                continue
            pos = int(idxs[0])

            score = hit["score"]
            metadata = kb_df.iloc[pos]
            retrieved_text = texts[pos]

            print(f"\n--- Result {i} (Score: {score:.4f}) ---")
            print(f"  📂 Source: {metadata['file']}, Page: {metadata['page']}")
            print(f"  🗓️ Year: {metadata['year']}, Quarter: {metadata['quarter']}")
            print(f"  📝 Section Hint: {metadata['section_hint']}")
            print("\n  📋 Retrieved Text Snippet:")
            print('-' * 30)
            print(str(retrieved_text).strip())
            print('-' * 30)
            
if __name__ == "__main__":
    run_sanity_check()