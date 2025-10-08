"""
Stage3.py — Benchmark Runner (Stage 3)

Runs the 3 standardized queries, times them, saves JSON, and prints prose answers with citations.

Artifacts written to OUT_DIR (default: data/):
  - bench_results.json      # structured results
  - bench_report.md         # human-readable answers with citations
"""
from __future__ import annotations
import os, json, time
from typing import List, Dict, Any

import pandas as pd

# Import Stage 2 API
from Stage2 import init_stage2, answer_with_llm, agentic_answer

OUT_DIR = os.environ.get("AGENT_CFO_OUT_DIR", "data")

# --- Standardized queries (exact spec) ---
QUERIES: List[str] = [
    # 1) NIM trend over last 5 quarters
    "Report the Net Interest Margin (NIM) over the last 5 quarters, with values, and add 1–2 lines of explanation.",
    # 2) Opex YoY with top 3 drivers
    "Show Operating Expenses (Opex) for the last 3 fiscal years, year-on-year comparison, and summarize the top 3 Opex drivers from the MD&A.",
    # 3) CTI ratio for last 3 years with working & implications
    "Calculate the Cost-to-Income Ratio (CTI) for the last 3 fiscal years; show your working and give 1–2 lines of implications.",
]


def _format_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for h in hits:
        out.append({
            "file": h.get("file"),
            "year": h.get("year"),
            "quarter": h.get("quarter"),
            "page": h.get("page"),
            "section_hint": h.get("section_hint"),
        })
    return out


def run_benchmark(top_k_retrieval: int = 12, top_ctx: int = 3, out_dir: str = OUT_DIR, print_prose: bool = False) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    init_stage2(out_dir)

    rows = []
    results: List[Dict[str, Any]] = []

    for q in QUERIES:
        t0 = time.perf_counter()
        try:
            out = agentic_answer(q, top_k_retrieval=top_k_retrieval, top_ctx=top_ctx)
        except Exception:
            out = answer_with_llm(q, top_k_retrieval=top_k_retrieval, top_ctx=top_ctx)
        lat_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        if print_prose:
            print(f"\n=== Question ===\n{q}")
            print("\n--- Answer ---\n")
            print(out["answer"].strip())
            if out.get("hits"):
                print("\n--- Citations (top ctx) ---")
                for h in _format_hits(out.get("hits", [])):
                    y = f" {h['year']}" if h.get('year') is not None else ""
                    qtr = f" {h['quarter']}Q{str(h['year'])[2:]}" if h.get('quarter') else ""
                    sec = f" — {h['section_hint']}" if h.get('section_hint') else ""
                    print(f"- {h['file']}{y}{qtr} — p.{h['page']}{sec}")
            print(f"\n(latency: {lat_ms} ms)")

        results.append({
            "query": q,
            "answer": out["answer"],
            "hits": _format_hits(out.get("hits", [])),
            "latency_ms": lat_ms,
        })
        rows.append({"Query": q, "Latency_ms": lat_ms})

    # Save JSON
    json_path = os.path.join(out_dir, "bench_results.json")
    with open(json_path, "w") as f:
        json.dump({"results": results}, f, indent=2)

    # Save simple markdown report
    md_lines = ["# Agent CFO — Benchmark Report\n"]
    for i, r in enumerate(results, start=1):
        md_lines.append(f"\n## Q{i}. {r['query']}")
        md_lines.append("\n**Answer**\n\n" + r["answer"].strip())
        if r.get("hits"):
            md_lines.append("\n**Citations (top ctx)**")
            for h in r["hits"]:
                y = f" {h['year']}" if h.get('year') is not None else ""
                qtr = f" {h['quarter']}Q{str(h['year'])[2:]}" if h.get('quarter') else ""
                sec = f" — {h['section_hint']}" if h.get('section_hint') else ""
                md_lines.append(f"- {h['file']}{y}{qtr} — p.{h['page']}{sec}")
    md_path = os.path.join(out_dir, "bench_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    df = pd.DataFrame(rows)
    if print_prose and not df.empty:
        p50 = float(df['Latency_ms'].quantile(0.5))
        p95 = float(df['Latency_ms'].quantile(0.95))
        print(f"\n=== Benchmark Summary ===\nSaved JSON: {json_path}\nSaved report: {md_path}\nLatency p50: {p50:.1f} ms, p95: {p95:.1f} ms")

    # Return a compact summary (and a DataFrame for notebook display if desired)
    return {"json_path": json_path, "md_path": md_path, "summary": df}


if __name__ == "__main__":
    run_benchmark(print_prose=True)