from __future__ import annotations

"""
Stage3.py — Benchmark Runner (Stage 3)

Runs the 3 standardized queries for both the baseline and agentic pipelines,
times them, saves JSON/Markdown reports, and prints prose answers with citations.

Artifacts written to OUT_DIR (default: data/):
  - bench_results_baseline.json / bench_results_agent.json
  - bench_report_baseline.md / bench_report_agent.md
"""
import os, json, time, inspect
from typing import List, Dict, Any

import pandas as pd

# Explicitly import Stage-2 entrypoints so we don't rely on globals
from g2 import init_stage2, answer_with_llm_baseline as answer_with_llm, answer_with_agent

OUT_DIR = os.environ.get("AGENT_CFO_OUT_DIR", "data")

# --- Standardized queries (exact spec) ---
QUERIES: List[str] = [
    # 1) NIM trend over last 5 quarters
    "Report the Gross Margin (or Net Interest Margin, if a bank) over the last 5 quarters, with values.",
    # 2) Opex YoY table only (absolute & % change)
    "Show Operating Expenses for the last 3 fiscal years, year-on-year comparison.",
    # 3) Operating Efficiency Ratio (Opex ÷ Operating Income) with working
    "Calculate the Operating Efficiency Ratio (Opex ÷ Operating Income) for the last 3 fiscal years, showing the working."
]


# --- Helper functions for answer call and output normalization ---
def _call_answer(func, query: str, dry_run: bool):
    """Call answer function with optional dry_run if supported."""
    try:
        params = inspect.signature(func).parameters
    except Exception:
        params = {}
    kwargs = {}
    if 'dry_run' in params:
        kwargs['dry_run'] = dry_run
    return func(query, **kwargs)

def _normalize_out(res) -> Dict[str, Any]:
    """Coerce answer result to a dict with keys: answer, hits, execution_log."""
    if isinstance(res, str):
        return {"answer": res, "hits": [], "execution_log": None}
    if isinstance(res, dict):
        ans = res.get("answer") or res.get("Answer") or str(res)
        hits = res.get("hits") or res.get("Hits") or []
        log  = res.get("execution_log") or res.get("ExecutionLog")
        return {"answer": ans, "hits": hits, "execution_log": log}
    return {"answer": str(res), "hits": [], "execution_log": None}


def _format_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Helper to format citation hits for JSON output."""
    out = []
    if not hits: return out
    for h in hits:
        out.append({
            "file": h.get("file"),
            "year": h.get("year"),
            "quarter": h.get("quarter"),
            "page": h.get("page"),
            "section_hint": h.get("section_hint"),
        })
    return out



def run_benchmark(
    print_prose: bool = True,
    use_agent: bool = False,
    out_dir: str = OUT_DIR,
    dry_run: bool = False  # <-- NEW TOGGLE
) -> Dict[str, Any]:
    """
    Runs the benchmark for either the baseline RAG or the agentic pipeline.
    
    Args:
        print_prose: Whether to print results to the console.
        use_agent: If True, uses answer_with_agent. If False, uses answer_with_llm.
        out_dir: The directory to save report files.
        dry_run: If True, prints prompts instead of calling the LLM API.
    """
    # Guard: this module is intentionally NOT importing Stage 2.
    # The caller/notebook must `import g2` first so that the following names
    # are available in the global namespace.
    if use_agent and 'answer_with_agent' not in globals():
        raise RuntimeError("answer_with_agent is not defined. Import Stage 2 (g2) in the caller before running Stage 3.")
    if not use_agent and 'answer_with_llm' not in globals():
        raise RuntimeError("answer_with_llm is not defined. Import Stage 2 (g2) in the caller before running Stage 3.")

    os.makedirs(out_dir, exist_ok=True)
    
    if use_agent:
        mode_name = "agent"
        answer_func = answer_with_agent
        print("\n" + "="*25 + f" RUNNING AGENT BENCHMARK " + "="*25)
    else:
        mode_name = "baseline"
        answer_func = answer_with_llm
        print("\n" + "="*24 + f" RUNNING BASELINE BENCHMARK " + "="*24)
    
    if dry_run:
        print("--- 🔬 DRY RUN MODE IS ON ---")

    json_path = os.path.join(out_dir, f"bench_results_{mode_name}.json")
    md_path = os.path.join(out_dir, f"bench_report_{mode_name}.md")

    results: List[Dict[str, Any]] = []
    latency_rows = []

    for q in QUERIES:
        t0 = time.perf_counter()
        raw = _call_answer(answer_func, q, dry_run=dry_run)
        out = _normalize_out(raw)
        lat_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        if print_prose:
            print(f"\n=== Question ===\n{q}")
            print("\n--- Answer ---\n")
            print(str(out["answer"]).strip())
            if out.get("hits"):
                print("\n--- Citations (top ctx) ---")
                for h in _format_hits(out.get("hits", [])):
                    y = f" {int(h['year'])}" if h.get('year') is not None else ""
                    qtr_val = h.get('quarter')
                    qtr = f" {int(qtr_val)}Q{str(y).strip()[2:]}" if qtr_val else ""
                    sec = f" — {h['section_hint']}" if h.get('section_hint') else ""
                    print(f"- {h['file']}{y}{qtr} — p.{h['page']}{sec}")
            print(f"\n(latency: {lat_ms} ms)")

        results.append({
            "query": q,
            "answer": out.get("answer"),
            "hits": _format_hits(out.get("hits", [])),
            "execution_log": out.get("execution_log"),
            "latency_ms": lat_ms,
        })
        latency_rows.append({"Query": q, "Latency_ms": lat_ms})

    # Saving logic remains the same...
    with open(json_path, "w") as f:
        json.dump({"results": results}, f, indent=2)

    md_lines = [f"# Agent CFO — {mode_name.title()} Benchmark Report\n"]
    for i, r in enumerate(results, start=1):
        md_lines.append(f"\n---\n\n## Q{i}. {r['query']}")
        md_lines.append("\n**Answer**\n\n" + r["answer"].strip())
        if r.get("hits"):
            md_lines.append("\n**Citations (top ctx)**")
            for h in r["hits"]:
                y = f" {int(h['year'])}" if h.get('year') is not None else ""
                qtr_val = h.get('quarter')
                qtr = f" {int(qtr_val)}Q{str(y).strip()[2:]}" if qtr_val else ""
                sec = f" — {h['section_hint']}" if h.get('section_hint') else ""
                md_lines.append(f"- {h['file']}{y}{qtr} — p.{h['page']}{sec}")
        if r.get("execution_log"):
            md_lines.append("\n**Execution Log**\n")
            md_lines.append("```json")
            md_lines.append(json.dumps(r["execution_log"], indent=2))
            md_lines.append("```")

    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    df = pd.DataFrame(latency_rows)
    if print_prose and not df.empty:
        p50 = float(df['Latency_ms'].quantile(0.5))
        p95 = float(df['Latency_ms'].quantile(0.95))
        print(f"\n=== {mode_name.upper()} Benchmark Summary ===")
        print(f"Saved JSON: {json_path}")
        print(f"Saved report: {md_path}")
        print(f"Latency p50: {p50:.1f} ms, p95: {p95:.1f} ms")

    return {"json_path": json_path, "md_path": md_path, "summary": df}

if __name__ == "__main__":
    # Ensure Stage 2 is initialized, then run baseline with prose printing
    try:
        init_stage2(out_dir=OUT_DIR)
        print("[Stage3] init_stage2() called successfully.")
    except Exception as e:
        print(f"[Stage3] init_stage2() failed: {e}")
    
    bench = run_benchmark(print_prose=True, use_agent=False, out_dir=OUT_DIR, dry_run=False)
    # Also echo the summary table at the end
    if isinstance(bench.get("summary"), pd.DataFrame) and not bench["summary"].empty:
        df = bench["summary"]
        p50 = float(df['Latency_ms'].quantile(0.5))
        p95 = float(df['Latency_ms'].quantile(0.95))
        print(f"\n=== BASELINE Benchmark Summary ===")
        print(f"Latency p50: {p50:.1f} ms, p95: {p95:.1f} ms")