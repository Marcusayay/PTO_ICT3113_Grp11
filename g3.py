from __future__ import annotations

"""
Stage3.py — Benchmark Runner (Stage 3)

Runs the 3 standardized queries for both the baseline and agentic pipelines,
times them, saves JSON/Markdown reports, and prints prose answers with citations.

Artifacts written to OUT_DIR (default: data/):
  - bench_results_baseline.json / bench_results_agent.json
  - bench_report_baseline.md / bench_report_agent.md
"""
import os, json, time, inspect, re, glob
from typing import List, Dict, Any

import pandas as pd

# Explicitly import Stage-2 entrypoints so we don't rely on globals
from g2 import init_stage2, answer_with_llm_baseline as answer_with_llm, answer_with_agent


OUT_DIR = os.environ.get("AGENT_CFO_OUT_DIR", "data")

# --- Structured NIM helpers ---
def _pkey_quarter_label(lbl: str):
    """
    Convert labels like '2Q25' or '3Q2024' into a sortable (year, quarter) tuple.
    Non-matching labels get (0,0) so they sort first.
    """
    m = re.match(r"^\s*([1-4])\s*Q\s*(\d{2}|\d{4})\s*$", lbl, re.IGNORECASE)
    if not m:
        return (0, 0)
    q = int(m.group(1))
    y = int(m.group(2))
    y = y if y >= 100 else (2000 + y)  # normalize 2-digit years
    return (y, q)

def _load_structured_nim_from_metrics(out_dir: str = OUT_DIR):
    """
    Prefer structured NIM from Stage-1 artifacts: data/*_metrics.json
    Returns newest-last list of dicts:
      [{"period": "2Q25", "value": 2.55, "source": "2Q25_CFO_presentation.pdf", "page": 6}, ...]
    We look for pages with chart_type == "line-like" and metric containing "net interest margin".
    """
    items = []
    for path in glob.glob(os.path.join(out_dir, "*_metrics.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        src = doc.get("source") or os.path.basename(path).replace("_metrics.json", ".pdf")
        pages = doc.get("pages") or []
        for pg in pages:
            if not isinstance(pg, dict):
                continue
            metric = (pg.get("metric") or "").lower()
            if pg.get("chart_type") == "line-like" and "net interest margin" in metric:
                extracted = pg.get("extracted") or {}
                for period, vals in extracted.items():
                    if not isinstance(vals, dict):
                        continue
                    # Prefer group_nim; fallback to commercial_nim
                    val = vals.get("group_nim")
                    if val is None:
                        val = vals.get("commercial_nim")
                    if val is None:
                        continue
                    try:
                        v = float(val)
                    except Exception:
                        continue
                    items.append({
                        "period": period,
                        "value": v,
                        "source": src,
                        "page": pg.get("page"),
                    })
    # sort by (year, quarter) and return newest last
    items.sort(key=lambda d: _pkey_quarter_label(d["period"]))
    return items

def _format_nim_answer_from_structured(items):
    """
    Build the exact prose/table answer format expected by Stage 3 for the NIM query,
    using the last 5 quarters from the provided structured items.
    """
    if not items:
        return None
    last5 = items[-5:]  # newest last
    # Build a simple markdown table
    lines = ["NIM (%) — last 5 quarters:", "Quarter | NIM (%)", "--------|--------"]
    for d in last5:
        # Keep raw float; formatting can be adjusted if you prefer 2 dp
        lines.append(f"{d['period']} | {d['value']}")
    answer_text = "\n".join(lines)
    # Citations: list each unique source/page once
    cites = []
    seen = set()
    for d in last5:
        src = d.get("source") or ""
        page = d.get("page")
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        if src:
            if page is not None:
                cites.append(f"- {src}, p.{page}")
            else:
                cites.append(f"- {src}")
    if cites:
        answer_text += "\n\nCitations:\n" + "\n".join(cites)
    return {
        "answer": answer_text,
        "hits": [],              # keep empty; we’re answering directly from structured cache
        "execution_log": {"used": "structured_metrics_json", "count": len(last5)}
    }
def answer_with_llm_wrapped(query: str, dry_run: bool = False):
    """
    Wrapper around answer_with_llm that prefers structured NIM from *_metrics.json
    when the query asks for Net Interest Margin. Falls back to Stage-2 otherwise.
    """
    ql = (query or "").lower()
    is_nim_query = ("net interest margin" in ql) or (" nim" in ql) or ql.endswith("nim") or ql.startswith("nim")
    if is_nim_query:
        # Only use structured cache if explicitly enabled
        if os.environ.get("USE_STRUCTURED_NIM", "0") in ("1", "true", "True"):
            structured = _load_structured_nim_from_metrics(OUT_DIR)
            if structured:
                return _format_nim_answer_from_structured(structured)
    # Fallback to the original Stage-2 baseline pipeline
    return answer_with_llm(query)

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
        answer_func = answer_with_llm_wrapped
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