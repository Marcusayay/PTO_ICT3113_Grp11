

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
g3x.py — Task runner over your FAISS/Marker KB (agentic tools) + optional ONLINE LLM answers

This runs 3 specific analyses using the tools/agent from g2x.py:

  1) NIM trend over last 5 quarters
     -> "Report the Gross Margin (or Net Interest Margin, if a bank) over the last 5 quarters, with values."
  2) Operating Expenses YoY table (absolute & % change) for last 3 fiscal years
     -> "Show Operating Expenses for the last 3 fiscal years, year-on-year comparison."
  3) Operating Efficiency Ratio (Opex ÷ Operating Income) with working
     -> "Calculate the Operating Efficiency Ratio (Opex ÷ Operating Income) for the last 3 fiscal years, showing the working."

All offline. Import and run from a notebook cell:
    from g3x import run_all
    run_all(base="./data_marker")
"""

from typing import Dict, List, Optional, Tuple
import math
import re
import os
from g2x import KBEnv, Agent, show_agent_result, _llm_single_call, baseline_answer_one_call, _llm_provider_info
# Feature flag for LLM summaries (set USE_LLM_SUMMARY=0/false in env to disable)
USE_LLM_SUMMARY = os.getenv("USE_LLM_SUMMARY", "1") not in ("0", "false", "False")
# ONLINE flag for baseline LLM calls (set ONLINE=0/false in env to disable)
ONLINE = os.getenv("ONLINE", "1") not in ("0", "false", "False")

# ---------- helpers ----------

def _llm_summary(
    question: str,
    agent: Agent,
    kb: KBEnv,
    res=None,
    k_ctx: int = 8,
    rows_override: Optional[List[dict]] = None
) -> str:
    """One LLM call to summarize/answer using extracted tables if present, else vector contexts."""
    lines = []
    # Prefer table rows from override if provided, else from the result
    rows = rows_override if rows_override is not None else []
    if not rows and res and getattr(res, 'final', None):
        rows = res.final.get("table_rows") or []
    if rows:
        lines.append("TABLE EXTRACTS:")
        for r in rows[:2]:
            # prefer quarters if any
            sq = r.get("series_q") or {}
            if sq:
                # sort quarters
                def _qkey(k):
                    m = re.match(r"([1-4])Q(20\d{2})$", k)
                    return (int(m.group(2)), int(m.group(1))) if m else (0,0)
                qkeys = sorted(sq.keys(), key=_qkey)[-5:]
                ser = ", ".join(f"{k}: {sq[k]}" for k in qkeys)
                lines.append(f"- {r['doc']} | {r['label']} | quarters(last5)={ser}")
            else:
                ys = sorted((r.get("series") or {}).keys())[-3:]
                ser = ", ".join(f"{y}: {r['series'][y]}" for y in ys)
                lines.append(f"- {r['doc']} | {r['label']} | years(last3)={ser}")
    # If nothing extracted, fall back to vector contexts
    if not lines:
        ctx = kb.search(question, k=k_ctx)
        if ctx is not None and not ctx.empty:
            lines.append("CONTEXT SNIPPETS:")
            for _, row in ctx.head(5).iterrows():
                text = str(row["text"]).replace("\n", " ").strip()
                if len(text) > 600:
                    text = text[:600] + "..."
                lines.append("- " + text)
    # Provide page-level hints for better citations
    if rows:
        hint_lines = []
        for r in rows[:4]:
            p = r.get('page')
            if p is not None:
                hint_lines.append(f"- {r.get('doc')}, page {int(p)}")
            else:
                hint_lines.append(f"- {r.get('doc')}, table {r.get('table_id')} row {r.get('row_id')} (no page)")
        if hint_lines:
            lines.append("CITATION HINTS:")
            lines.extend(hint_lines)
    # Build prompt
    context_block = "\n".join(lines) if lines else "(no structured context found)"
    prompt = (
        "USER QUESTION:\n" + question + "\n\n" +
        context_block +
        "\n\nINSTRUCTIONS:\n"
        "- You are given STRUCTURED TABLE ROWS and/or CONTEXT SNIPPETS above.\n"
        "- If STRUCTURED TABLE ROWS are present, you MUST use ONLY those numbers for your answer and calculations.\n"
        "- Do NOT claim data is missing if the numbers are present in the structured rows.\n"
        "- If the task asks for 'Operating Income' but the rows contain 'Total income' only, TREAT 'Total income' as the denominator for Operating Efficiency Ratio.\n"
        "- If a requested period truly does not appear in the structured rows, say so explicitly and do not infer.\n"
        "- Return a concise answer, followed by a tiny table if applicable."
    )
    print(f"[LLM] summary using {_llm_provider_info()}")
    return _llm_single_call(prompt)

# ---------- helpers ----------

def _last_n_quarters(series_q: Dict[str, float], n: int = 5) -> List[Tuple[str, float]]:
    if not series_q:
        return []
    def _qkey(k: str):
        m = re.match(r"([1-4])Q(20\d{2})$", k)
        if m:
            return (int(m.group(2)), int(m.group(1)))
        return (0, 0)
    keys = sorted(series_q.keys(), key=_qkey)
    last = keys[-n:]
    return [(k, series_q[k]) for k in last]

def _last_n_years(series: Dict[int, float], n: int = 3) -> List[Tuple[int, float]]:
    if not series:
        return []
    ys = sorted(series.keys())
    sel = ys[-n:]
    return [(y, series[y]) for y in sel]

def _pct(a: float, b: float) -> Optional[float]:
    b = float(b)
    if b == 0:
        return None
    return (float(a) - b) / b * 100.0

def _union_series(rows):
    """
    Merge {year->value} across many table rows from different docs and
    return (values, provenance) where provenance maps each year to a list
    of sources that contributed that year's value:
        provenance[year] = [{"doc":..., "table_id":..., "row_id":..., "page": ...}, ...]
    The first non-null value encountered for a year is kept as the value.
    """
    values = {}
    prov = {}
    for r in rows or []:
        doc = r.get("doc")
        tid = r.get("table_id")
        rid = r.get("row_id")
        page = r.get("page")
        series = r.get("series") or {}
        for y, v in series.items():
            if v is None:
                continue
            # record provenance regardless
            prov.setdefault(y, []).append({
                "doc": doc, "table_id": tid, "row_id": rid, "page": page
            })
            # keep the first seen value for this year
            if y not in values:
                values[y] = v
    return values, prov

def _last_n_years_map(series_map, n: int = 3):
    ys = sorted(series_map.keys())
    sel = ys[-n:]
    return [(y, series_map[y]) for y in sel]

# Helper to pick a representative source for a year
def _pick_source_for_year(prov_map, y):
    """
    Choose one representative source dict for a given year
    from the provenance map, preferring entries with a page number.
    """
    items = prov_map.get(y) or []
    if not items:
        return None
    with_page = [s for s in items if s.get("page") is not None]
    return (with_page[0] if with_page else items[0])

# ---------- Q1: NIM last 5 quarters ----------

def run_q1_nim_last5q(agent: Agent, kb: KBEnv):
    q = "Net Interest Margin over the last 5 quarters"
    res = agent.run(q, k_ctx=6)
    print("\n=== Q1) Net Interest Margin — last 5 quarters ===")
    # Try table rows with quarters
    rows = res.final.get("table_rows") or []
    picked = None
    for r in rows:
        if r.get("series_q"):
            picked = r
            break
    if not picked:
        print("⚠️ No quarterly NIM found in indexed tables.")
        # fall back to annual if available
        for r in rows:
            if r.get("series"):
                years = _last_n_years(r["series"], n=3)
                print("Fallback (years):", ", ".join(f"{y}: {v}" for y, v in years))
                break
        # LLM summary even if not found
        if USE_LLM_SUMMARY:
            print("\nLLM Summary (baseline, single call):")
            print(_llm_summary(q, agent, kb, res=res, k_ctx=8, rows_override=([picked] if picked else rows)))
        if ONLINE:
            print("\nLLM Answer (online, single call):")
            print(f"[LLM] baseline using {_llm_provider_info()}")
            tr = ([picked] if picked else rows)
            baseline_answer_one_call(kb, q, k_ctx=8, table_rows=tr)
        return res
    last5 = _last_n_quarters(picked["series_q"], n=5)
    if not last5:
        print("⚠️ No quarterly NIM found in indexed tables.")
        if USE_LLM_SUMMARY:
            print("\nLLM Summary (baseline, single call):")
            print(_llm_summary(q, agent, kb, res=res, k_ctx=8))
        if ONLINE:
            print("\nLLM Answer (online, single call):")
            print(f"[LLM] baseline using {_llm_provider_info()}")
            tr = ([picked] if picked else rows)
            baseline_answer_one_call(kb, q, k_ctx=8, table_rows=tr)
        return res
    print(f"Source: {picked['doc']} | label: {picked['label']}")
    print("Values (last 5): " + ", ".join(f"{k}: {v}" for k, v in last5))
    if USE_LLM_SUMMARY:
        print("\nLLM Summary (baseline, single call):")
        print(_llm_summary(q, agent, kb, res=res, k_ctx=8, rows_override=([picked] if picked else rows)))
    if ONLINE:
        print("\nLLM Answer (online, single call):")
        print(f"[LLM] baseline using {_llm_provider_info()}")
        tr = ([picked] if picked else rows)
        baseline_answer_one_call(kb, q, k_ctx=8, table_rows=tr)
    return res

# ---------- Q2: Opex last 3 fiscal years with YoY ----------

def run_q2_opex_yoy(agent: Agent, kb: KBEnv):
    q = "Operating Expenses last 3 fiscal years YoY"
    res = agent.run(q, k_ctx=6)
    print("\n=== Q2) Operating Expenses — last 3 fiscal years (YoY) ===")

    # Pull MANY rows then union across docs/tables to recover a continuous series
    rows = agent.table.get_metric_rows("operating expenses", limit=50)
    if not rows:
        rows = agent.table.get_metric_rows("total expenses", limit=50)

    combo, prov = _union_series(rows)
    # Build per-year rows with real provenance so citations show actual docs/pages
    years_for_report = sorted(combo.keys())[-3:] if combo else []
    rows_yearwise = []
    for y in years_for_report:
        src = _pick_source_for_year(prov, y)
        rows_yearwise.append({
            "doc": (src.get("doc") if src else "(unknown)"),
            "table_id": (src.get("table_id") if src else -1),
            "row_id": (src.get("row_id") if src else -1),
            "label": "Operating expenses",
            "series": {y: combo.get(y)},
            "series_q": {},
            "page": (src.get("page") if src and src.get("page") is not None else None),
        })
    # Fallback: if something went wrong, still provide a single combined row
    if not rows_yearwise:
        rows_yearwise = [{
            "doc": "(union)",
            "table_id": -1,
            "row_id": -1,
            "label": "Operating expenses",
            "series": combo,
            "series_q": {},
            "page": None
        }]
    if not combo:
        print("⚠️ No expenses series found across docs.")
        if USE_LLM_SUMMARY:
            print("\nLLM Summary (baseline, single call):")
            print(_llm_summary(q, agent, kb, res=res, k_ctx=8, rows_override=rows_yearwise))
        if ONLINE:
            print("\nLLM Answer (online, single call):")
            print(f"[LLM] baseline using {_llm_provider_info()}")
            baseline_answer_one_call(kb, q, k_ctx=8, table_rows=rows_yearwise)
        return res

    last3 = [(y, combo[y]) for y in years_for_report]
    if len(last3) < 2:
        print("⚠️ Not enough annual values to compute YoY.")
        if USE_LLM_SUMMARY:
            print("\nLLM Summary (baseline, single call):")
            print(_llm_summary(q, agent, kb, res=res, k_ctx=8, rows_override=rows_yearwise))
        if ONLINE:
            print("\nLLM Answer (online, single call):")
            print(f"[LLM] baseline using {_llm_provider_info()}")
            baseline_answer_one_call(kb, q, k_ctx=8, table_rows=rows_yearwise)
        return res

    print("Year | Opex | YoY %")
    print("-----|------|------")
    prev_val = None
    for y, v in last3:
        yoy = ((v - prev_val) / prev_val * 100.0) if prev_val not in (None, 0) else None
        yoy_s = f"{yoy:.2f}%" if yoy is not None else "—"
        print(f"{y} | {v} | {yoy_s}")
        prev_val = v

    # Show sources (doc & page) used for each year printed
    print("\nSources:")
    for y, _ in last3:
        src = _pick_source_for_year(prov, y)
        if src:
            p = src.get("page")
            ptxt = f"page {int(p)}" if p is not None else "no page"
            print(f"  {y}: {src.get('doc')} ({ptxt})")

    if USE_LLM_SUMMARY:
        print("\nLLM Summary (baseline, single call):")
        print(_llm_summary(q, agent, kb, res=res, k_ctx=8, rows_override=rows_yearwise))
    if ONLINE:
        print("\nLLM Answer (online, single call):")
        print(f"[LLM] baseline using {_llm_provider_info()}")
        baseline_answer_one_call(kb, q, k_ctx=8, table_rows=rows_yearwise)

    return res

# ---------- Q3: Operating Efficiency Ratio (Opex ÷ Operating Income) ----------

def run_q3_efficiency_ratio(agent: Agent, kb: KBEnv):
    print("\n=== Q3) Operating Efficiency Ratio — last 3 fiscal years ===")

    # Union Opex across docs/tables
    opex_rows = agent.table.get_metric_rows("operating expenses", limit=50) \
        or agent.table.get_metric_rows("total expenses", limit=50)
    opex, opex_prov = _union_series(opex_rows)

    # Union Income across docs/tables (prefer 'total income', else 'operating income')
    income_rows = agent.table.get_metric_rows("total income", limit=50) \
        or agent.table.get_metric_rows("operating income", limit=50)
    income, income_prov = _union_series(income_rows)

    # Build per-year rows for both Opex and Income so citations show real docs/pages
    rows_for_llm = []
    years_overlap = sorted(set(opex.keys()).intersection(income.keys()))[-3:]
    for y in years_overlap:
        s_ox = _pick_source_for_year(opex_prov, y)
        s_in = _pick_source_for_year(income_prov, y)
        rows_for_llm.append({
            "doc": (s_ox.get("doc") if s_ox else "(unknown)"),
            "table_id": (s_ox.get("table_id") if s_ox else -1),
            "row_id": (s_ox.get("row_id") if s_ox else -1),
            "label": "Operating expenses",
            "series": {y: opex.get(y)},
            "series_q": {},
            "page": (s_ox.get("page") if s_ox and s_ox.get("page") is not None else None)
        })
        rows_for_llm.append({
            "doc": (s_in.get("doc") if s_in else "(unknown)"),
            "table_id": (s_in.get("table_id") if s_in else -1),
            "row_id": (s_in.get("row_id") if s_in else -1),
            "label": "Total income",
            "series": {y: income.get(y)},
            "series_q": {},
            "page": (s_in.get("page") if s_in and s_in.get("page") is not None else None)
        })
    # Fallback to union-style rows if needed
    if not rows_for_llm:
        rep_year = max(opex.keys() & income.keys()) if (opex and income) else None
        rep_opex = _pick_source_for_year(opex_prov, rep_year) if rep_year else None
        rep_income = _pick_source_for_year(income_prov, rep_year) if rep_year else None
        rows_for_llm = [
            {
                "doc": (rep_opex.get("doc") if rep_opex else "(union)"),
                "table_id": (rep_opex.get("table_id") if rep_opex else -1),
                "row_id": (rep_opex.get("row_id") if rep_opex else -1),
                "label": "Operating expenses",
                "series": opex or {},
                "series_q": {},
                "page": (rep_opex.get("page") if rep_opex else None)
            },
            {
                "doc": (rep_income.get("doc") if rep_income else "(union)"),
                "table_id": (rep_income.get("table_id") if rep_income else -1),
                "row_id": (rep_income.get("row_id") if rep_income else -1),
                "label": "Total income",
                "series": income or {},
                "series_q": {},
                "page": (rep_income.get("page") if rep_income else None)
            },
        ]

    if not opex or not income:
        print("⚠️ Missing Opex or Income series across docs.")
        if USE_LLM_SUMMARY:
            print("\nLLM Summary (baseline, single call):")
            q = "Operating Efficiency Ratio (Opex / Operating Income) for the last 3 fiscal years"
            print(_llm_summary(q, agent, kb, res=None, k_ctx=8, rows_override=rows_for_llm))
        if ONLINE:
            print("\nLLM Answer (online, single call):")
            print(f"[LLM] baseline using {_llm_provider_info()}")
            q_llm = "Operating Efficiency Ratio (Opex / Operating Income) for the last 3 fiscal years"
            baseline_answer_one_call(kb, q_llm, k_ctx=8, table_rows=rows_for_llm)
        return None

    years = years_overlap
    if not years:
        print("⚠️ No overlapping years between Opex and Income.")
        if USE_LLM_SUMMARY:
            print("\nLLM Summary (baseline, single call):")
            q = "Operating Efficiency Ratio (Opex / Operating Income) for the last 3 fiscal years"
            print(_llm_summary(q, agent, kb, res=None, k_ctx=8, rows_override=rows_for_llm))
        if ONLINE:
            print("\nLLM Answer (online, single call):")
            print(f"[LLM] baseline using {_llm_provider_info()}")
            q_llm = "Operating Efficiency Ratio (Opex / Operating Income) for the last 3 fiscal years"
            baseline_answer_one_call(kb, q_llm, k_ctx=8, table_rows=rows_for_llm)
        return None

    print("Year | Opex | Income | Opex/Income %")
    print("-----|------|--------|---------------")
    for y in years:
        ov = opex.get(y)
        iv = income.get(y)
        ratio = (ov / iv * 100.0) if (iv not in (None, 0)) else None
        ratio_s = f"{ratio:.2f}%" if ratio is not None else "—"
        print(f"{y} | {ov} | {iv} | {ratio_s}")

    print("\nSources:")
    for y in years:
        s1 = _pick_source_for_year(opex_prov, y)
        s2 = _pick_source_for_year(income_prov, y)
        if s1:
            p1 = s1.get("page"); p1t = f"page {int(p1)}" if p1 is not None else "no page"
            print(f"  Opex {y}: {s1.get('doc')} ({p1t})")
        if s2:
            p2 = s2.get("page"); p2t = f"page {int(p2)}" if p2 is not None else "no page"
            print(f"  Income {y}: {s2.get('doc')} ({p2t})")

    if USE_LLM_SUMMARY:
        print("\nLLM Summary (baseline, single call):")
        q = "Operating Efficiency Ratio (Opex / Operating Income) for the last 3 fiscal years"
        print(_llm_summary(q, agent, kb, res=None, k_ctx=8, rows_override=rows_for_llm))
    if ONLINE:
        print("\nLLM Answer (online, single call):")
        q_llm = "Operating Efficiency Ratio (Opex / Operating Income) for the last 3 fiscal years"
        baseline_answer_one_call(kb, q_llm, k_ctx=8, table_rows=rows_for_llm)

    return {"years": years, "opex": opex, "income": income}

# ---------- Runner ----------

def run_all(base: str = "./data_marker"):
    kb = KBEnv(base=base)
    agent = Agent(kb)

    # Q1
    res1 = run_q1_nim_last5q(agent, kb)

    # Q2
    res2 = run_q2_opex_yoy(agent, kb)

    # Q3
    _ = run_q3_efficiency_ratio(agent, kb)

# Auto-run when executed directly (safe in notebooks too)
if __name__ == "__main__" or "__file__" not in globals():
    run_all(base="./data_marker")