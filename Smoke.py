# --- Smoke test for Agent CFO (Stage 2 already imported as g2) ---
# Consolidated & de-duplicated query set (original + standardized)

import os, json, pprint

# 0) Init Stage 2 from your built artifacts
import g2
g2.init_stage2(out_dir="data")

# 1) Define focused queries (consolidated)
QUERIES = [
    # Keep NIM phrasing (triggers Stage2 'nim' logic)
    "Report the Net Interest Margin (NIM) over the last 5 quarters, with values, and add 1–2 lines of explanation.",

    # Opex YoY w/ MD&A (original, richer)
    "Show Operating Expenses (Opex) for the last 3 fiscal years, year-on-year comparison, and summarize the top 3 Opex drivers from the MD&A.",

    # CTI (original)
    "Calculate the Cost-to-Income Ratio (CTI) for the last 3 fiscal years; show your working and give 1–2 lines of implications.",

    # Opex YoY table-only (standardized)
    "Show Operating Expenses for the last 3 fiscal years, year-on-year comparison.",

    # Operating Efficiency Ratio (new standardized)
    "Calculate the Operating Efficiency Ratio (Opex ÷ Operating Income) for the last 3 fiscal years, showing the working.",
]

def _s(x, maxlen=240):
    """Safe short-string: handles None and trims long outputs."""
    if x is None:
        return ""
    try:
        s = str(x)
    except Exception:
        s = repr(x)
    s = s.replace("\n", " ")
    return s[:maxlen]

def run_once(query: str, dry_run: bool):
    print("\n" + "="*90)
    print(("DRY RUN" if dry_run else "LIVE"), "→", query)
    out = g2.answer_with_agent(query, dry_run=dry_run)

    ans = out.get("answer", "")
    print("\n--- Answer ---\n", (ans or "").strip())

    exec_log = out.get("execution_log") or []
    if exec_log:
        print("\n--- Tool Execution Log (truncated) ---")
        for step in exec_log:
            step_name = step.get("step", "")
            tool_call = _s(step.get("tool_call"))
            result    = _s(step.get("result"))
            error     = _s(step.get("error"))

            if "Planning" in step_name:
                plan = step.get("plan") or []
                print("• Plan steps:", len(plan))
            elif tool_call.startswith("calculator("):
                print("•", tool_call, "→", result or error or "(no output)")
            elif tool_call.startswith("table_extraction("):
                print("•", tool_call, "→", result or "(no result)")
            elif tool_call.startswith("multi_document_compare("):
                print("•", tool_call, "→ [multi-doc compare output]")
            elif error:
                print("•", tool_call or step_name or "(unknown step)", "ERROR:", error)
            else:
                # Fallback for any step without a recognized shape
                if step_name or tool_call or result:
                    print("•", step_name or tool_call or "(step)", "→", result or "(no result)")

    return out

# 2) DRY RUN (plans only)
for q in QUERIES:
    run_once(q, dry_run=True)

# 3) LIVE RUNS (execute tools)
live_results = []
for q in QUERIES:
    live_results.append(run_once(q, dry_run=False))

# 4) Optional: Pull out the numeric values the agent stashed for calculators
#    (Helpful to verify that %, commas, bn/mn were sanitized correctly.)
def extract_state_vars(execution_log):
    vars_seen = {}
    for step in (execution_log or []):
        res = step.get("result")
        if not res:
            continue
        res_s = str(res)
        if "Value:" in res_s and "Source:" in res_s:
            # e.g., "Value: 37%, Source: ..."
            v = res_s.split("Value:", 1)[1].split("Source:", 1)[0].strip()
            vars_seen.setdefault("values", []).append(v)
    return vars_seen

print("\n" + "="*90)
print("EXTRACTED NUMERIC PREVIEW")
for i, r in enumerate(live_results, 1):
    vars_preview = extract_state_vars(r.get("execution_log"))
    print(f"\nQ{i}: {QUERIES[i-1][:60]}…")
    pprint.pp(vars_preview)