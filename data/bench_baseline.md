# Baseline Benchmark Report

**Pipeline**: Hybrid Search (BM25 + Vector + RRF + Rerank) -> Single LLM


---

## Q1. Report the Gross Margin (or Net Interest Margin, if a bank) over the last 5 quarters, with values.

**Answer**

**Answer**  
The context provides Net Interest Margin (NIM) figures only at the half‑year and annual levels for DBS.  Quarter‑specific NIM values for the most recent five quarters are not available in the excerpts supplied.

| Period | Metric | Value | Source |
|--------|--------|-------|--------|
| 1H 2023 | NIM | 2.14 % | 2Q24_performance_summary (row 6) |
| 2H 2023 | NIM | 2.16 % | 4Q24_performance_summary (row 6) |
| 1H 2024 | NIM | 2.14 % | 4Q24_performance_summary (row 6) |
| 2H 2024 | NIM | 2.13 % | 4Q24_performance_summary (row 6) |
| 1H 2025 | NIM | 2.08 % | 2Q25_performance_summary (row 6) |

**Missing data**

- Quarter‑level NIM for Q1 2024, Q2 2024, Q3 2024, Q4 2024, and Q4 2023 are not provided in the context.  
- No quarter‑level NIM values are available for the last five quarters; only half‑year and annual figures are cited.


**Citations**

- dbs-annual-report-2022 
- 2Q24_performance_summary p.9
- dbs-annual-report-2022 p.96
- dbs-annual-report-2022 p.96
- dbs-annual-report-2023 p.95

**Latency**: 6496.71 ms

---

## Q2. Show Operating Expenses for the last 3 fiscal years, year-on-year comparison.

**Answer**

**Answer**  
Operating expenses for the last three fiscal years are:

| Fiscal Year | Operating Expenses (USD bn) | YoY % Change |
|-------------|-----------------------------|--------------|
| 2024 | 8,895.0 | 10.0 % (from context) |
| 2023 | 8,056.0 | 13.6 % (calculated) |
| 2022 | 7,090.0 | – |

**YoY % Calculation (2023 vs 2022)**  
Ratio = 8,056.0 bn ÷ 7,090.0 bn = 1.1363 = **13.6 %**

**Sources**  
- 4Q24 performance summary, table #1 row 9: 2024 = 8,895.0 bn, 2023 = 8,056.0 bn【excerpt 1】  
- dbs‑annual‑report‑2022, table #195 row 11: 2022 = 7,090.0 bn【excerpt 8】

**Missing data**  
- No additional fiscal years beyond 2022 are required for this comparison.


**Citations**

- dbs-annual-report-2022 
- dbs-annual-report-2024 p.22
- dbs-annual-report-2022 
- dbs-annual-report-2024 p.22
- dbs-annual-report-2022 p.63

**Latency**: 8290.1 ms

---

## Q3. Calculate the Operating Efficiency Ratio (Opex ÷ Operating Income) for the last 3 fiscal years, showing the working.

**Answer**

**Answer**  
The only explicit operating‑efficiency ratio (Opex ÷ Operating Income) provided in the excerpts is for FY 2022, where the cost‑income ratio was 40 % (underlying 44 %). No comparable figures are quoted for FY 2023 or FY 2024, so those ratios cannot be calculated from the supplied context.

| Fiscal Year | Operating‑Efficiency Ratio (Opex ÷ Operating Income) |
|-------------|-----------------------------------------------------|
| 2022        | 40 % (cost‑income ratio) – *underlying 44 %* |

**Missing data**  
- FY 2023: No cost‑income ratio or Opex/Operating Income figures are provided.  
- FY 2024: No cost‑income ratio or Opex/Operating Income figures are provided.  

*Citation:* The 40 % figure comes from the excerpt stating “the cost‑income ratio of the overall business improved by 11 percentage points from the previous year to 40% on the back of higher rates and volumes” (context excerpt 1). The underlying 44 % figure is also quoted in the same excerpt. No other fiscal‑year ratios are mentioned in the supplied excerpts.


**Citations**

- dbs-annual-report-2022 
- 4Q24_performance_summary p.34
- 4Q24_performance_summary p.28
- 4Q24_performance_summary 
- 4Q24_performance_summary p.34

**Latency**: 5901.11 ms

---

## Summary

- P50: 6496.7 ms
- P95: 8110.8 ms