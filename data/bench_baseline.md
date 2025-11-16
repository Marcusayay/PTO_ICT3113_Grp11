# Baseline Benchmark Report

**Pipeline**: Hybrid Search (BM25 + Vector + RRF + Rerank) -> Single LLM


---

## Q1. Report the Gross Margin (or Net Interest Margin, if a bank) over the last 5 quarters, with values.

**Answer**

**Answer**  
The context supplies explicit net‑interest‑margin (NIM) figures only for two quarters: Q4 2024 (2.05 %) and Q4 2023 (2.61 %). No quarter‑specific NIM values are provided for Q3 2024, Q2 2024, or Q1 2024.

| Quarter | Net Interest Margin |
|---------|---------------------|
| Q4 2024 | 2.05 %  *(excerpt 1)* |
| Q4 2023 | 2.61 %  *(excerpt 2)* |

**Missing data**

- Q3 2024: No quarter‑specific NIM in the context.  
- Q2 2024: No quarter‑specific NIM in the context.  
- Q1 2024: No quarter‑specific NIM in the context.  

*(Half‑year and annual NIM figures are available in the context but were not requested.)*


**Citations**

- dbs-annual-report-2022 
- 2Q24_performance_summary p.9
- dbs-annual-report-2022 p.96
- dbs-annual-report-2022 p.96
- dbs-annual-report-2023 p.95

**Latency**: 2692.19 ms

---

## Q2. Show Operating Expenses for the last 3 fiscal years, year-on-year comparison.

**Answer**

**Answer**  
The context provides “Total expenses” for the three most recent fiscal years (2024, 2023, 2022). Treating these figures as the company’s operating expenses, the year‑on‑year changes are:

| Fiscal Year | Operating Expenses (USD m) | YoY Change |
|-------------|---------------------------|------------|
| 2024 | 9 018.0 | +8.8 % vs 2023 |
| 2023 | 8 291.0 | +16.9 % vs 2022 |
| 2022 | 7 090.0 | – (baseline) |

**Calculations**

- 2024 vs 2023:  
  Ratio = 9 018 ÷ 8 291 = 1.0877 → YoY % = 8.77 % ≈ **8.8 %**

- 2023 vs 2022:  
  Ratio = 8 291 ÷ 7 090 = 1.1694 → YoY % = 16.94 % ≈ **16.9 %**

**Sources**

- Total expenses 2024: 9 018.0, 2023: 8 291.0 – *[dbs‑annual‑report‑2024] table#188 row#13*  
- Total expenses 2023: 8 291.0, 2022: 7 090.0 – *[dbs‑annual‑report‑2023] table#197 row#11*  
- Total expenses 2022: 7 090.0, 2021: 6 569.0 – *[dbs‑annual‑report‑2022] table#195 row#11*  

**Missing data**  
None – all required figures for 2024, 2023, and 2022 are present in the provided excerpts.


**Citations**

- dbs-annual-report-2022 
- dbs-annual-report-2024 p.22
- dbs-annual-report-2022 
- dbs-annual-report-2024 p.22
- dbs-annual-report-2022 p.63

**Latency**: 2089.03 ms

---

## Q3. Calculate the Operating Efficiency Ratio (Opex ÷ Operating Income) for the last 3 fiscal years, showing the working.

**Answer**

**Answer**  
The context does not provide the necessary figures for Operating Expense (Opex) or Operating Income for the last three fiscal years, so the Operating Efficiency Ratio (Opex ÷ Operating Income) cannot be calculated from the supplied excerpts.

| Period | Metric | Value | Source |
|--------|--------|-------|--------|
| 2024 | Profit before changes in operating assets & liabilities | 14,080.0 | [4Q24_performance_summary] table#50 row#12 |
| 2023 | Profit before changes in operating assets & liabilities | 12,671.0 | [4Q24_performance_summary] table#50 row#12 |

**Missing data**

- **Operating Expense (Opex)** for 2024, 2023, and 2022 – not disclosed in the provided excerpts.  
- **Operating Income** (or an equivalent metric such as EBIT) for 2024, 2023, and 2022 – not disclosed in the provided excerpts.  

Without both Opex and Operating Income for each year, the Operating Efficiency Ratio cannot be computed.


**Citations**

- dbs-annual-report-2022 
- 4Q24_performance_summary p.34
- 4Q24_performance_summary p.28
- 4Q24_performance_summary 
- 4Q24_performance_summary p.34

**Latency**: 1372.66 ms

---

## Summary

- P50: 2089.0 ms
- P95: 2631.9 ms