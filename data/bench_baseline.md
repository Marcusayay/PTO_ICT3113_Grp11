# Baseline Benchmark Report

**Pipeline**: Hybrid Search (BM25 + Vector + RRF + Rerank) -> Single LLM


---

## Q1. Report the Gross Margin (or Net Interest Margin, if a bank) over the last 5 quarters, with values.

**Answer**

**Answer**  
The context provides Net Interest Margin (NIM) for only two of the last five quarters: Q4 2024 and Q4 2023.  NIM for the other three quarters (Q1–Q3 2024) is not disclosed in the excerpts supplied.

| Quarter | Net Interest Margin (NIM) | Source |
|---------|---------------------------|--------|
| Q4 2024 | 2.61 % (commercial book) | “net interest margin increased 48 basis points to 2.11%, with its fourth‑quarter net interest margin signiȴantly higher at 2.61%.” |
| Q4 2023 | 2.05 % | “net interest margin was 2.05% in the fourth quarter.” |

**Missing data**

- Q1 2024: No quarterly NIM disclosed.  
- Q2 2024: No quarterly NIM disclosed.  
- Q3 2024: No quarterly NIM disclosed.  

The only other quarterly‑level figures available in the context are half‑year aggregates (e.g., 2nd Half 2024: 2.13 %, 1st Half 2024: 2.14 %) which do not provide the specific quarterly values requested.


**Citations**

- dbs-annual-report-2022 
- 2Q24_performance_summary p.9
- dbs-annual-report-2022 p.96
- dbs-annual-report-2022 p.96
- dbs-annual-report-2023 p.95

**Latency**: 4236.25 ms

---

## Q2. Show Operating Expenses for the last 3 fiscal years, year-on-year comparison.

**Answer**

**Answer**  
The operating expenses reported for the last three fiscal years are:

| Fiscal Year | Operating Expenses (SGD m) | YoY % Change |
|-------------|---------------------------|--------------|
| 2022 | 7,090.0 | – |
| 2023 | 8,291.0 | **+16.9 %** |
| 2024 | 9,018.0 | **+8.8 %** |

*YoY % = (Current Year Expenses – Prior Year Expenses) ÷ Prior Year Expenses.*

**Citations**

- 2024 total expenses: 9,018.0 – *[dbs‑annual‑report‑2024] table#188 row#13*  
- 2023 total expenses: 8,291.0 – *[dbs‑annual‑report‑2023] table#197 row#11*  
- 2022 total expenses: 7,090.0 – *[dbs‑annual‑report‑2022] table#195 row#11*

**Missing data**  
No data gaps exist for the requested operating expenses of the last three fiscal years.


**Citations**

- dbs-annual-report-2022 
- dbs-annual-report-2024 p.22
- dbs-annual-report-2022 
- dbs-annual-report-2024 p.22
- dbs-annual-report-2022 p.63

**Latency**: 3514.83 ms

---

## Q3. Calculate the Operating Efficiency Ratio (Opex ÷ Operating Income) for the last 3 fiscal years, showing the working.

**Answer**

**Answer**  
The context does not provide the explicit figures for operating expenses (Opex) or operating income for the last three fiscal years, so the Operating Efficiency Ratio (Opex ÷ Operating Income) cannot be calculated from the supplied excerpts.

| Period | Metric | Value (from context) |
|--------|--------|----------------------|
| 2024 | Profit before changes in operating assets & liabilities | 14,080.0 |
| 2024 | Total income | 22,297.0 |
| 2024 | Income taxes paid | (1,438) |
| 2023 | Profit before changes in operating assets & liabilities | 12,671.0 |
| 2023 | Total income | 20,180.0 |
| 2023 | Income taxes paid | (1,319) |
| 2022 | Cost‑income ratio (overall) | 40 % |
| 2022 | Cost‑income ratio (underlying) | 44 % |
| 2022 | ROE (overall) | 37 % |
| 2022 | ROE (digital) | 39 % |
| 2022 | ROE (traditional) | 24 % |

**Missing data**

- **Operating expenses (Opex)** for 2024, 2023, and 2022 – not disclosed in the excerpts.  
- **Operating income** (or a clear proxy such as EBIT) for 2024, 2023, and 2022 – not disclosed in the excerpts.  
- **Operating income for 2022** – no figure provided.  

Without both Opex and operating income for each year, the requested Operating Efficiency Ratio cannot be computed.


**Citations**

- dbs-annual-report-2022 
- 4Q24_performance_summary p.34
- 4Q24_performance_summary p.28
- 4Q24_performance_summary p.4
- 4Q24_performance_summary p.34

**Latency**: 3147.98 ms

---

## Summary

- P50: 3514.8 ms
- P95: 4164.1 ms