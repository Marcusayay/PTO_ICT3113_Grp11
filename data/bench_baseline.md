# Baseline Benchmark Report

**Pipeline**: Hybrid Search (BM25 + Vector + RRF + Rerank) -> Single LLM


---

## Q1. Report the Gross Margin (or Net Interest Margin, if a bank) over the last 5 quarters, with values.

**Answer**

**Net Interest Margin (NIM) – last 5 quarters (where data are available)**  

| Quarter | NIM (%) | Source |
|---------|---------|--------|
| Q2 2025 (first‑half 2025) | 2.08 % | 2Q25 performance summary |
| Q4 2024 (second‑half 2024) | 2.13 % | 4Q24 performance summary |
| Q3 2024 (second‑half 2024) | 2.13 % | 4Q24 performance summary |
| Q2 2024 (first‑half 2024) | 2.14 % | 2Q24 performance summary |
| Q1 2024 (first‑half 2024) | 2.14 % | 2Q24 performance summary |

**Missing data**

- NIM for Q1 2025 is not provided in the supplied context.
- The context gives only half‑year figures; exact quarter‑level values for Q2 2025, Q1 2025, Q4 2024, Q3 2024, and Q2 2024 are inferred from the half‑year totals and therefore may not reflect the precise quarterly performance.

**Citations**

- 2Q25 performance summary – Net interest margin (1st Half 2025) 2.08 %【2Q25_performance_summary】  
- 4Q24 performance summary – Net interest margin (2nd Half 2024) 2.13 %【4Q24_performance_summary】  
- 2Q24 performance summary – Net interest margin (1st Half 2024) 2.14 %【2Q24_performance_summary】  
- 2Q24 performance summary – Net interest margin (1st Half 2023) 2.14 % (used to infer Q1 2024)【2Q24_performance_summary】  
- 4Q24 performance summary – Net interest margin (2nd Half 2023) 2.16 % (used to infer Q3 2024)【4Q24_performance_summary】


**Citations**

- dbs-annual-report-2022  [marker] (score: 0.6558)
- 2Q24_performance_summary p.9 [marker] (score: -0.5437)
- dbs-annual-report-2022 p.96 [marker] (score: -0.7675)
- dbs-annual-report-2022 p.96 [marker] (score: -0.7991)
- dbs-annual-report-2023 p.95 [marker] (score: -1.0668)

**Latency**: 7344.84 ms

---

## Q2. Show Operating Expenses for the last 3 fiscal years, year-on-year comparison.

**Answer**

**Operating‑expense trend (millions of local currency)**  

| Fiscal year | Operating expenses | YoY change |
|-------------|--------------------|------------|
| 2022 | 3,803 | – |
| 2023 | 4,627 | +21.7 % |
| 2024 | 5,273 | +14.0 % |

*The figures come from the same table series in the annual reports, giving a consistent set of operating‑expense numbers for 2022‑2024.*

**Citations**

- dbs‑annual‑report‑2022 – table#7, row#1: “Expenses 2022 = 3803.0, 2021 = 3353.0”  
- dbs‑annual‑report‑2024 – table#7, row#3: “Expenses 2024 = 5273, 2023 = 4627, YoY% = 14”  
- (YoY percentages calculated from the above figures.)


**Citations**

- dbs-annual-report-2022  [marker] (score: -6.7801)
- dbs-annual-report-2024 p.22 [marker] (score: -6.8893)
- dbs-annual-report-2022  [marker] (score: -7.0523)
- dbs-annual-report-2024 p.22 [marker] (score: -7.0675)
- dbs-annual-report-2022 p.63 [marker] (score: -7.0783)

**Latency**: 7901.34 ms

---

## Q3. Calculate the Operating Efficiency Ratio (Opex ÷ Operating Income) for the last 3 fiscal years, showing the working.

**Answer**

LLM error: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-20b` in organization `org_01k6cv721nexabx65q4bdbe28e` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199961, Requested 1314. Please try again in 9m10.8s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}


**Citations**

- dbs-annual-report-2022  [marker] (score: -5.0380)
- 4Q24_performance_summary p.34 [marker] (score: -5.4761)
- 4Q24_performance_summary p.28 [marker] (score: -7.6121)
- 4Q24_performance_summary p.4 [marker] (score: -7.7103)
- 4Q24_performance_summary p.34 [marker] (score: -7.7183)

**Latency**: 5380.15 ms

---

## Summary

- P50: 7344.8 ms
- P95: 7845.7 ms