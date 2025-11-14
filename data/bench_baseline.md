# Baseline Benchmark Report

**Pipeline**: Hybrid Search (BM25 + Vector + RRF + Rerank) -> Single LLM


---

## Q1. Report the Gross Margin (or Net Interest Margin, if a bank) over the last 5 quarters, with values.

**Answer**

**Net Interest Margin – last five quarters**

| Quarter | Net Interest Margin |
|---------|---------------------|
| Q4 2024 | 2.05 % |
| Q3 2024 | *value not disclosed in the provided context* |
| Q2 2024 | *value not disclosed in the provided context* |
| Q1 2024 | *value not disclosed in the provided context* |
| Q4 2023 | *value not disclosed in the provided context* |

**Citations**

- “net interest margin was 2.05% in the fourth quarter” – 4Q 2024 performance summary.  
- “2nd Half 2024: 2.13” – 4Q 2024 performance summary (table #4 row 6).  
- “1st Half 2024: 2.14” – 4Q 2024 performance summary (table #4 row 6).  
- “Year 2024: 2.13” – 4Q 2024 performance summary (table #4 row 6).


**Citations**

- dbs-annual-report-2022  [marker] (score: 0.6558)
- 2Q24_performance_summary p.9 [marker] (score: -0.5437)
- dbs-annual-report-2022 p.96 [marker] (score: -0.7675)
- dbs-annual-report-2022 p.96 [marker] (score: -0.7991)
- dbs-annual-report-2023 p.95 [marker] (score: -1.0668)

**Latency**: 4297.13 ms

---

## Q2. Show Operating Expenses for the last 3 fiscal years, year-on-year comparison.

**Answer**

LLM error: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-20b` in organization `org_01k6cv721nexabx65q4bdbe28e` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199995, Requested 898. Please try again in 6m25.776s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}


**Citations**

- dbs-annual-report-2022  [marker] (score: -6.7801)
- dbs-annual-report-2024 p.22 [marker] (score: -6.8893)
- dbs-annual-report-2022  [marker] (score: -7.0523)
- dbs-annual-report-2024 p.22 [marker] (score: -7.0675)
- dbs-annual-report-2022 p.63 [marker] (score: -7.0783)

**Latency**: 2432.75 ms

---

## Q3. Calculate the Operating Efficiency Ratio (Opex ÷ Operating Income) for the last 3 fiscal years, showing the working.

**Answer**

LLM error: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-20b` in organization `org_01k6cv721nexabx65q4bdbe28e` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199990, Requested 1467. Please try again in 10m29.424s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}


**Citations**

- dbs-annual-report-2022  [marker] (score: -5.0380)
- 4Q24_performance_summary p.34 [marker] (score: -5.4761)
- 4Q24_performance_summary p.28 [marker] (score: -7.6121)
- 4Q24_performance_summary p.4 [marker] (score: -7.7103)
- 4Q24_performance_summary p.34 [marker] (score: -7.7183)

**Latency**: 2166.13 ms

---

## Summary

- P50: 2432.8 ms
- P95: 4110.7 ms