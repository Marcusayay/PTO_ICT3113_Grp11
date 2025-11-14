# Baseline Benchmark Report

**Pipeline**: Hybrid Search (BM25 + Vector + RRF + Rerank) -> Single LLM


---

## Q1. Report the Gross Margin (or Net Interest Margin, if a bank) over the last 5 quarters, with values.

**Answer**

LLM error: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-20b` in organization `org_01k6cv721nexabx65q4bdbe28e` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199708, Requested 1214. Please try again in 6m38.304s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}


**Citations**

- dbs-annual-report-2022  [marker] (score: 0.6558)
- 2Q24_performance_summary p.9 [marker] (score: -0.5437)
- dbs-annual-report-2022 p.96 [marker] (score: -0.7675)
- dbs-annual-report-2022 p.96 [marker] (score: -0.7991)
- dbs-annual-report-2023 p.95 [marker] (score: -1.0668)

**Latency**: 5526.66 ms

---

## Q2. Show Operating Expenses for the last 3 fiscal years, year-on-year comparison.

**Answer**

LLM error: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-20b` in organization `org_01k6cv721nexabx65q4bdbe28e` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199694, Requested 991. Please try again in 4m55.92s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}


**Citations**

- dbs-annual-report-2022  [marker] (score: -6.7801)
- dbs-annual-report-2024 p.22 [marker] (score: -6.8893)
- dbs-annual-report-2022  [marker] (score: -7.0523)
- dbs-annual-report-2024 p.22 [marker] (score: -7.0675)
- dbs-annual-report-2022 p.63 [marker] (score: -7.0783)

**Latency**: 6339.09 ms

---

## Q3. Calculate the Operating Efficiency Ratio (Opex ÷ Operating Income) for the last 3 fiscal years, showing the working.

**Answer**

LLM error: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-20b` in organization `org_01k6cv721nexabx65q4bdbe28e` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199681, Requested 1314. Please try again in 7m9.84s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}


**Citations**

- dbs-annual-report-2022  [marker] (score: -5.0380)
- 4Q24_performance_summary p.34 [marker] (score: -5.4761)
- 4Q24_performance_summary p.28 [marker] (score: -7.6121)
- 4Q24_performance_summary p.4 [marker] (score: -7.7103)
- 4Q24_performance_summary p.34 [marker] (score: -7.7183)

**Latency**: 5343.19 ms

---

## Summary

- P50: 5526.7 ms
- P95: 6257.8 ms