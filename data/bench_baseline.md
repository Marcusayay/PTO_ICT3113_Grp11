# Two-Mode RAG — Baseline Benchmark Report

**Pipeline**: Hybrid Search (BM25 + Vector) → RRF → Reranking → Single LLM Call


---

## Q1. Report the Gross Margin (or Net Interest Margin, if a bank) over the last 5 quarters, with values.

**Answer**

The structured rows provide only annual Net Interest Margin figures, not quarterly data. Therefore, quarterly values for the last five quarters cannot be derived from the information supplied.

**Available annual Net Interest Margin (NIM) from the structured rows**

| Year | Net Interest Margin |
|------|---------------------|
| 2020 | 1.62 |
| 2021 | 1.45 |
| 2022 | 1.75 |
| 2023 | 2.15 |
| 2024 | 2.13 |

No quarterly figures are present in the provided data.


**Citations**

- dbs-annual-report-2022 —  [marker] (score: 0.6558)
- 2Q24_performance_summary — p.9 [marker] (score: -0.5437)
- dbs-annual-report-2022 — p.96 [marker] (score: -0.7675)
- dbs-annual-report-2022 — p.96 [marker] (score: -0.7991)
- dbs-annual-report-2023 — p.95 [marker] (score: -1.0668)

**Latency**: 4785.21 ms

---

## Q2. Show Operating Expenses for the last 3 fiscal years, year-on-year comparison.

**Answer**

I’m sorry, but the structured data you provided only contains figures for the 2Q 2025 period. There are no entries for the last three fiscal years, so I cannot provide a year‑on‑year comparison of operating expenses.


**Citations**

- dbs-annual-report-2022 —  [marker] (score: -6.7801)
- dbs-annual-report-2024 — p.22 [marker] (score: -6.8893)
- dbs-annual-report-2022 —  [marker] (score: -7.0523)
- dbs-annual-report-2024 — p.22 [marker] (score: -7.0675)
- dbs-annual-report-2022 — p.63 [marker] (score: -7.0783)

**Latency**: 5484.42 ms

---

## Q3. Calculate the Operating Efficiency Ratio (Opex ÷ Operating Income) for the last 3 fiscal years, showing the working.

**Answer**

I’m sorry, but the structured rows only contain data for 2Q 2025.  There are no figures for the last three fiscal years, so the Operating Efficiency Ratio cannot be calculated from the information provided.


**Citations**

- dbs-annual-report-2022 —  [marker] (score: -5.0380)
- 4Q24_performance_summary — p.34 [marker] (score: -5.4761)
- 4Q24_performance_summary — p.28 [marker] (score: -7.6121)
- 4Q24_performance_summary — p.4 [marker] (score: -7.7103)
- 4Q24_performance_summary — p.34 [marker] (score: -7.7183)

**Latency**: 6184.98 ms

---

## Summary

- **Queries**: 3
- **P50 Latency**: 5484.4 ms
- **P95 Latency**: 6114.9 ms
- **Mean Latency**: 5484.9 ms
- **Total Time**: 16454.6 ms