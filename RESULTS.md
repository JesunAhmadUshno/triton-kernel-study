# RESULTS

Device of record: BlackBox, NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM). Environment: torch 2.11.0+cu128, triton 3.9.0, Python on the machine of record. Rows 4096, float32, 5 warmup calls then 25 timed calls per arm per size, CUDA events around the call only.

| Cols | Arm | Median ms (IQR) | GB/s | Max abs err |
| --- | --- | --- | --- | --- |
| 1024 | eager | 0.0479 (0.0424 to 0.0523) | 700.9 | 0 |
| 1024 | compiled | 0.0642 (0.0634 to 0.0650) | 522.7 | 1.12e-08 |
| 1024 | triton | 0.0473 (0.0470 to 0.0479) | 709.0 | 7.45e-09 |
| 2048 | eager | 0.2167 (0.2159 to 0.2187) | 309.6 | 0 |
| 2048 | compiled | 0.2475 (0.2467 to 0.2494) | 271.2 | 7.45e-09 |
| 2048 | triton | 0.2081 (0.2000 to 0.2140) | 322.4 | 7.45e-09 |
| 4096 | eager | 0.4889 (0.4858 to 0.4926) | 274.5 | 0 |
| 4096 | compiled | 0.4603 (0.4587 to 0.4622) | 291.6 | 3.73e-09 |
| 4096 | triton | 0.4108 (0.4090 to 0.4144) | 326.8 | 3.73e-09 |
| 8192 | eager | 0.8602 (0.8544 to 0.8647) | 312.1 | 0 |
| 8192 | compiled | 0.8898 (0.8878 to 0.8984) | 301.7 | 1.4e-09 |
| 8192 | triton | 0.7895 (0.7857 to 0.7936) | 340.0 | 1.4e-09 |
| 16384 | eager | 1.5855 (1.5821 to 1.5897) | 338.6 | 0 |
| 16384 | compiled | 1.6620 (1.6552 to 1.6691) | 323.0 | 9.31e-10 |
| 16384 | triton | 1.5914 (1.5872 to 1.5988) | 337.4 | 9.31e-10 |

Ratios at the verdict size (16384 columns): triton over eager bandwidth 0.996; triton over compiled bandwidth 1.044.

## Verdicts (frozen rules, PREREGISTRATION.md)

- H1 (triton beats eager by at least 1.30x at 16384): REFUTED.
- H2 (triton within 0.85 to 1.18 of compiled at 16384): CONFIRMED.
- H3 (triton within 1e-5 of the eager reference at every size): CONFIRMED.

Context, not scored: the compiled arm's first-call wall time per size (compilation included) was 1024 cols: 2.10 s, 2048 cols: 0.57 s, 4096 cols: 0.00 s, 8192 cols: 0.00 s, 16384 cols: 0.00 s.
