# Gap-closure locked-clock repeat: triton-kernel-study, 2026-11-17

Run of record: DECISIONS.md D7. Same frozen protocol (rows 4096,
fp32, 5 warmup + 25 timed calls per arm per size, full ladder) in a
run-copy, environment of record (b4 venv: torch 2.11.0+cu128,
triton 3.9.0). The one change: SM clocks locked at 2,000 MHz with
nvidia-smi -lgc before the run. The lock is confirmed in the fresh
output's gpu_state_after (SM clock 2002 MHz); the committed run's
clocks floated (recorded range 1,755 to 2,595 MHz). AC power.
Fresh output: results/raw_runs_gapclosure_2026-11-17.json. The
committed results were not touched and no verdict is re-issued.
Correctness: every arm within tolerance at every size, both passes.

## Decisive cell, 16,384 columns (medians)

| Arm | Committed ms | Fresh ms | Committed GB/s | Fresh GB/s |
| --- | --- | --- | --- | --- |
| eager | 1.5855 | 1.7707 | 338.6 | 303.2 |
| compiled | 1.6620 | 1.6841 | 323.0 | 318.8 |
| triton | 1.5914 | 2.0385 | 337.4 | 263.4 |

Bandwidth ratios at 16,384 columns: triton/eager 0.996 committed,
0.869 fresh; triton/compiled 1.044 committed, 0.826 fresh.

## Full ladder (median ms, committed -> fresh)

| Cols | eager | compiled | triton |
| --- | --- | --- | --- |
| 1024 | 0.0479 -> 0.0467 | 0.0642 -> 0.0746 | 0.0473 -> 0.0588 |
| 2048 | 0.2167 -> 0.2175 | 0.2475 -> 0.2388 | 0.2081 -> 0.2050 |
| 4096 | 0.4889 -> 0.6012 | 0.4603 -> 0.4647 | 0.4108 -> 0.4152 |
| 8192 | 0.8602 -> 0.9705 | 0.8898 -> 0.9041 | 0.7895 -> 1.0378 |
| 16384 | 1.5855 -> 1.7707 | 1.6620 -> 1.6841 | 1.5914 -> 2.0385 |

## Reading

Locking the clocks did not reproduce the committed 0.996 point
value: under the lock the triton arm slowed more than either
PyTorch arm at the two largest sizes, and the size-16,384 ordering
became compiled, eager, triton. The direction of the frozen H1
verdict (the hand-written kernel does not beat eager by the 1.30
line) is strengthened, not weakened: 0.869 sits further below the
line than 0.996. The committed run's clocks floated up to 2,595
MHz, above the 2,000 MHz lock, so the two passes ran in different
clock regimes by design; no mechanism for the triton arm's larger
drop is claimed here. A future version of the paper should report
the locked-clock cell beside the original.
