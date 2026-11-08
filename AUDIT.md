# AUDIT

Self-audit of this repository against the standing project standard:
one real research question, hypotheses frozen before any measured run,
measured results only, verdicts published either way, receipts on every
claim.

## Claim audit

Every headline figure in README.md and RESULTS.md is computed by
scripts/make_results.py from results/raw_runs.json, and was
independently recounted from the raw times_ms arrays by separate code
before publication. The recount matched on every figure: medians and
effective bandwidths for all 15 arm-by-size cells, the verdict-size
ratios (triton over eager 0.9963, triton over compiled 1.0444), and
the tolerance flags (triton within 1e-5 at all five sizes, worst
error 7.45e-09). Every timed cell holds exactly 25 calls, as frozen.

Jev claim_check (gate record): model jev-1.13.0, noul 0.27, route
REVIEW (bands: clear at 0.85 or above, block at 0.10 or below, review
between), usage 964 input and 20 output tokens. Spec and raw result:
~/workspace/tools/jev/triton-kernel-study-claim-spec-2026-11-08.json
and triton-kernel-study-claim-results-2026-11-08.json; the record is
also committed at results/jev-claim-check.json. On REVIEW the figures
were hand-verified against the tables, as described above, and the
verdicts stand on that verification.

Protocol integrity: PREREGISTRATION.md is the first content commit
(70afc00), before the harness commit (dd600c8) and before any measured
run. The benchmark ran once, as registered. The only pre-run contact
with the machine beyond the gate check was the disclosed
torch.compile feasibility probe (DECISIONS.md D3), which recorded no
timing and kept no numbers.

## Reading of the result

The refuted H1 is the study's centre, and it is a bandwidth story,
not a kernel-quality story. At the verdict size the eager vendor
kernel already runs at 338.6 GB/s on a device whose own coalesced
copy ceiling measured 305.2 GB/s in gpu-first-principles-lab (M1,
2026-11-07, same machine): both kernels sit at the memory wall, and
at the wall there is no 1.30x to be had. The hand-written kernel
matches the wall (337.4 GB/s) and beats the compiler arm, which
Inductor itself reported it had split rather than fused (301.7 to
323.0 GB/s at the large sizes). At 4096 and 8192 columns the triton
arm is the fastest of the three; at the verdict size it ties eager.
Two cells carry a caveat the tables show but a reader could miss:
at 1024 columns the moved bytes (33.5 MB) are partly cache-resident,
so the 700 GB/s figures there are cache-assisted and are not DRAM
bandwidth; the frozen protocol judges only the verdict size, where
the working set is far beyond cache. The pre-run GPU snapshot
records near-idle clocks (180 MHz SM); timed cells at the verdict
size have tight interquartile ranges (1.5872 to 1.5988 ms), so the
clock ramp does not move the verdicts, but no clock was locked and
the threat stands named in the preregistration.

## Grade

6,700 of 10,000. Deductions, itemized:

- Single machine, single laptop GPU, clocks not locked: minus 800.
- One operation and one shape family (rows fixed at 4096, float32
  only): minus 700.
- One pass of the protocol; no repeated full run to price run to
  run variance: minus 500.
- The 1024-column cell is partly cache-resident, a weaker cell in
  the ladder (reported openly, not scored): minus 400.
- The compiled arm is bound to one Inductor version, whose
  per-size compile behaviour was observed, not controlled:
  minus 400.
- The study shows eager at the bandwidth wall but cannot fully
  separate an optimal eager kernel from an ordinary hand-written
  one beyond that argument: minus 300.
- Jev claim_check landed at REVIEW, not clear: minus 200.
