# triton-kernel-study

One hand-written Triton kernel, measured against the framework's
eager kernel and against the compiler's own generated kernel, on
one named GPU, under hypotheses frozen before the first run.

## The question

Writing the kernel yourself is widely assumed to be the fast
option by default. This study tests that assumption on fused
row-wise softmax (float32) against two baselines on the same
inputs: torch.softmax eager, and torch.compile. The full
protocol, the frozen verdict rules, and the named threats are in
PREREGISTRATION.md, which was this repository's first content
commit.

## Results

Measured on BlackBox, NVIDIA GeForce RTX 5050 Laptop GPU (8 GB),
torch 2.11.0+cu128, triton 3.9.0. Full tables in RESULTS.md; raw
record in results/raw_runs.json.

Verdicts under the frozen rules:

- H1 REFUTED. Hand-written does not beat eager by 1.30x. At the
  verdict size (16384 columns) the triton kernel reaches
  337.4 GB/s against eager's 338.6 GB/s, a ratio of 0.996. Both
  sit at the device's memory wall, and at the wall there is no
  1.30x to be had.
- H2 CONFIRMED. Triton lands inside the parity band against
  torch.compile: ratio 1.044 (band 0.85 to 1.18). The compiler
  arm is the slowest of the three at the large sizes.
- H3 CONFIRMED. The triton kernel matches the eager reference
  to at most 7.5e-09 at every size, far inside the frozen 1e-5
  tolerance.

Effective bandwidth (GB/s) by size:

| Cols | Eager | Compiled | Triton |
| --- | --- | --- | --- |
| 1024 | 700.9 | 522.7 | 709.0 |
| 2048 | 309.6 | 271.2 | 322.4 |
| 4096 | 274.5 | 291.6 | 326.8 |
| 8192 | 312.1 | 301.7 | 340.0 |
| 16384 | 338.6 | 323.0 | 337.4 |

The 1024-column row is partly cache-resident (33.5 MB moved) and
its figures are cache-assisted, not DRAM bandwidth; the frozen
protocol judges only the 16384-column size. Grade in AUDIT.md:
6,700 of 10,000, deductions itemized.

## Why this study exists

Every team adopting a kernel library, a compiler, or a
hand-written kernel is betting performance on folklore it has
not measured: that hand-written beats library, that the compiler
has caught up, that the eager call is slow. On named hardware,
with a frozen protocol, those bets become three numbers and
three verdicts. The societal angle is plain: small teams cannot
afford to spend engineering weeks chasing speed that a
measurement would have shown was never there, and cannot afford
to leave real speed unclaimed either. Measuring one kernel
honestly is how the decision stops being fashion.

## Layout

- PREREGISTRATION.md: frozen protocol and verdict rules.
- study/: the kernel and the benchmark runner.
- scripts/make_results.py: verdicts computed from the raw record.
- results/: the raw timed record and the computed summary.
- RESULTS.md, DECISIONS.md, METHODS.md, AUDIT.md, ROADMAP.md.

## Licence

Apache-2.0. See LICENSE.
