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

Measured results are published in RESULTS.md with the raw record
in results/raw_runs.json. The headline table and verdicts are
added to this README in the results commit, once the run exists.
No result is stated here before then.

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
