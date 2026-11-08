# PREREGISTRATION

Frozen before any measured run. Amendments only by a dated entry in
DECISIONS.md that states the reason; verdict rules never change after
the first measured run of this study.

## Research question

On one named GPU, what does a hand-written fused kernel in Triton
actually deliver against the two ways the same operation is normally
run: the framework's eager kernel and the compiler-generated kernel
produced by torch.compile? The claim under test is the folklore that
writing the kernel yourself is automatically faster. This study
measures that claim on one operation with genuine memory traffic and
publishes the result either way.

## Operation and kernel choice

Fused row-wise softmax, float32. Softmax is memory-bound, fuses
naturally (one read of the row, the max and sum reductions held in
registers, one write of the row), and is the canonical first kernel
of the Triton literature, which makes the hand-written arm a fair
representative of what a practitioner writes first.

Arms:

- TRITON: the hand-written kernel in this repository
  (study/softmax_triton.py), one program per row, BLOCK the
  smallest power of two at or above the column count, two passes
  over the row in registers (max, then exp and sum).
- EAGER: torch.softmax(x, dim=-1), the framework's own kernel.
- COMPILED: torch.compile applied to a plain function returning
  torch.softmax(x, dim=-1), default mode. Compilation happens
  inside the warmup calls; compile wall time per size is recorded
  as context and is never part of a timed figure.

Frozen num_warps rule for the Triton arm: 4 warps when BLOCK is at
most 1024, 8 warps when BLOCK is at most 4096, 16 warps above that.

## Machine and environment of record

- Machine: BlackBox (Windows 11 laptop), reached over the fleet
  SSH route.
- GPU: NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB VRAM, named on
  every results table. Numbers describe this device only.
- Environment: the existing user-space virtual environment on
  BlackBox (Python 3.14.7, torch 2.11.0+cu128, triton 3.9.0).
  No installs, no system changes.

## Shapes and data

- Rows: 4096, fixed across the ladder, so the ladder varies row
  length only.
- Columns ladder: 1024, 2048, 4096, 8192, 16384.
- Verdict size: 16384 columns (the largest size).
- One input tensor per size, float32, standard normal, generated
  on the device under torch.manual_seed(20261108), drawn in ladder
  order. The same tensor feeds all three arms and the reference.
- Reference output: the EAGER arm's output.

## Timing protocol

- Per arm per size: 5 warmup calls, then 25 timed calls.
- Timing by CUDA events around the call only; host work and
  compilation are outside the timed region.
- Report per arm per size: median and interquartile range of the
  25 call times in milliseconds, and effective bandwidth in GB/s
  computed from the median time as bytes moved divided by time,
  where bytes moved is 2 times rows times columns times 4 bytes
  (one read, one write of the tensor). GFLOP/s is not reported:
  softmax is a reduction, not a flop-dense operation, and a flop
  figure would flatter it.

## Correctness protocol

- For the TRITON and COMPILED arms at every size on the ladder:
  max absolute error of the full output against the reference,
  computed before that size's timing is accepted.
- Frozen tolerance: max absolute error at or below 1e-5.
- A fast wrong kernel is a failed kernel: any size failing the
  tolerance is reported as failed in RESULTS.md, never dropped,
  and fails H3.

## Hypotheses and frozen verdict rules

- H1 (hand-written beats eager): at the verdict size, the TRITON
  arm's effective bandwidth is at least 1.30 times the EAGER arm's.
  CONFIRMED if the ratio of medians is at least 1.30; REFUTED
  below it.
- H2 (hand-written vs compiler parity): at the verdict size, the
  ratio of TRITON to COMPILED effective bandwidth lies inside the
  band 0.85 to 1.18. CONFIRMED inside the band; REFUTED outside
  it, with the direction reported. If the COMPILED arm cannot
  execute on the machine of record, H2 is INCONCLUSIVE and the
  compiler failure is published as the evidence.
- H3 (correctness holds): the TRITON arm meets the frozen
  tolerance at every size on the ladder. CONFIRMED if all five
  sizes pass; REFUTED if any size fails.

Verdicts are computed by scripts/make_results.py from
results/raw_runs.json, never by hand.

## Named threats to validity

Single machine and single GPU, a laptop part whose clocks move
with power state; float32 only; one kernel family (row-wise, one
program per row); rows fixed at 4096; the COMPILED arm depends on
the Inductor version inside the pinned torch build. Every claim in
this repository is limited to the named device and environment.
