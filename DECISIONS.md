# DECISIONS

## D1. The kernel is fused row-wise softmax

One real kernel, chosen before any run: fused softmax is
memory-bound, fuses cleanly (one read, reductions in registers,
one write), and is the canonical first hand-written kernel in the
Triton literature, so the hand-written arm represents what a
practitioner writes first rather than an exotic best case. A GEMM
was not chosen: this portfolio's GEMM line is already measured in
gpu-first-principles-lab (M3b and M3c, the second against cuBLAS),
and repeating it here would measure the same claim twice.

## D2. Machine and environment of record

BlackBox, Jesun's Windows 11 laptop (RTX 5050 Laptop GPU, 8 GB),
through the existing user-space virtual environment that the GPU
lab studies already ran in (Python 3.14.7, torch 2.11.0+cu128,
triton 3.9.0). No installs were needed and nothing on the laptop
changed system-wide. Another machine's numbers would be a
different study, so no substitute machine was considered when the
gate was checked: the machine answered, the GPU answered, the
study ran there.

## D3. torch.compile probed before the preregistration froze

Whether Inductor can execute at all on a Windows laptop is an
environment fact, not a measurement. A throwaway probe (a small
compiled softmax, correctness only, no timing recorded or kept)
confirmed the compiled arm can run on the machine of record
before the hypotheses were frozen. The INCONCLUSIVE contingency
for compiler failure stays in the preregistration anyway: a probe
is not a promise about the full ladder.

## D4. Bandwidth, not GFLOP/s

Softmax is a reduction; its cost is data movement. The scored
figure is effective bandwidth, bytes moved (one read plus one
write of the tensor) over median time. A GFLOP/s figure for
softmax would mostly measure how the counter flatters the
operation, so none is reported.

## D5. Rows fixed at 4096

The ladder varies row length only. Fixing the row count keeps
total work per size proportional to columns, keeps every size
large enough to be DRAM-bound rather than launch-bound, and
keeps the comparison about the kernel's handling of row length.

## D6. The correctness call is one extra untimed call

Per arm per size, correctness is computed from one call made
before the warmup calls. That call is therefore also a warmup in
effect (for the compiled arm it is where compilation happens;
its wall time is recorded as context). The frozen count of 5
warmup and 25 timed calls is unchanged and runs after it.

## D7. Gap-closure locked-clock repeat (2026-11-17)

The audit records that the scored pass ran once, with SM clocks
unlocked (the record's clocks floated between 1,755 and 2,595
MHz), so no repeat prices the variance. This entry registers
the repeat before it runs. On BlackBox (the machine of
record), study/benchmark.py is executed on 2026-11-17 in a
run-copy of the GitHub clone at
C:\Users\Jesun\gap-closure-2026-11-17\triton-kernel-study,
with SM clocks locked at 2,000 MHz (nvidia-smi -lgc 2000,2000;
the driver accepted the lock in this session) and the laptop
on AC power. Environment of record: Python 3.14.7, torch
2.11.0+cu128, triton 3.9.0, driver 577.05. The frozen protocol
is unchanged: rows 4096, float32, 5 warmup calls then 25 timed
calls per arm per size, the full size ladder. The fresh raw
output is filed in this repo as the NEW file
results/raw_runs_gapclosure_2026-11-17.json;
results/raw_runs.json and RESULTS.md are not edited and no
verdict is re-issued. The comparison of record is the ratio at
16,384 columns: committed triton over eager bandwidth 0.996
(triton over compiled 1.044).
