# METHODS

## Protocol source

PREREGISTRATION.md is the protocol. It was the first content
commit in this repository (70afc00), before the harness and
before any measured run.

## Environment of record

- Machine: BlackBox (Windows 11 laptop), SSH over the fleet
  Tailscale proxy route.
- GPU: NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB VRAM.
- Python environment on BlackBox: the pre-existing user-space
  venv at C:\Users\jesun\b4-pytorch-depth\.venv (Python 3.14.7,
  torch 2.11.0+cu128, triton 3.9.0). Nothing was installed for
  this study.

## How the run was produced

1. study/ and scripts/ were copied to
   C:\Users\jesun\triton-kernel-study\ on BlackBox.
2. From that folder, in the venv above:
   python study\benchmark.py
   which writes results\raw_runs.json.
3. The raw file was copied back into this repository's
   results/ folder unchanged.
4. python scripts/make_results.py computes the verdicts from
   the raw file and writes results/summary.json and RESULTS.md.
   Verdicts are never computed by hand.

## Reproduction of the analysis (no GPU needed)

python -m pytest -q

The test suite recomputes the summary from the committed raw
record and checks it matches, checks every timed cell holds
exactly 25 calls, and checks the frozen lines of the
preregistration are intact.

## File map

- PREREGISTRATION.md: the frozen protocol and verdict rules.
- study/softmax_triton.py: the hand-written kernel (TRITON arm).
- study/benchmark.py: the runner that produced the raw record.
- scripts/make_results.py: verdicts and RESULTS.md generation.
- results/raw_runs.json: the raw timed record from the machine.
- results/summary.json: the computed verdicts and ratios.
- RESULTS.md: the published tables and verdicts.
- DECISIONS.md: choices made before and during the build.
- AUDIT.md: the self-audit and grade.
