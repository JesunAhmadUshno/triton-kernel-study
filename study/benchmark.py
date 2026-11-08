"""Benchmark runner for triton-kernel-study.

Implements PREREGISTRATION.md exactly: rows 4096, column ladder
1024, 2048, 4096, 8192, 16384, float32, one input tensor per size
drawn on the device under torch.manual_seed(20261108) in ladder
order and shared by all three arms; per arm per size one
correctness call against the eager reference (max absolute error,
frozen tolerance 1e-5), then 5 warmup calls, then 25 timed calls
bracketed by CUDA events; effective bandwidth from the median
time as 2 * rows * cols * 4 bytes over time.

Arms: TRITON (study/softmax_triton.py), EAGER (torch.softmax),
COMPILED (torch.compile of the same expression, default mode;
its warmup wall time, compilation included, is recorded as
context and never timed as a run).

Run on BlackBox (RTX 5050 Laptop GPU, 8 GB) in the recorded
environment. Writes results/raw_runs.json at the repo root.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent))
from softmax_triton import block_for, run_triton, warps_for  # noqa: E402

ROWS = 4096
COLS_LADDER = [1024, 2048, 4096, 8192, 16384]
WARMUPS = 5
TIMED = 25
SEED = 20261108
TOL = 1e-5


def eager_fn(x):
    return torch.softmax(x, dim=-1)


def gpu_state() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=clocks.sm,clocks.mem,utilization.gpu,"
             "temperature.gpu,power.draw",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        return {"nvidia_smi": out.stdout.strip()}
    except Exception as exc:  # context only, never fatal
        return {"nvidia_smi_error": repr(exc)}


def time_calls(fn, warmups: int, timed: int) -> list[float]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(timed):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def arm_stats(times_ms: list[float], rows: int, cols: int) -> dict:
    arr = np.array(times_ms)
    med = float(np.median(arr))
    bytes_moved = 2 * rows * cols * 4
    return {
        "times_ms": [float(t) for t in arr],
        "time_ms_median": med,
        "time_ms_q1": float(np.percentile(arr, 25)),
        "time_ms_q3": float(np.percentile(arr, 75)),
        "bandwidth_gb_s_median": bytes_moved / (med / 1000.0) / 1e9,
        "bytes_moved": bytes_moved,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available on this machine")
    compiled_fn = torch.compile(eager_fn)
    state_before = gpu_state()
    torch.manual_seed(SEED)
    results = {}
    for cols in COLS_LADDER:
        x = torch.randn((ROWS, cols), device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        ref = eager_fn(x)
        torch.cuda.synchronize()
        block = block_for(cols)

        arms = {}

        def run_eager():
            return eager_fn(x)

        def run_compiled():
            return compiled_fn(x)

        def run_tr():
            return run_triton(x, out)

        plan = (("eager", run_eager),
                ("compiled", run_compiled),
                ("triton", run_tr))
        for name, run in plan:
            t0 = time.perf_counter()
            handle = run()
            torch.cuda.synchronize()
            warm_wall = time.perf_counter() - t0
            result_tensor = out if name == "triton" else handle
            if name == "eager":
                max_err = 0.0  # the eager arm is the reference
            else:
                max_err = float((result_tensor - ref).abs().max().item())
            times = time_calls(run, WARMUPS, TIMED)
            record = {
                "max_abs_err_vs_eager": max_err,
                "within_tolerance": bool(max_err <= TOL),
                "first_call_wall_s": warm_wall,
                **arm_stats(times, ROWS, cols),
            }
            if name == "triton":
                record["block"] = block
                record["num_warps"] = warps_for(block)
                try:
                    record["registers_per_thread"] = int(handle.n_regs)
                    record["shared_bytes_per_block"] = int(
                        handle.metadata.shared)
                except Exception:
                    pass
            arms[name] = record
            print(cols, name, "median ms",
                  round(record["time_ms_median"], 4),
                  "GB/s", round(record["bandwidth_gb_s_median"], 1),
                  "err", max_err)
        results[str(cols)] = arms
    payload = {
        "study": "triton-kernel-study: fused softmax, triton vs eager "
                 "vs compiled",
        "device": torch.cuda.get_device_name(0),
        "machine": platform.node(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "rows": ROWS,
        "cols_ladder": COLS_LADDER,
        "verdict_cols": COLS_LADDER[-1],
        "warmup_calls": WARMUPS,
        "timed_calls": TIMED,
        "seed": SEED,
        "tolerance_max_abs_err": TOL,
        "gpu_state_before": state_before,
        "gpu_state_after": gpu_state(),
        "results_by_size": results,
    }
    root = Path(__file__).resolve().parent.parent
    out_path = root / "results" / "raw_runs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
