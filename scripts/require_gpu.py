#!/usr/bin/env python3
"""Refuse to start a session that would run on CPU.

Round 4 sessions 1 and 2 ran to the ten-hour wall on a Kaggle notebook whose
accelerator was never attached. ``torch`` was the CPU build, fp16 matmuls fell
back to software, and a caption that costs 7.8 s on two T4s took 724 s -- 93x
slower. Each session produced about fifty usable captions out of five hundred
before SIGTERM, the plan's other jobs never started, and twenty hours of quota
bought nothing.

``probe_host.py`` already reported the problem. The runner only echoed a
warning and carried on, which is the whole lesson: for an unattended
ten-hour session a missing GPU is not a warning.

Exit codes: 0 usable host, 3 no GPU, 4 torch missing.
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - any import failure is fatal here
        print(f"[fatal] torch did not import: {exc}", file=sys.stderr)
        return 4

    if not torch.cuda.is_available():
        print(
            "[fatal] no CUDA device visible; this session would run on CPU "
            "and finish roughly one job in ten.\n"
            f"        torch {torch.__version__}\n"
            "        Kaggle: Sidebar -> Accelerator = GPU T4 x2, then "
            "restart the session.\n"
            "        Set ALLOW_CPU=1 to override.",
            file=sys.stderr,
        )
        return 3

    names = ", ".join(torch.cuda.get_device_name(i)
                      for i in range(torch.cuda.device_count()))
    print(f"[probe] {torch.cuda.device_count()} GPU(s): {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
