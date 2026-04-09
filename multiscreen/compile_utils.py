"""Helpers for torch.compile setup, especially the Windows MSVC dance."""

from __future__ import annotations

import glob
import os
from pathlib import Path


def find_msvc_cl() -> str | None:
    """Find MSVC cl.exe for Triton/torch.compile on Windows.

    Returns the value of the CC environment variable if set, otherwise probes
    Visual Studio 2022 BuildTools install paths. Returns None if nothing is found.
    """
    if os.environ.get("CC"):
        return os.environ["CC"]
    bases = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio"),
        Path(r"C:\Program Files\Microsoft Visual Studio"),
    ]
    for base in bases:
        if not base.exists():
            continue
        # Prefer newer VS versions (sorted reverse)
        for vs in sorted(base.iterdir(), reverse=True):
            pattern = str(
                vs / "BuildTools" / "VC" / "Tools" / "MSVC"
                / "*" / "bin" / "Hostx64" / "x64" / "cl.exe"
            )
            matches = sorted(glob.glob(pattern))
            if matches:
                return matches[-1]
    return None


def setup_compile_env() -> str | None:
    """Auto-detect MSVC and set CC env var if not already set.

    Returns the path to cl.exe (or None on non-Windows / not found).
    Safe to call on Linux — it just returns None.
    """
    if os.environ.get("CC"):
        return os.environ["CC"]
    cl_path = find_msvc_cl()
    if cl_path:
        os.environ["CC"] = cl_path
    return cl_path
