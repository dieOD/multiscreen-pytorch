"""Helpers for torch.compile setup, especially the Windows MSVC dance."""

from __future__ import annotations

import glob
import os
import subprocess
import sys
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


def _find_vcvarsall() -> Path | None:
    """Locate vcvarsall.bat in the installed VS BuildTools."""
    bases = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio"),
        Path(r"C:\Program Files\Microsoft Visual Studio"),
    ]
    for base in bases:
        if not base.exists():
            continue
        for vs in sorted(base.iterdir(), reverse=True):
            candidate = vs / "BuildTools" / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
            if candidate.exists():
                return candidate
    return None


def load_vcvars_env() -> bool:
    """Load the full MSVC build environment into ``os.environ``.

    Runs ``vcvarsall.bat x64 && set`` in a subshell and copies every exported
    variable back. This sets INCLUDE, LIB, LIBPATH, PATH, etc. — the variables
    inductor needs when it invokes ``cl`` to compile the C++ kernel shims.

    Without this, ``torch.compile`` on Windows can fail with
    ``fatal error C1083: include file 'omp.h': No such file or directory``.

    Returns True on success, False on non-Windows / vcvarsall missing / failure.
    Idempotent — if ``VSCMD_VER`` is already set we assume the env is loaded.
    """
    if sys.platform != "win32":
        return False
    if os.environ.get("VSCMD_VER"):
        return True  # already loaded

    vcvarsall = _find_vcvarsall()
    if vcvarsall is None:
        return False

    try:
        result = subprocess.run(
            f'"{vcvarsall}" x64 >nul && set',
            shell=True, capture_output=True, text=True, check=False,
        )
    except OSError:
        return False

    if result.returncode != 0:
        return False

    for line in result.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ[k] = v
    return True


def setup_compile_env() -> str | None:
    """Auto-detect MSVC and set up the full build environment on Windows.

    On Windows: runs ``load_vcvars_env()`` (populates INCLUDE/LIB/PATH/...),
    then sets ``CC`` to cl.exe if unset. Returns the path to cl.exe.

    On Linux: does nothing and returns the existing CC (or None). Safe to
    call unconditionally.
    """
    if sys.platform == "win32":
        load_vcvars_env()
    if os.environ.get("CC"):
        return os.environ["CC"]
    cl_path = find_msvc_cl()
    if cl_path:
        os.environ["CC"] = cl_path
    return cl_path
