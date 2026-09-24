#!/usr/bin/env python3
"""Dry-run smoke test for run_full_production_pipeline.py."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    cmd = [
        sys.executable,
        str(project_root / "src" / "pipeline" / "run_full_production_pipeline.py"),
        "--dry-run",
        "--trials-image-standard",
        "2",
        "--trials-image-lossy",
        "2",
        "--trials-audio",
        "2",
        "--minutes-image-standard",
        "1",
        "--minutes-image-lossy",
        "1",
        "--minutes-audio",
        "1",
    ]
    subprocess.check_call(cmd)
    print("[+] Full production pipeline dry-run smoke test passed")


if __name__ == "__main__":
    main()

