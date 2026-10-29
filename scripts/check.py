"""Every check a change must pass before it is pushed.

uv run python scripts/check.py [--skip-gitleaks]
"""

import shutil
import subprocess
import sys

CHECKS = [
    ("lint", ["ruff", "check", "."]),
    ("format", ["ruff", "format", "--check", "."]),
    ("tests", ["pytest", "-q"]),
    ("dependencies", ["pip-audit", "--progress-spinner", "off"]),
]
GITLEAKS = ("secrets", ["gitleaks", "git", "--redact", "--no-banner", "."])


def main() -> int:
    checks = list(CHECKS)
    if "--skip-gitleaks" not in sys.argv:
        checks.append(GITLEAKS)
    failed = []
    for name, cmd in checks:
        exe = shutil.which(cmd[0])
        if exe is None:
            print(f"[{name}] {cmd[0]} not found", flush=True)
            failed.append(name)
            continue
        print(f"[{name}] {' '.join(cmd)}", flush=True)
        if subprocess.run([exe, *cmd[1:]], check=False).returncode != 0:  # noqa: S603
            failed.append(name)
    print("FAILED: " + ", ".join(failed) if failed else "all checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
