"""Small, dependency-free runtime preflight used by setup.ps1.

It deliberately performs no installation. The PowerShell wrapper owns Windows
services and WSL commands; this helper is useful for CI or a future launcher to
validate the resolved layout without guessing user-specific paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--reranker-script", type=Path, required=True)
    parser.add_argument("--report-script", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = {
        "project_root": str(args.project_root.resolve()),
        "runtime_dir": str(args.runtime_dir.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "requirements": str(args.requirements.resolve()),
        "reranker_script": str(args.reranker_script.resolve()),
        "report_script": str(args.report_script.resolve()),
        "requirements_present": args.requirements.is_file(),
        "reranker_present": args.reranker_script.is_file(),
        "report_present": args.report_script.is_file(),
    }
    print(json.dumps(result, indent=2))
    return 0 if all(result[k] for k in ("requirements_present", "reranker_present", "report_present")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
