#!/usr/bin/env python3
"""Independently audit actual M2 artifacts on CPU and write a fresh report."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formal", action="store_true", help="Require the complete frozen 200-anchor registry")
    parser.add_argument("--sonic", type=Path, help="Enable independent official G1 CPU FK, nominal pose and sole verification")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a fresh file; old audit reports are never overwritten")
    from g1_embodiedskills.m2_audit import audit_manifests
    try:
        model = None
        if args.sonic:
            import torch
            torch.set_num_threads(1)
            from g1_embodiedskills.kinematics import G1Kinematics
            model = G1Kinematics(args.sonic)
        report = audit_manifests(args.manifests, model=model, formal=args.formal)
        code = 0
    except Exception as error:
        report = {"schema": "g1_embodiedskills.m2_independent_audit.v1", "completed": True, "passed": False,
            "type": type(error).__name__, "error": str(error), "manifests": [str(path.resolve()) for path in args.manifests]}
        code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": report["passed"], "report": str(args.output.resolve()), "error": report.get("error")}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
