#!/usr/bin/env python3
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_PATH = ROOT_DIR / "reports" / "ga_release_gate_report.json"
STRICT_GA_CHECKLIST_PATH = ROOT_DIR / "docs" / "STRICT_GA_CHECKLIST.md"


@dataclass
class GateResult:
    name: str
    ok: bool
    command: Optional[str] = None
    summary: str = ""
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _python_bin() -> str:
    candidates = [
        ROOT_DIR / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return sys.executable or "python3"


def parse_checklist_verdict(text: str) -> str:
    match = re.search(r"Current verdict:\s*`([^`]+)`", text)
    if not match:
        return "UNKNOWN"
    return match.group(1).strip()


def parse_checklist_recommendation(text: str) -> Dict[str, str]:
    recommendations: Dict[str, str] = {}
    for label, value in re.findall(r"-\s*(safe for [^:]+):\s*(yes|no)", text, flags=re.IGNORECASE):
        recommendations[label.strip().lower()] = value.strip().lower()
    return recommendations


def read_strict_checklist(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path),
        "verdict": parse_checklist_verdict(text),
        "recommendations": parse_checklist_recommendation(text),
    }


def run_command(command: List[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def command_gate(name: str, command: List[str], *, cwd: Path, success_summary: str) -> GateResult:
    completed = run_command(command, cwd=cwd)
    ok = completed.returncode == 0
    summary = success_summary if ok else (completed.stderr.strip() or completed.stdout.strip() or "command failed")
    return GateResult(
        name=name,
        ok=ok,
        command=shlex.join(command),
        summary=summary.splitlines()[-1] if summary else "",
        details={
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout.strip().splitlines()[-20:],
            "stderr_tail": completed.stderr.strip().splitlines()[-20:],
        },
    )


def init_gate(
    *,
    python_bin: str,
    cwd: Path,
    corpus_root: Path,
    spring_version: str,
    micronaut_version: str,
) -> GateResult:
    command = [
        python_bin,
        "main.py",
        "init",
        "--mode",
        "trusted",
        "--corpus-root",
        str(corpus_root),
        "--spring-version",
        spring_version,
        "--micronaut-version",
        micronaut_version,
    ]
    completed = run_command(command, cwd=cwd)
    ok = completed.returncode == 0
    details: Dict[str, Any] = {
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout.strip().splitlines()[-30:],
        "stderr_tail": completed.stderr.strip().splitlines()[-20:],
    }

    report_path = corpus_root / "validated_patterns" / "release" / "chroma_audit_report.json"
    smoke_path = corpus_root / "validated_patterns" / "release" / "kb_smoke_report.json"
    if ok and report_path.exists():
        try:
            details["chroma_audit_report"] = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            details["chroma_audit_report_error"] = str(exc)
            ok = False
    if ok and smoke_path.exists():
        try:
            details["kb_smoke_report"] = json.loads(smoke_path.read_text(encoding="utf-8"))
        except Exception as exc:
            details["kb_smoke_report_error"] = str(exc)
            ok = False

    audit = details.get("chroma_audit_report") or {}
    smoke = details.get("kb_smoke_report") or {}
    if ok and audit:
        ok = bool(audit.get("distribution_ready")) and audit.get("trust_level") in {"high", "medium"}
    if ok and smoke:
        ok = bool(smoke.get("ok"))

    summary = "trusted init, KB smoke, and Chroma audit passed"
    if not ok:
        summary = "trusted init path is not release-clean"

    return GateResult(
        name="trusted_init_and_runtime_audit",
        ok=ok,
        command=shlex.join(command),
        summary=summary,
        details=details,
    )


def build_report(
    *,
    checklist: Dict[str, Any],
    gates: List[GateResult],
    spring_version: str,
    micronaut_version: str,
    release_tier: str = "ga",
) -> Dict[str, Any]:
    technical_pass = all(gate.ok for gate in gates)
    checklist_verdict = str(checklist.get("verdict") or "UNKNOWN")
    recommendations = dict(checklist.get("recommendations") or {})
    candidate_recommended = (
        str(recommendations.get("safe for pilot migrations with engineering review") or "").strip().lower() == "yes"
    )
    ga_ready = technical_pass and checklist_verdict == "GA READY"
    release_candidate_ready = technical_pass and candidate_recommended
    normalized_tier = str(release_tier or "ga").strip().lower()
    release_decision = "do_not_ship_as_ga"
    if normalized_tier == "candidate":
        release_decision = "ship_release_candidate" if release_candidate_ready else "do_not_ship_candidate"
    elif ga_ready:
        release_decision = "ship"
    return {
        "schema_version": 1,
        "spring_version": spring_version,
        "micronaut_version": micronaut_version,
        "release_tier": normalized_tier,
        "technical_gate_passed": technical_pass,
        "strict_checklist_verdict": checklist_verdict,
        "ga_ready": ga_ready,
        "release_candidate_ready": release_candidate_ready,
        "release_decision": release_decision,
        "checklist": checklist,
        "gates": [gate.to_dict() for gate in gates],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Spring2Naut GA release gate.")
    parser.add_argument("--spring-version", default="3.4.5")
    parser.add_argument("--micronaut-version", default="4.10.8")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--release-tier", choices=("ga", "candidate"), default="ga")
    args = parser.parse_args(argv)

    python_bin = _python_bin()
    checklist = read_strict_checklist(STRICT_GA_CHECKLIST_PATH)

    with tempfile.TemporaryDirectory(prefix="spring2naut-ga-gate-") as tmpdir:
        corpus_root = Path(tmpdir) / "corpus"
        gates = [
            command_gate(
                "fast_regression_suite",
                [python_bin, "scripts/run_regression_suite.py", "--tier", "fast"],
                cwd=ROOT_DIR,
                success_summary="fast regression suite passed",
            ),
            command_gate(
                "corpus_regression_suite",
                [python_bin, "scripts/run_regression_suite.py", "--tier", "corpus"],
                cwd=ROOT_DIR,
                success_summary="corpus regression suite passed",
            ),
            init_gate(
                python_bin=python_bin,
                cwd=ROOT_DIR,
                corpus_root=corpus_root,
                spring_version=args.spring_version,
                micronaut_version=args.micronaut_version,
            ),
        ]

    report = build_report(
        checklist=checklist,
        gates=gates,
        spring_version=args.spring_version,
        micronaut_version=args.micronaut_version,
        release_tier=args.release_tier,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 58)
    print("GA RELEASE GATE")
    print("=" * 58)
    print(f"Targeting: Spring {args.spring_version} -> Micronaut {args.micronaut_version}")
    for gate in gates:
        status = "PASS" if gate.ok else "FAIL"
        print(f"{status}: {gate.name} - {gate.summary}")
    print(f"Release Tier: {report['release_tier']}")
    print(f"Strict Checklist Verdict: {report['strict_checklist_verdict']}")
    print(f"Release Candidate Ready: {report['release_candidate_ready']}")
    print(f"GA Ready: {report['ga_ready']}")
    print(f"Release Decision: {report['release_decision']}")
    print(f"Report: {report_path}")

    if report["release_tier"] == "candidate":
        return 0 if report["release_candidate_ready"] else 1
    return 0 if report["ga_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
