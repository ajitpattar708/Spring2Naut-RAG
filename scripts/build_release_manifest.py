#!/usr/bin/env python3
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT_DIR = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_artifacts(dist_dir: Path) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    for path in sorted(p for p in dist_dir.iterdir() if p.is_file()):
        if path.name in {"SHA256SUMS", "release_manifest.json"}:
            continue
        artifacts.append(
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return artifacts


def _write_checksums(path: Path, artifacts: List[Dict[str, Any]]) -> None:
    lines = [f"{artifact['sha256']}  {artifact['filename']}" for artifact in artifacts]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(dist_dir: Path, report_path: Path) -> Dict[str, Any]:
    artifacts = _build_artifacts(dist_dir)
    if not artifacts:
        raise ValueError(f"No release artifacts found in {dist_dir}")

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }

    if report_path.exists():
        gate_report = _load_json(report_path)
        manifest["ga_gate_report"] = {
            "path": str(report_path.relative_to(ROOT_DIR)) if report_path.is_relative_to(ROOT_DIR) else str(report_path),
            "release_decision": gate_report.get("release_decision"),
            "ga_ready": gate_report.get("ga_ready"),
            "technical_gate_passed": gate_report.get("technical_gate_passed"),
            "strict_checklist_verdict": gate_report.get("strict_checklist_verdict"),
            "spring_version": gate_report.get("spring_version"),
            "micronaut_version": gate_report.get("micronaut_version"),
        }

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate release manifest and SHA256 checksums for built artifacts.")
    parser.add_argument("--dist-dir", default="dist")
    parser.add_argument("--report", default="reports/ga_release_gate_report.json")
    parser.add_argument("--output-manifest", default="dist/release_manifest.json")
    parser.add_argument("--output-checksums", default="dist/SHA256SUMS")
    args = parser.parse_args()

    dist_dir = Path(args.dist_dir)
    report_path = Path(args.report)
    manifest_path = Path(args.output_manifest)
    checksums_path = Path(args.output_checksums)

    manifest = build_manifest(dist_dir, report_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    checksums_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_checksums(checksums_path, manifest["artifacts"])

    print(f"Release manifest: {manifest_path}")
    print(f"Release checksums: {checksums_path}")
    print(f"Artifacts covered: {manifest['artifact_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
