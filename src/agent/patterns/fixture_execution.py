import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

from src.agent.patterns.fixture_packs import write_fixture_packs
from src.agent.patterns.fixture_registry import write_fixture_registry
from src.agent.patterns.repository import PatternCorpusRepository


REQUIRED_PACK_FILES = {"pack.json"}


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_fixture_execution(corpus_root: str = "corpus") -> Dict[str, object]:
    target_root = Path(corpus_root) / "validated_patterns" / "release" / "legacy_reviewed"
    registry_payload = _load_json(target_root / "fixture_registry.json")
    pack_index_payload = _load_json(target_root / "fixture_packs" / "index.json")

    requirement_map = {item["pattern_id"]: item for item in registry_payload.get("requirements", [])}
    covered_ids: Set[str] = set(pack_index_payload.get("covered_pattern_ids", []))
    pack_payloads = list(pack_index_payload.get("packs", []))

    issues: List[str] = []
    backlog_items: List[str] = []
    execution_items: List[Dict[str, object]] = []

    pack_by_pattern_id: Dict[str, Dict[str, object]] = {}
    for pack in pack_payloads:
        for pattern_id in pack.get("covered_pattern_ids", []):
            pack_by_pattern_id[pattern_id] = pack

    for pattern_id, requirement in requirement_map.items():
        priority = str(requirement.get("priority", "medium"))
        pack = pack_by_pattern_id.get(pattern_id)
        if not pack:
            message = f"Missing fixture pack coverage for {pattern_id}"
            if priority == "high":
                issues.append(message)
            else:
                backlog_items.append(message)
            execution_items.append(
                {
                    "pattern_id": pattern_id,
                    "fixture_kind": requirement.get("fixture_kind"),
                    "priority": priority,
                    "execution_status": "missing_pack",
                    "pack_id": None,
                }
            )
            continue

        pack_root = target_root / "fixture_packs" / str(pack["pack_id"])
        missing_pack_files = [name for name in REQUIRED_PACK_FILES if not (pack_root / name).exists()]
        source_files = pack.get("source_files", [])
        missing_source_files = [
            source_file["filename"]
            for source_file in source_files
            if not (pack_root / source_file["filename"]).exists()
        ]

        if missing_pack_files or missing_source_files:
            if missing_pack_files:
                issues.append(f"Pack {pack['pack_id']} missing metadata files: {', '.join(sorted(missing_pack_files))}")
            if missing_source_files:
                issues.append(f"Pack {pack['pack_id']} missing source files: {', '.join(sorted(missing_source_files))}")
            execution_status = "incomplete_pack"
        else:
            execution_status = "seeded_ready"

        execution_items.append(
            {
                "pattern_id": pattern_id,
                "fixture_kind": requirement.get("fixture_kind"),
                "priority": priority,
                "execution_status": execution_status,
                "pack_id": pack["pack_id"],
            }
        )

    covered_but_missing_index_ids = sorted(
        pattern_id
        for pattern_id in pack_by_pattern_id
        if pattern_id in requirement_map and pattern_id not in covered_ids
    )
    if covered_but_missing_index_ids:
        issues.append(
            "Fixture pack metadata exists but pack index coverage is missing for: "
            + ", ".join(covered_but_missing_index_ids)
        )

    ready_count = sum(1 for item in execution_items if item["execution_status"] == "seeded_ready")
    blocking_pattern_ids = sorted(
        item["pattern_id"]
        for item in execution_items
        if item["execution_status"] != "seeded_ready" and item["priority"] == "high"
    )
    backlog_pattern_ids = sorted(
        item["pattern_id"]
        for item in execution_items
        if item["execution_status"] != "seeded_ready" and item["priority"] != "high"
    )
    return {
        "ok": not issues,
        "requirement_count": len(requirement_map),
        "covered_requirement_count": len(covered_ids & set(requirement_map)),
        "seeded_ready_count": ready_count,
        "issues": issues,
        "backlog_items": backlog_items,
        "blocking_issue_count": len(issues),
        "backlog_count": len(backlog_items),
        "blocking_pattern_ids": blocking_pattern_ids,
        "backlog_pattern_ids": backlog_pattern_ids,
        "items": execution_items,
    }


def write_fixture_execution_report(corpus_root: str = "corpus") -> Dict[str, object]:
    repository = PatternCorpusRepository(root=corpus_root)
    repository.initialize_layout()
    write_fixture_registry(corpus_root=corpus_root)
    write_fixture_packs(corpus_root=corpus_root)

    target_root = Path(corpus_root) / "validated_patterns" / "release" / "legacy_reviewed"
    report = evaluate_fixture_execution(corpus_root=corpus_root)
    report_path = target_root / "fixture_execution_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "report_path": str(report_path),
        **report,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate that fixture registry items are backed by complete seeded fixture packs")
    parser.add_argument("--corpus-root", default="corpus", help="Corpus root directory")
    parser.add_argument("--write", action="store_true", help="Write fixture execution validation report")
    args = parser.parse_args()

    if args.write:
        print(json.dumps(write_fixture_execution_report(corpus_root=args.corpus_root), indent=2, sort_keys=True))
        return

    print(json.dumps({"message": "Use --write to materialize fixture execution report."}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
