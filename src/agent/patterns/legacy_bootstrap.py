import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from src.agent.core.config import resolve_default_enhanced_dataset_file
from src.agent.patterns.repository import PatternCorpusRepository
from src.agent.patterns.schema import (
    PatternEvidence,
    PatternType,
    SourceKind,
    ValidationStatus,
    VersionWindow,
    VersionedPattern,
)
from src.agent.rag.dataset_cleaner import clean_rules, load_dataset


def _pattern_type_for_rule(rule: Dict[str, object]) -> PatternType:
    migration_type = str(rule.get("migration_type", "")).strip().lower()
    category = str(rule.get("category", "code_patterns")).strip().lower()

    if migration_type == "annotation" or category == "annotations":
        return PatternType.ANNOTATION
    if migration_type == "configuration" or category == "configurations":
        return PatternType.CONFIGURATION
    if migration_type == "dependency" or category == "dependencies":
        return PatternType.DEPENDENCY
    if migration_type == "type" or category == "types":
        return PatternType.TYPE
    if migration_type == "import" or category == "imports":
        return PatternType.IMPORT
    return PatternType.CODE_PATTERN


def _legacy_pattern_id(rule: Dict[str, object], index: int) -> str:
    rule_id = str(rule.get("id", "")).strip()
    if rule_id:
        return f"legacy.{rule_id}"
    return f"legacy.rule_{index}"


def _resolve_readable_dataset_path(base_dataset_path: str = None) -> str:
    candidates = []
    if base_dataset_path:
        candidates.append(base_dataset_path)
    else:
        candidates.extend(
            [
                resolve_default_enhanced_dataset_file(),
                "migration_dataset_enhanced_cleaned.json",
                "migration_dataset_enhanced.json",
                "migration_dataset.json",
                "corpus/validated_patterns/release/runtime_dataset.json",
            ]
        )

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if load_dataset(candidate) is not None:
            return candidate

    return candidates[0] if candidates else resolve_default_enhanced_dataset_file()


def bootstrap_legacy_patterns(base_dataset_path: str = None) -> Dict[str, object]:
    resolved_path = _resolve_readable_dataset_path(base_dataset_path=base_dataset_path)
    dataset = load_dataset(resolved_path)
    if dataset is None:
        dataset = []
        resolved_path = "unavailable_legacy_base_dataset"

    cleaned_rules, summary = clean_rules(dataset)
    patterns: List[VersionedPattern] = []
    spring_version_counter: Counter[str] = Counter()
    micronaut_version_counter: Counter[str] = Counter()

    for index, rule in enumerate(cleaned_rules, start=1):
        spring_version = str(rule.get("spring_version", "")).strip() or "3.x"
        micronaut_version = str(rule.get("micronaut_version", "")).strip() or "4.x"
        spring_version_counter[spring_version] += 1
        micronaut_version_counter[micronaut_version] += 1

        pattern = VersionedPattern(
            pattern_id=_legacy_pattern_id(rule, index),
            pattern_type=_pattern_type_for_rule(rule),
            spring_pattern=str(rule.get("spring_pattern", "")).strip(),
            micronaut_pattern=str(rule.get("micronaut_pattern", "")).strip(),
            description=str(rule.get("description", "")).strip() or "Legacy runtime dataset rule",
            spring_versions=VersionWindow(spec=spring_version),
            micronaut_versions=VersionWindow(spec=micronaut_version),
            status=ValidationStatus.VALIDATED,
            confidence=0.8,
            complexity=str(rule.get("complexity", "medium")).strip() or "medium",
            category=str(rule.get("category", "code_patterns")).strip() or "code_patterns",
            source_kind=SourceKind.MANUAL,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.MANUAL,
                    source_ref=resolved_path,
                    title="Legacy runtime dataset bootstrap",
                    notes=str(rule.get("id", "")).strip() or f"bootstrap-index:{index}",
                )
            ],
            metadata={
                "bootstrap_source": "legacy_runtime_dataset",
                "legacy_rule_id": str(rule.get("id", "")).strip() or None,
            },
        )
        patterns.append(pattern)

    return {
        "schema_version": 1,
        "catalog_type": "legacy_runtime_bootstrap",
        "pattern_count": len(patterns),
        "patterns": [pattern.to_dict() for pattern in patterns],
        "source_dataset_path": resolved_path,
        "category_counts": summary.category_counts,
        "spring_version_counts": dict(sorted(spring_version_counter.items())),
        "micronaut_version_counts": dict(sorted(micronaut_version_counter.items())),
    }


def write_legacy_bootstrap(corpus_root: str = "corpus", base_dataset_path: str = None) -> Dict[str, object]:
    repository = PatternCorpusRepository(root=corpus_root)
    repository.initialize_layout()

    payload = bootstrap_legacy_patterns(base_dataset_path=base_dataset_path)
    archive_root = Path(corpus_root) / "validated_patterns" / "archives" / "legacy_runtime_bootstrap"
    patterns_root = archive_root / "patterns"
    patterns_root.mkdir(parents=True, exist_ok=True)

    index_path = archive_root / "index.json"
    index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    pattern_files = []
    for pattern_payload in payload["patterns"]:
        path = patterns_root / f"{pattern_payload['pattern_id']}.json"
        path.write_text(json.dumps(pattern_payload, indent=2), encoding="utf-8")
        pattern_files.append(str(path))

    summary_payload = {
        "pattern_count": payload["pattern_count"],
        "source_dataset_path": payload["source_dataset_path"],
        "category_counts": payload["category_counts"],
        "spring_version_counts": payload["spring_version_counts"],
        "micronaut_version_counts": payload["micronaut_version_counts"],
    }
    summary_path = archive_root / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return {
        "index_path": str(index_path),
        "summary_path": str(summary_path),
        "pattern_count": payload["pattern_count"],
        "pattern_file_count": len(pattern_files),
        "pattern_files_sample": pattern_files[:10],
        "source_dataset_path": payload["source_dataset_path"],
    }


def main():
    parser = argparse.ArgumentParser(description="Bootstrap the existing runtime dataset into versioned legacy corpus patterns")
    parser.add_argument("--corpus-root", default="corpus", help="Corpus root directory")
    parser.add_argument("--base-dataset", default=None, help="Optional base dataset path")
    parser.add_argument("--write", action="store_true", help="Write legacy bootstrap outputs into validated pattern archives")
    args = parser.parse_args()

    if args.write:
        print(
            json.dumps(
                write_legacy_bootstrap(
                    corpus_root=args.corpus_root,
                    base_dataset_path=args.base_dataset,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return

    print(
        json.dumps(
            bootstrap_legacy_patterns(base_dataset_path=args.base_dataset),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
