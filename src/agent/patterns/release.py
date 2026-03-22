import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.agent.core.config import resolve_default_enhanced_dataset_file
from src.agent.core.versioning import compare_versions
from src.agent.patterns.catalog_normalizer import curated_catalog_patterns, write_curated_catalog_patterns
from src.agent.patterns.legacy_promotion import write_legacy_promotion_outputs
from src.agent.patterns.legacy_review import write_legacy_review_outputs
from src.agent.patterns.official_normalizer import write_normalized_official_patterns
from src.agent.patterns.promotion import _load_pattern_index, write_promotion_outputs
from src.agent.patterns.repository import PatternCorpusRepository
from src.agent.patterns.schema import ValidationStatus, VersionedPattern
from src.agent.rag.dataset_cleaner import clean_rules, load_dataset, write_encrypted_dataset, write_json_dataset
from src.agent.rag.audit import audit_dataset


@dataclass(frozen=True)
class ReleaseExportReport:
    base_dataset_path: str
    base_rule_count: int
    official_pattern_count: int
    catalog_pattern_count: int
    legacy_promoted_count: int
    legacy_ga_ready_count: int
    approved_candidate_count: int
    superseded_base_rule_count: int
    collapsed_duplicate_rule_count: int
    release_rule_count: int
    runtime_dataset_path: str
    pattern_index_path: str
    audit_ok: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _rule_source_key(rule: Dict[str, object]) -> Tuple[str, str]:
    return (
        str(rule.get("category", "code_patterns")).strip() or "code_patterns",
        str(rule.get("spring_pattern", "")).strip(),
    )


def _version_window_from_rule(rule: Dict[str, object], prefix: str) -> Tuple[Optional[str], Optional[str]]:
    metadata = dict(rule.get("metadata", {}) or {})
    window = dict(metadata.get(f"{prefix}_version_window", {}) or {})
    minimum = window.get("minimum")
    maximum = window.get("maximum")
    version = str(rule.get(f"{prefix}_version", "")).strip()

    if version:
        if minimum is None or compare_versions(version, minimum) < 0:
            minimum = version
        if maximum is None or compare_versions(version, maximum) > 0:
            maximum = version

    return minimum, maximum


def _select_canonical_rule(rules: Sequence[Dict[str, object]]) -> Dict[str, object]:
    def score(rule: Dict[str, object]) -> Tuple[int, int, int]:
        metadata = dict(rule.get("metadata", {}) or {})
        source_kind = str(metadata.get("release_source_kind") or metadata.get("source_kind") or "")
        status = str(metadata.get("release_validation_status") or metadata.get("status") or "")
        official_score = 1 if source_kind == "official_doc" else 0
        validated_score = 1 if status == "validated" else 0
        description_score = len(str(rule.get("description", "")))
        return official_score, validated_score, description_score

    return max(rules, key=score)


def _collapse_retrieval_duplicates(rules: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], int]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}
    for rule in rules:
        key = (
            str(rule.get("category", "code_patterns")).strip() or "code_patterns",
            str(rule.get("spring_pattern", "")).strip(),
            str(rule.get("micronaut_pattern", "")).strip(),
        )
        grouped.setdefault(key, []).append(rule)

    collapsed: List[Dict[str, object]] = []
    collapsed_count = 0

    for rules_for_key in grouped.values():
        if len(rules_for_key) == 1:
            collapsed.append(dict(rules_for_key[0]))
            continue

        canonical = dict(_select_canonical_rule(rules_for_key))
        metadata = dict(canonical.get("metadata", {}) or {})
        spring_min, spring_max = None, None
        micronaut_min, micronaut_max = None, None
        source_ids: List[str] = []

        for rule in rules_for_key:
            source_id = str(rule.get("id", "")).strip()
            if source_id:
                source_ids.append(source_id)

            rule_spring_min, rule_spring_max = _version_window_from_rule(rule, "spring")
            rule_micronaut_min, rule_micronaut_max = _version_window_from_rule(rule, "micronaut")

            if rule_spring_min and (spring_min is None or compare_versions(rule_spring_min, spring_min) < 0):
                spring_min = rule_spring_min
            if rule_spring_max and (spring_max is None or compare_versions(rule_spring_max, spring_max) > 0):
                spring_max = rule_spring_max
            if rule_micronaut_min and (micronaut_min is None or compare_versions(rule_micronaut_min, micronaut_min) < 0):
                micronaut_min = rule_micronaut_min
            if rule_micronaut_max and (
                micronaut_max is None or compare_versions(rule_micronaut_max, micronaut_max) > 0
            ):
                micronaut_max = rule_micronaut_max

        metadata["spring_version_window"] = {
            "spec": metadata.get("spring_version_spec") or None,
            "minimum": spring_min,
            "maximum": spring_max,
        }
        metadata["micronaut_version_window"] = {
            "spec": metadata.get("micronaut_version_spec") or None,
            "minimum": micronaut_min,
            "maximum": micronaut_max,
        }
        metadata["merged_duplicate_count"] = len(rules_for_key)
        metadata["merged_rule_ids"] = sorted(source_ids)

        canonical["metadata"] = metadata
        canonical["spring_version"] = spring_min
        canonical["micronaut_version"] = micronaut_min
        collapsed.append(canonical)
        collapsed_count += len(rules_for_key) - 1

    return collapsed, collapsed_count


def _pattern_to_rule(pattern: VersionedPattern) -> Dict[str, object]:
    rule = pattern.to_migration_rule()
    payload = {key: value for key, value in rule.__dict__.items() if value is not None}
    metadata = dict(payload.get("metadata", {}))
    metadata.update(
        {
            "release_source_kind": pattern.source_kind.value,
            "release_validation_status": pattern.status.value,
        }
    )
    payload["metadata"] = metadata
    return payload


def _promote_candidates_for_release(patterns: Iterable[VersionedPattern]) -> List[VersionedPattern]:
    promoted: List[VersionedPattern] = []
    for pattern in patterns:
        promoted.append(
            replace(
                pattern,
                status=ValidationStatus.VALIDATED,
                metadata={
                    **pattern.metadata,
                    "approved_for_release": True,
                    "approved_from_status": pattern.status.value,
                },
            )
        )
    return promoted


def _resolve_release_dataset_path(base_dataset_path: Optional[str] = None) -> str:
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


def _select_approved_candidates(
    staged_patterns: Sequence[VersionedPattern],
    approved_pattern_ids: Optional[Sequence[str]],
) -> List[VersionedPattern]:
    if not approved_pattern_ids:
        return []

    staged_by_id = {pattern.pattern_id: pattern for pattern in staged_patterns}
    missing_ids = [pattern_id for pattern_id in approved_pattern_ids if pattern_id not in staged_by_id]
    if missing_ids:
        raise KeyError(f"Unknown staged pattern ids: {', '.join(sorted(missing_ids))}")

    selected = [staged_by_id[pattern_id] for pattern_id in approved_pattern_ids]
    return _promote_candidates_for_release(selected)


def build_validated_release(
    base_dataset,
    official_patterns: Sequence[VersionedPattern],
    catalog_patterns: Optional[Sequence[VersionedPattern]] = None,
    legacy_promoted_patterns: Optional[Sequence[VersionedPattern]] = None,
    approved_candidates: Optional[Sequence[VersionedPattern]] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, int], List[VersionedPattern]]:
    base_rules, _ = clean_rules(base_dataset or [])
    validated_patterns = [pattern for pattern in official_patterns if pattern.status == ValidationStatus.VALIDATED]
    validated_patterns.extend(list(catalog_patterns or []))
    validated_patterns.extend(list(legacy_promoted_patterns or []))
    validated_patterns.extend(list(approved_candidates or []))

    validated_rules = [_pattern_to_rule(pattern) for pattern in validated_patterns]
    validated_source_keys: Set[Tuple[str, str]] = {_rule_source_key(rule) for rule in validated_rules}

    filtered_base_rules = [rule for rule in base_rules if _rule_source_key(rule) not in validated_source_keys]
    superseded_base_rule_count = len(base_rules) - len(filtered_base_rules)

    merged_rules, _ = clean_rules(validated_rules + filtered_base_rules)
    merged_rules, collapsed_duplicate_rule_count = _collapse_retrieval_duplicates(merged_rules)
    stats = {
        "base_rule_count": len(base_rules),
        "official_pattern_count": len([p for p in official_patterns if p.status == ValidationStatus.VALIDATED]),
        "catalog_pattern_count": len(catalog_patterns or []),
        "legacy_promoted_count": len(legacy_promoted_patterns or []),
        "legacy_ga_ready_count": len(legacy_promoted_patterns or []),
        "approved_candidate_count": len(approved_candidates or []),
        "superseded_base_rule_count": superseded_base_rule_count,
        "release_rule_count": len(merged_rules),
        "collapsed_duplicate_rule_count": collapsed_duplicate_rule_count,
    }
    return merged_rules, stats, validated_patterns


def write_validated_release(
    corpus_root: str = "corpus",
    base_dataset_path: Optional[str] = None,
    approved_pattern_ids: Optional[Sequence[str]] = None,
    runtime_format: str = "json",
    output_path: Optional[str] = None,
) -> Dict[str, object]:
    repository = PatternCorpusRepository(root=corpus_root)
    repository.initialize_layout()
    write_normalized_official_patterns(corpus_root=corpus_root)
    write_curated_catalog_patterns(corpus_root=corpus_root)
    write_promotion_outputs(corpus_root=corpus_root)
    write_legacy_promotion_outputs(corpus_root=corpus_root)
    write_legacy_review_outputs(corpus_root=corpus_root)

    resolved_base_dataset = _resolve_release_dataset_path(base_dataset_path)
    base_dataset = load_dataset(resolved_base_dataset)
    if base_dataset is None:
        base_dataset = []
        resolved_base_dataset = "unavailable_release_base_dataset"

    official_index = Path(corpus_root) / "official_docs" / "normalized" / "index.json"
    catalog_index = Path(corpus_root) / "validated_patterns" / "release" / "catalog" / "index.json"
    legacy_promoted_index = Path(corpus_root) / "validated_patterns" / "release" / "legacy_reviewed" / "ga_ready" / "index.json"
    staged_index = Path(corpus_root) / "staged_patterns" / "candidates" / "index.json"
    release_root = Path(corpus_root) / "validated_patterns" / "release"
    release_root.mkdir(parents=True, exist_ok=True)

    official_patterns = _load_pattern_index(official_index)
    catalog_patterns = _load_pattern_index(catalog_index) if catalog_index.exists() else curated_catalog_patterns()
    legacy_promoted_patterns = _load_pattern_index(legacy_promoted_index) if legacy_promoted_index.exists() else []
    staged_patterns = _load_pattern_index(staged_index) if staged_index.exists() else []
    approved_candidates = _select_approved_candidates(staged_patterns, approved_pattern_ids)

    merged_rules, stats, validated_patterns = build_validated_release(
        base_dataset=base_dataset,
        official_patterns=official_patterns,
        catalog_patterns=catalog_patterns,
        legacy_promoted_patterns=legacy_promoted_patterns,
        approved_candidates=approved_candidates,
    )

    runtime_dataset_path = output_path or str(
        release_root / f"runtime_dataset.{ 'dat' if runtime_format == 'dat' else 'json' }"
    )
    if runtime_format == "dat":
        write_encrypted_dataset(runtime_dataset_path, merged_rules)
    else:
        write_json_dataset(runtime_dataset_path, merged_rules)

    pattern_index_payload = {
        "schema_version": 1,
        "catalog_type": "validated_release_export",
        "pattern_count": len(validated_patterns),
        "patterns": [pattern.to_dict() for pattern in validated_patterns],
        "runtime_dataset_path": runtime_dataset_path,
        "base_dataset_path": resolved_base_dataset,
        "catalog_pattern_count": len(catalog_patterns),
        "approved_candidate_ids": [pattern.pattern_id for pattern in approved_candidates],
    }
    pattern_index_path = release_root / "index.json"
    pattern_index_path.write_text(json.dumps(pattern_index_payload, indent=2), encoding="utf-8")

    audit = audit_dataset(merged_rules)
    report = ReleaseExportReport(
        base_dataset_path=resolved_base_dataset,
        base_rule_count=stats["base_rule_count"],
        official_pattern_count=stats["official_pattern_count"],
        catalog_pattern_count=stats["catalog_pattern_count"],
        legacy_promoted_count=stats["legacy_promoted_count"],
        legacy_ga_ready_count=stats["legacy_ga_ready_count"],
        approved_candidate_count=stats["approved_candidate_count"],
        superseded_base_rule_count=stats["superseded_base_rule_count"],
        collapsed_duplicate_rule_count=stats["collapsed_duplicate_rule_count"],
        release_rule_count=stats["release_rule_count"],
        runtime_dataset_path=runtime_dataset_path,
        pattern_index_path=str(pattern_index_path),
        audit_ok=audit.is_valid,
    )
    report_path = release_root / "release_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    return {
        **report.to_dict(),
        "approved_candidate_ids": [pattern.pattern_id for pattern in approved_candidates],
        "release_report_path": str(report_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Build a validated runtime release dataset from the corpus and existing dataset artifacts")
    parser.add_argument("--corpus-root", default="corpus", help="Corpus root directory")
    parser.add_argument("--base-dataset", default=None, help="Base runtime dataset to merge with validated patterns")
    parser.add_argument("--approved-pattern-id", action="append", dest="approved_pattern_ids", help="Staged pattern id approved for release")
    parser.add_argument("--runtime-format", choices=["json", "dat"], default="json", help="Release runtime dataset format")
    parser.add_argument("--output-path", default=None, help="Optional runtime dataset output path")
    parser.add_argument("--write", action="store_true", help="Write validated release outputs into the corpus")
    args = parser.parse_args()

    if args.write:
        print(
            json.dumps(
                write_validated_release(
                    corpus_root=args.corpus_root,
                    base_dataset_path=args.base_dataset,
                    approved_pattern_ids=args.approved_pattern_ids,
                    runtime_format=args.runtime_format,
                    output_path=args.output_path,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return

    print(json.dumps({"message": "Use --write to materialize validated release outputs."}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
