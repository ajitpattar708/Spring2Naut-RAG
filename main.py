import argparse
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Sequence

from src.agent.orchestrator import MigrationOrchestrator
from src.agent.core.config import MigrationConfig
from src.agent.core.models import MigrationRule, VersionCompatibilityMatrix
from src.agent.core.versioning import normalize_major_minor
from src.agent.agents.dependency_audit import DependencyCompatibilityAuditor
from src.agent.patterns.official_normalizer import write_normalized_official_patterns
from src.agent.patterns.promotion import _load_pattern_index
from src.agent.patterns.release import write_validated_release
from src.agent.rag.chroma_audit import write_chroma_audit_report
from src.agent.rag.dataset_cleaner import clean_rules, load_dataset, load_dataset_with_status, write_json_dataset
from src.agent.rag.kb_release_smoke import write_release_kb_smoke_report
from src.agent.rag.knowledge_base import LocalMigrationKnowledgeBase

KNOWN_SPRING_LINES = tuple(f"3.{minor}.x" for minor in range(0, 7))
KNOWN_MICRONAUT_LINES = tuple(f"4.{minor}.x" for minor in range(0, 11))
KNOWN_SPRING_ANCHORS = {line: line.replace(".x", ".0") for line in KNOWN_SPRING_LINES}
KNOWN_MICRONAUT_ANCHORS = {
    **{line: line.replace(".x", ".0") for line in KNOWN_MICRONAUT_LINES},
    "4.10.x": "4.10.8",
}
_LAST_RAW_DATASET_STATUS = None


@contextmanager
def _temporary_runtime_config(
    *,
    dataset_file: Optional[str] = None,
    enhanced_dataset_file: Optional[str] = None,
    vector_db_path: Optional[str] = None,
):
    original_dataset = MigrationConfig.DATASET_FILE
    original_enhanced = MigrationConfig.ENHANCED_DATASET_FILE
    original_vector_db_path = MigrationConfig.VECTOR_DB_PATH
    try:
        if dataset_file is not None:
            MigrationConfig.DATASET_FILE = dataset_file
        if enhanced_dataset_file is not None:
            MigrationConfig.ENHANCED_DATASET_FILE = enhanced_dataset_file
        if vector_db_path is not None:
            MigrationConfig.VECTOR_DB_PATH = vector_db_path
        yield
    finally:
        MigrationConfig.DATASET_FILE = original_dataset
        MigrationConfig.ENHANCED_DATASET_FILE = original_enhanced
        MigrationConfig.VECTOR_DB_PATH = original_vector_db_path


def _load_raw_dataset_rules():
    global _LAST_RAW_DATASET_STATUS
    cleaned_rules, summary, status = _load_raw_dataset_rules_with_status()
    _LAST_RAW_DATASET_STATUS = status
    return cleaned_rules, summary


def _load_raw_dataset_rules_with_status():
    community_data, community_status = load_dataset_with_status(MigrationConfig.DATASET_FILE)
    enhanced_data, enhanced_status = load_dataset_with_status(MigrationConfig.ENHANCED_DATASET_FILE)

    merged_rules = []
    for source_data in (community_data, enhanced_data):
        if isinstance(source_data, list):
            merged_rules.extend(source_data)
        elif isinstance(source_data, dict):
            for category, rules in source_data.items():
                if not isinstance(rules, list):
                    continue
                for rule in rules:
                    normalized = dict(rule)
                    normalized.setdefault("category", category)
                    merged_rules.append(normalized)

    cleaned_rules, summary = clean_rules(merged_rules)
    return cleaned_rules, summary, {
        "community": community_status,
        "enhanced": enhanced_status,
    }


def _materialize_dataset_rules(dataset) -> list:
    merged_rules = []
    if isinstance(dataset, list):
        merged_rules.extend(dataset)
    elif isinstance(dataset, dict):
        for category, rules in dataset.items():
            if not isinstance(rules, list):
                continue
            for rule in rules:
                normalized = dict(rule)
                normalized.setdefault("category", category)
                merged_rules.append(normalized)
    return merged_rules


def _load_pattern_rules(index_path: Path) -> list:
    if not index_path.exists():
        return []

    rules = []
    for pattern in _load_pattern_index(index_path):
        rule = pattern.to_migration_rule()
        rules.append({key: value for key, value in rule.__dict__.items() if value is not None})
    return rules


def _expand_known_version_values(
    version: str,
    known_lines: tuple[str, ...],
    anchor_versions: dict[str, str],
    major_prefix: str,
) -> list[str]:
    normalized = str(version or "").strip()
    if not normalized:
        return [anchor_versions[line] for line in known_lines]
    if normalized in anchor_versions:
        return [anchor_versions[normalized]]
    if normalized == f"{major_prefix}.x":
        return [anchor_versions[line] for line in known_lines]
    return [normalized]


def _expand_known_rules(rules: list) -> list:
    expanded = []
    for rule in rules:
        spring_versions = _expand_known_version_values(
            str(rule.get("spring_version", "")).strip(),
            KNOWN_SPRING_LINES,
            KNOWN_SPRING_ANCHORS,
            "3",
        )
        micronaut_versions = _expand_known_version_values(
            str(rule.get("micronaut_version", "")).strip(),
            KNOWN_MICRONAUT_LINES,
            KNOWN_MICRONAUT_ANCHORS,
            "4",
        )

        for spring_version in spring_versions:
            for micronaut_version in micronaut_versions:
                clone = dict(rule)
                clone["spring_version"] = spring_version
                clone["micronaut_version"] = micronaut_version
                metadata = dict(clone.get("metadata", {}) or {})
                metadata["known_version_matrix_expanded"] = True
                metadata["known_version_matrix_source"] = metadata.get("source_kind") or metadata.get("release_source_kind") or "unknown"
                clone["metadata"] = metadata
                expanded.append(clone)
    return expanded


def _initialize_persistent_kb(dataset_path: str, db_path: str) -> dict:
    db_root = Path(db_path)
    if db_root.exists():
        shutil.rmtree(db_root)
    db_root.parent.mkdir(parents=True, exist_ok=True)

    with _temporary_runtime_config(
        dataset_file=dataset_path,
        enhanced_dataset_file="__disable_enhanced_dataset__.json",
        vector_db_path=str(db_root),
    ):
        kb = LocalMigrationKnowledgeBase(db_path=str(db_root))

    collection_counts = {
        name: stats["count"]
        for name, stats in (
            (name, kb.get_collection_stats(name))
            for name in ("annotations", "dependencies", "configurations", "code_patterns", "imports", "types")
        )
    }
    return {
        "db_path": str(db_root),
        "collection_counts": collection_counts,
        "indexed_rule_count": int(sum(collection_counts.values())),
    }


def _count_version_compatible_rules(dataset_path: str, spring_version: str, micronaut_version: str) -> int:
    dataset = load_dataset(dataset_path)
    rules = _materialize_dataset_rules(dataset)
    compatible_count = 0

    for rule in rules:
        migration_rule = MigrationRule(
            spring_pattern=str(rule.get("spring_pattern", "")),
            micronaut_pattern=str(rule.get("micronaut_pattern", "")),
            category=str(rule.get("category", "code_patterns")),
            description=str(rule.get("description", "")),
            complexity=str(rule.get("complexity", "medium")),
            spring_version=rule.get("spring_version"),
            micronaut_version=rule.get("micronaut_version"),
            metadata=rule.get("metadata"),
        )
        if VersionCompatibilityMatrix.is_version_compatible(
            migration_rule,
            spring_version,
            micronaut_version,
        ):
            compatible_count += 1

    return compatible_count


def _filter_version_compatible_rules(dataset_path: str, spring_version: str, micronaut_version: str) -> list[dict]:
    dataset = load_dataset(dataset_path)
    rules = _materialize_dataset_rules(dataset)
    compatible_rules: list[dict] = []

    for rule in rules:
        migration_rule = MigrationRule(
            spring_pattern=str(rule.get("spring_pattern", "")),
            micronaut_pattern=str(rule.get("micronaut_pattern", "")),
            category=str(rule.get("category", "code_patterns")),
            description=str(rule.get("description", "")),
            complexity=str(rule.get("complexity", "medium")),
            spring_version=rule.get("spring_version"),
            micronaut_version=rule.get("micronaut_version"),
            metadata=rule.get("metadata"),
        )
        if VersionCompatibilityMatrix.is_version_compatible(
            migration_rule,
            spring_version,
            micronaut_version,
        ):
            compatible_rules.append(dict(rule))

    return compatible_rules


def _extract_ga(pattern: object) -> Optional[str]:
    text = str(pattern or "").strip()
    if not text or "@" in text:
        return None
    parts = [part.strip() for part in text.split(":")]
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return None
    return f"{parts[0]}:{parts[1]}"


def _target_version_window(version: str) -> dict[str, Optional[str]]:
    line = normalize_major_minor(version)
    return {
        "spec": f"{line}.x" if line else None,
        "minimum": version,
        "maximum": version,
    }


def _materialize_target_version_overlays(
    compatible_rules: list[dict],
    *,
    spring_version: str,
    micronaut_version: str,
) -> list[dict]:
    materialized_rules = [dict(rule) for rule in compatible_rules]
    compatibility_info = VersionCompatibilityMatrix.get_compatibility_info(spring_version, micronaut_version)
    version_specific_patterns = compatibility_info.get("version_specific_patterns", {})

    auditor = DependencyCompatibilityAuditor(object(), spring_version, micronaut_version)
    managed_dependencies = auditor.resolve_target_platform_managed_dependencies()
    target_platform_summary = auditor._build_target_platform_summary().get("summary", {})
    managed_lookup = {
        f"{item.group_id}:{item.artifact_id}": item
        for item in managed_dependencies
        if getattr(item, "group_id", "") and getattr(item, "artifact_id", "")
    }
    resolution_channel = str(target_platform_summary.get("target_platform_resolution_channel") or "unknown")
    target_line = normalize_major_minor(micronaut_version)
    spring_window = _target_version_window(spring_version)
    micronaut_window = _target_version_window(micronaut_version)

    for rule in materialized_rules:
        metadata = dict(rule.get("metadata") or {})
        metadata.setdefault("target_runtime_profile", "governed_target_pair")
        metadata["target_runtime_pair"] = f"spring:{spring_version}->micronaut:{micronaut_version}"
        metadata["target_runtime_spring_line"] = normalize_major_minor(spring_version)
        metadata["target_runtime_micronaut_line"] = target_line

        spring_pattern = str(rule.get("spring_pattern") or "").strip()
        pattern_overlay = version_specific_patterns.get(spring_pattern)
        if isinstance(pattern_overlay, dict):
            metadata["target_version_overlay"] = "api_compatibility_matrix"
            metadata["target_version_overlay_replacement"] = str(pattern_overlay.get("replacement") or "").strip()
            metadata["target_version_overlay_note"] = str(pattern_overlay.get("note") or "").strip()
            metadata["spring_version_window"] = dict(metadata.get("spring_version_window") or spring_window)
            metadata["micronaut_version_window"] = dict(metadata.get("micronaut_version_window") or micronaut_window)
            note = str(pattern_overlay.get("note") or "").strip()
            if note and note not in str(rule.get("description") or ""):
                base_description = str(rule.get("description") or "").strip()
                rule["description"] = f"{base_description} Target note: {note}".strip()

        replacement_ga = _extract_ga(rule.get("micronaut_pattern"))
        managed_dependency = managed_lookup.get(replacement_ga or "")
        if managed_dependency is not None:
            metadata["target_platform_overlay"] = "managed_dependency_alignment"
            metadata["target_platform_version"] = micronaut_version
            metadata["target_platform_resolution_channel"] = resolution_channel
            metadata["target_platform_managed_ga"] = replacement_ga
            metadata["target_platform_managed_version"] = str(getattr(managed_dependency, "version", "") or "").strip()
            metadata["spring_version_window"] = dict(metadata.get("spring_version_window") or spring_window)
            metadata["micronaut_version_window"] = dict(metadata.get("micronaut_version_window") or micronaut_window)

        rule["metadata"] = metadata

    existing_patterns = {
        str(rule.get("spring_pattern") or "").strip()
        for rule in materialized_rules
    }
    for spring_pattern, overlay in version_specific_patterns.items():
        if spring_pattern in existing_patterns or not isinstance(overlay, dict):
            continue
        materialized_rules.append(
            {
                "spring_pattern": spring_pattern,
                "micronaut_pattern": str(overlay.get("replacement") or spring_pattern),
                "category": "code_patterns",
                "description": str(overlay.get("note") or f"Target-specific Micronaut {micronaut_version} guidance."),
                "complexity": "medium",
                "metadata": {
                    "target_runtime_profile": "governed_target_pair",
                    "target_runtime_pair": f"spring:{spring_version}->micronaut:{micronaut_version}",
                    "target_version_overlay": "api_compatibility_matrix",
                    "target_version_overlay_replacement": str(overlay.get("replacement") or "").strip(),
                    "target_version_overlay_note": str(overlay.get("note") or "").strip(),
                    "spring_version_window": spring_window,
                    "micronaut_version_window": micronaut_window,
                },
            }
        )

    return materialized_rules


def _write_target_runtime_dataset(
    *,
    corpus_root: str,
    source_dataset_path: str,
    spring_version: str,
    micronaut_version: str,
) -> dict:
    compatible_rules = _filter_version_compatible_rules(
        source_dataset_path,
        spring_version,
        micronaut_version,
    )
    materialized_rules = _materialize_target_version_overlays(
        compatible_rules,
        spring_version=spring_version,
        micronaut_version=micronaut_version,
    )
    target_dir = Path(corpus_root) / "validated_patterns" / "release" / "target_runtime_datasets"
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = target_dir / f"{_version_pair_slug(spring_version, micronaut_version)}.json"
    write_json_dataset(str(dataset_path), materialized_rules)
    return {
        "dataset_path": str(dataset_path),
        "compatible_rule_count": len(materialized_rules),
    }


def _window_targets_line(value: Optional[str], line: str) -> bool:
    normalized = str(value or "").strip().lower()
    normalized_line = str(line or "").strip().lower()
    if not normalized or not normalized_line:
        return False
    return normalized == normalized_line or normalized.startswith(f"{normalized_line}.")


def _rule_window_summary(rule: MigrationRule, prefix: str) -> dict:
    window = VersionCompatibilityMatrix._window_from_metadata(rule, prefix)
    window = VersionCompatibilityMatrix._apply_legacy_rule_version(window, getattr(rule, f"{prefix}_version"))
    return window


def _categorize_compatible_rules(dataset_path: str, spring_version: str, micronaut_version: str) -> dict:
    dataset = load_dataset(dataset_path)
    rules = _materialize_dataset_rules(dataset)
    spring_line = normalize_major_minor(spring_version)
    micronaut_line = normalize_major_minor(micronaut_version)

    category_counts: dict[str, int] = {}
    compatible_rule_count = 0
    spring_line_specific_count = 0
    micronaut_line_specific_count = 0
    pair_line_specific_count = 0

    for rule in rules:
        migration_rule = MigrationRule(
            spring_pattern=str(rule.get("spring_pattern", "")),
            micronaut_pattern=str(rule.get("micronaut_pattern", "")),
            category=str(rule.get("category", "code_patterns")),
            description=str(rule.get("description", "")),
            complexity=str(rule.get("complexity", "medium")),
            spring_version=rule.get("spring_version"),
            micronaut_version=rule.get("micronaut_version"),
            metadata=rule.get("metadata"),
        )
        if not VersionCompatibilityMatrix.is_version_compatible(
            migration_rule,
            spring_version,
            micronaut_version,
        ):
            continue

        compatible_rule_count += 1
        category = migration_rule.category or "code_patterns"
        category_counts[category] = category_counts.get(category, 0) + 1

        spring_window = _rule_window_summary(migration_rule, "spring")
        micronaut_window = _rule_window_summary(migration_rule, "micronaut")
        spring_line_specific = any(
            _window_targets_line(spring_window.get(key), spring_line)
            for key in ("spec", "minimum", "maximum")
        )
        micronaut_line_specific = any(
            _window_targets_line(micronaut_window.get(key), micronaut_line)
            for key in ("spec", "minimum", "maximum")
        )

        if spring_line_specific:
            spring_line_specific_count += 1
        if micronaut_line_specific:
            micronaut_line_specific_count += 1
        if spring_line_specific and micronaut_line_specific:
            pair_line_specific_count += 1

    generic_compatible_rule_count = compatible_rule_count - pair_line_specific_count
    compatibility_mode = "line-aware" if pair_line_specific_count else "major-family-generic"

    return {
        "schema_version": 1,
        "spring_version": spring_version,
        "micronaut_version": micronaut_version,
        "spring_line": spring_line,
        "micronaut_line": micronaut_line,
        "compatible_rule_count": compatible_rule_count,
        "compatible_category_counts": dict(sorted(category_counts.items())),
        "spring_line_specific_rule_count": spring_line_specific_count,
        "micronaut_line_specific_rule_count": micronaut_line_specific_count,
        "pair_line_specific_rule_count": pair_line_specific_count,
        "generic_compatible_rule_count": generic_compatible_rule_count,
        "compatibility_mode": compatibility_mode,
    }


def _version_pair_slug(spring_version: str, micronaut_version: str) -> str:
    safe_spring = spring_version.replace(".", "_")
    safe_micronaut = micronaut_version.replace(".", "_")
    return f"spring_{safe_spring}__micronaut_{safe_micronaut}"


def _write_target_profile(
    *,
    corpus_root: str,
    dataset_path: str,
    spring_version: str,
    micronaut_version: str,
) -> dict:
    profile = _categorize_compatible_rules(
        dataset_path=dataset_path,
        spring_version=spring_version,
        micronaut_version=micronaut_version,
    )
    target_profiles_dir = Path(corpus_root) / "validated_patterns" / "release" / "target_profiles"
    target_profiles_dir.mkdir(parents=True, exist_ok=True)
    profile_path = target_profiles_dir / f"{_version_pair_slug(spring_version, micronaut_version)}.json"
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    profile["profile_path"] = str(profile_path)
    return profile


def _write_kb_manifest(
    *,
    db_path: str,
    mode: str,
    spring_version: str,
    micronaut_version: str,
    dataset_path: str,
    indexed_rule_count: int,
    compatible_rule_count: int,
    target_profile: Optional[dict] = None,
    target_platform_snapshot: Optional[dict] = None,
) -> str:
    manifest_path = Path(db_path) / "kb_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    spring_line = normalize_major_minor(spring_version)
    micronaut_line = normalize_major_minor(micronaut_version)
    manifest = {
        "schema_version": 2,
        "mode": mode,
        "spring_version": spring_version,
        "micronaut_version": micronaut_version,
        "spring_line": spring_line,
        "micronaut_line": micronaut_line,
        "dataset_path": dataset_path,
        "indexed_rule_count": indexed_rule_count,
        "compatible_rule_count": compatible_rule_count,
    }
    if target_profile:
        manifest["target_profile"] = target_profile
    if target_platform_snapshot:
        manifest["target_platform_snapshot"] = target_platform_snapshot
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(manifest_path)


def _write_target_platform_snapshot(
    *,
    corpus_root: str,
    spring_version: str,
    micronaut_version: str,
) -> dict:
    auditor = DependencyCompatibilityAuditor(object(), spring_version, micronaut_version)
    managed_dependencies = auditor.resolve_target_platform_managed_dependencies()
    target_platform_summary = auditor._build_target_platform_summary().get("summary", {})
    output_dir = Path(corpus_root) / "validated_patterns" / "release" / "target_platforms"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / f"{_version_pair_slug(spring_version, micronaut_version)}.json"
    payload = {
        "schema_version": 1,
        "spring_version": spring_version,
        "micronaut_version": micronaut_version,
        "managed_dependency_count": len(managed_dependencies),
        "target_platform_summary": target_platform_summary,
        "managed_dependencies": [item.to_dict() for item in managed_dependencies],
    }
    snapshot_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "snapshot_path": str(snapshot_path),
        "managed_dependency_count": len(managed_dependencies),
        "target_platform_summary": target_platform_summary,
    }


def _write_extended_dataset(corpus_root: str, rules) -> str:
    output_path = Path(corpus_root) / "validated_patterns" / "extended" / "runtime_dataset.json"
    write_json_dataset(str(output_path), rules)
    return str(output_path)


def _normalize_init_mode(mode: str) -> str:
    aliases = {
        "extended": "legacy",
        "both": "hybrid",
        "full": "all",
    }
    return aliases.get(str(mode or "").strip(), str(mode or "").strip())


def _raw_dataset_mode_note(mode: str) -> str:
    normalized_mode = _normalize_init_mode(mode)
    if normalized_mode in {"legacy", "hybrid", "max", "all"}:
        return "loaded"
    return "skipped (mode does not use legacy encrypted/raw datasets)"


def _legacy_mode_dataset_overrides(mode: str) -> dict[str, Optional[str]]:
    normalized_mode = _normalize_init_mode(mode)
    if normalized_mode != "all":
        return {}

    repo_root = Path.cwd()
    dataset_file = repo_root / "migration_dataset.json.dat"
    enhanced_dataset_file = repo_root / "migration_dataset_enhanced.json.dat"
    return {
        "dataset_file": str(dataset_file) if dataset_file.exists() else None,
        "enhanced_dataset_file": str(enhanced_dataset_file) if enhanced_dataset_file.exists() else None,
    }


def _write_candidate_dataset(corpus_root: str, trusted_dataset_path: str) -> dict:
    release_rules = _materialize_dataset_rules(load_dataset(trusted_dataset_path))
    staged_index = Path(corpus_root) / "staged_patterns" / "candidates" / "index.json"
    candidate_rules, summary = clean_rules(release_rules + _load_pattern_rules(staged_index))

    output_path = Path(corpus_root) / "validated_patterns" / "candidate" / "runtime_dataset.json"
    write_json_dataset(str(output_path), candidate_rules)
    return {"dataset_path": str(output_path), "rule_count": summary.output_rules}


def _write_max_dataset(corpus_root: str, raw_rules, trusted_dataset_path: str) -> dict:
    merged_rules = list(raw_rules)
    merged_rules.extend(_materialize_dataset_rules(load_dataset(trusted_dataset_path)))

    extra_index_paths = (
        Path(corpus_root) / "official_docs" / "normalized" / "index.json",
        Path(corpus_root) / "staged_patterns" / "candidates" / "index.json",
        Path(corpus_root) / "validated_patterns" / "release" / "legacy_promoted" / "index.json",
    )
    for index_path in extra_index_paths:
        merged_rules.extend(_expand_known_rules(_load_pattern_rules(index_path)))

    max_rules, summary = clean_rules(merged_rules)
    output_path = Path(corpus_root) / "validated_patterns" / "max" / "runtime_dataset.json"
    write_json_dataset(str(output_path), max_rules)
    return {"dataset_path": str(output_path), "rule_count": summary.output_rules}


def _run_migration(args: argparse.Namespace) -> int:
    print("-" * 50)
    print("Agentic Migration Initialized")
    print(f"Targeting: Spring {args.spring_version} -> Micronaut {args.micronaut_version}")
    print("-" * 50)

    orchestrator = MigrationOrchestrator(
        spring_version=args.spring_version,
        micronaut_version=args.micronaut_version,
        build_tool_override=args.build_tool,
    )
    orchestrator.migrate_project(args.input, args.output)
    report_path = Path(args.output) / "reports" / "migration_report.json"
    if report_path.exists():
        print(f"[INFO] Migration report written to {report_path}")
    print("-" * 50)
    return 0


def _run_init(args: argparse.Namespace) -> int:
    global _LAST_RAW_DATASET_STATUS
    print("-" * 50)
    print("Spring2Naut RAG Initialization")
    print(f"Corpus Root: {args.corpus_root}")
    print(f"Targeting: Spring {args.spring_version} -> Micronaut {args.micronaut_version}")
    print(f"Mode: {args.mode}")
    print("-" * 50)

    normalized_mode = _normalize_init_mode(args.mode)
    raw_rules = []
    raw_summary = None
    raw_status = None
    if normalized_mode in {"legacy", "hybrid", "max", "all"}:
        _LAST_RAW_DATASET_STATUS = None
        legacy_overrides = _legacy_mode_dataset_overrides(args.mode)
        with _temporary_runtime_config(
            dataset_file=legacy_overrides.get("dataset_file"),
            enhanced_dataset_file=legacy_overrides.get("enhanced_dataset_file"),
        ):
            raw_rules, raw_summary = _load_raw_dataset_rules()
            raw_status = _LAST_RAW_DATASET_STATUS

    official_result = write_normalized_official_patterns(corpus_root=args.corpus_root)
    release_result = write_validated_release(corpus_root=args.corpus_root, runtime_format="json")
    trusted_init_result = None
    legacy_init_result = None
    candidate_init_result = None
    max_init_result = None
    smoke_result = None
    audit_result = None
    candidate_dataset_result = None
    max_dataset_result = None
    trusted_compatible_rule_count = None
    trusted_manifest_path = None
    trusted_target_profile = None
    trusted_runtime_dataset_result = None
    trusted_target_platform_snapshot = None

    vector_root = Path(MigrationConfig.VECTOR_DB_PATH)
    trusted_db_path = str(vector_root)
    legacy_db_path = str(vector_root.parent / f"{vector_root.name}_legacy")
    candidate_db_path = str(vector_root.parent / f"{vector_root.name}_candidate")
    max_db_path = str(vector_root.parent / f"{vector_root.name}_max")

    if normalized_mode in {"trusted", "hybrid", "all"}:
        trusted_runtime_dataset_result = _write_target_runtime_dataset(
            corpus_root=args.corpus_root,
            source_dataset_path=release_result["runtime_dataset_path"],
            spring_version=args.spring_version,
            micronaut_version=args.micronaut_version,
        )
        trusted_compatible_rule_count = trusted_runtime_dataset_result["compatible_rule_count"]
        smoke_result = write_release_kb_smoke_report(
            corpus_root=args.corpus_root,
            spring_version=args.spring_version,
            micronaut_version=args.micronaut_version,
            runtime_dataset_path=trusted_runtime_dataset_result["dataset_path"],
            release_rule_count=trusted_compatible_rule_count,
        )
        trusted_init_result = _initialize_persistent_kb(
            dataset_path=trusted_runtime_dataset_result["dataset_path"],
            db_path=trusted_db_path,
        )
        trusted_target_profile = _write_target_profile(
            corpus_root=args.corpus_root,
            dataset_path=release_result["runtime_dataset_path"],
            spring_version=args.spring_version,
            micronaut_version=args.micronaut_version,
        )
        trusted_target_platform_snapshot = _write_target_platform_snapshot(
            corpus_root=args.corpus_root,
            spring_version=args.spring_version,
            micronaut_version=args.micronaut_version,
        )
        trusted_manifest_path = _write_kb_manifest(
            db_path=trusted_db_path,
            mode="trusted",
            spring_version=args.spring_version,
            micronaut_version=args.micronaut_version,
            dataset_path=trusted_runtime_dataset_result["dataset_path"],
            indexed_rule_count=trusted_init_result["indexed_rule_count"],
            compatible_rule_count=trusted_compatible_rule_count,
            target_profile=trusted_target_profile,
            target_platform_snapshot=trusted_target_platform_snapshot,
        )
        if not args.skip_audit:
            audit_result = write_chroma_audit_report(
                corpus_root=args.corpus_root,
                runtime_dataset_path=trusted_runtime_dataset_result["dataset_path"],
                release_rule_count=trusted_compatible_rule_count,
                spring_version=args.spring_version,
                micronaut_version=args.micronaut_version,
            )

    if normalized_mode == "legacy":
        legacy_dataset_path = _write_extended_dataset(args.corpus_root, raw_rules)
        legacy_init_result = _initialize_persistent_kb(
            dataset_path=legacy_dataset_path,
            db_path=legacy_db_path,
        )

    if normalized_mode == "hybrid":
        hybrid_rules = list(raw_rules)
        hybrid_rules.extend(_materialize_dataset_rules(load_dataset(trusted_runtime_dataset_result["dataset_path"])))
        hybrid_dataset_path = _write_extended_dataset(args.corpus_root, hybrid_rules)
        legacy_init_result = _initialize_persistent_kb(
            dataset_path=hybrid_dataset_path,
            db_path=legacy_db_path,
        )

    if normalized_mode == "candidate":
        candidate_dataset_result = _write_candidate_dataset(
            args.corpus_root,
            release_result["runtime_dataset_path"],
        )
        candidate_init_result = _initialize_persistent_kb(
            dataset_path=candidate_dataset_result["dataset_path"],
            db_path=candidate_db_path,
        )

    if normalized_mode in {"max", "all"}:
        max_dataset_result = _write_max_dataset(
            args.corpus_root,
            raw_rules,
            release_result["runtime_dataset_path"],
        )
        max_init_result = _initialize_persistent_kb(
            dataset_path=max_dataset_result["dataset_path"],
            db_path=max_db_path,
        )

    print("\n" + "=" * 50)
    print("INITIALIZATION SUMMARY")
    print("=" * 50)
    if raw_summary is not None:
        print(f"Raw Dataset Rules: {raw_summary.output_rules}")
    else:
        print(f"Raw Dataset Rules: {_raw_dataset_mode_note(args.mode)}")
    if raw_status is not None:
        community_status = raw_status.get("community", {})
        enhanced_status = raw_status.get("enhanced", {})
        community_key = f", key={community_status.get('key_source')}" if community_status.get("key_source") else ""
        enhanced_key = f", key={enhanced_status.get('key_source')}" if enhanced_status.get("key_source") else ""
        print(
            "Legacy Dataset Files: "
            f"community={community_status.get('actual_path', community_status.get('requested_path', 'unknown'))}, "
            f"enhanced={enhanced_status.get('actual_path', enhanced_status.get('requested_path', 'unknown'))}"
        )
        print(
            "Legacy Dataset Sources: "
            f"community={community_status.get('rule_count', 0)} "
            f"({community_status.get('reason', 'unknown')}{community_key}), "
            f"enhanced={enhanced_status.get('rule_count', 0)} "
            f"({enhanced_status.get('reason', 'unknown')}{enhanced_key})"
        )
    print(f"Governed Release Rules: {release_result['release_rule_count']}")
    print(f"Official Patterns: {official_result['pattern_count']}")
    if normalized_mode != args.mode:
        print(f"Mode Alias Normalized To: {normalized_mode}")
    if trusted_init_result is not None:
        print(f"Indexed Trusted Rules: {trusted_init_result['indexed_rule_count']}")
        if trusted_compatible_rule_count is not None:
            print(f"Trusted Rules Compatible With Target Pair: {trusted_compatible_rule_count}")
        if trusted_runtime_dataset_result is not None:
            print(f"Trusted Target Runtime Dataset: {trusted_runtime_dataset_result['dataset_path']}")
        if trusted_target_profile is not None:
            print(
                "Trusted Target Profile: "
                f"{trusted_target_profile['compatibility_mode']} "
                f"(spring_line={trusted_target_profile['spring_line']}, "
                f"micronaut_line={trusted_target_profile['micronaut_line']})"
            )
            print(
                "Trusted Line-Specific Rules: "
                f"spring={trusted_target_profile['spring_line_specific_rule_count']}, "
                f"micronaut={trusted_target_profile['micronaut_line_specific_rule_count']}, "
                f"pair={trusted_target_profile['pair_line_specific_rule_count']}"
            )
            category_counts = trusted_target_profile.get("compatible_category_counts", {})
            if category_counts:
                counts_text = ", ".join(f"{name}={count}" for name, count in category_counts.items())
                print(f"Trusted Compatible Rule Categories: {counts_text}")
            if trusted_target_profile.get("profile_path"):
                print(f"Trusted Target Profile Report: {trusted_target_profile['profile_path']}")
        if trusted_target_platform_snapshot is not None:
            print(
                "Target Platform Managed Dependencies: "
                f"{trusted_target_platform_snapshot['managed_dependency_count']}"
            )
            print(f"Target Platform Snapshot: {trusted_target_platform_snapshot['snapshot_path']}")
        print(f"Trusted DB Path: {trusted_init_result['db_path']}")
        if trusted_manifest_path is not None:
            print(f"Trusted KB Manifest: {trusted_manifest_path}")
    if legacy_init_result is not None:
        if normalized_mode == "hybrid":
            print(f"Hybrid Runtime Rules: {legacy_init_result['indexed_rule_count']}")
            print(f"Hybrid DB Path: {legacy_init_result['db_path']}")
        else:
            print(f"Indexed Legacy Rules: {legacy_init_result['indexed_rule_count']}")
            print(f"Legacy DB Path: {legacy_init_result['db_path']}")
    if candidate_init_result is not None:
        print(f"Candidate Runtime Rules: {candidate_dataset_result['rule_count']}")
        print(f"Indexed Candidate Rules: {candidate_init_result['indexed_rule_count']}")
        print(f"Candidate DB Path: {candidate_init_result['db_path']}")
    if max_init_result is not None:
        print(f"Max Runtime Rules: {max_dataset_result['rule_count']}")
        print(f"Indexed Max Rules: {max_init_result['indexed_rule_count']}")
        print(f"Max DB Path: {max_init_result['db_path']}")
    print(f"Governed Release Runtime Dataset: {release_result['runtime_dataset_path']}")
    if smoke_result is not None:
        print(f"KB Smoke OK: {smoke_result['ok']}")
        print(f"KB Smoke Report: {smoke_result['report_path']}")
    if legacy_init_result is not None:
        if normalized_mode == "hybrid":
            print(f"Hybrid Runtime Dataset: {Path(args.corpus_root) / 'validated_patterns' / 'extended' / 'runtime_dataset.json'}")
        else:
            print(f"Legacy Runtime Dataset: {Path(args.corpus_root) / 'validated_patterns' / 'extended' / 'runtime_dataset.json'}")
    if candidate_dataset_result is not None:
        print(f"Candidate Runtime Dataset: {candidate_dataset_result['dataset_path']}")
    if max_dataset_result is not None:
        print(f"Max Runtime Dataset: {max_dataset_result['dataset_path']}")
    if audit_result is not None:
        print(f"Chroma Audit Trust: {audit_result.get('trust_level', 'unknown')}")
        print(f"Distribution Ready: {audit_result.get('distribution_ready', False)}")
        if audit_result.get("report_path"):
            print(f"Chroma Audit Report: {audit_result['report_path']}")
        elif audit_result.get("reason"):
            print(f"Chroma Audit Note: {audit_result['reason']}")
    print("-" * 50)
    return 0


def _build_subcommand_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spring Boot to Micronaut Migration Agent")
    subparsers = parser.add_subparsers(dest="command")

    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Migrate a Spring Boot project to Micronaut",
    )
    migrate_parser.add_argument("input", help="Path to the source Spring Boot project directory")
    migrate_parser.add_argument("output", help="Path to the target Micronaut project directory")
    migrate_parser.add_argument("--spring-version", default="3.4.5", help="Source Spring Boot version")
    migrate_parser.add_argument("--micronaut-version", default="4.10.8", help="Target Micronaut version")
    migrate_parser.add_argument(
        "--build-tool",
        choices=["maven", "gradle"],
        help="Force Maven or Gradle when the source tree contains both build files",
    )

    init_parser = subparsers.add_parser(
        "init",
        help="Build the governed dataset, initialize the vector DB, and run smoke/audit checks",
    )
    init_parser.add_argument("--corpus-root", default="corpus", help="Corpus root directory")
    init_parser.add_argument("--spring-version", default="3.4.5", help="Spring version for release smoke/audit")
    init_parser.add_argument("--micronaut-version", default="4.10.8", help="Micronaut version for release smoke/audit")
    init_parser.add_argument(
        "--mode",
        choices=["trusted", "legacy", "hybrid", "candidate", "max", "all", "extended", "both", "full"],
        default="trusted",
        help=(
            "trusted = governed release DB only, legacy = old encrypted/raw dataset DB, "
            "hybrid = trusted plus legacy/raw, candidate = trusted plus staged candidates, "
            "max = widest local experimental DB, all = governed plus oldest legacy encrypted datasets, "
            "extended = legacy alias, both = hybrid alias, full = all alias"
        ),
    )
    init_parser.add_argument("--skip-audit", action="store_true", help="Skip the final Chroma audit report")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])

    try:
        parser = _build_subcommand_parser()
        args = parser.parse_args(argv)
        if args.command == "init":
            return _run_init(args)
        if args.command == "migrate":
            return _run_migration(args)
        parser.print_help()
        return 2
    except Exception as exc:
        print(f"\nCRITICAL ERROR: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
