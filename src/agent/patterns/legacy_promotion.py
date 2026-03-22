import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.agent.patterns.legacy_bootstrap import write_legacy_bootstrap
from src.agent.patterns.official_normalizer import write_normalized_official_patterns
from src.agent.patterns.promotion import _load_pattern_index, _pattern_key, _source_key
from src.agent.patterns.repository import PatternCorpusRepository
from src.agent.patterns.schema import PatternEvidence, SourceKind, ValidationStatus, VersionWindow, VersionedPattern


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "pattern"


def _derive_window(specs: Sequence[str]) -> VersionWindow:
    unique_specs = sorted({spec.strip() for spec in specs if spec and spec.strip()})
    if not unique_specs:
        return VersionWindow()
    if len(unique_specs) == 1:
        return VersionWindow(spec=unique_specs[0])

    majors = {spec.split(".")[0] for spec in unique_specs if "." in spec}
    if len(majors) == 1:
        major = next(iter(majors))
        return VersionWindow(spec=f"{major}.x")

    return VersionWindow(minimum=min(unique_specs), maximum=max(unique_specs))


def _runtime_equivalent_key(pattern: VersionedPattern) -> Tuple[str, str, str, Optional[str], Optional[str]]:
    return (
        pattern.category,
        pattern.spring_pattern.strip(),
        pattern.micronaut_pattern.strip(),
        pattern.spring_versions.spec,
        pattern.micronaut_versions.spec,
    )


@dataclass(frozen=True)
class LegacyPromotionReport:
    unique_mapping_count: int
    promoted_count: int
    duplicate_count: int
    conflict_count: int
    pending_review_count: int
    promoted_ids: List[str]
    duplicate_ids: List[str]
    conflict_ids: List[str]
    pending_review_ids: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "unique_mapping_count": self.unique_mapping_count,
            "promoted_count": self.promoted_count,
            "duplicate_count": self.duplicate_count,
            "conflict_count": self.conflict_count,
            "pending_review_count": self.pending_review_count,
            "promoted_ids": self.promoted_ids,
            "duplicate_ids": self.duplicate_ids,
            "conflict_ids": self.conflict_ids,
            "pending_review_ids": self.pending_review_ids,
        }


def consolidate_legacy_patterns(legacy_patterns: Iterable[VersionedPattern]) -> List[VersionedPattern]:
    grouped: Dict[Tuple[str, str, str, str], Dict[str, object]] = defaultdict(
        lambda: {
            "spring_versions": set(),
            "micronaut_versions": set(),
            "examples": [],
            "evidence": [],
            "source_kind": SourceKind.GENERATED,
            "status": ValidationStatus.VALIDATED,
            "confidence": 0.85,
            "complexity": "medium",
            "metadata": {},
            "pattern": None,
        }
    )

    for pattern in legacy_patterns:
        key = (
            pattern.pattern_type.value,
            pattern.spring_pattern.strip(),
            pattern.micronaut_pattern.strip(),
            pattern.category,
        )
        bucket = grouped[key]
        bucket["pattern"] = pattern
        if pattern.spring_versions.spec:
            bucket["spring_versions"].add(pattern.spring_versions.spec)
        if pattern.micronaut_versions.spec:
            bucket["micronaut_versions"].add(pattern.micronaut_versions.spec)
        bucket["examples"].extend(pattern.examples)
        bucket["evidence"].extend(pattern.evidence)
        bucket["metadata"] = {
            **bucket["metadata"],
            **pattern.metadata,
        }

    consolidated: List[VersionedPattern] = []
    for (pattern_type, spring_pattern, micronaut_pattern, category), bucket in grouped.items():
        source_pattern: VersionedPattern = bucket["pattern"]  # type: ignore[assignment]
        spring_specs = sorted(bucket["spring_versions"])
        micronaut_specs = sorted(bucket["micronaut_versions"])
        promoted_id = f"legacy_promoted.{pattern_type}.{_slugify(spring_pattern)}"

        evidence = [
            PatternEvidence(
                source_kind=SourceKind.GENERATED,
                source_ref="validated_patterns/archives/legacy_runtime_bootstrap",
                title="Legacy runtime bootstrap consolidation",
                notes=(
                    f"Support count {len(spring_specs) * len(micronaut_specs)} across "
                    f"{len(spring_specs)} Spring windows and {len(micronaut_specs)} Micronaut windows."
                ),
            )
        ]

        consolidated.append(
            VersionedPattern(
                pattern_id=promoted_id,
                pattern_type=source_pattern.pattern_type,
                spring_pattern=spring_pattern,
                micronaut_pattern=micronaut_pattern,
                description=source_pattern.description,
                spring_versions=_derive_window(spring_specs),
                micronaut_versions=_derive_window(micronaut_specs),
                status=ValidationStatus.VALIDATED,
                confidence=0.9,
                complexity=source_pattern.complexity,
                category=category,
                source_kind=SourceKind.GENERATED,
                evidence=evidence,
                examples=list(source_pattern.examples),
                metadata={
                    **dict(bucket["metadata"]),
                    "promotion_source": "legacy_runtime_bootstrap",
                    "support_count": len(spring_specs) * len(micronaut_specs),
                    "spring_version_support_count": len(spring_specs),
                    "micronaut_version_support_count": len(micronaut_specs),
                    "spring_version_specs": spring_specs,
                    "micronaut_version_specs": micronaut_specs,
                },
            )
        )

    return consolidated


def evaluate_legacy_promotions(
    official_patterns: Sequence[VersionedPattern],
    legacy_patterns: Sequence[VersionedPattern],
    minimum_support_count: int = 20,
    require_cross_version_support: bool = True,
) -> Tuple[List[VersionedPattern], LegacyPromotionReport]:
    official_by_exact = {_pattern_key(pattern): pattern for pattern in official_patterns}
    official_by_runtime_key = {_runtime_equivalent_key(pattern): pattern for pattern in official_patterns}
    official_by_source: Dict[Tuple[str, str], List[VersionedPattern]] = defaultdict(list)
    for pattern in official_patterns:
        official_by_source[_source_key(pattern)].append(pattern)

    consolidated = consolidate_legacy_patterns(legacy_patterns)
    promoted: List[VersionedPattern] = []
    duplicate_ids: List[str] = []
    conflict_ids: List[str] = []
    pending_review_ids: List[str] = []

    for pattern in consolidated:
        exact_key = _pattern_key(pattern)
        source_key = _source_key(pattern)
        support_count = int(pattern.metadata.get("support_count", 0))
        spring_support = int(pattern.metadata.get("spring_version_support_count", 0))
        micronaut_support = int(pattern.metadata.get("micronaut_version_support_count", 0))

        if exact_key in official_by_exact or _runtime_equivalent_key(pattern) in official_by_runtime_key:
            duplicate_ids.append(pattern.pattern_id)
            continue

        conflicting_targets = {
            official_pattern.micronaut_pattern
            for official_pattern in official_by_source.get(source_key, [])
            if official_pattern.micronaut_pattern != pattern.micronaut_pattern
        }
        if conflicting_targets:
            conflict_ids.append(pattern.pattern_id)
            continue

        if support_count < minimum_support_count:
            pending_review_ids.append(pattern.pattern_id)
            continue

        if require_cross_version_support and (spring_support < 2 or micronaut_support < 2):
            pending_review_ids.append(pattern.pattern_id)
            continue

        promoted.append(pattern)

    report = LegacyPromotionReport(
        unique_mapping_count=len(consolidated),
        promoted_count=len(promoted),
        duplicate_count=len(duplicate_ids),
        conflict_count=len(conflict_ids),
        pending_review_count=len(pending_review_ids),
        promoted_ids=[pattern.pattern_id for pattern in promoted],
        duplicate_ids=duplicate_ids,
        conflict_ids=conflict_ids,
        pending_review_ids=pending_review_ids,
    )
    return promoted, report


def write_legacy_promotion_outputs(
    corpus_root: str = "corpus",
    minimum_support_count: int = 20,
    require_cross_version_support: bool = True,
) -> Dict[str, object]:
    repository = PatternCorpusRepository(root=corpus_root)
    repository.initialize_layout()
    write_normalized_official_patterns(corpus_root=corpus_root)
    write_legacy_bootstrap(corpus_root=corpus_root)

    official_index = Path(corpus_root) / "official_docs" / "normalized" / "index.json"
    legacy_index = Path(corpus_root) / "validated_patterns" / "archives" / "legacy_runtime_bootstrap" / "index.json"
    target_root = Path(corpus_root) / "validated_patterns" / "release" / "legacy_promoted"
    target_root.mkdir(parents=True, exist_ok=True)

    official_patterns = _load_pattern_index(official_index)
    legacy_patterns = _load_pattern_index(legacy_index)
    promoted, report = evaluate_legacy_promotions(
        official_patterns=official_patterns,
        legacy_patterns=legacy_patterns,
        minimum_support_count=minimum_support_count,
        require_cross_version_support=require_cross_version_support,
    )

    index_payload = {
        "schema_version": 1,
        "catalog_type": "legacy_promoted_patterns",
        "pattern_count": len(promoted),
        "patterns": [pattern.to_dict() for pattern in promoted],
    }
    index_path = target_root / "index.json"
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    pattern_files = []
    for pattern in promoted:
        path = target_root / f"{pattern.pattern_id}.json"
        path.write_text(json.dumps(pattern.to_dict(), indent=2), encoding="utf-8")
        pattern_files.append(str(path))

    report_path = target_root / "promotion_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    return {
        "index_path": str(index_path),
        "pattern_count": len(promoted),
        "pattern_files": pattern_files,
        "report_path": str(report_path),
        "report": report.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description="Promote consolidated legacy runtime patterns into release-grade validated additions")
    parser.add_argument("--corpus-root", default="corpus", help="Corpus root directory")
    parser.add_argument("--minimum-support-count", type=int, default=20, help="Minimum cross-version support count")
    parser.add_argument(
        "--disable-cross-version-requirement",
        action="store_true",
        help="Allow mappings without multi-version support",
    )
    parser.add_argument("--write", action="store_true", help="Write promoted legacy patterns into the validated release area")
    args = parser.parse_args()

    if args.write:
        print(
            json.dumps(
                write_legacy_promotion_outputs(
                    corpus_root=args.corpus_root,
                    minimum_support_count=args.minimum_support_count,
                    require_cross_version_support=not args.disable_cross_version_requirement,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return

    print(
        json.dumps(
            {"message": "Use --write to materialize promoted legacy release patterns."},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
