import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.agent.patterns.legacy_promotion import write_legacy_promotion_outputs
from src.agent.patterns.promotion import _load_pattern_index
from src.agent.patterns.repository import PatternCorpusRepository
from src.agent.patterns.schema import VersionedPattern


GA_SAFE_IDS = {
    "legacy_promoted.annotation.repository",
    "legacy_promoted.annotation.configuration",
    "legacy_promoted.annotation.putmapping",
    "legacy_promoted.annotation.deletemapping",
    "legacy_promoted.annotation.patchmapping",
    "legacy_promoted.annotation.requestmapping",
    "legacy_promoted.annotation.autowired",
    "legacy_promoted.annotation.value",
    "legacy_promoted.annotation.requestheader",
    "legacy_promoted.code_pattern.setter_injection",
    "legacy_promoted.code_pattern.qualifier_injection",
}


RISK_REASONS = {
    "legacy_promoted.annotation.modelattribute": "Argument binding semantics differ and need fixture-backed controller tests.",
    "legacy_promoted.annotation.exceptionhandler": "Exception handling flow needs controller and error-route validation.",
    "legacy_promoted.annotation.controlleradvice": "Global advice behavior needs fixture-backed verification.",
    "legacy_promoted.annotation.enablewebmvc": "Framework enable/disable semantics should be verified with integration fixtures.",
    "legacy_promoted.annotation.enablecaching": "Caching enablement needs runtime bean and behavior validation.",
    "legacy_promoted.annotation.enablescheduling": "Scheduling activation must be verified with runtime fixtures.",
    "legacy_promoted.annotation.enableasync": "Async execution semantics require runtime validation.",
    "legacy_promoted.annotation.enablejparepositories": "JPA repository bootstrap behavior needs persistence fixtures.",
    "legacy_promoted.annotation.enablejpaauditing": "Auditing behavior needs persistence fixtures.",
    "legacy_promoted.configuration.resttemplate_configuration": "HTTP client configuration mapping needs integration fixtures.",
    "legacy_promoted.type.optional_responseentity": "Response wrapping semantics need endpoint fixtures.",
    "legacy_promoted.type.mono_flux_webflux": "Reactive type migration needs runtime and behavioral verification.",
    "legacy_promoted.type.filterchain": "Server filter ordering and request flow need integration fixtures.",
    "legacy_promoted.type.webmvcconfigurer": "MVC customization semantics differ and need framework-level fixtures.",
    "legacy_promoted.code_pattern.commandlinerunner": "Startup lifecycle mapping must be verified with application boot fixtures.",
    "legacy_promoted.code_pattern.applicationlistener": "Application event timing and scope need boot lifecycle fixtures.",
}


@dataclass(frozen=True)
class LegacyReviewReport:
    reviewed_count: int
    ga_ready_count: int
    needs_fixture_count: int
    ga_ready_ids: List[str]
    needs_fixture_ids: List[str]
    reasons_by_pattern_id: Dict[str, str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "reviewed_count": self.reviewed_count,
            "ga_ready_count": self.ga_ready_count,
            "needs_fixture_count": self.needs_fixture_count,
            "ga_ready_ids": self.ga_ready_ids,
            "needs_fixture_ids": self.needs_fixture_ids,
            "reasons_by_pattern_id": self.reasons_by_pattern_id,
        }


def review_legacy_promoted_patterns(patterns: Sequence[VersionedPattern]) -> Tuple[List[VersionedPattern], List[VersionedPattern], LegacyReviewReport]:
    ga_ready: List[VersionedPattern] = []
    needs_fixture: List[VersionedPattern] = []
    reasons: Dict[str, str] = {}

    for pattern in patterns:
        if pattern.pattern_id in GA_SAFE_IDS:
            ga_ready.append(pattern)
            continue

        reason = RISK_REASONS.get(
            pattern.pattern_id,
            "Pattern requires fixture-backed validation before GA release.",
        )
        reasons[pattern.pattern_id] = reason
        needs_fixture.append(pattern)

    report = LegacyReviewReport(
        reviewed_count=len(patterns),
        ga_ready_count=len(ga_ready),
        needs_fixture_count=len(needs_fixture),
        ga_ready_ids=[pattern.pattern_id for pattern in ga_ready],
        needs_fixture_ids=[pattern.pattern_id for pattern in needs_fixture],
        reasons_by_pattern_id=reasons,
    )
    return ga_ready, needs_fixture, report


def write_legacy_review_outputs(corpus_root: str = "corpus") -> Dict[str, object]:
    repository = PatternCorpusRepository(root=corpus_root)
    repository.initialize_layout()
    write_legacy_promotion_outputs(corpus_root=corpus_root)

    promoted_index = Path(corpus_root) / "validated_patterns" / "release" / "legacy_promoted" / "index.json"
    review_root = Path(corpus_root) / "validated_patterns" / "release" / "legacy_reviewed"
    ga_root = review_root / "ga_ready"
    fixture_root = review_root / "needs_fixture_validation"
    ga_root.mkdir(parents=True, exist_ok=True)
    fixture_root.mkdir(parents=True, exist_ok=True)

    promoted_patterns = _load_pattern_index(promoted_index)
    ga_ready, needs_fixture, report = review_legacy_promoted_patterns(promoted_patterns)

    ga_index_payload = {
        "schema_version": 1,
        "catalog_type": "legacy_reviewed_ga_ready_patterns",
        "pattern_count": len(ga_ready),
        "patterns": [pattern.to_dict() for pattern in ga_ready],
    }
    ga_index_path = ga_root / "index.json"
    ga_index_path.write_text(json.dumps(ga_index_payload, indent=2), encoding="utf-8")

    fixture_index_payload = {
        "schema_version": 1,
        "catalog_type": "legacy_reviewed_needs_fixture_patterns",
        "pattern_count": len(needs_fixture),
        "patterns": [pattern.to_dict() for pattern in needs_fixture],
        "reasons_by_pattern_id": report.reasons_by_pattern_id,
    }
    fixture_index_path = fixture_root / "index.json"
    fixture_index_path.write_text(json.dumps(fixture_index_payload, indent=2), encoding="utf-8")

    report_path = review_root / "review_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    return {
        "ga_index_path": str(ga_index_path),
        "needs_fixture_index_path": str(fixture_index_path),
        "report_path": str(report_path),
        "report": report.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description="Review promoted legacy patterns and separate GA-ready mappings from fixture-required mappings")
    parser.add_argument("--corpus-root", default="corpus", help="Corpus root directory")
    parser.add_argument("--write", action="store_true", help="Write reviewed legacy outputs")
    args = parser.parse_args()

    if args.write:
        print(json.dumps(write_legacy_review_outputs(corpus_root=args.corpus_root), indent=2, sort_keys=True))
        return

    print(json.dumps({"message": "Use --write to materialize reviewed legacy outputs."}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
