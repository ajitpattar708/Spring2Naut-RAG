import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from src.agent.patterns.official_normalizer import write_normalized_official_patterns
from src.agent.patterns.github_normalizer import write_normalized_github_candidates
from src.agent.patterns.repository import PatternCorpusRepository
from src.agent.patterns.schema import VersionedPattern


def _load_pattern_index(index_path: Path) -> List[VersionedPattern]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return [VersionedPattern.from_dict(item) for item in payload.get("patterns", [])]


def _pattern_key(pattern: VersionedPattern) -> Tuple[str, str, str]:
    return (
        pattern.pattern_type.value,
        pattern.spring_pattern.strip(),
        pattern.micronaut_pattern.strip(),
    )


def _source_key(pattern: VersionedPattern) -> Tuple[str, str]:
    return (
        pattern.pattern_type.value,
        pattern.spring_pattern.strip(),
    )


@dataclass
class PromotionReport:
    staged_count: int
    duplicate_count: int
    conflict_count: int
    pending_review_count: int
    staged_ids: List[str]
    duplicate_ids: List[str]
    conflict_ids: List[str]
    pending_review_ids: List[str]
    conflicts_by_source_pattern: Dict[str, List[str]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "staged_count": self.staged_count,
            "duplicate_count": self.duplicate_count,
            "conflict_count": self.conflict_count,
            "pending_review_count": self.pending_review_count,
            "staged_ids": self.staged_ids,
            "duplicate_ids": self.duplicate_ids,
            "conflict_ids": self.conflict_ids,
            "pending_review_ids": self.pending_review_ids,
            "conflicts_by_source_pattern": self.conflicts_by_source_pattern,
        }


def evaluate_promotions(
    official_patterns: List[VersionedPattern],
    github_patterns: List[VersionedPattern],
    confidence_threshold: float = 0.7,
) -> Tuple[List[VersionedPattern], PromotionReport]:
    official_by_exact = {_pattern_key(pattern): pattern for pattern in official_patterns}
    official_by_source: Dict[Tuple[str, str], List[VersionedPattern]] = defaultdict(list)
    for pattern in official_patterns:
        official_by_source[_source_key(pattern)].append(pattern)

    staged_patterns: List[VersionedPattern] = []
    duplicate_ids: List[str] = []
    conflict_ids: List[str] = []
    pending_review_ids: List[str] = []
    conflicts_by_source_pattern: Dict[str, List[str]] = {}

    for candidate in github_patterns:
        exact_key = _pattern_key(candidate)
        source_key = _source_key(candidate)
        source_label = f"{candidate.pattern_type.value}:{candidate.spring_pattern}"

        if exact_key in official_by_exact:
            duplicate_ids.append(candidate.pattern_id)
            continue

        conflicting_targets = sorted({pattern.micronaut_pattern for pattern in official_by_source.get(source_key, [])})
        if conflicting_targets:
            conflict_ids.append(candidate.pattern_id)
            conflicts_by_source_pattern[source_label] = conflicting_targets
            continue

        if candidate.confidence < confidence_threshold:
            pending_review_ids.append(candidate.pattern_id)
            continue

        staged_patterns.append(candidate)

    report = PromotionReport(
        staged_count=len(staged_patterns),
        duplicate_count=len(duplicate_ids),
        conflict_count=len(conflict_ids),
        pending_review_count=len(pending_review_ids),
        staged_ids=[pattern.pattern_id for pattern in staged_patterns],
        duplicate_ids=duplicate_ids,
        conflict_ids=conflict_ids,
        pending_review_ids=pending_review_ids,
        conflicts_by_source_pattern=conflicts_by_source_pattern,
    )
    return staged_patterns, report


def build_staged_payload(patterns: List[VersionedPattern]) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "catalog_type": "staged_candidate_patterns",
        "pattern_count": len(patterns),
        "patterns": [pattern.to_dict() for pattern in patterns],
    }


def write_promotion_outputs(corpus_root: str = "corpus", confidence_threshold: float = 0.7) -> Dict[str, object]:
    repository = PatternCorpusRepository(root=corpus_root)
    repository.initialize_layout()
    write_normalized_official_patterns(corpus_root=corpus_root)
    write_normalized_github_candidates(corpus_root=corpus_root)

    official_index = Path(corpus_root) / "official_docs" / "normalized" / "index.json"
    github_index = Path(corpus_root) / "github_candidates" / "normalized" / "index.json"
    staged_root = Path(corpus_root) / "staged_patterns" / "candidates"
    notes_root = Path(corpus_root) / "staged_patterns" / "review_notes"
    staged_root.mkdir(parents=True, exist_ok=True)
    notes_root.mkdir(parents=True, exist_ok=True)

    official_patterns = _load_pattern_index(official_index)
    github_patterns = _load_pattern_index(github_index)
    staged_patterns, report = evaluate_promotions(
        official_patterns,
        github_patterns,
        confidence_threshold=confidence_threshold,
    )

    staged_payload = build_staged_payload(staged_patterns)
    staged_index_path = staged_root / "index.json"
    staged_index_path.write_text(json.dumps(staged_payload, indent=2), encoding="utf-8")

    staged_pattern_paths = []
    for pattern in staged_patterns:
        path = staged_root / f"{pattern.pattern_id}.json"
        path.write_text(json.dumps(pattern.to_dict(), indent=2), encoding="utf-8")
        staged_pattern_paths.append(str(path))

    report_path = notes_root / "promotion_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    return {
        "staged_index_path": str(staged_index_path),
        "staged_pattern_count": len(staged_pattern_paths),
        "staged_pattern_files": staged_pattern_paths,
        "promotion_report_path": str(report_path),
        "report": report.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description="Promote normalized patterns into the staged corpus with conflict and confidence gating")
    parser.add_argument("--corpus-root", default="corpus", help="Corpus root directory")
    parser.add_argument("--confidence-threshold", type=float, default=0.7, help="Minimum confidence for staging")
    parser.add_argument("--write", action="store_true", help="Write staged promotion outputs into the corpus")
    args = parser.parse_args()

    if args.write:
        print(
            json.dumps(
                write_promotion_outputs(
                    corpus_root=args.corpus_root,
                    confidence_threshold=args.confidence_threshold,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return

    print(json.dumps({"message": "Use --write to materialize staged promotion outputs."}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
