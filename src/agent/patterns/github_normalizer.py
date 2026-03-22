import argparse
import json
from pathlib import Path
from typing import Dict, List

from src.agent.patterns.github_candidates import default_github_candidate_sources, write_github_candidate_catalog
from src.agent.patterns.repository import PatternCorpusRepository
from src.agent.patterns.schema import (
    PatternEvidence,
    PatternType,
    SourceKind,
    ValidationStatus,
    VersionWindow,
    VersionedPattern,
)


def _find_candidate(candidate_id: str):
    for candidate in default_github_candidate_sources():
        if candidate.candidate_id == candidate_id:
            return candidate
    raise KeyError(f"Unknown GitHub candidate id: {candidate_id}")


def curated_github_candidate_patterns() -> List[VersionedPattern]:
    petclinic = _find_candidate("spring_petclinic")
    micronaut_core = _find_candidate("micronaut_core")
    micronaut_guides = _find_candidate("micronaut_guides")
    spring_boot = _find_candidate("spring_boot")

    return [
        VersionedPattern(
            pattern_id="github.annotation.put_mapping",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@PutMapping",
            micronaut_pattern="@Put",
            description="Curated GitHub candidate for PUT endpoint annotation migration.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.CANDIDATE,
            confidence=0.74,
            complexity="low",
            category="annotations",
            source_kind=SourceKind.GITHUB_REPO,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.GITHUB_REPO,
                    source_ref=petclinic.repository_url,
                    title=petclinic.repository,
                    notes="Observed Spring MVC endpoint usage in representative app code.",
                ),
                PatternEvidence(
                    source_kind=SourceKind.GITHUB_REPO,
                    source_ref=micronaut_core.repository_url,
                    title=micronaut_core.repository,
                    notes="Observed Micronaut HTTP annotation target idiom.",
                ),
            ],
            metadata={"candidate_id": petclinic.candidate_id},
        ),
        VersionedPattern(
            pattern_id="github.annotation.delete_mapping",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@DeleteMapping",
            micronaut_pattern="@Delete",
            description="Curated GitHub candidate for DELETE endpoint annotation migration.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.CANDIDATE,
            confidence=0.74,
            complexity="low",
            category="annotations",
            source_kind=SourceKind.GITHUB_REPO,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.GITHUB_REPO,
                    source_ref=petclinic.repository_url,
                    title=petclinic.repository,
                    notes="Observed Spring MVC endpoint usage in representative app code.",
                ),
                PatternEvidence(
                    source_kind=SourceKind.GITHUB_REPO,
                    source_ref=micronaut_core.repository_url,
                    title=micronaut_core.repository,
                    notes="Observed Micronaut HTTP annotation target idiom.",
                ),
            ],
            metadata={"candidate_id": petclinic.candidate_id},
        ),
        VersionedPattern(
            pattern_id="github.annotation.autowired",
            pattern_type=PatternType.DEPENDENCY_INJECTION,
            spring_pattern="@Autowired",
            micronaut_pattern="@Inject",
            description="Curated GitHub candidate for dependency injection annotation migration.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.CANDIDATE,
            confidence=0.72,
            complexity="low",
            category="annotations",
            source_kind=SourceKind.GITHUB_REPO,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.GITHUB_REPO,
                    source_ref=spring_boot.repository_url,
                    title=spring_boot.repository,
                    notes="Observed Boot-side DI annotation usage.",
                ),
                PatternEvidence(
                    source_kind=SourceKind.GITHUB_REPO,
                    source_ref=micronaut_core.repository_url,
                    title=micronaut_core.repository,
                    notes="Observed Micronaut injection target idiom.",
                ),
            ],
            metadata={"candidate_id": spring_boot.candidate_id},
        ),
        VersionedPattern(
            pattern_id="github.annotation.value_property",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@Value",
            micronaut_pattern="@Property",
            description="Curated GitHub candidate for property injection annotation migration.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.CANDIDATE,
            confidence=0.68,
            complexity="medium",
            category="annotations",
            source_kind=SourceKind.GITHUB_REPO,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.GITHUB_REPO,
                    source_ref=spring_boot.repository_url,
                    title=spring_boot.repository,
                    notes="Observed Spring property injection usage.",
                ),
                PatternEvidence(
                    source_kind=SourceKind.GITHUB_REPO,
                    source_ref=micronaut_guides.repository_url,
                    title=micronaut_guides.repository,
                    notes="Observed Micronaut property injection idiom.",
                ),
            ],
            metadata={"candidate_id": spring_boot.candidate_id},
        ),
        VersionedPattern(
            pattern_id="github.di.qualifier_named",
            pattern_type=PatternType.DEPENDENCY_INJECTION,
            spring_pattern="Qualifier injection",
            micronaut_pattern="Named injection",
            description="Curated GitHub candidate for qualifier-to-named dependency injection migration.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.CANDIDATE,
            confidence=0.67,
            complexity="medium",
            category="code_patterns",
            source_kind=SourceKind.GITHUB_REPO,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.GITHUB_REPO,
                    source_ref=spring_boot.repository_url,
                    title=spring_boot.repository,
                    notes="Observed Spring bean qualification patterns.",
                ),
                PatternEvidence(
                    source_kind=SourceKind.GITHUB_REPO,
                    source_ref=micronaut_core.repository_url,
                    title=micronaut_core.repository,
                    notes="Observed Micronaut named injection idiom.",
                ),
            ],
            metadata={"candidate_id": spring_boot.candidate_id},
        ),
        VersionedPattern(
            pattern_id="github.config.configuration_properties",
            pattern_type=PatternType.CONFIGURATION,
            spring_pattern="@ConfigurationProperties",
            micronaut_pattern="@EachProperty",
            description="Curated GitHub candidate for configuration properties migration where collection-like property groups are present.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.CANDIDATE,
            confidence=0.64,
            complexity="medium",
            category="configurations",
            source_kind=SourceKind.GITHUB_REPO,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.GITHUB_REPO,
                    source_ref=spring_boot.repository_url,
                    title=spring_boot.repository,
                    notes="Observed Spring configuration properties usage.",
                ),
                PatternEvidence(
                    source_kind=SourceKind.GITHUB_REPO,
                    source_ref=micronaut_guides.repository_url,
                    title=micronaut_guides.repository,
                    notes="Observed Micronaut configuration pattern examples.",
                ),
            ],
            metadata={"candidate_id": spring_boot.candidate_id},
        ),
        VersionedPattern(
            pattern_id="github.type.response_entity_ok",
            pattern_type=PatternType.TYPE,
            spring_pattern="ResponseEntity.ok",
            micronaut_pattern="HttpResponse.ok",
            description="Curated GitHub candidate for common success response factory migration.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.CANDIDATE,
            confidence=0.71,
            complexity="low",
            category="types",
            source_kind=SourceKind.GITHUB_REPO,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.GITHUB_REPO,
                    source_ref=petclinic.repository_url,
                    title=petclinic.repository,
                    notes="Observed ResponseEntity return paths in application code.",
                ),
                PatternEvidence(
                    source_kind=SourceKind.GITHUB_REPO,
                    source_ref=micronaut_core.repository_url,
                    title=micronaut_core.repository,
                    notes="Observed Micronaut HttpResponse factory usage.",
                ),
            ],
            metadata={"candidate_id": petclinic.candidate_id},
        ),
        VersionedPattern(
            pattern_id="github.type.rest_template_http_client",
            pattern_type=PatternType.TYPE,
            spring_pattern="RestTemplate",
            micronaut_pattern="HttpClient",
            description="Curated GitHub candidate for blocking HTTP client migration.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.CANDIDATE,
            confidence=0.63,
            complexity="medium",
            category="types",
            source_kind=SourceKind.GITHUB_REPO,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.GITHUB_REPO,
                    source_ref=spring_boot.repository_url,
                    title=spring_boot.repository,
                    notes="Observed RestTemplate usage in Spring samples and tests.",
                ),
                PatternEvidence(
                    source_kind=SourceKind.GITHUB_REPO,
                    source_ref=micronaut_guides.repository_url,
                    title=micronaut_guides.repository,
                    notes="Observed Micronaut HttpClient usage in guides.",
                ),
            ],
            metadata={"candidate_id": spring_boot.candidate_id},
        ),
    ]


def build_normalized_github_payload() -> Dict[str, object]:
    patterns = curated_github_candidate_patterns()
    return {
        "schema_version": 1,
        "catalog_type": "github_normalized_candidates",
        "pattern_count": len(patterns),
        "patterns": [pattern.to_dict() for pattern in patterns],
    }


def write_normalized_github_candidates(corpus_root: str = "corpus") -> Dict[str, object]:
    repository = PatternCorpusRepository(root=corpus_root)
    repository.initialize_layout()
    write_github_candidate_catalog(corpus_root=corpus_root)

    normalized_root = Path(corpus_root) / "github_candidates" / "normalized"
    normalized_root.mkdir(parents=True, exist_ok=True)
    patterns_root = normalized_root / "patterns"
    patterns_root.mkdir(parents=True, exist_ok=True)

    patterns = curated_github_candidate_patterns()
    index_payload = build_normalized_github_payload()
    index_path = normalized_root / "index.json"
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    written_paths = []
    for pattern in patterns:
        path = patterns_root / f"{pattern.pattern_id}.json"
        path.write_text(json.dumps(pattern.to_dict(), indent=2), encoding="utf-8")
        written_paths.append(str(path))

    return {
        "index_path": str(index_path),
        "pattern_count": len(written_paths),
        "pattern_files": written_paths,
    }


def main():
    parser = argparse.ArgumentParser(description="Normalize curated GitHub candidates into versioned pattern candidates")
    parser.add_argument("--corpus-root", default="corpus", help="Corpus root directory")
    parser.add_argument("--write", action="store_true", help="Write normalized GitHub candidates into the corpus")
    args = parser.parse_args()

    if args.write:
        print(json.dumps(write_normalized_github_candidates(corpus_root=args.corpus_root), indent=2, sort_keys=True))
        return

    print(json.dumps(build_normalized_github_payload(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
