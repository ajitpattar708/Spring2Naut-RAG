import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

from src.agent.patterns.repository import PatternCorpusRepository
from src.agent.patterns.schema import SourceKind, VersionWindow


@dataclass(frozen=True)
class GitHubCandidateSource:
    candidate_id: str
    repository: str
    repository_url: str
    description: str
    owner_type: str
    source_kind: SourceKind
    spring_versions: VersionWindow = field(default_factory=VersionWindow)
    micronaut_versions: VersionWindow = field(default_factory=VersionWindow)
    include_paths: List[str] = field(default_factory=list)
    extraction_targets: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["source_kind"] = self.source_kind.value
        payload["spring_versions"] = asdict(self.spring_versions)
        payload["micronaut_versions"] = asdict(self.micronaut_versions)
        return payload


def default_github_candidate_sources() -> List[GitHubCandidateSource]:
    return [
        GitHubCandidateSource(
            candidate_id="spring_petclinic",
            repository="spring-projects/spring-petclinic",
            repository_url="https://github.com/spring-projects/spring-petclinic",
            description="Representative Spring Boot MVC application for discovering common controller, service, configuration, and testing patterns.",
            owner_type="official_org",
            source_kind=SourceKind.GITHUB_REPO,
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            include_paths=["src/main/java", "src/test/java", "src/main/resources"],
            extraction_targets=["mvc_annotations", "service_stereotypes", "config_properties", "tests"],
            notes=["Use as a high-signal Spring-side source for common app-layer patterns."],
        ),
        GitHubCandidateSource(
            candidate_id="micronaut_core",
            repository="micronaut-projects/micronaut-core",
            repository_url="https://github.com/micronaut-projects/micronaut-core",
            description="Micronaut core repository for target-framework idioms, annotation usage, bootstrap patterns, and response abstractions.",
            owner_type="official_org",
            source_kind=SourceKind.GITHUB_REPO,
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            include_paths=["http/src", "inject/src", "context/src", "test-suite"],
            extraction_targets=["http_annotations", "singleton_patterns", "http_response_types"],
            notes=["Use as a Micronaut-side target idiom source."],
        ),
        GitHubCandidateSource(
            candidate_id="micronaut_guides",
            repository="micronaut-projects/micronaut-guides",
            repository_url="https://github.com/micronaut-projects/micronaut-guides",
            description="Official Micronaut guides repository containing end-to-end application examples and migration-adjacent code layouts.",
            owner_type="official_org",
            source_kind=SourceKind.GITHUB_REPO,
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            include_paths=["guides", "buildSrc"],
            extraction_targets=["application_bootstrap", "controller_examples", "configuration_examples"],
            notes=["Good source for code-level Micronaut idioms matching official docs."],
        ),
        GitHubCandidateSource(
            candidate_id="spring_boot",
            repository="spring-projects/spring-boot",
            repository_url="https://github.com/spring-projects/spring-boot",
            description="Spring Boot framework repository to trace canonical Boot-side patterns and configuration names across active 3.x lines.",
            owner_type="official_org",
            source_kind=SourceKind.GITHUB_REPO,
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            include_paths=["spring-boot-project", "spring-boot-tests"],
            extraction_targets=["boot_annotations", "configuration_namespaces", "application_bootstrap"],
            notes=["Use for source-of-origin naming and configuration references."],
        ),
    ]


def build_github_candidate_catalog() -> Dict[str, object]:
    candidates = default_github_candidate_sources()
    return {
        "schema_version": 1,
        "catalog_type": "github_candidate_seed_catalog",
        "candidate_count": len(candidates),
        "candidates": [candidate.to_dict() for candidate in candidates],
    }


def write_github_candidate_catalog(corpus_root: str = "corpus") -> Dict[str, object]:
    repository = PatternCorpusRepository(root=corpus_root)
    repository.initialize_layout()

    raw_root = Path(corpus_root) / "github_candidates" / "raw"
    sources_root = raw_root / "sources"
    sources_root.mkdir(parents=True, exist_ok=True)

    catalog = build_github_candidate_catalog()
    catalog_path = raw_root / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    written_sources = []
    for candidate in default_github_candidate_sources():
        candidate_path = sources_root / f"{candidate.candidate_id}.json"
        candidate_path.write_text(json.dumps(candidate.to_dict(), indent=2), encoding="utf-8")
        written_sources.append(str(candidate_path))

    return {
        "catalog_path": str(catalog_path),
        "candidate_count": len(written_sources),
        "candidate_files": written_sources,
    }


def main():
    parser = argparse.ArgumentParser(description="Materialize curated GitHub candidate sources into the pattern corpus")
    parser.add_argument("--corpus-root", default="corpus", help="Corpus root directory")
    parser.add_argument("--write", action="store_true", help="Write the GitHub candidate catalog into the corpus")
    args = parser.parse_args()

    if args.write:
        print(json.dumps(write_github_candidate_catalog(corpus_root=args.corpus_root), indent=2, sort_keys=True))
        return

    print(json.dumps(build_github_candidate_catalog(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
