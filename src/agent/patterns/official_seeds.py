import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

from src.agent.patterns.repository import PatternCorpusRepository
from src.agent.patterns.schema import SourceKind, VersionWindow


@dataclass(frozen=True)
class OfficialSeedSource:
    seed_id: str
    title: str
    url: str
    owner: str
    source_kind: SourceKind
    description: str
    spring_versions: VersionWindow = field(default_factory=VersionWindow)
    micronaut_versions: VersionWindow = field(default_factory=VersionWindow)
    tags: List[str] = field(default_factory=list)
    extraction_targets: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["source_kind"] = self.source_kind.value
        payload["spring_versions"] = asdict(self.spring_versions)
        payload["micronaut_versions"] = asdict(self.micronaut_versions)
        return payload


def _spring_boot_release_seed(
    *,
    seed_id: str,
    title: str,
    url: str,
    spring_spec: str,
) -> OfficialSeedSource:
    return OfficialSeedSource(
        seed_id=seed_id,
        title=title,
        url=url,
        owner="Spring",
        source_kind=SourceKind.OFFICIAL_DOC,
        description="Official Spring Boot release notes for version-specific migration baselines, upgrade caveats, and dependency-management context.",
        spring_versions=VersionWindow(spec=spring_spec),
        micronaut_versions=VersionWindow(spec="4.x"),
        tags=["spring-boot", "release-notes", spring_spec],
        extraction_targets=[
            "release_highlights",
            "upgrade_notes",
            "dependency_management_baseline",
        ],
    )


def default_official_seed_sources() -> List[OfficialSeedSource]:
    return [
        OfficialSeedSource(
            seed_id="micronaut_spring_latest_guide",
            title="Micronaut for Spring (latest guide)",
            url="https://micronaut-projects.github.io/micronaut-spring/latest/guide/",
            owner="Micronaut",
            source_kind=SourceKind.OFFICIAL_DOC,
            description="Primary source for supported Spring annotations, interfaces, events, controller semantics, unsupported features, and HTTP client compatibility.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            tags=["spring", "micronaut", "compatibility", "supported-features"],
            extraction_targets=[
                "supported_annotations",
                "supported_interfaces",
                "supported_events",
                "mvc_controller_support",
                "unsupported_features",
            ],
        ),
        OfficialSeedSource(
            seed_id="micronaut_spring_5_11_guide",
            title="Micronaut for Spring 5.11.0 Guide",
            url="https://micronaut-projects.github.io/micronaut-spring/5.11.0/guide/",
            owner="Micronaut",
            source_kind=SourceKind.OFFICIAL_DOC,
            description="Versioned Micronaut Spring guide for explicit release-line capture and source stability during corpus generation.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            tags=["spring", "micronaut", "versioned-guide"],
            extraction_targets=[
                "supported_annotations",
                "spring_boot_annotations",
                "unsupported_features",
            ],
        ),
        OfficialSeedSource(
            seed_id="micronaut_spring_5_10_guide",
            title="Micronaut for Spring 5.10.0 Guide",
            url="https://micronaut-projects.github.io/micronaut-spring/5.10.0/guide/",
            owner="Micronaut",
            source_kind=SourceKind.OFFICIAL_DOC,
            description="Secondary versioned Micronaut Spring guide to help compare changes across adjacent compatible release lines.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            tags=["spring", "micronaut", "versioned-guide"],
            extraction_targets=[
                "supported_annotations",
                "spring_data_support",
                "spring_boot_annotations",
            ],
        ),
        OfficialSeedSource(
            seed_id="micronaut_guides_spring_boot_tag",
            title="Micronaut Guides: spring-boot tag",
            url="https://guides.micronaut.io/latest/tag-spring-boot.html",
            owner="Micronaut",
            source_kind=SourceKind.OFFICIAL_DOC,
            description="Official Micronaut migration/comparison guides for Spring Boot concepts and code patterns.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            tags=["guides", "spring-boot", "examples"],
            extraction_targets=[
                "application_class_patterns",
                "bean_definition_patterns",
                "uri_builder_patterns",
                "testing_patterns",
            ],
        ),
        OfficialSeedSource(
            seed_id="micronaut_guides_spring_tag",
            title="Micronaut Guides: spring tag",
            url="https://guides.micronaut.io/latest/tag-spring.html",
            owner="Micronaut",
            source_kind=SourceKind.OFFICIAL_DOC,
            description="Official Micronaut guides for Spring interoperability, including Spring Boot migration-oriented walkthroughs.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            tags=["guides", "spring", "examples"],
            extraction_targets=[
                "run_spring_boot_as_micronaut",
                "micronaut_data_from_spring_boot",
                "annotation_comparisons",
            ],
        ),
        OfficialSeedSource(
            seed_id="spring_boot_reference_index",
            title="Spring Boot Reference Index",
            url="https://docs.spring.io/spring-boot/reference/index.html",
            owner="Spring",
            source_kind=SourceKind.OFFICIAL_DOC,
            description="Official Spring Boot reference index listing active stable 3.3, 3.4, and 3.5 documentation lines for version-aware source targeting.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            tags=["spring-boot", "reference", "versions"],
            extraction_targets=[
                "active_stable_versions",
                "feature_reference_sections",
            ],
        ),
        OfficialSeedSource(
            seed_id="spring_boot_supported_versions",
            title="Spring Boot Supported Versions",
            url="https://github.com/spring-projects/spring-boot/wiki/Supported-Versions",
            owner="Spring",
            source_kind=SourceKind.OFFICIAL_DOC,
            description="Official Spring Boot support policy and maintenance windows used to distinguish actively supported 3.x release lines from archival reference lines.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            tags=["spring-boot", "support-policy", "versions"],
            extraction_targets=[
                "supported_release_lines",
                "maintenance_windows",
                "oss_support_status",
            ],
        ),
        _spring_boot_release_seed(
            seed_id="spring_boot_3_0_release_notes",
            title="Spring Boot 3.0 Release Notes",
            url="https://github.com/spring-projects/spring-boot/wiki/Spring-Boot-3.0-Release-Notes",
            spring_spec="3.0.x",
        ),
        _spring_boot_release_seed(
            seed_id="spring_boot_3_1_release_notes",
            title="Spring Boot 3.1 Release Notes",
            url="https://github.com/spring-projects/spring-boot/wiki/Spring-Boot-3.1-Release-Notes",
            spring_spec="3.1.x",
        ),
        _spring_boot_release_seed(
            seed_id="spring_boot_3_2_release_notes",
            title="Spring Boot 3.2 Release Notes",
            url="https://github.com/spring-projects/spring-boot/wiki/Spring-Boot-3.2-Release-Notes",
            spring_spec="3.2.x",
        ),
        _spring_boot_release_seed(
            seed_id="spring_boot_3_3_release_notes",
            title="Spring Boot 3.3 Release Notes",
            url="https://github.com/spring-projects/spring-boot/wiki/Spring-Boot-3.3-Release-Notes",
            spring_spec="3.3.x",
        ),
        _spring_boot_release_seed(
            seed_id="spring_boot_3_4_release_notes",
            title="Spring Boot 3.4 Release Notes",
            url="https://github.com/spring-projects/spring-boot/wiki/Spring-Boot-3.4-Release-Notes",
            spring_spec="3.4.x",
        ),
        _spring_boot_release_seed(
            seed_id="spring_boot_3_5_release_notes",
            title="Spring Boot 3.5 Release Notes",
            url="https://github.com/spring-projects/spring-boot/wiki/Spring-Boot-3.5-Release-Notes",
            spring_spec="3.5.x",
        ),
        OfficialSeedSource(
            seed_id="micronaut_release_archive",
            title="Micronaut Release Announcements Archive",
            url="https://micronaut.io/category/release-announcements/",
            owner="Micronaut",
            source_kind=SourceKind.OFFICIAL_DOC,
            description="Official Micronaut release archive for mapping migration-corpus changes to Micronaut 4.x release lines.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            tags=["micronaut", "release-history", "version-matrix"],
            extraction_targets=[
                "release_line_catalog",
                "version_change_tracking",
            ],
        ),
        OfficialSeedSource(
            seed_id="micronaut_framework_4_9_0_release",
            title="Micronaut Framework 4.9.0 Released",
            url="https://micronaut.io/2025/06/30/micronaut-framework-4-9-0-released/",
            owner="Micronaut",
            source_kind=SourceKind.OFFICIAL_DOC,
            description="Official Micronaut 4.9.0 release notes for version-specific target-framework capabilities and migration-aware feature tracking.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.9.x"),
            tags=["micronaut", "release-notes", "4.9.x"],
            extraction_targets=[
                "target_capabilities",
                "version_specific_features",
                "migration_notes",
            ],
        ),
        OfficialSeedSource(
            seed_id="micronaut_framework_4_10_0_release",
            title="Micronaut Framework 4.10.0 Released",
            url="https://micronaut.io/2025/10/22/micronaut-framework-4-10-0-released/",
            owner="Micronaut",
            source_kind=SourceKind.OFFICIAL_DOC,
            description="Official Micronaut 4.10.0 release notes for new target-side features and compatibility context relevant to migration outputs.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.10.x"),
            tags=["micronaut", "release-notes", "4.10.x"],
            extraction_targets=[
                "target_capabilities",
                "version_specific_features",
                "migration_notes",
            ],
        ),
        OfficialSeedSource(
            seed_id="micronaut_framework_4_10_1_release",
            title="Micronaut Framework 4.10.1 Released",
            url="https://micronaut.io/2025/10/29/micronaut-framework-4-10-1/",
            owner="Micronaut",
            source_kind=SourceKind.OFFICIAL_DOC,
            description="Official Micronaut 4.10.1 release notes for patch-line compatibility context and module release composition.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.10.x"),
            tags=["micronaut", "release-notes", "4.10.x", "patch-release"],
            extraction_targets=[
                "release_line_catalog",
                "module_versions",
                "compatibility_context",
            ],
        ),
    ]


def build_official_seed_catalog() -> Dict[str, object]:
    seeds = default_official_seed_sources()
    return {
        "schema_version": 1,
        "catalog_type": "official_doc_seed_catalog",
        "seed_count": len(seeds),
        "seeds": [seed.to_dict() for seed in seeds],
    }


def write_official_seed_catalog(corpus_root: str = "corpus") -> Dict[str, object]:
    repository = PatternCorpusRepository(root=corpus_root)
    repository.initialize_layout()

    raw_root = Path(corpus_root) / "official_docs" / "raw"
    sources_root = raw_root / "sources"
    sources_root.mkdir(parents=True, exist_ok=True)

    catalog = build_official_seed_catalog()
    catalog_path = raw_root / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    written_sources = []
    for seed in default_official_seed_sources():
        seed_path = sources_root / f"{seed.seed_id}.json"
        seed_path.write_text(json.dumps(seed.to_dict(), indent=2), encoding="utf-8")
        written_sources.append(str(seed_path))

    return {
        "catalog_path": str(catalog_path),
        "seed_count": len(written_sources),
        "seed_files": written_sources,
    }


def main():
    parser = argparse.ArgumentParser(description="Materialize official documentation seed sources into the pattern corpus")
    parser.add_argument("--corpus-root", default="corpus", help="Corpus root directory")
    parser.add_argument("--write", action="store_true", help="Write the official seed catalog into the corpus")
    args = parser.parse_args()

    if args.write:
        print(json.dumps(write_official_seed_catalog(corpus_root=args.corpus_root), indent=2, sort_keys=True))
        return

    print(json.dumps(build_official_seed_catalog(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
