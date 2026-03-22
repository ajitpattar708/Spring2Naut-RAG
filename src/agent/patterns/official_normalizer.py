import argparse
import json
from pathlib import Path
from typing import Dict, List

from src.agent.patterns.official_seeds import default_official_seed_sources, write_official_seed_catalog
from src.agent.patterns.repository import PatternCorpusRepository
from src.agent.patterns.schema import (
    PatternEvidence,
    PatternType,
    SourceKind,
    ValidationStatus,
    VersionWindow,
    VersionedPattern,
)


def _find_seed(seed_id: str):
    for seed in default_official_seed_sources():
        if seed.seed_id == seed_id:
            return seed
    raise KeyError(f"Unknown official seed id: {seed_id}")


def curated_official_patterns() -> List[VersionedPattern]:
    latest_guide = _find_seed("micronaut_spring_latest_guide")
    spring_boot_ref = _find_seed("spring_boot_reference_index")
    guides_tag = _find_seed("micronaut_guides_spring_boot_tag")
    versioned_guide = _find_seed("micronaut_spring_5_11_guide")
    release_4_9 = _find_seed("micronaut_framework_4_9_0_release")
    release_4_10 = _find_seed("micronaut_framework_4_10_0_release")
    release_4_10_1 = _find_seed("micronaut_framework_4_10_1_release")

    return [
        VersionedPattern(
            pattern_id="official.annotation.rest_controller",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@RestController",
            micronaut_pattern="@Controller",
            description="Migrate Spring REST controllers to Micronaut controllers.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.97,
            complexity="low",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="Controller compatibility and MVC migration baseline.",
                ),
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=release_4_10.url,
                    title=release_4_10.title,
                    notes="Micronaut 4.10.x target release context for current migration outputs.",
                ),
            ],
            metadata={"seed_id": latest_guide.seed_id, "release_seed_ids": [release_4_10.seed_id]},
        ),
        VersionedPattern(
            pattern_id="official.annotation.get_mapping",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@GetMapping",
            micronaut_pattern="@Get",
            description="Map Spring GET endpoint annotations to Micronaut GET annotations.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.96,
            complexity="low",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="HTTP endpoint mapping compatibility.",
                )
            ],
            metadata={"seed_id": latest_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.post_mapping",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@PostMapping",
            micronaut_pattern="@Post",
            description="Map Spring POST endpoint annotations to Micronaut POST annotations.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.96,
            complexity="low",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="HTTP endpoint mapping compatibility.",
                )
            ],
            metadata={"seed_id": latest_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.request_param",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@RequestParam",
            micronaut_pattern="@QueryValue",
            description="Map Spring request parameter binding to Micronaut query binding.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.93,
            complexity="low",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=versioned_guide.url,
                    title=versioned_guide.title,
                    notes="Argument binding compatibility.",
                ),
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=release_4_10_1.url,
                    title=release_4_10_1.title,
                    notes="Micronaut 4.10.x patch-line release context for supported target behavior.",
                ),
            ],
            metadata={"seed_id": versioned_guide.seed_id, "release_seed_ids": [release_4_10_1.seed_id]},
        ),
        VersionedPattern(
            pattern_id="official.annotation.request_body",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@RequestBody",
            micronaut_pattern="@Body",
            description="Map Spring request body binding to Micronaut body binding.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.93,
            complexity="low",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=versioned_guide.url,
                    title=versioned_guide.title,
                    notes="Argument binding compatibility.",
                ),
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=release_4_10_1.url,
                    title=release_4_10_1.title,
                    notes="Micronaut 4.10.x patch-line release context for supported target behavior.",
                ),
            ],
            metadata={"seed_id": versioned_guide.seed_id, "release_seed_ids": [release_4_10_1.seed_id]},
        ),
        VersionedPattern(
            pattern_id="official.annotation.path_variable",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@PathVariable",
            micronaut_pattern="@PathVariable",
            description="Map Spring path-variable binding to Micronaut path-variable binding.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.93,
            complexity="low",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=versioned_guide.url,
                    title=versioned_guide.title,
                    notes="Argument binding compatibility.",
                )
            ],
            metadata={"seed_id": versioned_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.request_header",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@RequestHeader",
            micronaut_pattern="@Header",
            description="Map Spring request-header binding to Micronaut header binding.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.93,
            complexity="low",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=versioned_guide.url,
                    title=versioned_guide.title,
                    notes="Argument binding compatibility.",
                )
            ],
            metadata={"seed_id": versioned_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.put_mapping",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@PutMapping",
            micronaut_pattern="@Put",
            description="Map Spring PUT endpoint annotations to Micronaut PUT annotations.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.96,
            complexity="low",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="HTTP endpoint mapping compatibility.",
                )
            ],
            metadata={"seed_id": latest_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.delete_mapping",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@DeleteMapping",
            micronaut_pattern="@Delete",
            description="Map Spring DELETE endpoint annotations to Micronaut DELETE annotations.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.96,
            complexity="low",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="HTTP endpoint mapping compatibility.",
                )
            ],
            metadata={"seed_id": latest_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.patch_mapping",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@PatchMapping",
            micronaut_pattern="@Patch",
            description="Map Spring PATCH endpoint annotations to Micronaut PATCH annotations.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.96,
            complexity="low",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="HTTP endpoint mapping compatibility.",
                )
            ],
            metadata={"seed_id": latest_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.value",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@Value",
            micronaut_pattern="@Property",
            description="Map Spring value placeholder injection to Micronaut property injection.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.92,
            complexity="medium",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="Official Spring compatibility baseline for property injection migration.",
                )
            ],
            metadata={"seed_id": latest_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.configuration",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@Configuration",
            micronaut_pattern="@Factory",
            description="Map Spring configuration classes to Micronaut factories for bean definitions.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.93,
            complexity="medium",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=guides_tag.url,
                    title=guides_tag.title,
                    notes="Guide-backed bean factory migration pattern.",
                )
            ],
            metadata={"seed_id": guides_tag.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.service",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@Service",
            micronaut_pattern="@Singleton",
            description="Replace Spring service stereotype with Micronaut singleton scope.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.95,
            complexity="low",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=guides_tag.url,
                    title=guides_tag.title,
                    notes="Guide-backed bean and service conversion pattern.",
                ),
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=release_4_9.url,
                    title=release_4_9.title,
                    notes="Micronaut 4.9.x release context for Spring compatibility updates.",
                ),
            ],
            metadata={"seed_id": guides_tag.seed_id, "release_seed_ids": [release_4_9.seed_id]},
        ),
        VersionedPattern(
            pattern_id="official.annotation.component",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@Component",
            micronaut_pattern="@Singleton",
            description="Replace Spring component stereotype with Micronaut singleton scope.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.94,
            complexity="low",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=guides_tag.url,
                    title=guides_tag.title,
                    notes="Guide-backed bean conversion pattern.",
                )
            ],
            metadata={"seed_id": guides_tag.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.repository",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@Repository",
            micronaut_pattern="@Singleton",
            description="Replace Spring repository stereotype with Micronaut singleton scope.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.94,
            complexity="low",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=guides_tag.url,
                    title=guides_tag.title,
                    notes="Guide-backed bean conversion pattern for repository stereotypes.",
                )
            ],
            metadata={"seed_id": guides_tag.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.autowired",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@Autowired",
            micronaut_pattern="jakarta.inject.Inject",
            description="Replace Spring autowiring annotations with Jakarta inject for Micronaut beans.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.94,
            complexity="low",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="Official Spring compatibility baseline for dependency injection migration.",
                )
            ],
            metadata={"seed_id": latest_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.qualifier",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@Qualifier",
            micronaut_pattern="jakarta.inject.Named",
            description="Replace Spring qualifier annotations with Jakarta named qualifiers for Micronaut beans.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.92,
            complexity="low",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="Official Spring compatibility baseline for qualifier and bean selection migration.",
                )
            ],
            metadata={"seed_id": latest_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.bean",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@Bean",
            micronaut_pattern="io.micronaut.context.annotation.Bean",
            description="Keep explicit bean factory methods as Micronaut bean definitions.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.93,
            complexity="low",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=guides_tag.url,
                    title=guides_tag.title,
                    notes="Guide-backed bean factory migration pattern.",
                )
            ],
            metadata={"seed_id": guides_tag.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.primary",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@Primary",
            micronaut_pattern="io.micronaut.context.annotation.Primary",
            description="Carry primary bean preference into Micronaut's primary bean annotation.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.91,
            complexity="low",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=guides_tag.url,
                    title=guides_tag.title,
                    notes="Guide-backed bean selection compatibility pattern.",
                )
            ],
            metadata={"seed_id": guides_tag.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.configuration_properties",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@ConfigurationProperties",
            micronaut_pattern="io.micronaut.context.annotation.ConfigurationProperties",
            description="Preserve grouped configuration binding with Micronaut configuration properties.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.93,
            complexity="medium",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="Official Spring compatibility baseline for configuration binding migration.",
                )
            ],
            metadata={"seed_id": latest_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.validated",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@Validated",
            micronaut_pattern="io.micronaut.validation.Validated",
            description="Retain bean and method validation intent with Micronaut validation annotations.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.92,
            complexity="low",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="Official Spring compatibility baseline for validation migration.",
                )
            ],
            metadata={"seed_id": latest_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.transactional",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@Transactional",
            micronaut_pattern="jakarta.transaction.Transactional",
            description="Retain transactional intent using Micronaut's Jakarta transaction support.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.9,
            complexity="medium",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="Official Spring compatibility baseline for transaction annotation migration.",
                )
            ],
            metadata={"seed_id": latest_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.cacheable",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@Cacheable",
            micronaut_pattern="io.micronaut.cache.annotation.Cacheable",
            description="Preserve cache read-through intent with Micronaut cache annotations.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.91,
            complexity="medium",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="Official Spring compatibility baseline for cache annotation migration.",
                )
            ],
            metadata={"seed_id": latest_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.cache_put",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@CachePut",
            micronaut_pattern="io.micronaut.cache.annotation.CachePut",
            description="Preserve cache update semantics with Micronaut cache put annotations.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.9,
            complexity="medium",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="Official Spring compatibility baseline for cache update annotation migration.",
                )
            ],
            metadata={"seed_id": latest_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.cache_evict",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@CacheEvict",
            micronaut_pattern="io.micronaut.cache.annotation.CacheInvalidate",
            description="Map Spring cache eviction to Micronaut cache invalidation annotations.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.9,
            complexity="medium",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="Official Spring compatibility baseline for cache eviction annotation migration.",
                )
            ],
            metadata={"seed_id": latest_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.scheduled",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@Scheduled",
            micronaut_pattern="io.micronaut.scheduling.annotation.Scheduled",
            description="Keep scheduled task intent with Micronaut scheduling annotations.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.9,
            complexity="medium",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="Official Spring compatibility baseline for scheduling annotation migration.",
                )
            ],
            metadata={"seed_id": latest_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.async",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@Async",
            micronaut_pattern="io.micronaut.scheduling.annotation.Async",
            description="Keep asynchronous execution intent with Micronaut async annotations.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.9,
            complexity="medium",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="Official Spring compatibility baseline for async annotation migration.",
                )
            ],
            metadata={"seed_id": latest_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.exception_handler",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@ExceptionHandler",
            micronaut_pattern="io.micronaut.http.annotation.Error",
            description="Map Spring exception handler annotations to Micronaut error handler annotations.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.9,
            complexity="medium",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="Official controller compatibility baseline for exception handling migration.",
                )
            ],
            metadata={"seed_id": latest_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.annotation.response_status",
            pattern_type=PatternType.ANNOTATION,
            spring_pattern="@ResponseStatus",
            micronaut_pattern="io.micronaut.http.annotation.Status",
            description="Map Spring response status annotations to Micronaut status annotations.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.9,
            complexity="medium",
            category="annotations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="Official controller compatibility baseline for status mapping migration.",
                )
            ],
            metadata={"seed_id": latest_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.di.autowired_field",
            pattern_type=PatternType.DEPENDENCY_INJECTION,
            spring_pattern="Field injection",
            micronaut_pattern="Constructor injection",
            description="Promote constructor injection over Spring field injection for Micronaut services.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.91,
            complexity="medium",
            category="code_patterns",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=guides_tag.url,
                    title=guides_tag.title,
                    notes="Guide-backed bean construction best practice.",
                )
            ],
            metadata={"seed_id": guides_tag.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.config.spring_application_name",
            pattern_type=PatternType.CONFIGURATION,
            spring_pattern="spring.application.name",
            micronaut_pattern="micronaut.application.name",
            description="Carry application name into Micronaut configuration namespace.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.9,
            complexity="low",
            category="configurations",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=spring_boot_ref.url,
                    title=spring_boot_ref.title,
                    notes="Spring Boot config source paired with Micronaut config namespace migration.",
                ),
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="Official Micronaut-for-Spring compatibility baseline.",
                ),
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=release_4_10.url,
                    title=release_4_10.title,
                    notes="Micronaut 4.10.x release context for current supported target behavior.",
                ),
            ],
            metadata={"seed_id": spring_boot_ref.seed_id, "release_seed_ids": [release_4_10.seed_id]},
        ),
        VersionedPattern(
            pattern_id="official.application.run_bootstrap",
            pattern_type=PatternType.APPLICATION,
            spring_pattern="SpringApplication.run",
            micronaut_pattern="Micronaut.run",
            description="Replace the Spring Boot application bootstrap entrypoint with Micronaut.run.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.96,
            complexity="low",
            category="code_patterns",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=guides_tag.url,
                    title=guides_tag.title,
                    notes="Official application bootstrap migration pattern.",
                ),
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=release_4_10.url,
                    title=release_4_10.title,
                    notes="Micronaut 4.10.x release context for current bootstrap behavior.",
                ),
            ],
            metadata={"seed_id": guides_tag.seed_id, "release_seed_ids": [release_4_10.seed_id]},
        ),
        VersionedPattern(
            pattern_id="official.type.response_entity",
            pattern_type=PatternType.TYPE,
            spring_pattern="ResponseEntity",
            micronaut_pattern="HttpResponse",
            description="Replace Spring ResponseEntity with Micronaut HttpResponse.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.92,
            complexity="medium",
            category="types",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="HTTP response abstraction migration pattern.",
                ),
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=release_4_10_1.url,
                    title=release_4_10_1.title,
                    notes="Micronaut 4.10.x patch-line release context for current response APIs.",
                ),
            ],
            metadata={"seed_id": latest_guide.seed_id, "release_seed_ids": [release_4_10_1.seed_id]},
        ),
        VersionedPattern(
            pattern_id="official.code.spring_data_page",
            pattern_type=PatternType.CODE_PATTERN,
            spring_pattern="org.springframework.data.domain.Page",
            micronaut_pattern="io.micronaut.data.model.Page",
            description="Replace Spring Data Page imports with Micronaut Data Page while preserving paging semantics.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.92,
            complexity="medium",
            category="code_patterns",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=versioned_guide.url,
                    title=versioned_guide.title,
                    notes="Official Spring compatibility guide includes Spring Data support on Micronaut 4.x lines.",
                )
            ],
            metadata={"seed_id": versioned_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.code.spring_data_pageable",
            pattern_type=PatternType.CODE_PATTERN,
            spring_pattern="org.springframework.data.domain.Pageable",
            micronaut_pattern="io.micronaut.data.model.Pageable",
            description="Replace Spring Data Pageable imports with Micronaut Data Pageable for version-aware pagination migration.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.92,
            complexity="medium",
            category="code_patterns",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=versioned_guide.url,
                    title=versioned_guide.title,
                    notes="Official Spring compatibility guide includes Spring Data repository and pagination support.",
                )
            ],
            metadata={"seed_id": versioned_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.code.spring_data_page_request",
            pattern_type=PatternType.CODE_PATTERN,
            spring_pattern="org.springframework.data.domain.PageRequest",
            micronaut_pattern="Pageable.from(page, size)",
            description="Rewrite Spring PageRequest factory usage to Micronaut Data Pageable factory calls.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.9,
            complexity="medium",
            category="code_patterns",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=versioned_guide.url,
                    title=versioned_guide.title,
                    notes="Official Spring compatibility guide and Micronaut Data pagination guidance inform the PageRequest rewrite shape.",
                )
            ],
            metadata={"seed_id": versioned_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.code.spring_ui_model",
            pattern_type=PatternType.CODE_PATTERN,
            spring_pattern="org.springframework.ui.Model",
            micronaut_pattern="java.util.Map<String, Object>",
            description="For server-side view controllers, migrate Spring Model arguments to a plain mutable model map in Micronaut views.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.88,
            complexity="medium",
            category="code_patterns",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="Official Spring compatibility guide documents Model support alongside Micronaut views integration.",
                ),
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=guides_tag.url,
                    title=guides_tag.title,
                    notes="Official guides provide server-side view migration context.",
                ),
            ],
            metadata={"seed_id": latest_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.code.spring_ui_model_map",
            pattern_type=PatternType.CODE_PATTERN,
            spring_pattern="org.springframework.ui.ModelMap",
            micronaut_pattern="java.util.Map<String, Object>",
            description="For view controllers, collapse Spring ModelMap usage into a simple mutable map compatible with Micronaut view rendering.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.88,
            complexity="medium",
            category="code_patterns",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=latest_guide.url,
                    title=latest_guide.title,
                    notes="Official Spring compatibility guide documents ModelMap usage with Micronaut views.",
                )
            ],
            metadata={"seed_id": latest_guide.seed_id},
        ),
        VersionedPattern(
            pattern_id="official.code.model_and_view",
            pattern_type=PatternType.CODE_PATTERN,
            spring_pattern="org.springframework.web.servlet.ModelAndView",
            micronaut_pattern="io.micronaut.views.ModelAndView",
            description="Switch Spring MVC ModelAndView to Micronaut views ModelAndView for template rendering flows.",
            spring_versions=VersionWindow(spec="3.x"),
            micronaut_versions=VersionWindow(spec="4.x"),
            status=ValidationStatus.VALIDATED,
            confidence=0.9,
            complexity="medium",
            category="code_patterns",
            source_kind=SourceKind.OFFICIAL_DOC,
            evidence=[
                PatternEvidence(
                    source_kind=SourceKind.OFFICIAL_DOC,
                    source_ref=guides_tag.url,
                    title=guides_tag.title,
                    notes="Official guides cover server-side view rendering and view-model return patterns on Micronaut.",
                )
            ],
            metadata={"seed_id": guides_tag.seed_id},
        ),
    ]


def build_normalized_official_payload() -> Dict[str, object]:
    patterns = curated_official_patterns()
    return {
        "schema_version": 1,
        "catalog_type": "official_normalized_patterns",
        "pattern_count": len(patterns),
        "patterns": [pattern.to_dict() for pattern in patterns],
    }


def write_normalized_official_patterns(corpus_root: str = "corpus") -> Dict[str, object]:
    repository = PatternCorpusRepository(root=corpus_root)
    repository.initialize_layout()
    write_official_seed_catalog(corpus_root=corpus_root)

    normalized_root = Path(corpus_root) / "official_docs" / "normalized"
    normalized_root.mkdir(parents=True, exist_ok=True)
    patterns_root = normalized_root / "patterns"
    patterns_root.mkdir(parents=True, exist_ok=True)

    patterns = curated_official_patterns()
    index_payload = build_normalized_official_payload()
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
    parser = argparse.ArgumentParser(description="Normalize official documentation seeds into versioned pattern candidates")
    parser.add_argument("--corpus-root", default="corpus", help="Corpus root directory")
    parser.add_argument("--write", action="store_true", help="Write normalized official patterns into the corpus")
    args = parser.parse_args()

    if args.write:
        print(json.dumps(write_normalized_official_patterns(corpus_root=args.corpus_root), indent=2, sort_keys=True))
        return

    print(json.dumps(build_normalized_official_payload(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
