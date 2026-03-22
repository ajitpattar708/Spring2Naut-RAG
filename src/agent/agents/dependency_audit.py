from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

from src.agent.core.config import MigrationConfig, resolve_maven_local_repository
from src.agent.core.interfaces import KnowledgeService
from src.agent.core.versioning import compare_versions, includes_version, normalize_major_minor


@dataclass(frozen=True)
class DependencyCoordinate:
    group_id: str
    artifact_id: str
    version: str = ""
    scope: str = ""
    packaging: str = ""
    classifier: str = ""
    source: str = "direct"
    depth: int = 0
    omitted: bool = False
    omitted_for: str = ""
    parent_chain: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def ga(self) -> str:
        return f"{self.group_id}:{self.artifact_id}"

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["ga"] = self.ga
        payload["parent_chain"] = list(self.parent_chain)
        return payload


@dataclass(frozen=True)
class DependencyFinding:
    severity: str
    code: str
    dependency: str
    message: str
    source: str
    version: str = ""
    depth: int = 0
    suggested_action: str = ""
    related_dependencies: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Optional[Dict[str, object]] = None

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["related_dependencies"] = list(self.related_dependencies)
        return payload


@dataclass(frozen=True)
class ProjectBuildContext:
    build_tool: str
    maven_parent_ga: str = ""
    maven_parent_version: str = ""
    maven_managed_dependency_gas: Tuple[str, ...] = field(default_factory=tuple)
    maven_managed_dependency_coords: Tuple[str, ...] = field(default_factory=tuple)
    gradle_platform_gas: Tuple[str, ...] = field(default_factory=tuple)
    gradle_platform_coords: Tuple[str, ...] = field(default_factory=tuple)

    def uses_micronaut_maven_parent(self) -> bool:
        return self.maven_parent_ga == "io.micronaut.platform:micronaut-parent"

    def uses_micronaut_maven_platform(self) -> bool:
        return any(ga == "io.micronaut.platform:micronaut-platform" for ga in self.maven_managed_dependency_gas)

    def uses_micronaut_gradle_platform(self) -> bool:
        return any(ga == "io.micronaut.platform:micronaut-platform" for ga in self.gradle_platform_gas)

    def uses_spring_boot_maven_parent(self) -> bool:
        return self.maven_parent_ga == "org.springframework.boot:spring-boot-starter-parent"

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["maven_managed_dependency_gas"] = list(self.maven_managed_dependency_gas)
        payload["maven_managed_dependency_coords"] = list(self.maven_managed_dependency_coords)
        payload["gradle_platform_gas"] = list(self.gradle_platform_gas)
        payload["gradle_platform_coords"] = list(self.gradle_platform_coords)
        return payload


@dataclass(frozen=True)
class CompatibilityCatalogEntry:
    ga: str
    severity: str
    replacement: str
    rationale: str
    target_status: str
    spring_spec: str = "3.x"
    micronaut_spec: str = "4.x"
    replacement_version: str = ""
    automated_migration_supported: bool = False
    version_management: str = "platform_managed"
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    def matches(self, dependency: DependencyCoordinate, spring_version: str, micronaut_version: str) -> bool:
        if dependency.ga != self.ga and dependency.ga not in self.aliases:
            return False
        return includes_version(spring_version, spec=self.spring_spec) and includes_version(
            micronaut_version,
            spec=self.micronaut_spec,
        )


class MavenCentralResolver:
    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        base_url: Optional[str] = None,
        artifact_base_url: Optional[str] = None,
        local_repository: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.enabled = MigrationConfig.MAVEN_CENTRAL_VERIFY if enabled is None else enabled
        self.base_url = base_url or MigrationConfig.MAVEN_CENTRAL_SEARCH_URL
        self.artifact_base_url = artifact_base_url or MigrationConfig.MAVEN_CENTRAL_ARTIFACT_BASE_URL
        self.local_repository = Path(local_repository or resolve_maven_local_repository()).expanduser()
        self.timeout = MigrationConfig.MAVEN_CENTRAL_TIMEOUT if timeout is None else timeout
        self._cache: Dict[str, Dict[str, object]] = {}
        self._descriptor_cache: Dict[Tuple[str, str], Dict[str, object]] = {}
        self._pom_text_cache: Dict[Tuple[str, str], Dict[str, object]] = {}

    def verify_artifact(self, ga: str) -> Dict[str, object]:
        if ga in self._cache:
            return self._cache[ga]

        local_versions = self._find_local_versions(ga)
        if local_versions:
            result = {
                "checked": True,
                "source": "local_maven_repo",
                "ga": ga,
                "available": True,
                "latest_version": local_versions[-1],
                "reason": "ok_local",
                "local_versions": local_versions,
                "local_repository": str(self.local_repository),
            }
            self._cache[ga] = result
            return result

        if not self.enabled:
            result = {
                "checked": False,
                "source": "maven_central",
                "ga": ga,
                "available": None,
                "latest_version": "",
                "reason": "verification_disabled",
            }
            self._cache[ga] = result
            return result

        if ":" not in ga:
            result = {
                "checked": False,
                "source": "maven_central",
                "ga": ga,
                "available": None,
                "latest_version": "",
                "reason": "invalid_ga",
            }
            self._cache[ga] = result
            return result

        group_id, artifact_id = ga.split(":", 1)
        query = f'g:"{group_id}" AND a:"{artifact_id}"'
        params = urllib.parse.urlencode({"q": query, "rows": 1, "wt": "json"})
        url = f"{self.base_url}?{params}"

        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            docs = ((payload or {}).get("response") or {}).get("docs") or []
            if not docs:
                result = {
                    "checked": True,
                    "source": "maven_central",
                    "ga": ga,
                    "available": False,
                    "latest_version": "",
                    "reason": "not_found",
                }
            else:
                doc = docs[0]
                result = {
                    "checked": True,
                    "source": "maven_central",
                    "ga": ga,
                    "available": True,
                    "latest_version": str(doc.get("latestVersion") or ""),
                    "reason": "ok",
                }
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            result = {
                "checked": False,
                "source": "maven_central",
                "ga": ga,
                "available": None,
                "latest_version": "",
                "reason": "network_unavailable",
            }

        self._cache[ga] = result
        return result

    def inspect_artifact(
        self,
        ga: str,
        version: str = "",
        *,
        include_children: bool = True,
    ) -> Dict[str, object]:
        requested_version = str(version or "").strip()
        cache_key = (ga, f"{requested_version}|children={include_children}")
        if cache_key in self._descriptor_cache:
            return self._descriptor_cache[cache_key]

        local_pom_path = self._local_pom_path(ga, requested_version) if requested_version else None
        verification = self.verify_artifact(ga)
        resolved_version = requested_version or str(verification.get("latest_version") or "")
        base_result = {
            "checked": verification.get("checked", False),
            "source": "local_maven_repo_pom" if local_pom_path is not None else "maven_central_pom",
            "ga": ga,
            "available": verification.get("available"),
            "requested_version": requested_version,
            "resolved_version": resolved_version,
            "pom_available": None,
            "reason": verification.get("reason", ""),
            "resolution_channel": "local_maven_repo" if local_pom_path is not None else "maven_central",
            "declared_dependency_count": 0,
            "compile_dependency_count": 0,
            "runtime_dependency_count": 0,
            "declared_compile_dependencies": [],
            "declared_runtime_dependencies": [],
            "spring_declared_dependencies": [],
            "javax_declared_dependencies": [],
            "child_inspection_enabled": include_children,
            "child_dependency_candidates_count": 0,
            "child_dependency_inspected_count": 0,
            "child_dependency_unresolved_count": 0,
            "inspected_child_artifacts": [],
            "transitive_spring_declared_dependencies": [],
            "transitive_javax_declared_dependencies": [],
        }

        if local_pom_path is not None:
            try:
                pom_text = local_pom_path.read_text(encoding="utf-8")
                descriptor = self._parse_pom_declared_dependencies(ga, resolved_version, pom_text)
                if include_children:
                    descriptor = {
                        **descriptor,
                        **self._inspect_declared_dependency_children(
                            descriptor.get("declared_runtime_dependencies", []),
                        ),
                    }
                result = {
                    **base_result,
                    **descriptor,
                    "checked": True,
                    "pom_available": True,
                    "reason": "ok_local",
                    "pom_path": str(local_pom_path),
                    "local_repository": str(self.local_repository),
                }
            except OSError:
                result = {
                    **base_result,
                    "checked": False,
                    "pom_available": None,
                    "reason": "local_read_error",
                    "pom_path": str(local_pom_path),
                    "local_repository": str(self.local_repository),
                }
            except ET.ParseError:
                result = {
                    **base_result,
                    "checked": True,
                    "pom_available": False,
                    "reason": "pom_parse_error",
                    "pom_path": str(local_pom_path),
                    "local_repository": str(self.local_repository),
                }
            self._descriptor_cache[cache_key] = result
            return result

        if not self.enabled:
            base_result["reason"] = "verification_disabled"
            self._descriptor_cache[cache_key] = base_result
            return base_result

        if ":" not in ga:
            base_result["reason"] = "invalid_ga"
            self._descriptor_cache[cache_key] = base_result
            return base_result

        if verification.get("available") is not True:
            self._descriptor_cache[cache_key] = base_result
            return base_result

        if not resolved_version:
            base_result["reason"] = "version_unresolved"
            self._descriptor_cache[cache_key] = base_result
            return base_result

        group_id, artifact_id = ga.split(":", 1)
        pom_url = self._pom_url(group_id, artifact_id, resolved_version)

        try:
            with urllib.request.urlopen(pom_url, timeout=self.timeout) as response:
                pom_text = response.read().decode("utf-8")
            descriptor = self._parse_pom_declared_dependencies(ga, resolved_version, pom_text)
            if include_children:
                descriptor = {
                    **descriptor,
                    **self._inspect_declared_dependency_children(
                        descriptor.get("declared_runtime_dependencies", []),
                    ),
                }
            result = {
                **base_result,
                **descriptor,
                "checked": True,
                "pom_available": True,
                "reason": "ok",
                "pom_url": pom_url,
            }
        except (urllib.error.URLError, TimeoutError, OSError):
            result = {
                **base_result,
                "checked": False,
                "pom_available": None,
                "reason": "network_unavailable",
                "pom_url": pom_url,
            }
        except ET.ParseError:
            result = {
                **base_result,
                "checked": True,
                "pom_available": False,
                "reason": "pom_parse_error",
                "pom_url": pom_url,
            }

        self._descriptor_cache[cache_key] = result
        return result

    def fetch_pom_text(self, ga: str, version: str) -> Dict[str, object]:
        requested_version = str(version or "").strip()
        cache_key = (ga, requested_version)
        if cache_key in self._pom_text_cache:
            return self._pom_text_cache[cache_key]

        result = {
            "checked": False,
            "ga": ga,
            "version": requested_version,
            "pom_available": None,
            "reason": "",
            "pom_text": "",
        }

        if ":" not in ga or not requested_version:
            result["reason"] = "invalid_coordinate"
            self._pom_text_cache[cache_key] = result
            return result

        local_pom_path = self._local_pom_path(ga, requested_version)
        if local_pom_path is not None:
            try:
                pom_text = local_pom_path.read_text(encoding="utf-8")
                result.update(
                    {
                        "checked": True,
                        "source": "local_maven_repo",
                        "pom_available": True,
                        "reason": "ok_local",
                        "pom_text": pom_text,
                        "pom_path": str(local_pom_path),
                        "local_repository": str(self.local_repository),
                    }
                )
            except OSError:
                result.update(
                    {
                        "checked": False,
                        "source": "local_maven_repo",
                        "pom_available": None,
                        "reason": "local_read_error",
                        "pom_path": str(local_pom_path),
                        "local_repository": str(self.local_repository),
                    }
                )
            self._pom_text_cache[cache_key] = result
            return result

        if not self.enabled:
            result["reason"] = "verification_disabled"
            self._pom_text_cache[cache_key] = result
            return result

        group_id, artifact_id = ga.split(":", 1)
        pom_url = self._pom_url(group_id, artifact_id, requested_version)

        try:
            with urllib.request.urlopen(pom_url, timeout=self.timeout) as response:
                pom_text = response.read().decode("utf-8")
            result.update(
                {
                    "checked": True,
                    "pom_available": True,
                    "reason": "ok",
                    "pom_text": pom_text,
                    "pom_url": pom_url,
                }
            )
        except (urllib.error.URLError, TimeoutError, OSError):
            result.update(
                {
                    "checked": False,
                    "pom_available": None,
                    "reason": "network_unavailable",
                    "pom_url": pom_url,
                }
            )

        self._pom_text_cache[cache_key] = result
        return result

    def _find_local_versions(self, ga: str) -> List[str]:
        if ":" not in ga or not self.local_repository.exists():
            return []

        group_id, artifact_id = ga.split(":", 1)
        artifact_dir = self.local_repository / Path(*group_id.split(".")) / artifact_id
        if not artifact_dir.exists() or not artifact_dir.is_dir():
            return []

        versions: List[str] = []
        for child in artifact_dir.iterdir():
            if not child.is_dir():
                continue
            pom_path = child / f"{artifact_id}-{child.name}.pom"
            if pom_path.exists():
                versions.append(child.name)

        return sorted(versions, key=self._local_version_sort_key)

    def _local_pom_path(self, ga: str, version: str) -> Optional[Path]:
        if ":" not in ga or not version or not self.local_repository.exists():
            return None

        group_id, artifact_id = ga.split(":", 1)
        pom_path = self.local_repository / Path(*group_id.split(".")) / artifact_id / version / f"{artifact_id}-{version}.pom"
        if pom_path.exists():
            return pom_path
        return None

    def _local_version_sort_key(self, version: str) -> Tuple[int, int, int, str]:
        tokens = re.split(r"[.-]", str(version or "").strip().lower())
        numeric: List[int] = []
        suffix_tokens: List[str] = []
        for token in tokens:
            if token.isdigit() and len(numeric) < 3 and not suffix_tokens:
                numeric.append(int(token))
            elif token:
                suffix_tokens.append(token)

        while len(numeric) < 3:
            numeric.append(-1)

        suffix_weight = 0 if not suffix_tokens else -1
        suffix_value = ".".join(suffix_tokens)
        return numeric[0], numeric[1], numeric[2], f"{suffix_weight}:{suffix_value}"

    def _inspect_declared_dependency_children(
        self,
        dependencies: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        child_candidates = [
            item
            for item in dependencies
            if self._is_resolvable_declared_version(str(item.get("version") or ""))
        ]
        unresolved_count = len(dependencies) - len(child_candidates)
        inspected_children: List[Dict[str, object]] = []
        transitive_spring: List[str] = []
        transitive_javax: List[str] = []

        for item in child_candidates[:8]:
            child_ga = str(item.get("ga") or "")
            child_version = str(item.get("version") or "")
            if not child_ga:
                continue
            child_descriptor = self.inspect_artifact(
                child_ga,
                version=child_version,
                include_children=False,
            )
            inspected_children.append(
                {
                    "ga": child_ga,
                    "resolved_version": child_descriptor.get("resolved_version", child_version),
                    "compile_dependency_count": child_descriptor.get("compile_dependency_count", 0),
                    "runtime_dependency_count": child_descriptor.get("runtime_dependency_count", 0),
                    "spring_declared_dependencies": child_descriptor.get("spring_declared_dependencies", []),
                    "javax_declared_dependencies": child_descriptor.get("javax_declared_dependencies", []),
                    "pom_available": child_descriptor.get("pom_available"),
                    "reason": child_descriptor.get("reason", ""),
                }
            )
            transitive_spring.extend(list(child_descriptor.get("spring_declared_dependencies") or []))
            transitive_javax.extend(list(child_descriptor.get("javax_declared_dependencies") or []))

        return {
            "child_dependency_candidates_count": len(dependencies),
            "child_dependency_inspected_count": len(inspected_children),
            "child_dependency_unresolved_count": unresolved_count,
            "inspected_child_artifacts": inspected_children,
            "transitive_spring_declared_dependencies": sorted(set(transitive_spring)),
            "transitive_javax_declared_dependencies": sorted(set(transitive_javax)),
        }

    def _is_resolvable_declared_version(self, version: str) -> bool:
        cleaned = str(version or "").strip()
        return bool(cleaned and not cleaned.startswith("${"))

    def _pom_url(self, group_id: str, artifact_id: str, version: str) -> str:
        group_path = group_id.replace(".", "/")
        return f"{self.artifact_base_url.rstrip('/')}/{group_path}/{artifact_id}/{version}/{artifact_id}-{version}.pom"

    def _parse_pom_declared_dependencies(
        self,
        ga: str,
        version: str,
        pom_text: str,
    ) -> Dict[str, object]:
        root = ET.fromstring(pom_text)
        namespace = ""
        if root.tag.startswith("{") and "}" in root.tag:
            namespace = root.tag[1:].split("}", 1)[0]
        ns = {"maven": namespace} if namespace else {}
        prefix = "maven:" if namespace else ""

        dependencies_node = root.find(f"{prefix}dependencies", ns) if namespace else root.find("dependencies")
        declared_dependencies: List[Dict[str, object]] = []
        compile_dependencies: List[Dict[str, object]] = []
        runtime_dependencies: List[Dict[str, object]] = []
        spring_dependencies: List[str] = []
        javax_dependencies: List[str] = []

        if dependencies_node is not None:
            for dep in dependencies_node.findall(f"{prefix}dependency", ns) if namespace else dependencies_node.findall("dependency"):
                group_id = (dep.findtext(f"{prefix}groupId", default="", namespaces=ns) if namespace else dep.findtext("groupId", default="") or "").strip()
                artifact_id = (dep.findtext(f"{prefix}artifactId", default="", namespaces=ns) if namespace else dep.findtext("artifactId", default="") or "").strip()
                dep_version = (dep.findtext(f"{prefix}version", default="", namespaces=ns) if namespace else dep.findtext("version", default="") or "").strip()
                scope = ((dep.findtext(f"{prefix}scope", default="", namespaces=ns) if namespace else dep.findtext("scope", default="")) or "").strip() or "compile"
                optional_text = ((dep.findtext(f"{prefix}optional", default="", namespaces=ns) if namespace else dep.findtext("optional", default="")) or "").strip().lower()
                optional = optional_text == "true"
                if not group_id or not artifact_id:
                    continue

                coordinate = {
                    "ga": f"{group_id}:{artifact_id}",
                    "group_id": group_id,
                    "artifact_id": artifact_id,
                    "version": dep_version,
                    "scope": scope,
                    "optional": optional,
                }
                declared_dependencies.append(coordinate)
                if scope in {"compile", "provided"}:
                    compile_dependencies.append(coordinate)
                if scope in {"compile", "runtime"}:
                    runtime_dependencies.append(coordinate)
                if group_id.startswith("org.springframework") or artifact_id.startswith("spring-") or "spring" in group_id:
                    spring_dependencies.append(f"{group_id}:{artifact_id}")
                if group_id.startswith("javax") or artifact_id.startswith("javax") or group_id.startswith("jakarta.servlet"):
                    javax_dependencies.append(f"{group_id}:{artifact_id}")

        return {
            "artifact_ga": ga,
            "artifact_version": version,
            "declared_dependency_count": len(declared_dependencies),
            "compile_dependency_count": len(compile_dependencies),
            "runtime_dependency_count": len(runtime_dependencies),
            "declared_compile_dependencies": compile_dependencies,
            "declared_runtime_dependencies": runtime_dependencies,
            "spring_declared_dependencies": sorted(set(spring_dependencies)),
            "javax_declared_dependencies": sorted(set(javax_dependencies)),
        }


class DependencyCompatibilityAuditor:
    """
    Extracts direct and transitive dependencies and flags migration risks
    for a target Spring Boot -> Micronaut version pair.
    """

    _TREE_LINE_RE = re.compile(
        r"^(?P<prefix>(?:\|  |   )*)(?P<connector>\+-|\\-)\s(?P<coords>.+)$"
    )
    _GRADLE_TREE_LINE_RE = re.compile(
        r"^(?P<prefix>(?:(?:\|    )|(?:     ))*)(?P<connector>\+---|\\---)\s(?P<coords>.+)$"
    )
    _GRADLE_STRING_DEP_RE = re.compile(
        r"""^\s*(?P<config>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\(\s*)?['"](?P<ga>[^'"]+)['"]\s*\)?"""
    )
    _GRADLE_MAP_DEP_RE = re.compile(
        r"""^\s*(?P<config>[A-Za-z_][A-Za-z0-9_]*)\s+(?:group\s*:\s*['"](?P<group>[^'"]+)['"]\s*,\s*name\s*:\s*['"](?P<name>[^'"]+)['"](?:\s*,\s*version\s*:\s*['"](?P<version>[^'"]+)['"])?).*$"""
    )
    _SPRING_MARKERS = (
        "org.springframework",
        "org.springframework.boot",
        "org.springframework.cloud",
        "spring-boot",
        "spring-cloud",
        "springframework",
    )
    _KNOWN_THIRD_PARTY_RISK_MARKERS = (
        "springfox",
        "springdoc",
        "sleuth",
        "autoconfigure",
    )
    _SOURCE_PLATFORM_MANAGED_VERSION_ALLOWLIST = {
        "com.h2database:h2",
        "org.assertj:assertj-core",
        "org.hamcrest:hamcrest",
        "org.junit.jupiter:junit-jupiter",
        "org.junit.jupiter:junit-jupiter-api",
        "org.junit.jupiter:junit-jupiter-engine",
        "org.mockito:mockito-core",
        "org.mockito:mockito-junit-jupiter",
    }
    _TARGET_PLATFORM_MANAGED_VERSION_ALLOWLIST = {
        "com.h2database:h2",
        "org.assertj:assertj-core",
        "org.hamcrest:hamcrest",
        "org.junit.jupiter:junit-jupiter",
        "org.junit.jupiter:junit-jupiter-api",
        "org.junit.jupiter:junit-jupiter-engine",
        "org.mockito:mockito-core",
        "org.mockito:mockito-junit-jupiter",
    }
    _LEGACY_JAVAX_REPLACEMENTS = {
        "javax.validation:validation-api": "jakarta.validation:jakarta.validation-api",
        "javax.persistence:javax.persistence-api": "jakarta.persistence:jakarta.persistence-api",
        "javax.annotation:javax.annotation-api": "jakarta.annotation:jakarta.annotation-api",
        "javax.xml.bind:jaxb-api": "jakarta.xml.bind:jakarta.xml.bind-api",
        "javax.activation:activation": "jakarta.activation:jakarta.activation-api",
        "javax.servlet:javax.servlet-api": "jakarta.servlet:jakarta.servlet-api",
        "javax.cache:cache-api": "jakarta.cache:jakarta.cache-api",
    }
    _COMPATIBILITY_CATALOG: Tuple[CompatibilityCatalogEntry, ...] = (
        CompatibilityCatalogEntry(
            ga="org.springframework.boot:spring-boot-starter-web",
            severity="review",
            replacement="io.micronaut:micronaut-http-server-netty",
            rationale="Spring MVC web starter usage should migrate to Micronaut's Netty-based HTTP server stack for standard service endpoints.",
            target_status="replacement_available",
            automated_migration_supported=True,
            version_management="platform_managed",
            notes="Review controller binding, filters, exception handling, and serialization behavior after migration.",
        ),
        CompatibilityCatalogEntry(
            ga="org.springframework.boot:spring-boot-starter-test",
            severity="review",
            replacement="io.micronaut.test:micronaut-test-junit5",
            rationale="Spring Boot test starter wiring should move to Micronaut Test for Micronaut-native test execution.",
            target_status="replacement_available",
            automated_migration_supported=True,
            version_management="platform_managed",
            notes="Re-check slice tests, mock wiring, and embedded-server assumptions after migration.",
        ),
        CompatibilityCatalogEntry(
            ga="org.springframework.boot:spring-boot-starter-data-jpa",
            severity="review",
            replacement="io.micronaut.data:micronaut-data-hibernate-jpa",
            rationale="Spring Data JPA starter usage usually maps to Micronaut Data with Hibernate JPA, but repository semantics still need validation.",
            target_status="replacement_available",
            automated_migration_supported=True,
            version_management="platform_managed",
            notes="Validate entity scanning, transaction boundaries, auditing, and repository behavior after migration.",
        ),
        CompatibilityCatalogEntry(
            ga="org.springframework.boot:spring-boot-starter-jdbc",
            severity="review",
            replacement="io.micronaut.sql:micronaut-jdbc-hikari",
            rationale="Spring JDBC starter usage generally migrates to Micronaut JDBC with Hikari-backed datasource management for standard blocking SQL access.",
            target_status="replacement_available",
            automated_migration_supported=True,
            version_management="platform_managed",
            notes="Review datasource naming, transaction boundaries, SQL exception translation, and any JdbcTemplate-heavy code paths after migration.",
        ),
        CompatibilityCatalogEntry(
            ga="org.springframework.boot:spring-boot-starter-data-r2dbc",
            severity="review",
            replacement="io.micronaut.data:micronaut-data-r2dbc",
            rationale="Spring Data R2DBC starter usage usually maps to Micronaut Data R2DBC for reactive repository and database-client workloads.",
            target_status="replacement_available",
            automated_migration_supported=True,
            version_management="platform_managed",
            notes="Validate reactive transaction behavior, connection-factory setup, repository semantics, and backpressure-sensitive flows after migration.",
        ),
        CompatibilityCatalogEntry(
            ga="org.springframework.boot:spring-boot-starter-cache",
            severity="review",
            replacement="io.micronaut.cache:micronaut-cache-caffeine",
            rationale="Spring cache starter usage typically migrates to Micronaut Cache with a concrete cache provider such as Caffeine for standard local-cache scenarios.",
            target_status="replacement_available",
            automated_migration_supported=True,
            version_management="platform_managed",
            notes="Review provider choice, cache names, TTL/eviction behavior, and any provider-specific Ehcache or Coherence configuration after migration.",
        ),
        CompatibilityCatalogEntry(
            ga="org.ehcache:ehcache",
            severity="review",
            replacement="io.micronaut.cache:micronaut-cache-ehcache",
            rationale="Direct Ehcache usage in Spring-era applications can usually be re-homed onto Micronaut Cache Ehcache integration while preserving provider-specific cache configuration.",
            target_status="replacement_available",
            automated_migration_supported=True,
            version_management="platform_managed",
            notes="Preserve `ehcache.xml` or equivalent provider settings and review any JAXB-related exclusions or startup wiring after migration.",
        ),
        CompatibilityCatalogEntry(
            ga="io.springfox:springfox-boot-starter",
            severity="blocking",
            replacement="io.micronaut.openapi:micronaut-openapi",
            rationale="Springfox is Spring MVC-centric and is not a drop-in runtime fit for Micronaut 4 services.",
            target_status="replacement_available",
            automated_migration_supported=False,
            aliases=("io.springfox:springfox-swagger2",),
            notes="Regenerate OpenAPI/Swagger support with Micronaut OpenAPI and re-check any UI exposure strategy.",
        ),
        CompatibilityCatalogEntry(
            ga="org.springdoc:springdoc-openapi-starter-webmvc-ui",
            severity="blocking",
            replacement="io.micronaut.openapi:micronaut-openapi",
            rationale="springdoc WebMVC starters assume Spring MVC infrastructure and must be replaced for Micronaut-native API documentation.",
            target_status="replacement_available",
            automated_migration_supported=False,
            aliases=(
                "org.springdoc:springdoc-openapi-starter-webmvc-api",
                "org.springdoc:springdoc-openapi-ui",
            ),
            notes="Revisit generated spec endpoints and UI wiring after migration.",
        ),
        CompatibilityCatalogEntry(
            ga="org.springframework.cloud:spring-cloud-starter-gateway",
            severity="blocking",
            replacement="io.micronaut:micronaut-http-client",
            rationale="Spring Cloud Gateway route/filter pipelines do not migrate as a drop-in dependency swap.",
            target_status="manual_redesign",
            automated_migration_supported=False,
            notes="Use Micronaut HTTP client/server primitives and redesign custom route predicates and filters explicitly.",
        ),
        CompatibilityCatalogEntry(
            ga="org.springframework.cloud:spring-cloud-starter-gateway-mvc",
            severity="blocking",
            replacement="io.micronaut:micronaut-http-client",
            rationale="Spring Cloud Gateway MVC still depends on Spring MVC route and filter infrastructure and does not translate as a simple dependency replacement.",
            target_status="manual_redesign",
            automated_migration_supported=False,
            notes="Remove the starter and redesign proxy routes, predicates, and filter logic with Micronaut HTTP primitives explicitly.",
        ),
        CompatibilityCatalogEntry(
            ga="org.springframework.cloud:spring-cloud-starter-openfeign",
            severity="blocking",
            replacement="io.micronaut:micronaut-http-client",
            rationale="OpenFeign starter wiring depends on Spring Cloud integration and needs client-interface migration to Micronaut @Client patterns.",
            target_status="manual_redesign",
            automated_migration_supported=False,
            notes="Rewrite declarative clients using Micronaut @Client interfaces and validate codecs, error handling, and interceptors.",
        ),
        CompatibilityCatalogEntry(
            ga="redis.clients:jedis",
            severity="review",
            replacement="io.micronaut.redis:micronaut-redis-lettuce",
            rationale="Jedis can remain technically usable, but the current migration path is standardized on Micronaut Redis with Lettuce for Micronaut 4 services.",
            target_status="replacement_available",
            automated_migration_supported=True,
            version_management="platform_managed",
            notes="Validate Redis topology, pooling, and serialization settings after the client change.",
        ),
        CompatibilityCatalogEntry(
            ga="org.springframework.boot:spring-boot-starter-security",
            severity="review",
            replacement="io.micronaut.security:micronaut-security",
            rationale="Spring Security starter wiring can be migrated to Micronaut Security for many standard authn/authz cases.",
            target_status="replacement_available",
            automated_migration_supported=True,
            version_management="platform_managed",
            notes="Re-check custom filter chains, method security, and any OAuth/JWT configuration after migration.",
        ),
        CompatibilityCatalogEntry(
            ga="org.springframework.boot:spring-boot-starter-validation",
            severity="review",
            replacement="io.micronaut.validation:micronaut-validation",
            rationale="Bean validation starter usage maps to Micronaut validation support in most common migration paths.",
            target_status="replacement_available",
            automated_migration_supported=True,
            version_management="platform_managed",
            notes="Verify any custom validator factories and binding error handling after migration.",
        ),
        CompatibilityCatalogEntry(
            ga="org.springframework.boot:spring-boot-starter-actuator",
            severity="review",
            replacement="io.micronaut:micronaut-management",
            rationale="Actuator endpoints generally migrate to Micronaut management endpoints, but custom exposure policies should be reviewed.",
            target_status="replacement_available",
            automated_migration_supported=True,
            version_management="platform_managed",
            notes="Revisit endpoint exposure, health groups, and metrics security in the target environment.",
        ),
        CompatibilityCatalogEntry(
            ga="org.springframework.boot:spring-boot-starter-aop",
            severity="review",
            replacement="io.micronaut:micronaut-aop",
            rationale="Basic proxy and interceptor use can often be carried to Micronaut AOP, though custom aspects still require review.",
            target_status="replacement_available",
            automated_migration_supported=True,
            version_management="platform_managed",
            notes="Validate ordering, pointcuts, and any runtime proxy assumptions after migration.",
        ),
        CompatibilityCatalogEntry(
            ga="org.springframework.boot:spring-boot-starter-data-redis",
            severity="review",
            replacement="io.micronaut.redis:micronaut-redis-lettuce",
            rationale="Spring Data Redis starter usage should be re-homed onto Micronaut Redis with Lettuce on Micronaut 4.",
            target_status="replacement_available",
            automated_migration_supported=True,
            version_management="platform_managed",
            notes="Re-check serialization, cache integration, and connection factory settings after migration.",
        ),
        CompatibilityCatalogEntry(
            ga="org.springframework.kafka:spring-kafka",
            severity="review",
            replacement="io.micronaut.kafka:micronaut-kafka",
            rationale="Common producer/consumer integrations can move to Micronaut Kafka, but listener semantics and error strategies must still be validated.",
            target_status="replacement_available",
            automated_migration_supported=True,
            version_management="platform_managed",
            notes="Review consumer groups, serializers, retries, and transactional semantics after migration.",
        ),
        CompatibilityCatalogEntry(
            ga="org.springframework.amqp:spring-rabbit",
            severity="review",
            replacement="io.micronaut.rabbitmq:micronaut-rabbitmq",
            rationale="Spring AMQP integrations often map to Micronaut RabbitMQ, though listener container behavior still needs runtime validation.",
            target_status="replacement_available",
            automated_migration_supported=True,
            version_management="platform_managed",
            notes="Validate queue declarations, listener concurrency, retries, and publisher confirms after migration.",
        ),
    )

    def __init__(self, knowledge_base: KnowledgeService, spring_version: str, micronaut_version: str):
        self.kb = knowledge_base
        self.spring_version = spring_version
        self.micronaut_version = micronaut_version
        self.maven_central = MavenCentralResolver()
        self._target_platform_managed_dependencies_cache: Optional[List[DependencyCoordinate]] = None
        self._target_platform_resolution_metadata_cache: Optional[Dict[str, object]] = None

    def _emit_progress(self, message: str) -> None:
        print(f"    [AUDIT] {message}", flush=True)

    def _metadata_budget_remaining(self, started_at: float) -> float:
        return max(0.0, MigrationConfig.BUILD_METADATA_TOTAL_BUDGET - (time.monotonic() - started_at))

    def _refresh_maven_local_repository(self, project_path: Optional[str]) -> None:
        resolved = resolve_maven_local_repository(project_path or "")
        self.maven_central.local_repository = Path(resolved).expanduser()

    def audit_maven_project(
        self,
        pom_path: str,
        project_path: Optional[str] = None,
        dependency_tree_text: Optional[str] = None,
        dependency_tree_path: Optional[str] = None,
        runtime_dependency_tree_text: Optional[str] = None,
        runtime_dependency_tree_path: Optional[str] = None,
        effective_pom_text: Optional[str] = None,
        effective_pom_path: Optional[str] = None,
    ) -> Dict[str, object]:
        self._refresh_maven_local_repository(project_path or os.path.dirname(pom_path))
        direct_dependencies = self.extract_direct_maven_dependencies(pom_path)
        build_context = self.extract_maven_build_context(pom_path)

        notes: List[str] = []
        compile_tree_text = dependency_tree_text
        runtime_tree_text = runtime_dependency_tree_text
        effective_text = effective_pom_text
        tree_source = "unavailable"
        effective_pom_source = "unavailable"
        metadata_started_at = time.monotonic()
        metadata_timed_out = False

        if not MigrationConfig.BUILD_METADATA_ENABLED:
            notes.append(
                "Resolved build metadata collection disabled; audit used deterministic local build-file evidence only."
            )

        if MigrationConfig.BUILD_METADATA_ENABLED and dependency_tree_path and compile_tree_text is None and os.path.exists(dependency_tree_path):
            with open(dependency_tree_path, "r", encoding="utf-8") as handle:
                compile_tree_text = handle.read()
            tree_source = "file"

        if MigrationConfig.BUILD_METADATA_ENABLED and runtime_dependency_tree_path and runtime_tree_text is None and os.path.exists(runtime_dependency_tree_path):
            with open(runtime_dependency_tree_path, "r", encoding="utf-8") as handle:
                runtime_tree_text = handle.read()
            tree_source = "file" if tree_source == "unavailable" else tree_source

        if MigrationConfig.BUILD_METADATA_ENABLED and effective_pom_path and effective_text is None and os.path.exists(effective_pom_path):
            with open(effective_pom_path, "r", encoding="utf-8") as handle:
                effective_text = handle.read()
            effective_pom_source = "file"

        if MigrationConfig.BUILD_METADATA_ENABLED and compile_tree_text is None and project_path and self._metadata_budget_remaining(metadata_started_at) > 0:
            self._emit_progress("Collecting Maven compile dependency tree...")
            compile_tree_text, command_note = self.collect_maven_dependency_tree(project_path, scope="compile")
            if compile_tree_text:
                tree_source = "generated"
                self._emit_progress("Maven compile dependency tree collected.")
            elif command_note:
                self._emit_progress(command_note)
                notes.append(command_note)
                metadata_timed_out = "timed out" in command_note.lower()

        if MigrationConfig.BUILD_METADATA_ENABLED and runtime_tree_text is None and project_path and not metadata_timed_out and self._metadata_budget_remaining(metadata_started_at) > 0:
            self._emit_progress("Collecting Maven runtime dependency tree...")
            runtime_tree_text, command_note = self.collect_maven_dependency_tree(project_path, scope="runtime")
            if runtime_tree_text and tree_source == "unavailable":
                tree_source = "generated"
                self._emit_progress("Maven runtime dependency tree collected.")
            elif command_note:
                self._emit_progress(command_note)
                notes.append(command_note)
                metadata_timed_out = "timed out" in command_note.lower()
        elif MigrationConfig.BUILD_METADATA_ENABLED and runtime_tree_text is None and project_path and metadata_timed_out:
            skip_note = "Skipping remaining Maven resolved metadata commands after timeout; audit continued with direct POM evidence only."
            self._emit_progress(skip_note)
            notes.append(skip_note)

        if MigrationConfig.BUILD_METADATA_ENABLED and effective_text is None and project_path and not metadata_timed_out and self._metadata_budget_remaining(metadata_started_at) > 0:
            self._emit_progress("Collecting Maven effective POM...")
            effective_text, command_note = self.collect_maven_effective_pom(project_path)
            if effective_text:
                effective_pom_source = "generated"
                self._emit_progress("Maven effective POM collected.")
            elif command_note:
                self._emit_progress(command_note)
                notes.append(command_note)
        elif MigrationConfig.BUILD_METADATA_ENABLED and effective_text is None and project_path and not metadata_timed_out:
            skip_note = (
                f"Skipping remaining Maven resolved metadata commands after "
                f"{MigrationConfig.BUILD_METADATA_TOTAL_BUDGET:g}s total budget was exhausted; "
                "audit continued with direct POM evidence only."
            )
            self._emit_progress(skip_note)
            notes.append(skip_note)

        if compile_tree_text is not None and runtime_tree_text is not None and tree_source == "unavailable":
            tree_source = "provided_scopes"
        elif compile_tree_text is not None and tree_source == "unavailable":
            tree_source = "provided_compile"

        if effective_text is not None and effective_pom_source == "unavailable":
            effective_pom_source = "provided_text"

        compile_dependencies = self.parse_maven_dependency_tree(compile_tree_text or "")
        runtime_dependencies = self.parse_maven_dependency_tree(runtime_tree_text or "")
        transitive_dependencies = self._merge_dependency_coordinates(compile_dependencies, runtime_dependencies)
        resolved_direct_dependencies = self.extract_direct_maven_dependencies_from_text(effective_text or "")
        managed_dependencies = self.extract_maven_managed_dependencies_from_text(effective_text or "")

        findings = self._audit_dependencies(direct_dependencies, transitive_dependencies, build_context)
        findings = self._enrich_maven_catalog_findings(
            findings,
            resolved_direct_dependencies=resolved_direct_dependencies,
            managed_dependencies=managed_dependencies,
        )
        report = self._build_audit_report(
            direct_dependencies=direct_dependencies,
            transitive_dependencies=transitive_dependencies,
            findings=findings,
            dependency_tree_source=tree_source,
            notes=notes,
            build_context=build_context,
            resolved_direct_dependencies=resolved_direct_dependencies,
            resolved_dependency_scopes={
                "compile": [item.to_dict() for item in compile_dependencies],
                "runtime": [item.to_dict() for item in runtime_dependencies],
            },
            effective_pom_source=effective_pom_source,
        )
        report["resolved_evidence"] = {
            "compile_dependency_tree_text": compile_tree_text or "",
            "runtime_dependency_tree_text": runtime_tree_text or "",
            "effective_pom_text": effective_text or "",
        }
        report.update(
            {
                "spring_version": self.spring_version,
                "micronaut_version": self.micronaut_version,
            }
        )
        return report

    def audit_gradle_project(
        self,
        build_file_path: str,
        project_path: Optional[str] = None,
        dependency_tree_text: Optional[str] = None,
        dependency_tree_path: Optional[str] = None,
        runtime_dependency_tree_text: Optional[str] = None,
        runtime_dependency_tree_path: Optional[str] = None,
    ) -> Dict[str, object]:
        direct_dependencies = self.extract_direct_gradle_dependencies(build_file_path)
        build_context = self.extract_gradle_build_context(build_file_path)
        platform_dependencies = self.extract_gradle_platform_dependencies(build_file_path)

        notes: List[str] = []
        compile_tree_text = dependency_tree_text
        runtime_tree_text = runtime_dependency_tree_text
        tree_source = "unavailable"

        if not MigrationConfig.BUILD_METADATA_ENABLED:
            notes.append(
                "Resolved build metadata collection disabled; audit used deterministic local build-file evidence only."
            )

        if MigrationConfig.BUILD_METADATA_ENABLED and dependency_tree_path and compile_tree_text is None and os.path.exists(dependency_tree_path):
            with open(dependency_tree_path, "r", encoding="utf-8") as handle:
                compile_tree_text = handle.read()
            tree_source = "file"

        if MigrationConfig.BUILD_METADATA_ENABLED and runtime_dependency_tree_path and runtime_tree_text is None and os.path.exists(runtime_dependency_tree_path):
            with open(runtime_dependency_tree_path, "r", encoding="utf-8") as handle:
                runtime_tree_text = handle.read()
            tree_source = "file" if tree_source == "unavailable" else tree_source

        if MigrationConfig.BUILD_METADATA_ENABLED and compile_tree_text is None and project_path:
            self._emit_progress("Collecting Gradle compileClasspath dependency graph...")
            compile_tree_text, command_note = self.collect_gradle_dependency_tree(
                project_path,
                configuration="compileClasspath",
            )
            if compile_tree_text:
                tree_source = "generated"
                self._emit_progress("Gradle compileClasspath dependency graph collected.")
            elif command_note:
                self._emit_progress(command_note)
                notes.append(command_note)

        if MigrationConfig.BUILD_METADATA_ENABLED and runtime_tree_text is None and project_path:
            self._emit_progress("Collecting Gradle runtimeClasspath dependency graph...")
            runtime_tree_text, command_note = self.collect_gradle_dependency_tree(
                project_path,
                configuration="runtimeClasspath",
            )
            if runtime_tree_text and tree_source == "unavailable":
                tree_source = "generated"
                self._emit_progress("Gradle runtimeClasspath dependency graph collected.")
            elif command_note:
                self._emit_progress(command_note)
                notes.append(command_note)

        if compile_tree_text is not None and runtime_tree_text is not None and tree_source == "unavailable":
            tree_source = "provided_scopes"
        elif compile_tree_text is not None and tree_source == "unavailable":
            tree_source = "provided_compile"
        elif runtime_tree_text is not None and tree_source == "unavailable":
            tree_source = "provided_runtime"

        compile_dependencies = self.parse_gradle_dependency_tree(compile_tree_text or "")
        runtime_dependencies = self.parse_gradle_dependency_tree(runtime_tree_text or "")
        transitive_dependencies = self._merge_dependency_coordinates_without_scope(
            compile_dependencies,
            runtime_dependencies,
        )
        resolved_direct_dependencies = self._extract_resolved_direct_dependencies_from_graphs(
            compile_dependencies,
            runtime_dependencies,
        )
        managed_dependencies = self._resolve_gradle_managed_dependencies(platform_dependencies)
        findings = self._audit_dependencies(direct_dependencies, transitive_dependencies, build_context)
        findings = self._enrich_maven_catalog_findings(
            findings,
            resolved_direct_dependencies=resolved_direct_dependencies,
            managed_dependencies=managed_dependencies,
        )
        report = self._build_audit_report(
            direct_dependencies=direct_dependencies,
            transitive_dependencies=transitive_dependencies,
            findings=findings,
            dependency_tree_source=tree_source,
            notes=notes,
            build_context=build_context,
            resolved_direct_dependencies=resolved_direct_dependencies,
            resolved_dependency_scopes={
                "compile": [item.to_dict() for item in compile_dependencies],
                "runtime": [item.to_dict() for item in runtime_dependencies],
            },
        )
        report["resolved_evidence"] = {
            "compile_dependency_tree_text": compile_tree_text or "",
            "runtime_dependency_tree_text": runtime_tree_text or "",
        }
        report.update(
            {
                "spring_version": self.spring_version,
                "micronaut_version": self.micronaut_version,
            }
        )
        return report

    def _build_audit_report(
        self,
        *,
        direct_dependencies: Sequence[DependencyCoordinate],
        transitive_dependencies: Sequence[DependencyCoordinate],
        findings: Sequence[DependencyFinding],
        dependency_tree_source: str,
        notes: Sequence[str],
        build_context: ProjectBuildContext,
        resolved_direct_dependencies: Sequence[DependencyCoordinate] = (),
        resolved_dependency_scopes: Optional[Dict[str, List[Dict[str, object]]]] = None,
        effective_pom_source: str = "unavailable",
    ) -> Dict[str, object]:
        severity_counts = {
            "blocking": len([item for item in findings if item.severity == "blocking"]),
            "review": len([item for item in findings if item.severity == "review"]),
            "info": len([item for item in findings if item.severity == "info"]),
        }

        overall_status = "pass"
        if severity_counts["blocking"] > 0:
            overall_status = "blocking"
        elif severity_counts["review"] > 0:
            overall_status = "review"

        target_platform_summary = self._build_target_platform_summary(build_context)
        evidence_quality = self._derive_evidence_quality(
            dependency_tree_source=dependency_tree_source,
            effective_pom_source=effective_pom_source,
            resolved_dependency_scopes=resolved_dependency_scopes or {"compile": [], "runtime": []},
            target_platform_summary=target_platform_summary.get("summary", {}),
        )
        all_notes = list(notes)
        platform_summary = dict(target_platform_summary.get("summary", {}) or {})
        platform_evidence_level = str(platform_summary.get("target_platform_evidence_level") or "")
        resolution_channel = str(platform_summary.get("target_platform_resolution_channel") or "")
        if platform_evidence_level == "exact_resolved":
            if resolution_channel == "local_maven_repo":
                all_notes.append(
                    "Exact Micronaut target platform managed dependencies were resolved for this audit from the local Maven repository."
                )
            else:
                all_notes.append(
                    "Exact Micronaut target platform managed dependencies were resolved for this audit."
                )
        elif platform_evidence_level == "configured_target_line":
            all_notes.append(
                "Micronaut target line is configured locally for the requested version, but full managed dependency inventory was not resolved in this runtime."
            )
        else:
            all_notes.append(
                "Micronaut target platform evidence could not be proven for this audit runtime."
            )
        dependency_graph_summary = self._build_dependency_graph_summary(
            transitive_dependencies=transitive_dependencies,
            findings=findings,
            target_platform_managed_dependencies=target_platform_summary.get("managed_dependencies", []),
        )

        return {
            "ok": severity_counts["blocking"] == 0,
            "status": overall_status,
            "spring_version": self.spring_version,
            "micronaut_version": self.micronaut_version,
            "direct_dependency_count": len(direct_dependencies),
            "transitive_dependency_count": len(transitive_dependencies),
            "resolved_direct_dependency_count": len(resolved_direct_dependencies),
            "dependency_tree_source": dependency_tree_source,
            "effective_pom_source": effective_pom_source,
            "evidence_quality": evidence_quality,
            "build_context": build_context.to_dict(),
            "notes": all_notes,
            "severity_counts": severity_counts,
            "direct_dependencies": [item.to_dict() for item in direct_dependencies],
            "transitive_dependencies": [item.to_dict() for item in transitive_dependencies],
            "resolved_direct_dependencies": [item.to_dict() for item in resolved_direct_dependencies],
            "resolved_dependency_scopes": resolved_dependency_scopes or {"compile": [], "runtime": []},
            "resolved_dependency_scope_counts": {
                scope: len(items) for scope, items in (resolved_dependency_scopes or {"compile": [], "runtime": []}).items()
            },
            "target_platform_summary": target_platform_summary.get("summary", {}),
            "dependency_graph_summary": dependency_graph_summary,
            "repository_intelligence_summary": self._build_repository_intelligence_summary(findings),
            "findings": [item.to_dict() for item in findings],
        }

    def resolve_target_platform_managed_dependencies(self) -> List[DependencyCoordinate]:
        if self._target_platform_managed_dependencies_cache is not None:
            return list(self._target_platform_managed_dependencies_cache)

        if not self.micronaut_version:
            self._target_platform_managed_dependencies_cache = []
            self._target_platform_resolution_metadata_cache = {
                "source": "unconfigured",
                "resolution_channel": "unconfigured",
                "imported_bom_count": 0,
                "unresolved_placeholder_count": 0,
                "visited_boms": [],
            }
            return []

        snapshot_managed, snapshot_metadata = self._load_target_platform_managed_dependencies_snapshot()
        if snapshot_managed:
            self._target_platform_managed_dependencies_cache = list(snapshot_managed)
            self._target_platform_resolution_metadata_cache = {
                "source": str(snapshot_metadata.get("source") or "init_snapshot"),
                "resolution_channel": str(snapshot_metadata.get("resolution_channel") or "init_snapshot"),
                "imported_bom_count": int(snapshot_metadata.get("imported_bom_count", 0) or 0),
                "unresolved_placeholder_count": int(snapshot_metadata.get("unresolved_placeholder_count", 0) or 0),
                "visited_boms": list(snapshot_metadata.get("visited_boms", []) or []),
                "snapshot_path": str(snapshot_metadata.get("snapshot_path") or ""),
            }
            return list(snapshot_managed)

        managed, metadata = self._resolve_target_platform_managed_dependencies_with_metadata(
            "io.micronaut.platform:micronaut-platform",
            self.micronaut_version,
        )
        source = "micronaut_platform_recursive"
        if not managed:
            managed, metadata = self._resolve_target_platform_managed_dependencies_with_metadata(
                "io.micronaut.platform:micronaut-parent",
                self.micronaut_version,
            )
            source = "micronaut_parent_recursive" if managed else "unresolved"

        self._target_platform_managed_dependencies_cache = list(managed)
        self._target_platform_resolution_metadata_cache = {
            "source": source,
            "resolution_channel": str(metadata.get("resolution_channel") or "unknown"),
            "imported_bom_count": int(metadata.get("imported_bom_count", 0) or 0),
            "unresolved_placeholder_count": int(metadata.get("unresolved_placeholder_count", 0) or 0),
            "visited_boms": list(metadata.get("visited_boms", []) or []),
        }
        return list(managed)

    def _load_target_platform_managed_dependencies_snapshot(
        self,
    ) -> Tuple[List[DependencyCoordinate], Dict[str, object]]:
        snapshot_path = str(MigrationConfig.TARGET_PLATFORM_MANAGED_FILE or "").strip()
        if not snapshot_path:
            return [], {}

        candidate = Path(snapshot_path).expanduser()
        if not candidate.exists():
            return [], {}

        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return [], {}

        if str(payload.get("micronaut_version") or "").strip() != self.micronaut_version:
            return [], {}

        raw_dependencies = payload.get("managed_dependencies", [])
        if not isinstance(raw_dependencies, list):
            return [], {}

        managed_dependencies: List[DependencyCoordinate] = []
        for item in raw_dependencies:
            if not isinstance(item, dict):
                continue
            group_id = str(item.get("group_id") or "").strip()
            artifact_id = str(item.get("artifact_id") or "").strip()
            if not group_id or not artifact_id:
                continue
            managed_dependencies.append(
                DependencyCoordinate(
                    group_id=group_id,
                    artifact_id=artifact_id,
                    version=str(item.get("version") or "").strip(),
                    scope=str(item.get("scope") or "").strip(),
                    packaging=str(item.get("packaging") or "").strip(),
                    classifier=str(item.get("classifier") or "").strip(),
                    source=str(item.get("source") or "direct"),
                    depth=int(item.get("depth", 0) or 0),
                    omitted=bool(item.get("omitted", False)),
                    omitted_for=str(item.get("omitted_for") or "").strip(),
                    parent_chain=tuple(item.get("parent_chain", []) or ()),
                )
            )

        metadata = dict(payload.get("target_platform_summary", {}) or {})
        metadata["snapshot_path"] = str(candidate)
        if not metadata.get("source"):
            metadata["source"] = "init_snapshot"
        if not metadata.get("resolution_channel"):
            metadata["resolution_channel"] = "init_snapshot"
        return managed_dependencies, metadata

    def _build_target_platform_summary(self, build_context: Optional[ProjectBuildContext] = None) -> Dict[str, object]:
        managed_dependencies = self.resolve_target_platform_managed_dependencies()
        metadata = dict(self._target_platform_resolution_metadata_cache or {})
        source = str(metadata.get("source") or "")
        resolution_channel = str(metadata.get("resolution_channel") or "unknown")
        resolved = bool(managed_dependencies)
        locally_configured = self._build_context_targets_exact_micronaut_version(build_context)
        evidence_level = "exact_resolved" if resolved else "configured_target_line" if locally_configured else "none"
        if resolved and source:
            summary_source = source
        elif locally_configured:
            summary_source = "configured_target_line"
        else:
            summary_source = source or ("unresolved" if not resolved else "unknown")
        return {
            "managed_dependencies": managed_dependencies,
            "summary": {
                "target_platform_ga": (
                    "io.micronaut.platform:micronaut-parent"
                    if source == "micronaut_parent_recursive"
                    else "io.micronaut.platform:micronaut-platform"
                ),
                "target_platform_version": self.micronaut_version,
                "target_platform_managed_dependency_count": len(managed_dependencies),
                "target_platform_resolved": resolved,
                "target_platform_locally_configured": locally_configured,
                "target_platform_evidence_level": evidence_level,
                "target_platform_source": summary_source,
                "target_platform_resolution_channel": resolution_channel,
                "target_platform_imported_bom_count": int(metadata.get("imported_bom_count", 0) or 0),
                "target_platform_unresolved_placeholder_count": int(
                    metadata.get("unresolved_placeholder_count", 0) or 0
                ),
                "target_platform_visited_bom_count": len(list(metadata.get("visited_boms", []) or [])),
            },
        }

    def _build_context_targets_exact_micronaut_version(
        self,
        build_context: Optional[ProjectBuildContext],
    ) -> bool:
        if build_context is None:
            return False

        if (
            build_context.maven_parent_ga == "io.micronaut.platform:micronaut-parent"
            and str(build_context.maven_parent_version or "").strip() == self.micronaut_version
        ):
            return True

        target_platform_coord = f"io.micronaut.platform:micronaut-platform:{self.micronaut_version}"
        if target_platform_coord in set(build_context.maven_managed_dependency_coords or ()):
            return True
        if target_platform_coord in set(build_context.gradle_platform_coords or ()):
            return True

        return False

    def _build_dependency_graph_summary(
        self,
        *,
        transitive_dependencies: Sequence[DependencyCoordinate],
        findings: Sequence[DependencyFinding],
        target_platform_managed_dependencies: Sequence[DependencyCoordinate],
    ) -> Dict[str, object]:
        max_depth = max((item.depth for item in transitive_dependencies), default=0)
        deep_transitives = [item for item in transitive_dependencies if item.depth >= 4]
        very_deep_transitives = [item for item in transitive_dependencies if item.depth >= 6]
        deep_spring = [item for item in deep_transitives if self._is_spring_dependency(item)]
        deep_javax = [
            item for item in deep_transitives
            if item.group_id.startswith("javax.") or item.ga.startswith("javax:")
        ]
        drift_findings = [item for item in findings if item.code == "micronaut_version_drift"]
        target_managed_lookup = {item.ga for item in target_platform_managed_dependencies}
        target_managed_direct_or_transitive = [
            item for item in transitive_dependencies
            if item.ga in target_managed_lookup
        ]

        return {
            "max_transitive_depth": max_depth,
            "deep_transitive_dependency_count": len(deep_transitives),
            "very_deep_transitive_dependency_count": len(very_deep_transitives),
            "deep_transitive_spring_dependency_count": len(deep_spring),
            "deep_transitive_javax_dependency_count": len(deep_javax),
            "micronaut_version_drift_count": len(drift_findings),
            "target_platform_managed_dependency_hits": len(target_managed_direct_or_transitive),
        }

    def _derive_evidence_quality(
        self,
        *,
        dependency_tree_source: str,
        effective_pom_source: str,
        resolved_dependency_scopes: Dict[str, List[Dict[str, object]]],
        target_platform_summary: Optional[Dict[str, object]] = None,
    ) -> str:
        has_compile = len(resolved_dependency_scopes.get("compile", [])) > 0
        has_runtime = len(resolved_dependency_scopes.get("runtime", [])) > 0
        has_effective_pom = effective_pom_source != "unavailable"
        platform_summary = dict(target_platform_summary or {})
        platform_exact = bool(platform_summary.get("target_platform_resolved"))
        platform_channel = str(platform_summary.get("target_platform_resolution_channel") or "")

        if platform_exact and platform_channel == "local_maven_repo":
            if has_compile and has_runtime and has_effective_pom:
                return "maven_resolved_full_with_local_target_platform"
            if has_compile or has_runtime or has_effective_pom or dependency_tree_source != "unavailable":
                return "local_target_platform_exact_plus_partial_resolved_evidence"
            return "local_target_platform_exact_only"

        if platform_exact:
            if has_compile and has_runtime and has_effective_pom:
                return "maven_resolved_full_with_exact_target_platform"
            if has_compile or has_runtime or has_effective_pom or dependency_tree_source != "unavailable":
                return "exact_target_platform_plus_partial_resolved_evidence"
            return "exact_target_platform_only"

        if has_compile and has_runtime and has_effective_pom:
            return "maven_resolved_full"
        if has_compile and has_runtime:
            return "resolved_scopes_only"
        if has_compile and (has_runtime or has_effective_pom):
            return "maven_resolved_partial"
        if dependency_tree_source != "unavailable" or has_effective_pom:
            return "partial_resolved_evidence"
        return "raw_build_only"

    def _build_repository_intelligence_summary(
        self,
        findings: Sequence[DependencyFinding],
    ) -> Dict[str, object]:
        repository_findings = [
            item for item in findings if item.code.startswith("repository_")
        ]
        codes = [item.code for item in repository_findings]
        affected_dependencies = sorted({item.dependency for item in repository_findings})
        return {
            "finding_count": len(repository_findings),
            "affected_dependency_count": len(affected_dependencies),
            "affected_dependencies": affected_dependencies,
            "severity_counts": {
                "blocking": len([item for item in repository_findings if item.severity == "blocking"]),
                "review": len([item for item in repository_findings if item.severity == "review"]),
                "info": len([item for item in repository_findings if item.severity == "info"]),
            },
            "codes": sorted(set(codes)),
        }

    def extract_maven_build_context(self, pom_path: str) -> ProjectBuildContext:
        ns = {"maven": "http://maven.apache.org/POM/4.0.0"}
        tree = ET.parse(pom_path)
        root = tree.getroot()

        parent = root.find("maven:parent", ns)
        parent_group = ((parent.findtext("maven:groupId", default="", namespaces=ns) if parent is not None else "") or "").strip()
        parent_artifact = ((parent.findtext("maven:artifactId", default="", namespaces=ns) if parent is not None else "") or "").strip()
        parent_version = ((parent.findtext("maven:version", default="", namespaces=ns) if parent is not None else "") or "").strip()
        parent_ga = f"{parent_group}:{parent_artifact}" if parent_group and parent_artifact else ""

        managed_dependency_gas: List[str] = []
        managed_dependency_coords: List[str] = []
        dep_mgmt = root.find("maven:dependencyManagement", ns)
        if dep_mgmt is not None:
            for dep in dep_mgmt.findall(".//maven:dependency", ns):
                group = (dep.findtext("maven:groupId", default="", namespaces=ns) or "").strip()
                artifact = (dep.findtext("maven:artifactId", default="", namespaces=ns) or "").strip()
                version = (dep.findtext("maven:version", default="", namespaces=ns) or "").strip()
                if group and artifact:
                    managed_dependency_gas.append(f"{group}:{artifact}")
                    if version:
                        managed_dependency_coords.append(f"{group}:{artifact}:{version}")

        return ProjectBuildContext(
            build_tool="maven",
            maven_parent_ga=parent_ga,
            maven_parent_version=parent_version,
            maven_managed_dependency_gas=tuple(sorted(set(managed_dependency_gas))),
            maven_managed_dependency_coords=tuple(sorted(set(managed_dependency_coords))),
        )

    def extract_direct_maven_dependencies(self, pom_path: str) -> List[DependencyCoordinate]:
        tree = ET.parse(pom_path)
        return self._extract_direct_maven_dependencies_from_root(tree.getroot())

    def extract_direct_maven_dependencies_from_text(self, pom_text: str) -> List[DependencyCoordinate]:
        if not pom_text.strip():
            return []
        try:
            root = ET.fromstring(pom_text)
        except ET.ParseError:
            return []
        return self._extract_direct_maven_dependencies_from_root(root)

    def extract_maven_managed_dependencies_from_text(self, pom_text: str) -> List[DependencyCoordinate]:
        if not pom_text.strip():
            return []
        try:
            root = ET.fromstring(pom_text)
        except ET.ParseError:
            return []

        ns = {"maven": "http://maven.apache.org/POM/4.0.0"}
        managed_node = root.find("maven:dependencyManagement/maven:dependencies", ns)
        if managed_node is None:
            return []

        coordinates: List[DependencyCoordinate] = []
        for dep in managed_node.findall("maven:dependency", ns):
            group = (dep.findtext("maven:groupId", default="", namespaces=ns) or "").strip()
            artifact = (dep.findtext("maven:artifactId", default="", namespaces=ns) or "").strip()
            version = (dep.findtext("maven:version", default="", namespaces=ns) or "").strip()
            scope = (dep.findtext("maven:scope", default="", namespaces=ns) or "").strip()
            packaging = (dep.findtext("maven:type", default="", namespaces=ns) or "").strip()
            if not group or not artifact:
                continue
            coordinates.append(
                DependencyCoordinate(
                    group_id=group,
                    artifact_id=artifact,
                    version=version,
                    scope=scope,
                    packaging=packaging,
                    source="managed",
                )
            )

        return coordinates

    def extract_direct_gradle_dependencies_from_tree_text(self, tree_text: str) -> List[DependencyCoordinate]:
        dependencies = self.parse_gradle_dependency_tree(tree_text)
        return [item for item in dependencies if item.depth == 1]

    def _extract_direct_maven_dependencies_from_root(self, root: ET.Element) -> List[DependencyCoordinate]:
        ns = {"maven": "http://maven.apache.org/POM/4.0.0"}
        dependencies_node = root.find("maven:dependencies", ns)
        if dependencies_node is None:
            return []

        coordinates: List[DependencyCoordinate] = []
        for dep in dependencies_node.findall("maven:dependency", ns):
            group = (dep.findtext("maven:groupId", default="", namespaces=ns) or "").strip()
            artifact = (dep.findtext("maven:artifactId", default="", namespaces=ns) or "").strip()
            version = (dep.findtext("maven:version", default="", namespaces=ns) or "").strip()
            scope = (dep.findtext("maven:scope", default="", namespaces=ns) or "").strip()
            packaging = (dep.findtext("maven:type", default="", namespaces=ns) or "").strip()
            if not group or not artifact:
                continue
            coordinates.append(
                DependencyCoordinate(
                    group_id=group,
                    artifact_id=artifact,
                    version=version,
                    scope=scope,
                    packaging=packaging,
                    source="direct",
                )
            )

        return coordinates

    def extract_direct_gradle_dependencies(self, build_file_path: str) -> List[DependencyCoordinate]:
        with open(build_file_path, "r", encoding="utf-8") as handle:
            content = handle.read()

        coordinates: List[DependencyCoordinate] = []
        seen: set[tuple[str, str, str, str]] = set()
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue

            string_match = self._GRADLE_STRING_DEP_RE.match(line)
            if string_match:
                ga = string_match.group("ga")
                if ga.startswith("project(") or ga.startswith("files("):
                    continue
                parts = [item.strip() for item in ga.split(":")]
                if len(parts) >= 2:
                    version = parts[2] if len(parts) >= 3 else ""
                    coordinate = DependencyCoordinate(
                        group_id=parts[0],
                        artifact_id=parts[1],
                        version=version,
                        scope=string_match.group("config"),
                        source="direct",
                    )
                    dedupe_key = (coordinate.group_id, coordinate.artifact_id, coordinate.version, coordinate.scope)
                    if dedupe_key not in seen:
                        seen.add(dedupe_key)
                        coordinates.append(coordinate)
                continue

            map_match = self._GRADLE_MAP_DEP_RE.match(line)
            if map_match:
                coordinate = DependencyCoordinate(
                    group_id=map_match.group("group"),
                    artifact_id=map_match.group("name"),
                    version=map_match.group("version") or "",
                    scope=map_match.group("config"),
                    source="direct",
                )
                dedupe_key = (coordinate.group_id, coordinate.artifact_id, coordinate.version, coordinate.scope)
                if dedupe_key not in seen:
                    seen.add(dedupe_key)
                    coordinates.append(coordinate)

        return coordinates

    def extract_gradle_build_context(self, build_file_path: str) -> ProjectBuildContext:
        with open(build_file_path, "r", encoding="utf-8") as handle:
            content = handle.read()

        platform_gas = []
        platform_coords = []
        for match in re.finditer(
            r"""(?:platform|enforcedPlatform)\s*\(\s*['"](?P<coords>[^'"]+)['"]\s*\)""",
            content,
        ):
            coords = match.group("coords").strip()
            parts = [item.strip() for item in coords.split(":")]
            if len(parts) >= 2:
                platform_gas.append(f"{parts[0]}:{parts[1]}")
            if len(parts) >= 3:
                platform_coords.append(f"{parts[0]}:{parts[1]}:{parts[2]}")

        return ProjectBuildContext(
            build_tool="gradle",
            gradle_platform_gas=tuple(sorted(set(platform_gas))),
            gradle_platform_coords=tuple(sorted(set(platform_coords))),
        )

    def extract_gradle_platform_dependencies(self, build_file_path: str) -> List[DependencyCoordinate]:
        with open(build_file_path, "r", encoding="utf-8") as handle:
            content = handle.read()

        coordinates: List[DependencyCoordinate] = []
        seen: set[tuple[str, str, str]] = set()
        for match in re.finditer(
            r"""(?:platform|enforcedPlatform)\s*\(\s*['"](?P<coords>[^'"]+)['"]\s*\)""",
            content,
        ):
            coords = match.group("coords").strip()
            parts = [item.strip() for item in coords.split(":")]
            if len(parts) < 3:
                continue
            coordinate = DependencyCoordinate(
                group_id=parts[0],
                artifact_id=parts[1],
                version=parts[2],
                scope="platform",
                source="managed",
            )
            dedupe_key = (coordinate.group_id, coordinate.artifact_id, coordinate.version)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            coordinates.append(coordinate)

        return coordinates

    def collect_maven_dependency_tree(self, project_path: str, scope: str = "compile") -> Tuple[Optional[str], Optional[str]]:
        base_command = ["mvn", "-q", "dependency:tree", "-DoutputType=text", f"-Dscope={scope}"]
        if os.path.exists(os.path.join(project_path, "mvnw")):
            base_command = ["./mvnw", "-q", "dependency:tree", "-DoutputType=text", f"-Dscope={scope}"]

        command_variants = []
        if MigrationConfig.BUILD_METADATA_OFFLINE_FIRST:
            command_variants.append(base_command[:1] + ["-o"] + base_command[1:])
        if not MigrationConfig.BUILD_METADATA_OFFLINE_FIRST or MigrationConfig.BUILD_METADATA_ALLOW_ONLINE_FALLBACK:
            command_variants.append(base_command)

        failure_notes: List[str] = []
        for index, command in enumerate(command_variants):
            try:
                result = subprocess.run(
                    command,
                    cwd=project_path,
                    capture_output=True,
                    text=True,
                    shell=False,
                    timeout=MigrationConfig.BUILD_METADATA_COMMAND_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                return (
                    None,
                    f"Maven dependency tree generation timed out after "
                    f"{MigrationConfig.BUILD_METADATA_COMMAND_TIMEOUT:g}s for scope={scope}; "
                    "audit continued with reduced dependency evidence.",
                )
            except Exception as exc:
                failure_notes.append(f"Dependency tree command failed to start: {exc}")
                continue

            output = (result.stdout or "") + "\n" + (result.stderr or "")
            if result.returncode == 0:
                return output, None

            is_offline_attempt = "-o" in command
            mode = "offline" if is_offline_attempt else "online"
            failure_notes.append(
                f"Maven dependency tree generation failed in {mode} mode for scope={scope}; audit continued with reduced dependency evidence."
            )
            if is_offline_attempt and not MigrationConfig.BUILD_METADATA_ALLOW_ONLINE_FALLBACK:
                break
            if index == len(command_variants) - 1:
                break

        return None, failure_notes[-1] if failure_notes else (
            f"Maven dependency tree generation failed for scope={scope}; audit continued with reduced dependency evidence."
        )

    def collect_maven_effective_pom(self, project_path: str) -> Tuple[Optional[str], Optional[str]]:
        base_command = ["mvn", "-q", "help:effective-pom", "-DforceStdout"]
        if os.path.exists(os.path.join(project_path, "mvnw")):
            base_command = ["./mvnw", "-q", "help:effective-pom", "-DforceStdout"]

        command_variants = []
        if MigrationConfig.BUILD_METADATA_OFFLINE_FIRST:
            command_variants.append(base_command[:1] + ["-o"] + base_command[1:])
        if not MigrationConfig.BUILD_METADATA_OFFLINE_FIRST or MigrationConfig.BUILD_METADATA_ALLOW_ONLINE_FALLBACK:
            command_variants.append(base_command)

        failure_notes: List[str] = []
        for index, command in enumerate(command_variants):
            try:
                result = subprocess.run(
                    command,
                    cwd=project_path,
                    capture_output=True,
                    text=True,
                    shell=False,
                    timeout=MigrationConfig.BUILD_METADATA_COMMAND_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                return (
                    None,
                    f"Effective POM generation timed out after "
                    f"{MigrationConfig.BUILD_METADATA_COMMAND_TIMEOUT:g}s; "
                    "audit continued without resolved direct dependency metadata.",
                )
            except Exception as exc:
                failure_notes.append(f"Effective POM command failed to start: {exc}")
                continue

            output = (result.stdout or "") + "\n" + (result.stderr or "")
            if result.returncode == 0:
                xml_match = re.search(r"(<project[\s\S]*</project>)", output)
                if xml_match:
                    return xml_match.group(1), None
                failure_notes.append("Effective POM command succeeded but no XML payload was detected.")
            else:
                is_offline_attempt = "-o" in command
                mode = "offline" if is_offline_attempt else "online"
                failure_notes.append(
                    f"Effective POM generation failed in {mode} mode; audit continued without resolved direct dependency metadata."
                )
                if is_offline_attempt and not MigrationConfig.BUILD_METADATA_ALLOW_ONLINE_FALLBACK:
                    break
                if index == len(command_variants) - 1:
                    break

        return None, failure_notes[-1] if failure_notes else (
            "Effective POM generation failed; audit continued without resolved direct dependency metadata."
        )

    def collect_gradle_dependency_tree(
        self,
        project_path: str,
        configuration: str = "runtimeClasspath",
    ) -> Tuple[Optional[str], Optional[str]]:
        command = ["gradle", "-q", "dependencies", "--configuration", configuration]
        if os.name != "nt" and os.path.exists(os.path.join(project_path, "gradlew")):
            command = ["./gradlew", "-q", "dependencies", "--configuration", configuration]

        try:
            result = subprocess.run(
                command,
                cwd=project_path,
                capture_output=True,
                text=True,
                shell=False,
                timeout=MigrationConfig.BUILD_METADATA_COMMAND_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return (
                None,
                f"Gradle dependency tree generation timed out after "
                f"{MigrationConfig.BUILD_METADATA_COMMAND_TIMEOUT:g}s for configuration={configuration}; "
                "audit continued with reduced dependency evidence.",
            )
        except Exception as exc:
            return None, f"Dependency tree command failed to start: {exc}"

        output = (result.stdout or "") + "\n" + (result.stderr or "")
        if result.returncode != 0:
            return None, (
                f"Gradle dependency tree generation failed for configuration={configuration}; "
                "audit continued with reduced dependency evidence."
            )

        return output, None

    def parse_maven_dependency_tree(self, tree_text: str) -> List[DependencyCoordinate]:
        dependencies: List[DependencyCoordinate] = []
        stack: Dict[int, DependencyCoordinate] = {}

        for raw_line in tree_text.splitlines():
            line = raw_line.strip("\n")
            if not line:
                continue
            if line.startswith("[INFO] "):
                line = line[len("[INFO] ") :]
            match = self._TREE_LINE_RE.match(line)
            if not match:
                continue

            depth = len(match.group("prefix")) // 3 + 1
            coords = match.group("coords")
            omitted = " omitted for " in coords
            omitted_for = ""
            if " (" in coords:
                coords, suffix = coords.split(" (", 1)
                suffix = suffix.rstrip(")")
                if "omitted for" in suffix:
                    omitted = True
                    omitted_for = suffix

            dependency = self._parse_tree_coordinate(
                coords=coords.strip(),
                depth=depth,
                omitted=omitted,
                omitted_for=omitted_for,
                stack=stack,
            )
            if dependency is None:
                continue

            stack[depth] = dependency
            for key in list(stack.keys()):
                if key > depth:
                    del stack[key]
            dependencies.append(dependency)

        return dependencies

    def parse_gradle_dependency_tree(self, tree_text: str) -> List[DependencyCoordinate]:
        dependencies: List[DependencyCoordinate] = []
        stack: Dict[int, DependencyCoordinate] = {}
        in_dependency_section = False
        current_scope = "runtimeClasspath"

        for raw_line in tree_text.splitlines():
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue
            if "runtimeClasspath" in stripped or "compileClasspath" in stripped:
                in_dependency_section = True
                current_scope = "compileClasspath" if "compileClasspath" in stripped else "runtimeClasspath"
                stack = {}
                continue
            if not in_dependency_section:
                continue
            if not (stripped.startswith("+---") or stripped.startswith("\\---") or stripped.startswith("|") or stripped.startswith("project ")):
                # End the current dependency section when Gradle moves on to the next heading.
                if stripped.endswith(" -") or stripped.startswith("---"):
                    in_dependency_section = False
                continue

            match = self._GRADLE_TREE_LINE_RE.match(line)
            if not match:
                continue

            depth = len(match.group("prefix")) // 5 + 1
            coords = match.group("coords").strip()
            omitted = " -> " in coords
            omitted_for = ""
            if " (" in coords:
                coords, suffix = coords.split(" (", 1)
                suffix = suffix.rstrip(")")
                if suffix:
                    omitted_for = suffix

            dependency = self._parse_gradle_tree_coordinate(
                coords=coords.strip(),
                depth=depth,
                omitted=omitted,
                omitted_for=omitted_for,
                stack=stack,
                scope=current_scope,
            )
            if dependency is None:
                continue

            stack[depth] = dependency
            for key in list(stack.keys()):
                if key > depth:
                    del stack[key]
            dependencies.append(dependency)

        return dependencies

    def _parse_tree_coordinate(
        self,
        coords: str,
        depth: int,
        omitted: bool,
        omitted_for: str,
        stack: Dict[int, DependencyCoordinate],
    ) -> Optional[DependencyCoordinate]:
        parts = [item.strip() for item in coords.split(":")]
        if len(parts) < 4:
            return None

        group_id = parts[0]
        artifact_id = parts[1]
        packaging = parts[2] if len(parts) >= 3 else ""
        classifier = ""
        version = ""
        scope = ""

        if len(parts) == 4:
            version = parts[3]
        elif len(parts) == 5:
            version = parts[3]
            scope = parts[4]
        else:
            classifier = parts[3]
            version = parts[4]
            scope = parts[5]

        parent_chain = tuple(
            item.ga
            for level, item in sorted(stack.items())
            if level < depth
        )

        return DependencyCoordinate(
            group_id=group_id,
            artifact_id=artifact_id,
            version=version,
            scope=scope,
            packaging=packaging,
            classifier=classifier,
            source="transitive",
            depth=depth,
            omitted=omitted,
            omitted_for=omitted_for,
            parent_chain=parent_chain,
        )

    def _parse_gradle_tree_coordinate(
        self,
        coords: str,
        depth: int,
        omitted: bool,
        omitted_for: str,
        stack: Dict[int, DependencyCoordinate],
        scope: str = "runtimeClasspath",
    ) -> Optional[DependencyCoordinate]:
        if coords.startswith("project "):
            return None

        resolved_version = ""
        if " -> " in coords:
            coords, resolved_version = [item.strip() for item in coords.split(" -> ", 1)]
            omitted = True
            omitted_for = omitted_for or f"resolved to {resolved_version}"

        parts = [item.strip() for item in coords.split(":")]
        if len(parts) < 2:
            return None

        group_id = parts[0]
        artifact_id = parts[1]
        version = resolved_version or (parts[2] if len(parts) >= 3 else "")

        parent_chain = tuple(
            item.ga
            for level, item in sorted(stack.items())
            if level < depth
        )

        return DependencyCoordinate(
            group_id=group_id,
            artifact_id=artifact_id,
            version=version,
            scope=scope,
            packaging="jar",
            source="transitive",
            depth=depth,
            omitted=omitted,
            omitted_for=omitted_for,
            parent_chain=parent_chain,
        )

    def _audit_dependencies(
        self,
        direct_dependencies: Sequence[DependencyCoordinate],
        transitive_dependencies: Sequence[DependencyCoordinate],
        build_context: ProjectBuildContext,
    ) -> List[DependencyFinding]:
        findings: List[DependencyFinding] = []
        all_dependencies = list(direct_dependencies) + list(transitive_dependencies)

        for dependency in direct_dependencies:
            findings.extend(self._audit_direct_dependency(dependency, build_context))

        for dependency in transitive_dependencies:
            findings.extend(self._audit_transitive_dependency(dependency, build_context))

        findings.extend(self._audit_version_conflicts(all_dependencies))
        findings.extend(self._audit_duplicate_spring_footprint(all_dependencies))

        deduped: Dict[Tuple[str, str, str, str], DependencyFinding] = {}
        for finding in findings:
            key = (finding.severity, finding.code, finding.dependency, finding.message)
            deduped[key] = finding
        return list(deduped.values())

    def _merge_dependency_coordinates(
        self,
        *groups: Sequence[DependencyCoordinate],
    ) -> List[DependencyCoordinate]:
        merged: List[DependencyCoordinate] = []
        seen: set[Tuple[str, str, str, str, int, Tuple[str, ...]]] = set()
        for group in groups:
            for dependency in group:
                key = (
                    dependency.group_id,
                    dependency.artifact_id,
                    dependency.version,
                    dependency.scope,
                    dependency.depth,
                    dependency.parent_chain,
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(dependency)
        return merged

    def _merge_dependency_coordinates_without_scope(
        self,
        *groups: Sequence[DependencyCoordinate],
    ) -> List[DependencyCoordinate]:
        merged: List[DependencyCoordinate] = []
        seen: set[Tuple[str, str, str, int, Tuple[str, ...], bool, str]] = set()
        for group in groups:
            for dependency in group:
                key = (
                    dependency.group_id,
                    dependency.artifact_id,
                    dependency.version,
                    dependency.depth,
                    dependency.parent_chain,
                    dependency.omitted,
                    dependency.omitted_for,
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(dependency)
        return merged

    def _extract_resolved_direct_dependencies_from_graphs(
        self,
        *groups: Sequence[DependencyCoordinate],
    ) -> List[DependencyCoordinate]:
        direct_groups = []
        for group in groups:
            direct_groups.append([item for item in group if item.depth == 1])
        return self._merge_dependency_coordinates_without_scope(*direct_groups)

    def _resolve_gradle_managed_dependencies(
        self,
        platform_dependencies: Sequence[DependencyCoordinate],
    ) -> List[DependencyCoordinate]:
        resolved: List[DependencyCoordinate] = []
        seen: set[tuple[str, str, str]] = set()

        for platform in platform_dependencies:
            if not platform.version:
                continue
            for dependency in self._fetch_maven_managed_dependencies(platform.ga, platform.version):
                dedupe_key = (dependency.group_id, dependency.artifact_id, dependency.version)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                resolved.append(dependency)

        return resolved

    def _fetch_maven_managed_dependencies(
        self,
        ga: str,
        version: str,
    ) -> List[DependencyCoordinate]:
        managed, _ = self._resolve_target_platform_managed_dependencies_with_metadata(ga, version)
        return managed

    def _resolve_target_platform_managed_dependencies_with_metadata(
        self,
        ga: str,
        version: str,
    ) -> Tuple[List[DependencyCoordinate], Dict[str, object]]:
        return self._fetch_maven_managed_dependencies_recursive(ga, version, visited=set())

    def _fetch_maven_managed_dependencies_recursive(
        self,
        ga: str,
        version: str,
        *,
        visited: set[Tuple[str, str]],
    ) -> Tuple[List[DependencyCoordinate], Dict[str, object]]:
        coordinate_key = (ga, version)
        if not ga or not version or coordinate_key in visited:
            return [], {
                "imported_bom_count": 0,
                "unresolved_placeholder_count": 0,
                "visited_boms": [f"{ga}:{version}"] if ga and version else [],
                "resolution_channels": [],
            }

        visited.add(coordinate_key)
        pom_result = self.maven_central.fetch_pom_text(ga, version)
        if pom_result.get("pom_available") is not True:
            return [], {
                "imported_bom_count": 0,
                "unresolved_placeholder_count": 0,
                "visited_boms": [f"{ga}:{version}"],
                "resolution_channels": [],
            }

        pom_text = str(pom_result.get("pom_text") or "")
        if not pom_text.strip():
            return [], {
                "imported_bom_count": 0,
                "unresolved_placeholder_count": 0,
                "visited_boms": [f"{ga}:{version}"],
                "resolution_channels": [],
            }

        try:
            root = ET.fromstring(pom_text)
        except ET.ParseError:
            return [], {
                "imported_bom_count": 0,
                "unresolved_placeholder_count": 0,
                "visited_boms": [f"{ga}:{version}"],
                "resolution_channels": [],
            }

        ns, prefix = self._pom_namespace(root)
        managed_node = root.find(f"{prefix}dependencyManagement/{prefix}dependencies", ns)
        if managed_node is None:
            return [], {
                "imported_bom_count": 0,
                "unresolved_placeholder_count": 0,
                "visited_boms": [f"{ga}:{version}"],
                "resolution_channels": [],
            }

        properties = self._extract_pom_properties(
            root,
            ga=ga,
            version=version,
            visited=visited,
        )

        coordinates: List[DependencyCoordinate] = []
        seen: set[Tuple[str, str, str]] = set()
        metadata = {
            "imported_bom_count": 0,
            "unresolved_placeholder_count": 0,
            "visited_boms": [f"{ga}:{version}"],
            "resolution_channels": [str(pom_result.get("source") or "").strip()],
        }

        for dep in managed_node.findall(f"{prefix}dependency", ns):
            group = self._resolve_maven_property_value(
                (dep.findtext(f"{prefix}groupId", default="", namespaces=ns) or "").strip(),
                properties,
            ).strip()
            artifact = self._resolve_maven_property_value(
                (dep.findtext(f"{prefix}artifactId", default="", namespaces=ns) or "").strip(),
                properties,
            ).strip()
            dep_version = self._resolve_maven_property_value(
                (dep.findtext(f"{prefix}version", default="", namespaces=ns) or "").strip(),
                properties,
            ).strip()
            scope = self._resolve_maven_property_value(
                (dep.findtext(f"{prefix}scope", default="", namespaces=ns) or "").strip(),
                properties,
            ).strip()
            packaging = self._resolve_maven_property_value(
                (dep.findtext(f"{prefix}type", default="", namespaces=ns) or "").strip(),
                properties,
            ).strip()

            if not group or not artifact:
                metadata["unresolved_placeholder_count"] += 1
                continue

            if not dep_version or "${" in dep_version:
                metadata["unresolved_placeholder_count"] += 1
                continue

            if scope == "import" and (packaging or "jar") == "pom":
                imported_coordinates, imported_metadata = self._fetch_maven_managed_dependencies_recursive(
                    f"{group}:{artifact}",
                    dep_version,
                    visited=visited,
                )
                metadata["imported_bom_count"] += 1 + int(imported_metadata.get("imported_bom_count", 0) or 0)
                metadata["unresolved_placeholder_count"] += int(
                    imported_metadata.get("unresolved_placeholder_count", 0) or 0
                )
                metadata["visited_boms"].extend(list(imported_metadata.get("visited_boms", []) or []))
                metadata["resolution_channels"].extend(list(imported_metadata.get("resolution_channels", []) or []))
                for dependency in imported_coordinates:
                    dedupe_key = (dependency.group_id, dependency.artifact_id, dependency.version)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    coordinates.append(dependency)
                continue

            dedupe_key = (group, artifact, dep_version)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            coordinates.append(
                DependencyCoordinate(
                    group_id=group,
                    artifact_id=artifact,
                    version=dep_version,
                    scope=scope or "managed",
                    packaging=packaging,
                    source="managed",
                )
            )

        metadata["visited_boms"] = sorted(set(str(item) for item in metadata["visited_boms"] if item))
        channels = sorted(set(str(item) for item in metadata.get("resolution_channels", []) if item))
        metadata["resolution_channels"] = channels
        if len(channels) == 1:
            metadata["resolution_channel"] = channels[0]
        elif len(channels) > 1:
            metadata["resolution_channel"] = "mixed"
        else:
            metadata["resolution_channel"] = "unknown"
        return coordinates, metadata

    def _pom_namespace(self, root: ET.Element) -> Tuple[Dict[str, str], str]:
        namespace = ""
        if root.tag.startswith("{") and "}" in root.tag:
            namespace = root.tag[1:].split("}", 1)[0]
        if namespace:
            return {"maven": namespace}, "maven:"
        return {}, ""

    def _extract_pom_properties(
        self,
        root: ET.Element,
        *,
        ga: str,
        version: str,
        visited: set[Tuple[str, str]],
    ) -> Dict[str, str]:
        ns, prefix = self._pom_namespace(root)
        group_id_default = ga.split(":", 1)[0] if ":" in ga else ""
        artifact_id_default = ga.split(":", 1)[1] if ":" in ga else ""

        inherited_properties: Dict[str, str] = {}
        parent_group = ""
        parent_artifact = ""
        parent_version = ""
        parent = root.find(f"{prefix}parent", ns)
        if parent is not None:
            parent_group = (parent.findtext(f"{prefix}groupId", default="", namespaces=ns) or "").strip()
            parent_artifact = (parent.findtext(f"{prefix}artifactId", default="", namespaces=ns) or "").strip()
            parent_version = (parent.findtext(f"{prefix}version", default="", namespaces=ns) or "").strip()
            if parent_group and parent_artifact and parent_version:
                parent_pom = self.maven_central.fetch_pom_text(f"{parent_group}:{parent_artifact}", parent_version)
                if parent_pom.get("pom_available") is True:
                    parent_text = str(parent_pom.get("pom_text") or "")
                    if parent_text.strip():
                        try:
                            parent_root = ET.fromstring(parent_text)
                        except ET.ParseError:
                            parent_root = None
                        if parent_root is not None:
                            inherited_properties.update(
                                self._extract_pom_properties(
                                    parent_root,
                                    ga=f"{parent_group}:{parent_artifact}",
                                    version=parent_version,
                                    visited=visited,
                                )
                            )

        project_group = (
            self._resolve_maven_property_value(
                (root.findtext(f"{prefix}groupId", default="", namespaces=ns) or "").strip(),
                inherited_properties,
            ).strip()
            or parent_group
            or group_id_default
        )
        project_artifact = (
            self._resolve_maven_property_value(
                (root.findtext(f"{prefix}artifactId", default="", namespaces=ns) or "").strip(),
                inherited_properties,
            ).strip()
            or artifact_id_default
        )
        project_version = (
            self._resolve_maven_property_value(
                (root.findtext(f"{prefix}version", default="", namespaces=ns) or "").strip(),
                inherited_properties,
            ).strip()
            or version
        )

        properties = dict(inherited_properties)
        properties.update(
            {
                "project.groupId": project_group,
                "pom.groupId": project_group,
                "groupId": project_group,
                "project.artifactId": project_artifact,
                "pom.artifactId": project_artifact,
                "artifactId": project_artifact,
                "project.version": project_version,
                "pom.version": project_version,
                "version": project_version,
                "parent.groupId": parent_group,
                "project.parent.groupId": parent_group,
                "parent.artifactId": parent_artifact,
                "project.parent.artifactId": parent_artifact,
                "parent.version": parent_version,
                "project.parent.version": parent_version,
            }
        )

        properties_node = root.find(f"{prefix}properties", ns)
        raw_properties: Dict[str, str] = {}
        if properties_node is not None:
            for child in list(properties_node):
                tag_name = child.tag.split("}", 1)[-1]
                raw_properties[tag_name] = (child.text or "").strip()

        unresolved = dict(raw_properties)
        for _ in range(5):
            changed = False
            for key, value in list(unresolved.items()):
                resolved = self._resolve_maven_property_value(value, properties).strip()
                if resolved != value:
                    changed = True
                properties[key] = resolved
                del unresolved[key]
            if not changed:
                break
        for key, value in raw_properties.items():
            properties[key] = self._resolve_maven_property_value(value, properties).strip()

        return properties

    def _resolve_maven_property_value(self, value: str, properties: Dict[str, str]) -> str:
        resolved = str(value or "")
        for _ in range(8):
            changed = False

            def replace(match: re.Match) -> str:
                nonlocal changed
                key = match.group(1)
                replacement = properties.get(key)
                if replacement is None:
                    return match.group(0)
                changed = True
                return str(replacement)

            updated = re.sub(r"\$\{([^}]+)\}", replace, resolved)
            resolved = updated
            if not changed:
                break
        return resolved

    def _audit_direct_dependency(
        self,
        dependency: DependencyCoordinate,
        build_context: ProjectBuildContext,
    ) -> List[DependencyFinding]:
        findings: List[DependencyFinding] = []
        auto_replacement_candidate = self._is_source_auto_replacement_candidate(dependency, build_context)

        kb_finding = self._kb_recommendation_finding(dependency)
        if kb_finding is not None:
            findings.append(kb_finding)

        catalog_finding = self._catalog_compatibility_finding(dependency)
        if catalog_finding is not None:
            findings.append(catalog_finding)

        if self._is_spring_dependency(dependency):
            migrated_target_context = (
                build_context.uses_micronaut_maven_parent()
                or build_context.uses_micronaut_maven_platform()
                or build_context.uses_micronaut_gradle_platform()
            )
            if migrated_target_context or not auto_replacement_candidate:
                findings.append(
                    DependencyFinding(
                        severity="blocking" if migrated_target_context else "review",
                        code="spring_direct_dependency" if migrated_target_context else "spring_source_dependency",
                        dependency=dependency.ga,
                        version=dependency.version,
                        source=dependency.source,
                        message=(
                            "Direct Spring dependency remains in the migrated project and should not survive a Micronaut migration unchanged."
                            if migrated_target_context
                            else "Direct Spring dependency is present in the source project and should be replaced or removed during migration."
                        ),
                        suggested_action=(
                            "Replace with a reviewed Micronaut equivalent or remove it if the behavior is no longer needed."
                            if migrated_target_context
                            else "Use the migration report to confirm this dependency is replaced in the migrated build."
                        ),
                    )
                )

        findings.extend(self._audit_common_dependency_signals(dependency, build_context))
        return findings

    def _audit_transitive_dependency(
        self,
        dependency: DependencyCoordinate,
        build_context: ProjectBuildContext,
    ) -> List[DependencyFinding]:
        findings: List[DependencyFinding] = []
        catalog_finding = self._catalog_compatibility_finding(dependency)
        if catalog_finding is not None:
            findings.append(self._downgrade_transitive_catalog_finding(catalog_finding, dependency))

        if self._is_spring_dependency(dependency):
            findings.append(
                DependencyFinding(
                    severity="review",
                    code="spring_transitive_dependency",
                    dependency=dependency.ga,
                    version=dependency.version,
                    source=dependency.source,
                    depth=dependency.depth,
                    related_dependencies=dependency.parent_chain,
                    message="Spring ecosystem code still appears transitively and may indicate an incomplete migration or a third-party starter that pulls Spring back onto the classpath.",
                    suggested_action="Trace the parent chain and replace the third-party dependency, starter, or bridge that reintroduces Spring APIs.",
                )
            )

        if dependency.omitted:
            findings.append(
                DependencyFinding(
                    severity="review",
                    code="omitted_dependency_conflict",
                    dependency=dependency.ga,
                    version=dependency.version,
                    source=dependency.source,
                    depth=dependency.depth,
                    related_dependencies=dependency.parent_chain,
                    message=f"Dependency graph reported a resolved or omitted transitive conflict: {dependency.omitted_for or 'conflict detected'}.",
                    suggested_action="Review the resolved version and make sure the surviving version is valid for the Micronaut target line.",
                )
            )

        findings.extend(self._audit_common_dependency_signals(dependency, build_context))
        findings.extend(self._audit_scope_specific_signals(dependency))
        return findings

    def _audit_common_dependency_signals(
        self,
        dependency: DependencyCoordinate,
        build_context: ProjectBuildContext,
    ) -> List[DependencyFinding]:
        findings: List[DependencyFinding] = []
        lower_group = dependency.group_id.lower()
        lower_artifact = dependency.artifact_id.lower()
        auto_replacement_candidate = self._is_source_auto_replacement_candidate(dependency, build_context)

        if lower_group.startswith("javax.") or lower_artifact.startswith("javax.") or lower_group == "javax":
            replacement = self._LEGACY_JAVAX_REPLACEMENTS.get(dependency.ga, "")
            suggested_action = (
                f"Replace with {replacement} and keep the Jakarta-era version aligned with the target Micronaut line."
                if replacement
                else "Verify whether this library has a Jakarta-compatible version or alternative before treating the migration as complete."
            )
            findings.append(
                DependencyFinding(
                    severity="review",
                    code="javax_dependency_review",
                    dependency=dependency.ga,
                    version=dependency.version,
                    source=dependency.source,
                    depth=dependency.depth,
                    related_dependencies=dependency.parent_chain,
                    message="Legacy javax-based dependency detected; Micronaut 4 era projects often need Jakarta-aligned dependencies instead.",
                    suggested_action=suggested_action,
                )
            )

        if dependency.group_id.startswith("io.micronaut") and dependency.version:
            version_finding = self._micronaut_version_finding(dependency)
            if version_finding is not None:
                findings.append(version_finding)

        if dependency.version.startswith("${"):
            findings.append(
                DependencyFinding(
                    severity="info",
                    code="property_managed_version",
                    dependency=dependency.ga,
                    version=dependency.version,
                    source=dependency.source,
                    depth=dependency.depth,
                    related_dependencies=dependency.parent_chain,
                    message="Dependency version is property-managed and could not be fully evaluated from the raw POM alone.",
                    suggested_action="Compare the resolved effective version or dependency tree before finalizing enterprise sign-off.",
                )
            )

        if not dependency.version and dependency.source == "direct":
            if self._is_version_managed_by_target_platform(dependency, build_context):
                findings.append(
                    DependencyFinding(
                        severity="info",
                        code="platform_managed_version",
                        dependency=dependency.ga,
                        source=dependency.source,
                        message="Direct dependency has no explicit version because it is managed by the configured Micronaut parent/BOM.",
                        suggested_action="Keep the dependency versionless unless you intentionally need to override the Micronaut-managed line.",
                    )
                )
            elif self._is_version_managed_by_source_platform(dependency, build_context):
                findings.append(
                    DependencyFinding(
                        severity="info",
                        code="source_platform_managed_version",
                        dependency=dependency.ga,
                        source=dependency.source,
                        message="Direct dependency has no explicit version because it is likely managed by the source parent/BOM.",
                        suggested_action="Verify the resolved version after migration, but this source-side versionless declaration alone is not a migration risk if the source platform managed it intentionally.",
                    )
                )
            elif not auto_replacement_candidate:
                findings.append(
                    DependencyFinding(
                        severity="review",
                        code="missing_direct_version",
                        dependency=dependency.ga,
                        source=dependency.source,
                        message="Direct dependency has no explicit version in the raw build file and may rely on a BOM or parent that changed during migration.",
                        suggested_action="Verify the resolved version after the Micronaut parent/BOM switch so you do not inherit an unintended library line.",
                    )
                )

        if self._is_third_party_starter_like_dependency(dependency) and not auto_replacement_candidate:
            findings.append(
                DependencyFinding(
                    severity="review",
                    code="third_party_starter_review",
                    dependency=dependency.ga,
                    version=dependency.version,
                    source=dependency.source,
                    depth=dependency.depth,
                    related_dependencies=dependency.parent_chain,
                    message="Third-party starter/autoconfigure-style dependency detected; these often hide framework-specific transitive assumptions.",
                    suggested_action="Check whether this library has a Micronaut-native module or whether it still brings Spring infrastructure transitively.",
                )
            )

        return findings

    def _is_source_auto_replacement_candidate(
        self,
        dependency: DependencyCoordinate,
        build_context: ProjectBuildContext,
    ) -> bool:
        target_context = (
            build_context.uses_micronaut_maven_parent()
            or build_context.uses_micronaut_maven_platform()
            or build_context.uses_micronaut_gradle_platform()
        )
        if target_context:
            return False

        entry = self.get_catalog_entry(dependency)
        if entry is None:
            return False

        return bool(
            self._is_spring_dependency(dependency)
            and entry.automated_migration_supported
            and entry.target_status == "replacement_available"
        )

    def _audit_scope_specific_signals(self, dependency: DependencyCoordinate) -> List[DependencyFinding]:
        findings: List[DependencyFinding] = []
        scope = (dependency.scope or "").lower()
        if "runtime" not in scope:
            return findings

        if self._is_spring_dependency(dependency):
            findings.append(
                DependencyFinding(
                    severity="review",
                    code="runtime_scope_spring_review",
                    dependency=dependency.ga,
                    version=dependency.version,
                    source=dependency.source,
                    depth=dependency.depth,
                    related_dependencies=dependency.parent_chain,
                    message="Dependency appears on the runtime classpath and still belongs to the Spring ecosystem.",
                    suggested_action="Review runtime behavior carefully; runtime-only Spring carryover is a strong signal that the migrated service may still depend on Spring infrastructure.",
                )
            )
        return findings

    def _kb_recommendation_finding(self, dependency: DependencyCoordinate) -> Optional[DependencyFinding]:
        try:
            rules = self.kb.search_dependency(
                dependency.artifact_id,
                spring_version=self.spring_version,
                micronaut_version=self.micronaut_version,
            )
        except Exception:
            return None

        if not rules:
            return None

        rule = rules[0]
        suggested_action = rule.micronaut_pattern
        severity = "review"
        if self._is_spring_dependency(dependency):
            severity = "blocking"
        return DependencyFinding(
            severity=severity,
            code="kb_dependency_mapping",
            dependency=dependency.ga,
            version=dependency.version,
            source=dependency.source,
            message="A reviewed dependency mapping exists for this artifact and should be applied consistently with the target Micronaut version.",
            suggested_action=f"Preferred mapping: {suggested_action}",
        )

    def _catalog_compatibility_finding(self, dependency: DependencyCoordinate) -> Optional[DependencyFinding]:
        entry = self.get_catalog_entry(dependency)
        if entry is None:
            return None

        code = "compatibility_catalog_replacement"
        message = entry.rationale
        suggested_action = f"Preferred Micronaut migration target: {entry.replacement}."
        if entry.target_status == "manual_redesign":
            code = "compatibility_catalog_manual_redesign"
            suggested_action = (
                f"Use {entry.replacement} as a starting point only if it fits; "
                "manual design migration is required."
            )
        verification = self.maven_central.verify_artifact(entry.replacement) if ":" in entry.replacement else {
            "checked": False,
            "source": "maven_central",
            "ga": entry.replacement,
            "available": None,
            "latest_version": "",
            "reason": "non_maven_coordinate",
        }
        repository_descriptor = self.maven_central.inspect_artifact(
            entry.replacement,
            version=entry.replacement_version,
        ) if ":" in entry.replacement else {
            "checked": False,
            "source": "maven_central_pom",
            "ga": entry.replacement,
            "available": None,
            "requested_version": entry.replacement_version,
            "resolved_version": entry.replacement_version,
            "pom_available": None,
            "reason": "non_maven_coordinate",
            "declared_dependency_count": 0,
            "compile_dependency_count": 0,
            "runtime_dependency_count": 0,
            "declared_compile_dependencies": [],
            "declared_runtime_dependencies": [],
            "spring_declared_dependencies": [],
            "javax_declared_dependencies": [],
        }
        if verification.get("available") is True:
            if entry.version_management == "platform_managed":
                suggested_action = (
                    f"{suggested_action} Verified in Maven Central. "
                    "Prefer the Micronaut platform-managed version instead of pinning a direct version."
                )
            else:
                suggested_action = f"{suggested_action} Verified in Maven Central."
        elif verification.get("available") is False:
            suggested_action = (
                f"{suggested_action} Maven Central verification did not find the suggested artifact; "
                "review the replacement manually before auto-upgrading."
            )
        if repository_descriptor.get("pom_available") is True:
            suggested_action = (
                f"{suggested_action} Repository POM inspection found "
                f"compile={repository_descriptor.get('compile_dependency_count', 0)} "
                f"and runtime={repository_descriptor.get('runtime_dependency_count', 0)} "
                "declared dependencies."
            )
        if entry.notes:
            suggested_action = f"{suggested_action} {entry.notes}"

        return DependencyFinding(
            severity=entry.severity,
            code=code,
            dependency=dependency.ga,
            version=dependency.version,
            source=dependency.source,
            depth=dependency.depth,
            related_dependencies=dependency.parent_chain,
            message=message,
            suggested_action=suggested_action,
            metadata={
                "repository_verification": verification,
                "repository_dependency_intelligence": repository_descriptor,
                "repository_verified": verification.get("available") is True,
                "verification_reason": verification.get("reason", ""),
                "latest_repository_version": str(verification.get("latest_version") or ""),
                "bom_compatible_recommended_version": "",
                "bom_compatible_version_source": "",
                "platform_reference_version": self.micronaut_version if entry.version_management == "platform_managed" else "",
                "recommended_upgrade_version": entry.replacement_version or (
                    str(verification.get("latest_version") or "") if entry.version_management != "platform_managed" else ""
                ),
                "suggested_replacement": entry.replacement,
                "replacement_version_management": entry.version_management,
            },
        )

    def _enrich_maven_catalog_findings(
        self,
        findings: Sequence[DependencyFinding],
        *,
        resolved_direct_dependencies: Sequence[DependencyCoordinate],
        managed_dependencies: Sequence[DependencyCoordinate],
    ) -> List[DependencyFinding]:
        resolved_lookup = self._dependency_lookup_by_ga(resolved_direct_dependencies)
        managed_lookup = self._dependency_lookup_by_ga(managed_dependencies)
        enriched: List[DependencyFinding] = []

        for finding in findings:
            if finding.code not in {
                "compatibility_catalog_replacement",
                "compatibility_catalog_manual_redesign",
            }:
                enriched.append(finding)
                continue

            metadata = dict(finding.metadata or {})
            replacement = str(metadata.get("suggested_replacement") or "")
            version_management = str(metadata.get("replacement_version_management") or "")
            repository_descriptor = dict(metadata.get("repository_dependency_intelligence") or {})
            latest_repository_version = str(
                metadata.get("latest_repository_version")
                or ((metadata.get("repository_verification") or {}).get("latest_version") if isinstance(metadata.get("repository_verification"), dict) else "")
                or ""
            )

            bom_version = ""
            bom_source = ""
            if replacement in resolved_lookup:
                bom_version = resolved_lookup[replacement].version
                bom_source = "resolved_direct_dependency"
            elif replacement in managed_lookup:
                bom_version = managed_lookup[replacement].version
                bom_source = "effective_dependency_management"

            metadata["latest_repository_version"] = latest_repository_version
            metadata["bom_compatible_recommended_version"] = bom_version
            metadata["bom_compatible_version_source"] = bom_source

            if bom_version:
                metadata["recommended_upgrade_version"] = bom_version
            elif version_management == "platform_managed":
                metadata["recommended_upgrade_version"] = ""

            suggested_action = finding.suggested_action
            if bom_version:
                suggested_action = (
                    f"{suggested_action} Resolved build metadata found a Micronaut-platform-compatible version: "
                    f"{bom_version}."
                )
            elif version_management == "platform_managed":
                suggested_action = (
                    f"{suggested_action} Target this through the Micronaut {self.micronaut_version} platform line "
                    "instead of pinning a direct artifact version."
                )

            if latest_repository_version:
                suggested_action = (
                    f"{suggested_action} Maven Central latest reported version: {latest_repository_version} "
                    "(informational only; do not override reviewed BOM/platform guidance blindly)."
                )

            spring_declared_dependencies = list(repository_descriptor.get("spring_declared_dependencies") or [])
            javax_declared_dependencies = list(repository_descriptor.get("javax_declared_dependencies") or [])
            transitive_spring_declared_dependencies = list(
                repository_descriptor.get("transitive_spring_declared_dependencies") or []
            )
            transitive_javax_declared_dependencies = list(
                repository_descriptor.get("transitive_javax_declared_dependencies") or []
            )
            child_dependency_unresolved_count = int(
                repository_descriptor.get("child_dependency_unresolved_count") or 0
            )
            child_micronaut_version_drift: List[str] = []
            for child in list(repository_descriptor.get("inspected_child_artifacts") or []):
                child_ga = str(child.get("ga") or "")
                child_version = str(child.get("resolved_version") or "")
                if not child_ga.startswith("io.micronaut:") or not child_version:
                    continue
                if normalize_major_minor(child_version) != normalize_major_minor(self.micronaut_version):
                    child_micronaut_version_drift.append(f"{child_ga}:{child_version}")
            if spring_declared_dependencies:
                suggested_action = (
                    f"{suggested_action} Repository dependency inspection also found Spring-linked declared dependencies "
                    f"({', '.join(spring_declared_dependencies[:3])}); review the replacement carefully before auto-upgrading."
                )
            if javax_declared_dependencies:
                suggested_action = (
                    f"{suggested_action} Repository dependency inspection found legacy servlet/javax-style declared "
                    f"dependencies ({', '.join(javax_declared_dependencies[:3])}); verify Jakarta-era compatibility."
                )
            if transitive_spring_declared_dependencies:
                suggested_action = (
                    f"{suggested_action} One-level child dependency inspection found downstream Spring-linked "
                    f"dependencies ({', '.join(transitive_spring_declared_dependencies[:3])}); this replacement may still "
                    "reintroduce Spring assumptions transitively."
                )
            if transitive_javax_declared_dependencies:
                suggested_action = (
                    f"{suggested_action} One-level child dependency inspection found downstream `javax`-style "
                    f"dependencies ({', '.join(transitive_javax_declared_dependencies[:3])}); verify Jakarta compatibility."
                )
            if child_micronaut_version_drift:
                suggested_action = (
                    f"{suggested_action} Child artifact inspection found Micronaut modules on a different major/minor line "
                    f"({', '.join(child_micronaut_version_drift[:3])}); align these to the target Micronaut {self.micronaut_version} line."
                )
            if child_dependency_unresolved_count:
                suggested_action = (
                    f"{suggested_action} Repository child inspection skipped {child_dependency_unresolved_count} declared "
                    "dependencies because their versions were inherited or property-managed in the published POM."
                )

            enriched_finding = DependencyFinding(
                severity=finding.severity,
                code=finding.code,
                dependency=finding.dependency,
                version=finding.version,
                source=finding.source,
                depth=finding.depth,
                related_dependencies=finding.related_dependencies,
                message=finding.message,
                suggested_action=suggested_action,
                metadata=metadata,
            )
            enriched.append(enriched_finding)
            enriched.extend(
                self._repository_descriptor_findings(
                    enriched_finding,
                    repository_descriptor=repository_descriptor,
                    child_micronaut_version_drift=child_micronaut_version_drift,
                )
            )

        deduped: Dict[Tuple[str, str, str, str], DependencyFinding] = {}
        for finding in enriched:
            key = (finding.severity, finding.code, finding.dependency, finding.message)
            deduped[key] = finding
        return list(deduped.values())

    def _repository_descriptor_findings(
        self,
        base_finding: DependencyFinding,
        *,
        repository_descriptor: Dict[str, object],
        child_micronaut_version_drift: Sequence[str],
    ) -> List[DependencyFinding]:
        findings: List[DependencyFinding] = []
        metadata = dict(base_finding.metadata or {})
        spring_declared_dependencies = list(repository_descriptor.get("spring_declared_dependencies") or [])
        javax_declared_dependencies = list(repository_descriptor.get("javax_declared_dependencies") or [])
        transitive_spring_declared_dependencies = list(
            repository_descriptor.get("transitive_spring_declared_dependencies") or []
        )
        transitive_javax_declared_dependencies = list(
            repository_descriptor.get("transitive_javax_declared_dependencies") or []
        )
        child_dependency_unresolved_count = int(repository_descriptor.get("child_dependency_unresolved_count") or 0)

        if spring_declared_dependencies:
            findings.append(
                DependencyFinding(
                    severity="blocking",
                    code="repository_declared_spring_risk",
                    dependency=base_finding.dependency,
                    version=base_finding.version,
                    source=base_finding.source,
                    depth=base_finding.depth,
                    related_dependencies=base_finding.related_dependencies,
                    message=(
                        "Replacement artifact's published POM still declares direct Spring-linked dependencies."
                    ),
                    suggested_action=(
                        f"Review the replacement carefully; direct Spring-linked declared dependencies were found: "
                        f"{', '.join(spring_declared_dependencies[:3])}."
                    ),
                    metadata=metadata,
                )
            )
        if javax_declared_dependencies:
            findings.append(
                DependencyFinding(
                    severity="review",
                    code="repository_declared_javax_risk",
                    dependency=base_finding.dependency,
                    version=base_finding.version,
                    source=base_finding.source,
                    depth=base_finding.depth,
                    related_dependencies=base_finding.related_dependencies,
                    message=(
                        "Replacement artifact's published POM still declares legacy `javax`-style dependencies."
                    ),
                    suggested_action=(
                        f"Verify Jakarta-era compatibility; declared legacy dependencies were found: "
                        f"{', '.join(javax_declared_dependencies[:3])}."
                    ),
                    metadata=metadata,
                )
            )
        if transitive_spring_declared_dependencies:
            findings.append(
                DependencyFinding(
                    severity="review",
                    code="repository_transitive_spring_risk",
                    dependency=base_finding.dependency,
                    version=base_finding.version,
                    source=base_finding.source,
                    depth=base_finding.depth,
                    related_dependencies=base_finding.related_dependencies,
                    message=(
                        "One-level child repository inspection found downstream Spring-linked dependencies."
                    ),
                    suggested_action=(
                        f"Review the replacement carefully; one-level child inspection found downstream Spring-linked "
                        f"dependencies: {', '.join(transitive_spring_declared_dependencies[:3])}."
                    ),
                    metadata=metadata,
                )
            )
        if transitive_javax_declared_dependencies:
            findings.append(
                DependencyFinding(
                    severity="review",
                    code="repository_transitive_javax_risk",
                    dependency=base_finding.dependency,
                    version=base_finding.version,
                    source=base_finding.source,
                    depth=base_finding.depth,
                    related_dependencies=base_finding.related_dependencies,
                    message=(
                        "One-level child repository inspection found downstream legacy `javax`-style dependencies."
                    ),
                    suggested_action=(
                        f"Verify Jakarta-era compatibility; one-level child inspection found downstream legacy "
                        f"dependencies: {', '.join(transitive_javax_declared_dependencies[:3])}."
                    ),
                    metadata=metadata,
                )
            )
        if child_micronaut_version_drift:
            findings.append(
                DependencyFinding(
                    severity="review",
                    code="repository_child_micronaut_version_drift",
                    dependency=base_finding.dependency,
                    version=base_finding.version,
                    source=base_finding.source,
                    depth=base_finding.depth,
                    related_dependencies=base_finding.related_dependencies,
                    message=(
                        "One-level child repository inspection found Micronaut modules on a different major/minor line."
                    ),
                    suggested_action=(
                        f"Align the replacement's child Micronaut modules to the target Micronaut {self.micronaut_version} "
                        f"line: {', '.join(child_micronaut_version_drift[:3])}."
                    ),
                    metadata=metadata,
                )
            )
        if child_dependency_unresolved_count:
            findings.append(
                DependencyFinding(
                    severity="info",
                    code="repository_child_version_unresolved",
                    dependency=base_finding.dependency,
                    version=base_finding.version,
                    source=base_finding.source,
                    depth=base_finding.depth,
                    related_dependencies=base_finding.related_dependencies,
                    message=(
                        "One-level child repository inspection could not inspect some declared dependencies because "
                        "their versions were inherited or property-managed."
                    ),
                    suggested_action=(
                        f"Review the published POM or resolve the artifact in a real build; "
                        f"{child_dependency_unresolved_count} child dependencies could not be inspected automatically."
                    ),
                    metadata=metadata,
                )
            )

        return findings

    def _dependency_lookup_by_ga(
        self,
        dependencies: Sequence[DependencyCoordinate],
    ) -> Dict[str, DependencyCoordinate]:
        lookup: Dict[str, DependencyCoordinate] = {}
        for dependency in dependencies:
            if not dependency.version:
                continue
            lookup[dependency.ga] = dependency
        return lookup

    def get_catalog_entry(self, dependency: DependencyCoordinate) -> Optional[CompatibilityCatalogEntry]:
        return next(
            (
                item
                for item in self._COMPATIBILITY_CATALOG
                if item.matches(dependency, self.spring_version, self.micronaut_version)
            ),
            None,
        )

    def _downgrade_transitive_catalog_finding(
        self,
        finding: DependencyFinding,
        dependency: DependencyCoordinate,
    ) -> DependencyFinding:
        severity = finding.severity
        if severity == "blocking":
            severity = "review"

        return DependencyFinding(
            severity=severity,
            code=finding.code,
            dependency=finding.dependency,
            version=finding.version,
            source=dependency.source,
            depth=dependency.depth,
            related_dependencies=dependency.parent_chain,
            message=(
                f"Transitive dependency matched the curated compatibility catalog. {finding.message}"
            ),
            suggested_action=finding.suggested_action,
        )

    def _micronaut_version_finding(self, dependency: DependencyCoordinate) -> Optional[DependencyFinding]:
        target_mm = normalize_major_minor(self.micronaut_version)
        dependency_mm = normalize_major_minor(dependency.version)
        if not target_mm or not dependency_mm or target_mm == dependency_mm:
            return None

        severity = "review"
        message = "Micronaut module version does not align with the requested Micronaut target line."
        if dependency_mm.split(".")[0] != target_mm.split(".")[0]:
            severity = "blocking"
            message = "Micronaut module major version does not match the requested target line."

        return DependencyFinding(
            severity=severity,
            code="micronaut_version_drift",
            dependency=dependency.ga,
            version=dependency.version,
            source=dependency.source,
            depth=dependency.depth,
            related_dependencies=dependency.parent_chain,
            message=message,
            suggested_action=f"Align the dependency to the Micronaut {target_mm}.x line or let the Micronaut platform BOM manage it.",
        )

    def _audit_version_conflicts(
        self,
        dependencies: Sequence[DependencyCoordinate],
    ) -> List[DependencyFinding]:
        version_index: Dict[str, set[str]] = {}
        for dependency in dependencies:
            if not dependency.version or dependency.version.startswith("${"):
                continue
            version_index.setdefault(dependency.ga, set()).add(dependency.version)

        findings: List[DependencyFinding] = []
        for ga, versions in sorted(version_index.items()):
            if len(versions) <= 1:
                continue
            findings.append(
                DependencyFinding(
                    severity="review",
                    code="multi_version_conflict",
                    dependency=ga,
                    message="Multiple versions of the same dependency appeared across the resolved graph.",
                    source="graph",
                    suggested_action=f"Resolve to one reviewed version. Seen versions: {', '.join(sorted(versions))}",
                    related_dependencies=tuple(sorted(versions)),
                )
            )
        return findings

    def _audit_duplicate_spring_footprint(
        self,
        dependencies: Sequence[DependencyCoordinate],
    ) -> List[DependencyFinding]:
        spring_deps = [dependency for dependency in dependencies if self._is_spring_dependency(dependency)]
        if not spring_deps:
            return []

        return [
            DependencyFinding(
                severity="review",
                code="spring_footprint_present",
                dependency="spring-ecosystem",
                source="graph",
                message="Spring libraries are still present somewhere in the resolved dependency graph.",
                suggested_action="Do not treat the migrated service as dependency-clean until all intentional Spring carryovers are reviewed.",
                related_dependencies=tuple(sorted({item.ga for item in spring_deps})[:10]),
            )
        ]

    def _is_spring_dependency(self, dependency: DependencyCoordinate) -> bool:
        group = dependency.group_id.lower()
        artifact = dependency.artifact_id.lower()
        text = f"{group}:{artifact}"
        return any(marker in text for marker in self._SPRING_MARKERS)

    def _is_version_managed_by_target_platform(
        self,
        dependency: DependencyCoordinate,
        build_context: ProjectBuildContext,
    ) -> bool:
        uses_target_platform = False
        if build_context.build_tool == "maven":
            uses_target_platform = (
                build_context.uses_micronaut_maven_parent() or build_context.uses_micronaut_maven_platform()
            )
        elif build_context.build_tool == "gradle":
            uses_target_platform = build_context.uses_micronaut_gradle_platform()

        if not uses_target_platform:
            return False
        if dependency.group_id.startswith("io.micronaut"):
            return True
        if dependency.ga in self._TARGET_PLATFORM_MANAGED_VERSION_ALLOWLIST:
            return True

        managed_lookup = {item.ga for item in self.resolve_target_platform_managed_dependencies()}
        return dependency.ga in managed_lookup

    def _is_version_managed_by_source_platform(
        self,
        dependency: DependencyCoordinate,
        build_context: ProjectBuildContext,
    ) -> bool:
        if build_context.build_tool != "maven":
            return False
        if dependency.ga in set(build_context.maven_managed_dependency_gas or ()):
            return True
        return build_context.uses_spring_boot_maven_parent() and dependency.ga in self._SOURCE_PLATFORM_MANAGED_VERSION_ALLOWLIST

    def _is_third_party_starter_like_dependency(self, dependency: DependencyCoordinate) -> bool:
        lower_group = dependency.group_id.lower()
        lower_artifact = dependency.artifact_id.lower()

        if self._is_spring_dependency(dependency):
            return False
        if lower_group.startswith("org.webjars"):
            return False
        if any(marker in lower_artifact or marker in lower_group for marker in self._KNOWN_THIRD_PARTY_RISK_MARKERS):
            return True
        if "starter" in lower_artifact:
            return True
        return False


def _severity_rank(severity: str) -> int:
    ranks = {
        "none": -1,
        "info": 0,
        "review": 1,
        "blocking": 2,
    }
    return ranks.get((severity or "").lower(), -1)


def _has_findings_at_or_above(report: Dict[str, object], threshold: str) -> bool:
    threshold_rank = _severity_rank(threshold)
    if threshold_rank < 0:
        return False

    for severity, count in dict(report.get("severity_counts", {})).items():
        if _severity_rank(severity) >= threshold_rank and int(count) > 0:
            return True
    return False


def write_dependency_audit_report(
    *,
    build_file_path: str,
    spring_version: str = "3.x",
    micronaut_version: str = "4.x",
    project_path: Optional[str] = None,
    dependency_tree_text: Optional[str] = None,
    dependency_tree_path: Optional[str] = None,
    runtime_dependency_tree_text: Optional[str] = None,
    runtime_dependency_tree_path: Optional[str] = None,
    effective_pom_text: Optional[str] = None,
    effective_pom_path: Optional[str] = None,
    report_path: Optional[str] = None,
    fail_on: str = "none",
) -> Dict[str, object]:
    auditor = DependencyCompatibilityAuditor(
        _NullKnowledgeBase(),
        spring_version=spring_version,
        micronaut_version=micronaut_version,
    )

    if build_file_path.endswith("pom.xml"):
        report = auditor.audit_maven_project(
            build_file_path,
            project_path=project_path,
            dependency_tree_text=dependency_tree_text,
            dependency_tree_path=dependency_tree_path,
            runtime_dependency_tree_text=runtime_dependency_tree_text,
            runtime_dependency_tree_path=runtime_dependency_tree_path,
            effective_pom_text=effective_pom_text,
            effective_pom_path=effective_pom_path,
        )
    elif build_file_path.endswith(".gradle") or build_file_path.endswith(".gradle.kts"):
        report = auditor.audit_gradle_project(
            build_file_path,
            project_path=project_path,
            dependency_tree_text=dependency_tree_text,
            dependency_tree_path=dependency_tree_path,
            runtime_dependency_tree_text=runtime_dependency_tree_text,
            runtime_dependency_tree_path=runtime_dependency_tree_path,
        )
    else:
        raise ValueError(f"Unsupported build file: {build_file_path}")

    report = {
        **report,
        "build_file_path": build_file_path,
        "report_generated_for": os.path.basename(build_file_path),
        "fail_on": fail_on,
        "failed_threshold": _has_findings_at_or_above(report, fail_on),
    }

    resolved_evidence_payload = dict(report.get("resolved_evidence") or {})
    resolved_evidence_paths: Dict[str, str] = {}
    if report_path:
        report_dir = os.path.dirname(report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        if resolved_evidence_payload:
            report_stem = os.path.splitext(os.path.basename(report_path))[0] if report_path else "dependency_audit"
            evidence_dir = os.path.join(report_dir or os.getcwd(), "dependency_evidence", report_stem)
            os.makedirs(evidence_dir, exist_ok=True)
            evidence_files = {
                "compile_dependency_tree_text": "compile_dependency_tree.txt",
                "runtime_dependency_tree_text": "runtime_dependency_tree.txt",
                "effective_pom_text": "effective_pom.xml",
            }
            for payload_key, filename in evidence_files.items():
                text = str(resolved_evidence_payload.get(payload_key, "") or "")
                if not text:
                    continue
                destination = os.path.join(evidence_dir, filename)
                with open(destination, "w", encoding="utf-8") as handle:
                    handle.write(text)
                resolved_evidence_paths[payload_key] = destination
        report["resolved_evidence_paths"] = resolved_evidence_paths
        report.pop("resolved_evidence", None)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    if report_path:
        report["report_path"] = report_path

    if report_path and "resolved_evidence" not in report:
        report["resolved_evidence_paths"] = resolved_evidence_paths
    elif not report_path:
        report["resolved_evidence_paths"] = {}

    return report


class _NullKnowledgeBase(KnowledgeService):
    def search_annotation(self, spring_annotation: str, **kwargs):
        return []

    def search_dependency(self, spring_dep: str, **kwargs):
        return []

    def search_configuration(self, spring_prop: str, **kwargs):
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit direct and transitive Maven or Gradle dependencies for Micronaut migration risks.")
    parser.add_argument("--build-file", required=True, help="Path to pom.xml, build.gradle, or build.gradle.kts")
    parser.add_argument("--project-path", help="Optional project root used to run dependency graph commands automatically")
    parser.add_argument("--dependency-tree", help="Optional path to a saved mvn dependency:tree output")
    parser.add_argument("--runtime-dependency-tree", help="Optional path to a saved Maven runtime-scope dependency:tree output")
    parser.add_argument("--effective-pom", help="Optional path to a saved Maven effective POM")
    parser.add_argument("--report", help="Optional path to write the dependency audit report JSON")
    parser.add_argument(
        "--fail-on",
        default="none",
        choices=["none", "info", "review", "blocking"],
        help="Exit non-zero when findings at or above this severity are present",
    )
    parser.add_argument("--spring-version", default="3.x", help="Source Spring version")
    parser.add_argument("--micronaut-version", default="4.x", help="Target Micronaut version")
    args = parser.parse_args()

    report = write_dependency_audit_report(
        build_file_path=args.build_file,
        spring_version=args.spring_version,
        micronaut_version=args.micronaut_version,
        project_path=args.project_path,
        dependency_tree_text=None,
        dependency_tree_path=args.dependency_tree,
        runtime_dependency_tree_text=None,
        runtime_dependency_tree_path=args.runtime_dependency_tree,
        effective_pom_text=None,
        effective_pom_path=args.effective_pom,
        report_path=args.report,
        fail_on=args.fail_on,
    )
    print(json.dumps(report, indent=2))
    if report.get("failed_threshold"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
