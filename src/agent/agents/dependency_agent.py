import os
import re
from xml.etree import ElementTree as ET
from typing import Dict, List, Optional

from src.agent.agents.dependency_audit import (
    DependencyCompatibilityAuditor,
    write_dependency_audit_report,
)
from src.agent.core.interfaces import KnowledgeService
from src.agent.core.versioning import normalize_major_minor

class DependencyAgent:
    """
    Expert agent for migrating build configuration files (Maven and Gradle).
    Handles dependency mappings, parent POM updates, and plugin conversions.
    """
    LEGACY_TRANSITIVE_EXCLUSION_ALLOWLIST = {
        "javax.validation:validation-api",
        "javax.servlet:javax.servlet-api",
        "javax.persistence:javax.persistence-api",
        "javax.annotation:javax.annotation-api",
        "javax.cache:cache-api",
        "javax.xml.bind:jaxb-api",
    }
    DEFAULT_MICRONAUT_4_GRADLE_PLUGIN_VERSION = "4.6.2"
    
    def __init__(self, knowledge_base: KnowledgeService, spring_version: str, micronaut_version: str):
        self.kb = knowledge_base
        self.spring_version = spring_version
        self.micronaut_version = micronaut_version
        self.auditor = DependencyCompatibilityAuditor(knowledge_base, spring_version, micronaut_version)
        self.maven_dependency_mappings = {
            "org.springframework.boot:spring-boot-starter-web": "io.micronaut:micronaut-http-server-netty",
            "org.springframework.boot:spring-boot-starter-test": "io.micronaut.test:micronaut-test-junit5",
            "org.springframework.boot:spring-boot-starter-validation": "io.micronaut.validation:micronaut-validation",
            "org.springframework.boot:spring-boot-starter-security": "io.micronaut.security:micronaut-security",
            "org.springframework.boot:spring-boot-starter-data-jpa": "io.micronaut.data:micronaut-data-hibernate-jpa",
            "org.springframework.boot:spring-boot-starter-thymeleaf": "io.micronaut.views:micronaut-views-thymeleaf",
            "org.springframework.boot:spring-boot-starter-cache": "io.micronaut.cache:micronaut-cache-caffeine",
            "org.springframework.boot:spring-boot-starter-actuator": "io.micronaut:micronaut-management",
            "org.ehcache:ehcache": "io.micronaut.cache:micronaut-cache-ehcache",
            "javax.validation:validation-api": "jakarta.validation:jakarta.validation-api",
            "javax.persistence:javax.persistence-api": "jakarta.persistence:jakarta.persistence-api",
            "javax.annotation:javax.annotation-api": "jakarta.annotation:jakarta.annotation-api",
            "javax.xml.bind:jaxb-api": "jakarta.xml.bind:jakarta.xml.bind-api",
            "javax.activation:activation": "jakarta.activation:jakarta.activation-api",
        }
        self.gradle_dependency_mappings = {
            "org.springframework.boot:spring-boot-starter-web": "io.micronaut:micronaut-http-server-netty",
            "org.springframework.boot:spring-boot-starter-test": "io.micronaut.test:micronaut-test-junit5",
            "org.springframework.boot:spring-boot-starter-validation": "io.micronaut.validation:micronaut-validation",
            "org.springframework.boot:spring-boot-starter-security": "io.micronaut.security:micronaut-security",
            "org.springframework.boot:spring-boot-starter-data-jpa": "io.micronaut.data:micronaut-data-hibernate-jpa",
            "org.springframework.boot:spring-boot-starter-thymeleaf": "io.micronaut.views:micronaut-views-thymeleaf",
            "javax.validation:validation-api": "jakarta.validation:jakarta.validation-api",
            "javax.persistence:javax.persistence-api": "jakarta.persistence:jakarta.persistence-api",
            "javax.annotation:javax.annotation-api": "jakarta.annotation:jakarta.annotation-api",
            "javax.xml.bind:jaxb-api": "jakarta.xml.bind:jakarta.xml.bind-api",
            "javax.activation:activation": "jakarta.activation:jakarta.activation-api",
        }
        self.maven_plugins_to_remove = {
            ("org.springframework.boot", "spring-boot-maven-plugin"),
            ("io.spring.javaformat", "spring-javaformat-maven-plugin"),
            ("pl.project13.maven", "git-commit-id-plugin"),
            ("org.apache.maven.plugins", "maven-checkstyle-plugin"),
            ("org.jacoco", "jacoco-maven-plugin"),
            ("org.eclipse.m2e", "lifecycle-mapping"),
        }
        self.supplemental_test_dependencies = (
            ("io.micronaut", "micronaut-http-client", None),
            ("org.assertj", "assertj-core", "3.26.3"),
            ("org.hamcrest", "hamcrest", "2.2"),
            ("org.mockito", "mockito-core", "5.12.0"),
            ("org.mockito", "mockito-junit-jupiter", "5.12.0"),
        )
        self.supplemental_runtime_dependencies = {
            "web": (
                ("io.micronaut", "micronaut-jackson-databind", None),
            ),
            "data": (
                ("io.micronaut.sql", "micronaut-jdbc-hikari", None),
            ),
            "validation": (
                ("io.micronaut.beanvalidation", "micronaut-hibernate-validator", None),
            ),
        }
        self.source_managed_runtime_preserve_allowlist = {
            "com.mysql:mysql-connector-j",
            "org.postgresql:postgresql",
            "org.mariadb.jdbc:mariadb-java-client",
        }

    def _resolve_micronaut_gradle_plugin_version(self) -> str:
        override = str(os.getenv("MICRONAUT_GRADLE_PLUGIN_VERSION") or "").strip()
        if override:
            return override

        target_line = normalize_major_minor(self.micronaut_version)
        if target_line.startswith("4."):
            return self.DEFAULT_MICRONAUT_4_GRADLE_PLUGIN_VERSION
        return self.micronaut_version

    def _write_gradle_properties(self, gradle_path: str, output_path: str) -> bool:
        source_properties_path = os.path.join(os.path.dirname(gradle_path), "gradle.properties")
        output_properties_path = os.path.join(os.path.dirname(output_path), "gradle.properties")
        existing_lines: List[str] = []
        if os.path.exists(source_properties_path):
            try:
                with open(source_properties_path, "r", encoding="utf-8") as handle:
                    existing_lines = handle.read().splitlines()
            except OSError:
                existing_lines = []

        updated_lines: List[str] = []
        saw_micronaut_version = False
        for line in existing_lines:
            if re.match(r"^\s*micronautVersion\s*=", line):
                updated_lines.append(f"micronautVersion={self.micronaut_version}")
                saw_micronaut_version = True
            else:
                updated_lines.append(line)

        if not saw_micronaut_version:
            if updated_lines and updated_lines[-1].strip():
                updated_lines.append("")
            updated_lines.append(f"micronautVersion={self.micronaut_version}")

        os.makedirs(os.path.dirname(output_properties_path), exist_ok=True)
        content = "\n".join(updated_lines).rstrip() + "\n"
        with open(output_properties_path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return True

    def _minimum_supported_java_version(self) -> int:
        target_line = normalize_major_minor(self.micronaut_version)
        if target_line.startswith("4."):
            return 17
        return 11

    def _normalize_gradle_java_compatibility(self, content: str, *, is_kts: bool) -> tuple[str, Dict[str, str]]:
        changes: Dict[str, str] = {}
        source_match = re.search(r"(?m)^\s*sourceCompatibility\s*=\s*['\"]?([^'\"]+)['\"]?\s*$", content)
        target_match = re.search(r"(?m)^\s*targetCompatibility\s*=\s*['\"]?([^'\"]+)['\"]?\s*$", content)
        if not source_match and not target_match:
            return content, changes

        source_value = source_match.group(1).strip() if source_match else ""
        target_value = target_match.group(1).strip() if target_match else ""

        normalized = re.sub(r"(?m)^\s*sourceCompatibility\s*=\s*['\"]?([^'\"]+)['\"]?\s*$\n?", "", content)
        normalized = re.sub(r"(?m)^\s*targetCompatibility\s*=\s*['\"]?([^'\"]+)['\"]?\s*$\n?", "", normalized)

        minimum_java = self._minimum_supported_java_version()

        def normalize_java_major(raw: str) -> int:
            value = str(raw or "").strip()
            if value.startswith("JavaVersion.VERSION_"):
                digits = re.sub(r"[^0-9]", "", value)
                return int(digits) if digits else minimum_java
            digits = re.sub(r"[^0-9]", "", value)
            if digits:
                parsed = int(digits)
                if parsed == 1 and value.startswith("1."):
                    legacy = re.sub(r"[^0-9]", "", value.split(".", 1)[1])
                    return int(legacy) if legacy else minimum_java
                return parsed
            return minimum_java

        def to_java_version_expr(raw: str) -> str:
            value = raw.strip()
            if value.startswith("JavaVersion.VERSION_"):
                version_major = normalize_java_major(value)
                return f"JavaVersion.VERSION_{max(version_major, minimum_java)}"
            digits = re.sub(r"[^0-9]", "_", value).strip("_")
            if digits:
                version_major = normalize_java_major(value)
                return f"JavaVersion.VERSION_{max(version_major, minimum_java)}"
            return f"JavaVersion.VERSION_{minimum_java}"

        java_block_lines = ["java {"]
        if source_value:
            java_block_lines.append(f"    sourceCompatibility = {to_java_version_expr(source_value)}")
        if target_value:
            java_block_lines.append(f"    targetCompatibility = {to_java_version_expr(target_value)}")
        java_block_lines.append("}")
        java_block = "\n".join(java_block_lines)

        existing_java_block = re.search(r"(?s)java\s*\{.*?\}", normalized)
        if existing_java_block:
            java_body_lines: List[str] = []
            if source_value:
                java_body_lines.append(f"    sourceCompatibility = {to_java_version_expr(source_value)}")
            if target_value:
                java_body_lines.append(f"    targetCompatibility = {to_java_version_expr(target_value)}")
            if java_body_lines:
                insertion = "\n" + "\n".join(java_body_lines)
                normalized = (
                    normalized[: existing_java_block.end() - 1]
                    + insertion
                    + "\n}"
                    + normalized[existing_java_block.end():]
                )
        else:
            plugins_block = re.search(r"(?s)plugins\s*\{.*?\}", normalized)
            insertion = f"\n\n{java_block}"
            if plugins_block:
                insert_at = plugins_block.end()
                normalized = normalized[:insert_at] + insertion + normalized[insert_at:]
            else:
                normalized = java_block + "\n\n" + normalized.lstrip()

        changes["gradle.java.compatibility"] = "Normalized sourceCompatibility/targetCompatibility into java block"
        return normalized, changes

    def _remove_redundant_gradle_java_apply(self, content: str) -> tuple[str, bool]:
        if not re.search(r"""(?ms)plugins\s*\{.*?\bid\s+['"]java['"].*?\}""", content):
            return content, False

        updated, replacements = re.subn(
            r"""(?m)^\s*apply\s+plugin\s*:\s*['"]java['"]\s*\n?""",
            "",
            content,
        )
        return updated, replacements > 0

    def _scan_gradle_braces(self, line: str) -> tuple[int, int, int]:
        open_count = 0
        close_count = 0
        leading_close_count = 0
        in_single = False
        in_double = False
        escaped = False
        saw_non_whitespace = False

        for char in line:
            if escaped:
                escaped = False
                continue
            if char == "\\" and (in_single or in_double):
                escaped = True
                continue
            if char == "'" and not in_double:
                in_single = not in_single
                continue
            if char == '"' and not in_single:
                in_double = not in_double
                continue
            if in_single or in_double:
                continue
            if char.isspace() and not saw_non_whitespace:
                continue
            if char == "}":
                close_count += 1
                if not saw_non_whitespace:
                    leading_close_count += 1
                saw_non_whitespace = True
                continue
            if char == "{":
                open_count += 1
                saw_non_whitespace = True
                continue
            if not char.isspace():
                saw_non_whitespace = True

        return open_count, close_count, leading_close_count

    def _normalize_gradle_groovy_formatting(self, content: str) -> str:
        lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        formatted_lines: List[str] = []
        indent_level = 0
        previous_blank = False

        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped:
                if formatted_lines and not previous_blank:
                    formatted_lines.append("")
                previous_blank = True
                continue

            open_count, close_count, leading_close_count = self._scan_gradle_braces(stripped)
            indent_before = max(indent_level - leading_close_count, 0)
            formatted_lines.append(f"{'  ' * indent_before}{stripped}")
            indent_level = max(indent_level + open_count - close_count, 0)
            previous_blank = False

        while formatted_lines and not formatted_lines[-1]:
            formatted_lines.pop()
        return "\n".join(formatted_lines) + "\n"

    def _source_requires_direct_jcache_api(self, project_path: Optional[str]) -> bool:
        if not project_path or not os.path.isdir(project_path):
            return True

        allowed_imports = {
            "javax.cache.configuration.MutableConfiguration",
            "javax.cache.configuration.Configuration",
        }
        allowed_fqcns = (
            "javax.cache.configuration.MutableConfiguration",
            "javax.cache.configuration.Configuration",
        )

        for root_dir, _, files in os.walk(project_path):
            for filename in files:
                if not filename.endswith(".java"):
                    continue
                source_path = os.path.join(root_dir, filename)
                try:
                    with open(source_path, "r", encoding="utf-8") as handle:
                        content = handle.read()
                except OSError:
                    continue
                if "javax.cache" not in content:
                    continue

                for imported in re.findall(r"(?m)^\s*import\s+(javax\.cache\.[^;]+);", content):
                    if imported not in allowed_imports:
                        return True

                content_without_imports = re.sub(r"(?m)^\s*import\s+javax\.cache\.[^;]+;\n?", "", content)
                for match in re.findall(r"javax\.cache\.[A-Za-z0-9_$.]+", content_without_imports):
                    if not any(match.startswith(prefix) for prefix in allowed_fqcns):
                        return True

        return False

    def _should_drop_direct_jcache_api_dependency(self, project_path: Optional[str], original_content: str) -> bool:
        if "javax.cache:cache-api" not in original_content:
            return False
        if "org.springframework.boot:spring-boot-starter-cache" not in original_content:
            return False
        return not self._source_requires_direct_jcache_api(project_path)

    def _remove_maven_plugins(self, root: ET.Element, ns: Dict[str, str], changes: Dict[str, str]) -> None:
        plugin_containers = root.findall(".//maven:plugins", ns)
        for plugins in plugin_containers:
            plugins_to_remove = []
            for plugin in plugins.findall("maven:plugin", ns):
                group = plugin.findtext("maven:groupId", default="", namespaces=ns).strip()
                artifact = plugin.findtext("maven:artifactId", default="", namespaces=ns).strip()
                if (group, artifact) in self.maven_plugins_to_remove:
                    plugins_to_remove.append(plugin)
                    changes[artifact] = "Removed"
                    continue
                if group and "spring" in group.lower():
                    plugins_to_remove.append(plugin)
                    changes[artifact or group] = "Removed"
            for plugin in plugins_to_remove:
                plugins.remove(plugin)

    def _remove_spring_repositories(self, root: ET.Element, ns: Dict[str, str], changes: Dict[str, str]) -> None:
        for container_tag in ("repositories", "pluginRepositories"):
            for container in root.findall(f".//maven:{container_tag}", ns):
                for repo in list(container):
                    url = repo.findtext("maven:url", default="", namespaces=ns).strip().lower()
                    repo_id = repo.findtext("maven:id", default="", namespaces=ns).strip()
                    repo_name = repo.findtext("maven:name", default="", namespaces=ns).strip()
                    descriptor = repo_id or repo_name or url or container_tag
                    if "spring.io" in url:
                        container.remove(repo)
                        changes[f"{container_tag}:{descriptor}"] = "Removed"

    def _remove_maven_properties(self, root: ET.Element, ns: Dict[str, str], property_names: set[str]) -> None:
        properties = root.find("maven:properties", ns)
        if properties is None:
            return
        for child in list(properties):
            tag_name = child.tag.split("}", 1)[-1]
            if tag_name in property_names:
                properties.remove(child)

    def _remove_non_runtime_profiles(self, root: ET.Element, ns: Dict[str, str], changes: Dict[str, str]) -> None:
        profiles = root.find("maven:profiles", ns)
        if profiles is None:
            return
        for profile in list(profiles.findall("maven:profile", ns)):
            profile_id = profile.findtext("maven:id", default="", namespaces=ns).strip()
            if profile_id == "m2e":
                profiles.remove(profile)
                changes["profile:m2e"] = "Removed"

    def _remove_empty_maven_nodes(self, root: ET.Element, ns: Dict[str, str]) -> None:
        changed = True
        while changed:
            changed = False
            parent_map = {child: parent for parent in root.iter() for child in list(parent)}
            for node in list(root.iter()):
                parent = parent_map.get(node)
                if parent is None:
                    continue
                if list(node):
                    continue
                if (node.text or "").strip():
                    continue
                tag_name = node.tag.split("}", 1)[-1]
                if tag_name in {
                    "plugins",
                    "pluginRepositories",
                    "repositories",
                    "dependencies",
                    "dependencyManagement",
                    "pluginManagement",
                    "build",
                    "profiles",
                    "profile",
                    "properties",
                }:
                    parent.remove(node)
                    changed = True

    def _cleanup_maven_metadata(self, root: ET.Element, ns: Dict[str, str], changes: Dict[str, str]) -> None:
        self._remove_maven_plugins(root, ns, changes)
        self._remove_spring_repositories(root, ns, changes)
        self._remove_maven_properties(
            root,
            ns,
            {"spring-format.version", "nohttp-checkstyle.version", "jacoco.version"},
        )
        self._remove_non_runtime_profiles(root, ns, changes)
        self._remove_empty_maven_nodes(root, ns)

    def _ensure_maven_dependency(
        self,
        dependencies_node: ET.Element,
        ns: Dict[str, str],
        *,
        group_id: str,
        artifact_id: str,
        scope: Optional[str] = None,
        version: Optional[str] = None,
    ) -> bool:
        tag_prefix = f"{{{ns['maven']}}}"
        for dep in dependencies_node.findall("maven:dependency", ns):
            current_group = dep.findtext("maven:groupId", default="", namespaces=ns).strip()
            current_artifact = dep.findtext("maven:artifactId", default="", namespaces=ns).strip()
            if current_group != group_id or current_artifact != artifact_id:
                continue
            if scope:
                scope_node = dep.find("maven:scope", ns)
                if scope_node is None:
                    scope_node = ET.SubElement(dep, f"{tag_prefix}scope")
                scope_node.text = scope
            return False

        dep = ET.SubElement(dependencies_node, f"{tag_prefix}dependency")
        ET.SubElement(dep, f"{tag_prefix}groupId").text = group_id
        ET.SubElement(dep, f"{tag_prefix}artifactId").text = artifact_id
        if version:
            ET.SubElement(dep, f"{tag_prefix}version").text = version
        if scope:
            ET.SubElement(dep, f"{tag_prefix}scope").text = scope
        return True

    def _ensure_maven_compiler_annotation_processor(
        self,
        root: ET.Element,
        ns: Dict[str, str],
        *,
        group_id: str,
        artifact_id: str,
    ) -> bool:
        tag_prefix = f"{{{ns['maven']}}}"
        build = root.find("maven:build", ns)
        if build is None:
            build = ET.SubElement(root, f"{tag_prefix}build")
        plugins = build.find("maven:plugins", ns)
        if plugins is None:
            plugins = ET.SubElement(build, f"{tag_prefix}plugins")

        compiler_plugin = None
        for plugin in plugins.findall("maven:plugin", ns):
            current_group = plugin.findtext("maven:groupId", default="", namespaces=ns).strip()
            current_artifact = plugin.findtext("maven:artifactId", default="", namespaces=ns).strip()
            normalized_group = current_group or "org.apache.maven.plugins"
            if normalized_group == "org.apache.maven.plugins" and current_artifact == "maven-compiler-plugin":
                compiler_plugin = plugin
                break

        if compiler_plugin is None:
            compiler_plugin = ET.SubElement(plugins, f"{tag_prefix}plugin")
            ET.SubElement(compiler_plugin, f"{tag_prefix}groupId").text = "org.apache.maven.plugins"
            ET.SubElement(compiler_plugin, f"{tag_prefix}artifactId").text = "maven-compiler-plugin"

        configuration = compiler_plugin.find("maven:configuration", ns)
        if configuration is None:
            configuration = ET.SubElement(compiler_plugin, f"{tag_prefix}configuration")

        use_dep_mgmt = configuration.find("maven:annotationProcessorPathsUseDepMgmt", ns)
        if use_dep_mgmt is None:
            use_dep_mgmt = ET.SubElement(configuration, f"{tag_prefix}annotationProcessorPathsUseDepMgmt")
        use_dep_mgmt.text = "true"

        processor_paths = configuration.find("maven:annotationProcessorPaths", ns)
        if processor_paths is None:
            processor_paths = ET.SubElement(configuration, f"{tag_prefix}annotationProcessorPaths")
            processor_paths.set("combine.children", "append")

        for path in processor_paths.findall("maven:path", ns):
            current_group = path.findtext("maven:groupId", default="", namespaces=ns).strip()
            current_artifact = path.findtext("maven:artifactId", default="", namespaces=ns).strip()
            if current_group == group_id and current_artifact == artifact_id:
                return False

        path = ET.SubElement(processor_paths, f"{tag_prefix}path")
        ET.SubElement(path, f"{tag_prefix}groupId").text = group_id
        ET.SubElement(path, f"{tag_prefix}artifactId").text = artifact_id
        return True

    def _ensure_gradle_test_dependency(self, content: str, notation: str) -> str:
        if notation in content:
            return content

        dependency_block_pattern = re.compile(r"dependencies\s*\{")
        return dependency_block_pattern.sub(
            lambda match: match.group(0) + f"\n    testImplementation '{notation}'",
            content,
            count=1,
        )

    def _ensure_gradle_dependency(self, content: str, notation: str, configuration: str = "implementation") -> str:
        if notation in content:
            return content

        dependency_block_pattern = re.compile(r"dependencies\s*\{")
        return dependency_block_pattern.sub(
            lambda match: match.group(0) + f"\n    {configuration} '{notation}'",
            content,
            count=1,
        )

    def _target_platform_managed_lookup(self) -> Dict[str, object]:
        managed_dependencies = self.auditor.resolve_target_platform_managed_dependencies()
        return {item.ga: item for item in managed_dependencies}

    def _build_target_platform_override_plan(
        self,
        audit_report: Optional[Dict[str, object]],
    ) -> List[Dict[str, str]]:
        if not isinstance(audit_report, dict):
            return []

        managed_lookup = self._target_platform_managed_lookup()
        if not managed_lookup:
            return []

        plan_by_ga: Dict[str, Dict[str, str]] = {}
        target_line = normalize_major_minor(self.micronaut_version)

        for item in list(audit_report.get("transitive_dependencies", []) or []):
            if not isinstance(item, dict):
                continue
            ga = str(item.get("ga") or "")
            version = str(item.get("version") or "").strip()
            if not ga.startswith("io.micronaut:") or not version:
                continue
            managed = managed_lookup.get(ga)
            managed_version = str(getattr(managed, "version", "") or "").strip() if managed else ""
            if not managed_version or managed_version == version:
                continue
            if normalize_major_minor(version) == target_line:
                continue
            plan_by_ga[ga] = {
                "ga": ga,
                "current_version": version,
                "target_version": managed_version,
                "reason": "transitive_micronaut_version_drift",
            }

        for item in list(audit_report.get("findings", []) or []):
            if not isinstance(item, dict):
                continue
            ga = str(item.get("dependency") or "")
            if not ga.startswith("io.micronaut:"):
                continue
            code = str(item.get("code") or "")
            managed = managed_lookup.get(ga)
            managed_version = str(getattr(managed, "version", "") or "").strip() if managed else ""
            current_version = str(item.get("version") or "").strip()
            if not managed_version:
                continue
            if code == "micronaut_version_drift" and current_version and managed_version != current_version:
                plan_by_ga[ga] = {
                    "ga": ga,
                    "current_version": current_version,
                    "target_version": managed_version,
                    "reason": "direct_or_transitive_micronaut_version_drift",
                }

        return [plan_by_ga[key] for key in sorted(plan_by_ga.keys())]

    def _ensure_maven_dependency_management_override(
        self,
        root: ET.Element,
        ns: Dict[str, str],
        *,
        group_id: str,
        artifact_id: str,
        version: str,
    ) -> bool:
        tag_prefix = f"{{{ns['maven']}}}"
        dep_mgmt = root.find("maven:dependencyManagement", ns)
        if dep_mgmt is None:
            dep_mgmt = ET.SubElement(root, f"{tag_prefix}dependencyManagement")
        dependencies = dep_mgmt.find("maven:dependencies", ns)
        if dependencies is None:
            dependencies = ET.SubElement(dep_mgmt, f"{tag_prefix}dependencies")

        for dep in dependencies.findall("maven:dependency", ns):
            current_group = dep.findtext("maven:groupId", default="", namespaces=ns).strip()
            current_artifact = dep.findtext("maven:artifactId", default="", namespaces=ns).strip()
            if current_group != group_id or current_artifact != artifact_id:
                continue
            version_node = dep.find("maven:version", ns)
            if version_node is None:
                version_node = ET.SubElement(dep, f"{tag_prefix}version")
            previous_version = (version_node.text or "").strip()
            version_node.text = version
            return previous_version != version

        dep = ET.SubElement(dependencies, f"{tag_prefix}dependency")
        ET.SubElement(dep, f"{tag_prefix}groupId").text = group_id
        ET.SubElement(dep, f"{tag_prefix}artifactId").text = artifact_id
        ET.SubElement(dep, f"{tag_prefix}version").text = version
        return True

    def _ensure_maven_dependency_exclusion(
        self,
        dependency_node: ET.Element,
        ns: Dict[str, str],
        *,
        group_id: str,
        artifact_id: str,
    ) -> bool:
        tag_prefix = f"{{{ns['maven']}}}"
        exclusions = dependency_node.find("maven:exclusions", ns)
        if exclusions is None:
            exclusions = ET.SubElement(dependency_node, f"{tag_prefix}exclusions")
        for exclusion in exclusions.findall("maven:exclusion", ns):
            current_group = exclusion.findtext("maven:groupId", default="", namespaces=ns).strip()
            current_artifact = exclusion.findtext("maven:artifactId", default="", namespaces=ns).strip()
            if current_group == group_id and current_artifact == artifact_id:
                return False
        exclusion = ET.SubElement(exclusions, f"{tag_prefix}exclusion")
        ET.SubElement(exclusion, f"{tag_prefix}groupId").text = group_id
        ET.SubElement(exclusion, f"{tag_prefix}artifactId").text = artifact_id
        return True

    def _apply_maven_target_platform_override_plan(
        self,
        root: ET.Element,
        ns: Dict[str, str],
        changes: Dict[str, str],
        override_plan: List[Dict[str, str]],
    ) -> None:
        for item in override_plan:
            ga = str(item.get("ga") or "")
            target_version = str(item.get("target_version") or "").strip()
            current_version = str(item.get("current_version") or "").strip()
            if ":" not in ga or not target_version:
                continue
            group_id, artifact_id = ga.split(":", 1)
            changed = self._ensure_maven_dependency_management_override(
                root,
                ns,
                group_id=group_id,
                artifact_id=artifact_id,
                version=target_version,
            )
            if changed:
                changes[f"transitive-align:{artifact_id}"] = (
                    f"Pinned {ga} from {current_version or 'unresolved'} to target-managed {target_version} "
                    f"via dependencyManagement override"
                )

    def _build_legacy_transitive_exclusion_plan(
        self,
        audit_report: Optional[Dict[str, object]],
    ) -> Dict[str, List[Dict[str, str]]]:
        if not isinstance(audit_report, dict):
            return {}

        plan: Dict[str, Dict[str, Dict[str, str]]] = {}
        for item in list(audit_report.get("transitive_dependencies", []) or []):
            if not isinstance(item, dict):
                continue
            ga = str(item.get("ga") or "")
            if ga not in self.LEGACY_TRANSITIVE_EXCLUSION_ALLOWLIST:
                continue
            parent_chain = list(item.get("parent_chain", []) or [])
            if not parent_chain:
                continue
            root_ga = str(parent_chain[0] or "")
            if not root_ga or root_ga.startswith("org.springframework:") or root_ga.startswith("io.micronaut:"):
                continue
            version = str(item.get("version") or "").strip()
            group_id, artifact_id = ga.split(":", 1)
            plan.setdefault(root_ga, {})
            plan[root_ga][ga] = {
                "ga": ga,
                "group_id": group_id,
                "artifact_id": artifact_id,
                "version": version,
                "reason": "legacy_javax_transitive_carryover",
            }

        return {
            root_ga: [entries[key] for key in sorted(entries.keys())]
            for root_ga, entries in sorted(plan.items())
        }

    def _apply_maven_legacy_transitive_exclusions(
        self,
        dependencies_node: ET.Element,
        ns: Dict[str, str],
        changes: Dict[str, str],
        exclusion_plan: Dict[str, List[Dict[str, str]]],
    ) -> None:
        if not exclusion_plan:
            return
        for dep in dependencies_node.findall("maven:dependency", ns):
            group_id = dep.findtext("maven:groupId", default="", namespaces=ns).strip()
            artifact_id = dep.findtext("maven:artifactId", default="", namespaces=ns).strip()
            if not group_id or not artifact_id:
                continue
            root_ga = f"{group_id}:{artifact_id}"
            for exclusion in exclusion_plan.get(root_ga, []):
                changed = self._ensure_maven_dependency_exclusion(
                    dep,
                    ns,
                    group_id=str(exclusion.get("group_id") or ""),
                    artifact_id=str(exclusion.get("artifact_id") or ""),
                )
                if changed:
                    changes[f"transitive-exclude:{artifact_id}:{exclusion['artifact_id']}"] = (
                        f"Excluded legacy transitive {exclusion['ga']} from {root_ga}"
                    )

    def _ensure_gradle_constraint(self, content: str, notation: str) -> str:
        escaped = re.escape(notation)
        if re.search(rf"constraints\s*\{{[\s\S]*?[\"']{escaped}[\"']", content):
            return content

        dependency_block_pattern = re.compile(r"dependencies\s*\{")

        def replace(match: re.Match) -> str:
            return (
                match.group(0)
                + "\n    constraints {\n"
                + f"        implementation(\"{notation}\")\n"
                + "    }"
            )

        if dependency_block_pattern.search(content):
            return dependency_block_pattern.sub(replace, content, count=1)
        return content

    def _apply_gradle_dependency_exclusion(
        self,
        content: str,
        *,
        root_ga: str,
        excluded_group: str,
        excluded_artifact: str,
    ) -> str:
        escaped_root = re.escape(root_ga)
        kts_pattern = re.compile(
            rf'(?P<indent>^[ \t]*)(?P<config>[A-Za-z_][A-Za-z0-9_]*)\("(?P<coords>{escaped_root}(?::[^"]+)?)"\)(?!\s*\{{)',
            re.MULTILINE,
        )
        groovy_pattern = re.compile(
            rf"(?P<indent>^[ \t]*)(?P<config>[A-Za-z_][A-Za-z0-9_]*)\s+'(?P<coords>{escaped_root}(?::[^']+)?)'(?!\s*\{{)",
            re.MULTILINE,
        )

        def replace_kts(match: re.Match) -> str:
            indent = match.group("indent")
            config = match.group("config")
            coords = match.group("coords")
            return (
                f'{indent}{config}("{coords}") {{\n'
                f'{indent}    exclude(group = "{excluded_group}", module = "{excluded_artifact}")\n'
                f"{indent}}}"
            )

        updated = kts_pattern.sub(replace_kts, content, count=1)
        if updated != content:
            return updated

        def replace_groovy(match: re.Match) -> str:
            indent = match.group("indent")
            config = match.group("config")
            coords = match.group("coords")
            return (
                f"{indent}{config} '{coords}' {{\n"
                f"{indent}    exclude group: '{excluded_group}', module: '{excluded_artifact}'\n"
                f"{indent}}}"
            )

        return groovy_pattern.sub(replace_groovy, content, count=1)

    def _apply_gradle_target_platform_override_plan(
        self,
        content: str,
        changes: Dict[str, str],
        override_plan: List[Dict[str, str]],
    ) -> str:
        updated_content = content
        for item in override_plan:
            ga = str(item.get("ga") or "")
            target_version = str(item.get("target_version") or "").strip()
            current_version = str(item.get("current_version") or "").strip()
            if ":" not in ga or not target_version:
                continue
            notation = f"{ga}:{target_version}"
            next_content = self._ensure_gradle_constraint(updated_content, notation)
            if next_content != updated_content:
                artifact_id = ga.split(":", 1)[1]
                changes[f"transitive-align:{artifact_id}"] = (
                    f"Pinned {ga} from {current_version or 'unresolved'} to target-managed {target_version} "
                    f"via Gradle dependency constraint"
                )
            updated_content = next_content
        return updated_content

    def _apply_gradle_legacy_transitive_exclusions(
        self,
        content: str,
        changes: Dict[str, str],
        exclusion_plan: Dict[str, List[Dict[str, str]]],
    ) -> str:
        updated_content = content
        for root_ga, exclusions in exclusion_plan.items():
            for exclusion in exclusions:
                next_content = self._apply_gradle_dependency_exclusion(
                    updated_content,
                    root_ga=root_ga,
                    excluded_group=str(exclusion.get("group_id") or ""),
                    excluded_artifact=str(exclusion.get("artifact_id") or ""),
                )
                if next_content != updated_content:
                    root_artifact = root_ga.split(":", 1)[1]
                    changes[f"transitive-exclude:{root_artifact}:{exclusion['artifact_id']}"] = (
                        f"Excluded legacy transitive {exclusion['ga']} from {root_ga}"
                    )
                updated_content = next_content
        return updated_content

    def _align_maven_dependencies_to_target_platform(
        self,
        dependencies_node: ET.Element,
        ns: Dict[str, str],
        changes: Dict[str, str],
    ) -> None:
        managed_lookup = self._target_platform_managed_lookup()
        if not managed_lookup:
            return

        for dep in dependencies_node.findall("maven:dependency", ns):
            group_id = dep.findtext("maven:groupId", default="", namespaces=ns).strip()
            artifact_id = dep.findtext("maven:artifactId", default="", namespaces=ns).strip()
            if not group_id or not artifact_id:
                continue
            ga = f"{group_id}:{artifact_id}"
            managed = managed_lookup.get(ga)
            if not managed:
                continue
            version = dep.find("maven:version", ns)
            if version is None:
                continue
            dep.remove(version)
            managed_version = str(getattr(managed, "version", "") or "").strip()
            changes[f"platform-align:{artifact_id}"] = (
                f"Aligned {ga} to Micronaut target platform {self.micronaut_version}"
                f"{f' (managed version {managed_version})' if managed_version else ''}"
            )

    def _align_gradle_dependencies_to_target_platform(
        self,
        content: str,
        changes: Dict[str, str],
    ) -> str:
        managed_lookup = self._target_platform_managed_lookup()
        if not managed_lookup:
            return content

        pattern = re.compile(
            r'(?P<prefix>\b(?:implementation|api|runtimeOnly|testImplementation|annotationProcessor)\s*\(?\s*[\'"])(?P<ga>[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+):(?P<version>[^\'")]+)(?P<suffix>[\'"]\)?)'
        )

        def replace(match: re.Match) -> str:
            ga = match.group("ga")
            if ga not in managed_lookup:
                return match.group(0)
            artifact_id = ga.split(":", 1)[1]
            managed_version = str(getattr(managed_lookup.get(ga), "version", "") or "").strip()
            changes[f"platform-align:{artifact_id}"] = (
                f"Aligned {ga} to Micronaut target platform {self.micronaut_version}"
                f"{f' (managed version {managed_version})' if managed_version else ''}"
            )
            return f"{match.group('prefix')}{ga}{match.group('suffix')}"

        return pattern.sub(replace, content)

    def _replace_gradle_string_dependency(
        self,
        content: str,
        source_ga: str,
        target_ga: str,
    ) -> str:
        quoted_source = re.escape(source_ga)
        return re.sub(
            rf'([\'"])({quoted_source})(:[^\'"]+)?([\'"])',
            lambda match: f"{match.group(1)}{target_ga}{match.group(4)}",
            content,
        )

    def _replace_gradle_map_dependency(
        self,
        content: str,
        source_group: str,
        source_artifact: str,
        target_group: str,
        target_artifact: str,
    ) -> str:
        pattern = re.compile(
            rf"""group\s*:\s*(['"]){re.escape(source_group)}\1\s*,\s*name\s*:\s*(['"]){re.escape(source_artifact)}\2(?:\s*,\s*version\s*:\s*['"][^'"]+['"])?"""
        )
        return pattern.sub(
            lambda match: f"group: {match.group(1)}{target_group}{match.group(1)}, name: {match.group(2)}{target_artifact}{match.group(2)}",
            content,
        )

    def _apply_gradle_catalog_mappings(self, content: str, build_file_path: str, changes: Dict[str, str]) -> str:
        direct_dependencies = self.auditor.extract_direct_gradle_dependencies(build_file_path)
        for dependency in direct_dependencies:
            entry = self.auditor.get_catalog_entry(dependency)
            if not entry or not entry.automated_migration_supported or ":" not in entry.replacement:
                continue

            source_ga = dependency.ga
            target_ga = entry.replacement
            target_group, target_artifact = target_ga.split(":", 1)
            content = self._replace_gradle_string_dependency(content, source_ga, target_ga)
            content = self._replace_gradle_map_dependency(
                content,
                dependency.group_id,
                dependency.artifact_id,
                target_group,
                target_artifact,
            )
            changes[dependency.artifact_id] = (
                f"{target_ga} (catalog auto-upgrade for Micronaut {self.micronaut_version})"
            )
        return content

    def _remove_gradle_dependency(self, content: str, source_ga: str) -> str:
        quoted_source = re.escape(source_ga)
        content = re.sub(
            rf'(?m)^[ \t]*(?:implementation|api|runtimeOnly|testImplementation|developmentOnly|compileOnly|annotationProcessor)\s*\(?\s*[\'"]{quoted_source}(?::[^\'"]+)?[\'"]\)?\s*\n?',
            "",
            content,
        )
        return content

    def _replace_gradle_dependency_version(
        self,
        content: str,
        source_ga: str,
        version: str,
    ) -> str:
        if not version:
            return content

        quoted_source = re.escape(source_ga)
        content = re.sub(
            rf'(?m)^([ \t]*(?:implementation|api|runtimeOnly|testImplementation|developmentOnly|compileOnly|annotationProcessor)\s*\(?\s*[\'"]){quoted_source}([\'"]\)?\s*)$',
            rf"\1{source_ga}:{version}\2",
            content,
        )
        group_id, artifact_id = source_ga.split(":", 1)
        content = re.sub(
            rf"""group\s*:\s*(['"]){re.escape(group_id)}\1\s*,\s*name\s*:\s*(['"]){re.escape(artifact_id)}\2(?!\s*,\s*version\s*:)""",
            lambda match: (
                f"group: {match.group(1)}{group_id}{match.group(1)}, "
                f"name: {match.group(2)}{artifact_id}{match.group(2)}, version: '{version}'"
            ),
            content,
        )
        return content

    def migrate_project_config(
        self,
        source_path: str,
        output_path: str,
        audit_report: Optional[Dict[str, object]] = None,
        project_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Detects the build tool and applies the corresponding migration logic.
        """
        if source_path.endswith('pom.xml'):
            return self.migrate_maven_pom(
                source_path,
                output_path,
                audit_report=audit_report,
                project_path=project_path,
            )
        elif source_path.endswith('.gradle') or source_path.endswith('.gradle.kts'):
            return self.migrate_gradle(
                source_path,
                output_path,
                audit_report=audit_report,
                project_path=project_path,
            )
        return {}

    def audit_project_dependencies(
        self,
        source_path: str,
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
        if source_path.endswith("pom.xml"):
            return write_dependency_audit_report(
                build_file_path=source_path,
                spring_version=self.spring_version,
                micronaut_version=self.micronaut_version,
                project_path=project_path,
                dependency_tree_text=dependency_tree_text,
                dependency_tree_path=dependency_tree_path,
                runtime_dependency_tree_text=runtime_dependency_tree_text,
                runtime_dependency_tree_path=runtime_dependency_tree_path,
                effective_pom_text=effective_pom_text,
                effective_pom_path=effective_pom_path,
                report_path=report_path,
                fail_on=fail_on,
            )
        if source_path.endswith(".gradle") or source_path.endswith(".gradle.kts"):
            return write_dependency_audit_report(
                build_file_path=source_path,
                spring_version=self.spring_version,
                micronaut_version=self.micronaut_version,
                project_path=project_path,
                dependency_tree_text=dependency_tree_text,
                dependency_tree_path=dependency_tree_path,
                runtime_dependency_tree_text=runtime_dependency_tree_text,
                runtime_dependency_tree_path=runtime_dependency_tree_path,
                report_path=report_path,
                fail_on=fail_on,
            )

        return {
            "ok": True,
            "status": "unavailable",
            "notes": ["Dependency audit currently supports Maven and Gradle build files only."],
            "direct_dependency_count": 0,
            "transitive_dependency_count": 0,
            "severity_counts": {"blocking": 0, "review": 0, "info": 0},
            "direct_dependencies": [],
            "transitive_dependencies": [],
            "findings": [],
            "spring_version": self.spring_version,
            "micronaut_version": self.micronaut_version,
        }

    def migrate_maven_pom(
        self,
        pom_path: str,
        output_path: str,
        audit_report: Optional[Dict[str, object]] = None,
        project_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Comprehensive migration of Maven pom.xml.
        Handles parent, Bill of Materials (BOM), and specific dependency mappings.
        """
        changes = {}
        # Namespace handling for Maven POM
        ns = {"maven": "http://maven.apache.org/POM/4.0.0"}
        ET.register_namespace('', ns["maven"])
        
        try:
            tree = ET.parse(pom_path)
            root = tree.getroot()
            override_plan = self._build_target_platform_override_plan(audit_report)
            exclusion_plan = self._build_legacy_transitive_exclusion_plan(audit_report)
            
            # Step 1: Update Parent POM (Spring Boot -> Micronaut)
            parent = root.find("maven:parent", ns)
            if parent is not None:
                group_id = parent.find("maven:groupId", ns)
                artifact_id = parent.find("maven:artifactId", ns)
                version = parent.find("maven:version", ns)
                
                if group_id is not None and "spring" in group_id.text.lower():
                    group_id.text = "io.micronaut.platform"
                    artifact_id.text = "micronaut-parent"
                    version.text = self.micronaut_version
                    changes["parent"] = "Updated to micronaut-parent"

            # Step 2: Mapping individual dependencies
            dependencies_node = root.find("maven:dependencies", ns)
            if dependencies_node is not None:
                direct_coordinates = self.auditor.extract_direct_maven_dependencies(pom_path)
                had_spring_test_starter = any(
                    item.ga == "org.springframework.boot:spring-boot-starter-test"
                    for item in direct_coordinates
                )
                had_spring_data_jpa_starter = any(
                    item.ga == "org.springframework.boot:spring-boot-starter-data-jpa"
                    for item in direct_coordinates
                )
                had_spring_web_starter = any(
                    item.ga == "org.springframework.boot:spring-boot-starter-web"
                    for item in direct_coordinates
                )
                had_spring_validation_starter = any(
                    item.ga == "org.springframework.boot:spring-boot-starter-validation"
                    for item in direct_coordinates
                )
                uses_spring_sql_init = self._project_uses_spring_sql_init(project_path)
                # First pass: find and remove spring-specific management nodes
                dep_mgmt = root.find("maven:dependencyManagement", ns)
                if dep_mgmt is not None:
                    # Search for spring/cloud BOMs and remove
                    for dep in dep_mgmt.findall(".//maven:dependency", ns):
                        art = dep.find("maven:artifactId", ns)
                        if art is not None and ("spring-boot-dependencies" in art.text or "spring-cloud-dependencies" in art.text):
                            changes[art.text] = "Removed from Management"
                            # Need to find the parent of this dependency node and remove it
                            # Usually it's <dependencies> inside <dependencyManagement>
                            parent_node = root.find(".//maven:dependencyManagement/maven:dependencies", ns)
                            if parent_node is not None:
                                parent_node.remove(dep)
                
                # Step 3: Individual Dependency Replacement (RAG-based)
                # We'll collect nodes to remove to avoid concurrent modification issues
                to_remove = []
                
                for dep in dependencies_node.findall("maven:dependency", ns):
                    group = dep.find("maven:groupId", ns)
                    artifact = dep.find("maven:artifactId", ns)
                    version = dep.find("maven:version", ns)
                    
                    if artifact is not None:
                        artifact_id = artifact.text
                        group_id = group.text if group is not None else ""
                        current_dependency = next(
                            (
                                item
                                for item in direct_coordinates
                                if item.group_id == group_id and item.artifact_id == artifact_id
                            ),
                            None,
                        )

                        if current_dependency is not None:
                            explicit_replacement = self.maven_dependency_mappings.get(current_dependency.ga)
                            if explicit_replacement:
                                replacement_group, replacement_artifact = explicit_replacement.split(":", 1)
                                if group is not None:
                                    group.text = replacement_group
                                artifact.text = replacement_artifact
                                if version is not None:
                                    dep.remove(version)
                                changes[artifact_id] = explicit_replacement
                                continue
                            if current_dependency.ga == "org.springframework.boot:spring-boot-devtools":
                                to_remove.append(dep)
                                changes[artifact_id] = "Removed"
                                continue

                            catalog_entry = self.auditor.get_catalog_entry(current_dependency)
                            if catalog_entry and catalog_entry.automated_migration_supported and ":" in catalog_entry.replacement:
                                replacement_group, replacement_artifact = catalog_entry.replacement.split(":", 1)
                                if group is not None:
                                    group.text = replacement_group
                                artifact.text = replacement_artifact
                                if catalog_entry.version_management == "platform_managed":
                                    if version is not None:
                                        dep.remove(version)
                                else:
                                    if version is None:
                                        version = ET.SubElement(dep, "version")
                                    version.text = catalog_entry.replacement_version
                                changes[artifact_id] = (
                                    f"{catalog_entry.replacement} "
                                    f"(catalog auto-upgrade for Micronaut {self.micronaut_version})"
                                )
                                continue

                        if group_id.startswith("io.micronaut") and version is not None:
                            dep.remove(version)
                            changes[artifact_id] = (
                                f"Aligned to Micronaut platform-managed version for {self.micronaut_version}"
                            )
                            continue
                        
                        # Search for Micronaut equivalent
                        rules = self.kb.search_dependency(
                            artifact_id,
                            spring_version=self.spring_version,
                            micronaut_version=self.micronaut_version,
                        )
                        
                        if rules:
                            rule = rules[0]
                            if ":" in rule.micronaut_pattern:
                                m_group, m_art = rule.micronaut_pattern.split(":")
                                if group is not None: group.text = m_group
                                artifact.text = m_art
                                if version is not None: dep.remove(version)
                                changes[artifact_id] = rule.micronaut_pattern
                                continue
                            elif rule.micronaut_pattern == "REMOVE":
                                to_remove.append(dep)
                                changes[artifact_id] = "Removed"
                                continue
                        
                        # EXPERT FALLBACKS: Handle known Spring/Managed orphans
                        # 1. Broad Spring Detection (any group containing 'spring')
                        if "spring" in group_id.lower() or "spring" in artifact_id.lower():
                             if "web" in artifact_id:
                                 if group is not None: group.text = "io.micronaut"
                                 artifact.text = "micronaut-http-server-netty"
                                 if version is not None: dep.remove(version)
                                 changes[artifact_id] = "io.micronaut:micronaut-http-server-netty"
                             elif "test" in artifact_id:
                                 if group is not None: group.text = "io.micronaut.test"
                                 artifact.text = "micronaut-test-junit5"
                                 if version is not None: dep.remove(version)
                                 # Ensure scope is test
                                 scope = dep.find("maven:scope", ns)
                                 if scope is None:
                                     scope = ET.SubElement(dep, "scope")
                                     scope.text = "test"
                                 changes[artifact_id] = "io.micronaut.test:micronaut-test-junit5"
                             elif "cloud-gateway" in artifact_id:
                                 if group is not None: group.text = "io.micronaut"
                                 artifact.text = "micronaut-http-client"
                                 # Standard client for basic gateway proxy logic
                                 if version is not None: dep.remove(version)
                                 changes[artifact_id] = "io.micronaut:micronaut-http-client"
                             else:
                                 # Kill any other spring artifacts that will fail the build
                                 to_remove.append(dep)
                                 changes[artifact_id] = f"Removed {artifact_id} (Orphaned Spring dependency)"
                             continue

                        # 2. Handle non-Spring orphans (dependencies with no version that Micronaut doesn't manage)
                        if version is None:
                            # Known orphans from Spring BOM
                            if "jedis" in artifact_id.lower():
                                if group is not None: group.text = "io.micronaut.redis"
                                artifact.text = "micronaut-redis-lettuce"
                                # Micronaut-Redis-Lettuce needs version if not managed by Micronaut-BOM
                                version_node = ET.SubElement(dep, "version")
                                version_node.text = "6.4.1" # Stable version for Micronaut 4
                                changes[artifact_id] = "io.micronaut.redis:micronaut-redis-lettuce (Migrated from Jedis)"
                            elif "ehcache" in artifact_id.lower():
                                if group is not None: group.text = "io.micronaut.cache"
                                artifact.text = "micronaut-cache-ehcache"
                                version_node = ET.SubElement(dep, "version")
                                version_node.text = "4.0.0" 
                                changes[artifact_id] = "io.micronaut.cache:micronaut-cache-ehcache"
                            elif "h2" == artifact_id.lower():
                                # Micronaut manages H2 version
                                pass
                            elif "lombok" == artifact_id.lower():
                                # Micronaut manages Lombok version
                                pass
                            elif current_dependency is not None and current_dependency.ga in self.source_managed_runtime_preserve_allowlist:
                                source_managed_version = self._resolve_source_managed_version(
                                    dependency_ga=current_dependency.ga,
                                    source_path=pom_path,
                                    build_tool="maven",
                                )
                                if source_managed_version:
                                    version_node = ET.SubElement(dep, "version")
                                    version_node.text = source_managed_version
                                    changes[artifact_id] = (
                                        f"Preserved {current_dependency.ga} with source-managed version {source_managed_version}"
                                    )
                                else:
                                    to_remove.append(dep)
                                    changes[artifact_id] = (
                                        f"Removed {artifact_id} (No version specified and source-managed version could not be resolved)"
                                    )
                            else:
                                # For unknown orphans, we must either remove them or they will break the build
                                # Better to remove and let the user add them back with a version if they really need them
                                to_remove.append(dep)
                                changes[artifact_id] = f"Removed {artifact_id} (No version specified and not managed by Micronaut)"

                # Finalize removals
                for dep in to_remove:
                    try:
                        dependencies_node.remove(dep)
                    except ValueError:
                        pass # Already removed

                if dep_mgmt is not None:
                    dep_mgmt_dependencies = dep_mgmt.find("maven:dependencies", ns)
                    if dep_mgmt_dependencies is not None and not dep_mgmt_dependencies.findall("maven:dependency", ns):
                        root.remove(dep_mgmt)

                if had_spring_test_starter:
                    for group_id, artifact_id, version in self.supplemental_test_dependencies:
                        if self._ensure_maven_dependency(
                            dependencies_node,
                            ns,
                            group_id=group_id,
                            artifact_id=artifact_id,
                            scope="test",
                            version=version,
                        ):
                            changes[f"test-support:{artifact_id}"] = (
                                f"Added {group_id}:{artifact_id} for migrated test compilation"
                            )

                if had_spring_web_starter:
                    for group_id, artifact_id, version in self.supplemental_runtime_dependencies["web"]:
                        if self._ensure_maven_dependency(
                            dependencies_node,
                            ns,
                            group_id=group_id,
                            artifact_id=artifact_id,
                            version=version,
                        ):
                            changes[f"runtime-support:{artifact_id}"] = (
                                f"Added {group_id}:{artifact_id} for Micronaut JSON mapper/runtime HTTP support"
                            )

                if had_spring_data_jpa_starter:
                    for group_id, artifact_id, version in self.supplemental_runtime_dependencies["data"]:
                        if self._ensure_maven_dependency(
                            dependencies_node,
                            ns,
                            group_id=group_id,
                            artifact_id=artifact_id,
                            version=version,
                        ):
                            changes[f"runtime-support:{artifact_id}"] = (
                                f"Added {group_id}:{artifact_id} for Micronaut JDBC datasource/transaction support"
                            )

                if had_spring_validation_starter:
                    for group_id, artifact_id, version in self.supplemental_runtime_dependencies["validation"]:
                        if self._ensure_maven_dependency(
                            dependencies_node,
                            ns,
                            group_id=group_id,
                            artifact_id=artifact_id,
                            version=version,
                        ):
                            changes[f"runtime-support:{artifact_id}"] = (
                                f"Added {group_id}:{artifact_id} for Jakarta Validation provider support"
                            )

                if uses_spring_sql_init:
                    if self._ensure_maven_dependency(
                        dependencies_node,
                        ns,
                        group_id="io.micronaut.flyway",
                        artifact_id="micronaut-flyway",
                    ):
                        changes["runtime-support:micronaut-flyway"] = (
                            "Added io.micronaut.flyway:micronaut-flyway for Spring SQL init compatibility via Flyway"
                        )

                if had_spring_data_jpa_starter:
                    if self._ensure_maven_compiler_annotation_processor(
                        root,
                        ns,
                        group_id="io.micronaut.data",
                        artifact_id="micronaut-data-processor",
                    ):
                        changes["annotation-processor:micronaut-data-processor"] = (
                            "Added Micronaut Data annotation processor for repository/query metadata generation"
                        )

                self._align_maven_dependencies_to_target_platform(
                    dependencies_node,
                    ns,
                    changes,
                )
                self._apply_maven_legacy_transitive_exclusions(
                    dependencies_node,
                    ns,
                    changes,
                    exclusion_plan,
                )

            if override_plan:
                self._apply_maven_target_platform_override_plan(
                    root,
                    ns,
                    changes,
                    override_plan,
                )

            self._cleanup_maven_metadata(root, ns, changes)
            
            # Save the updated POM
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            
        except Exception as e:
            # Errors logged without emoticons
            print(f"Error during Maven migration: {e}")
            
        return changes

    def migrate_gradle(
        self,
        gradle_path: str,
        output_path: str,
        audit_report: Optional[Dict[str, object]] = None,
        project_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Migrates Gradle build scripts using regex-based pattern replacement.
        """
        changes = {}
        try:
            with open(gradle_path, 'r', encoding='utf-8') as f:
                content = f.read()
            original_content = content
            exclusion_plan = self._build_legacy_transitive_exclusion_plan(audit_report)
            uses_spring_sql_init = self._project_uses_spring_sql_init(project_path)

            is_kts = gradle_path.endswith(".kts")
            gradle_plugin_version = self._resolve_micronaut_gradle_plugin_version()
            plugin_line = (
                f'id("io.micronaut.application") version "{gradle_plugin_version}"'
                if is_kts
                else f'id "io.micronaut.application" version "{gradle_plugin_version}"'
            )
            content, plugin_replacements = re.subn(
                r'^\s*id\s*\(?["\']org\.springframework\.boot["\']\)?[^\n]*$',
                plugin_line,
                content,
                flags=re.MULTILINE,
            )
            if plugin_replacements:
                changes["plugin"] = "Spring Boot -> Micronaut Application"
                if gradle_plugin_version != self.micronaut_version:
                    changes["micronaut.gradle.plugin"] = (
                        f"Resolved Micronaut Gradle plugin {gradle_plugin_version} for target Micronaut {self.micronaut_version}"
                    )

            content = re.sub(
                r'^\s*id\s*\(?["\']io\.spring\.dependency-management["\']\)?[^\n]*\n?',
                "",
                content,
                flags=re.MULTILINE,
            )
            content, java_changes = self._normalize_gradle_java_compatibility(content, is_kts=is_kts)
            changes.update(java_changes)

            if "micronaut {" not in content:
                content += f'\nmicronaut {{\n    version = "{self.micronaut_version}"\n}}\n'
                changes["micronaut.version"] = self.micronaut_version

            if "micronaut-platform" not in content:
                dependency_block_pattern = r"dependencies\s*\{"
                bom_line = (
                    f'implementation(platform("io.micronaut.platform:micronaut-platform:{self.micronaut_version}"))'
                    if is_kts
                    else f'implementation platform("io.micronaut.platform:micronaut-platform:{self.micronaut_version}")'
                )
                content, bom_insertions = re.subn(
                    dependency_block_pattern,
                    lambda match: match.group(0) + f"\n    {bom_line}",
                    content,
                    count=1,
                )
                if bom_insertions:
                    changes["micronaut.bom"] = f"Added Micronaut BOM {self.micronaut_version}"

            for spring_dep, micronaut_dep in self.gradle_dependency_mappings.items():
                if spring_dep in content:
                    content = content.replace(spring_dep, micronaut_dep)
                    changes[spring_dep.split(":")[1]] = micronaut_dep

            if "org.springframework.boot:spring-boot-starter-test" in original_content:
                for notation in (
                    "io.micronaut:micronaut-http-client",
                    "org.assertj:assertj-core:3.26.3",
                    "org.hamcrest:hamcrest:2.2",
                    "org.junit.jupiter:junit-jupiter-engine:5.10.2",
                    "org.junit.platform:junit-platform-launcher:1.10.2",
                    "org.mockito:mockito-core:5.12.0",
                    "org.mockito:mockito-junit-jupiter:5.12.0",
                ):
                    configuration = (
                        "testRuntimeOnly"
                        if notation.startswith("org.junit.platform:junit-platform-launcher:")
                        or notation.startswith("org.junit.jupiter:junit-jupiter-engine:")
                        else "testImplementation"
                    )
                    updated = self._ensure_gradle_dependency(content, notation, configuration)
                    if updated != content:
                        changes[f"test-support:{notation.split(':')[1]}"] = (
                            f"Added {notation} for migrated test compilation"
                        )
                    content = updated

            if "org.springframework.boot:spring-boot-starter-web" in original_content:
                updated = self._ensure_gradle_dependency(
                    content,
                    "io.micronaut:micronaut-jackson-databind",
                    "implementation",
                )
                if updated != content:
                    changes["runtime-support:micronaut-jackson-databind"] = (
                        "Added io.micronaut:micronaut-jackson-databind for Micronaut JSON mapper/runtime HTTP support"
                    )
                content = updated

            if "org.springframework.boot:spring-boot-starter-data-jpa" in original_content:
                updated = self._ensure_gradle_dependency(
                    content,
                    "io.micronaut.sql:micronaut-jdbc-hikari",
                    "implementation",
                )
                if updated != content:
                    changes["runtime-support:micronaut-jdbc-hikari"] = (
                        "Added io.micronaut.sql:micronaut-jdbc-hikari for Micronaut JDBC datasource/transaction support"
                    )
                content = updated

            if "org.springframework.boot:spring-boot-starter-validation" in original_content:
                updated = self._ensure_gradle_dependency(
                    content,
                    "io.micronaut.beanvalidation:micronaut-hibernate-validator",
                    "implementation",
                )
                if updated != content:
                    changes["runtime-support:micronaut-hibernate-validator"] = (
                        "Added io.micronaut.beanvalidation:micronaut-hibernate-validator for Jakarta Validation provider support"
                    )
                content = updated

            if uses_spring_sql_init:
                updated = self._ensure_gradle_dependency(
                    content,
                    "io.micronaut.flyway:micronaut-flyway",
                    "implementation",
                )
                if updated != content:
                    changes["runtime-support:micronaut-flyway"] = (
                        "Added io.micronaut.flyway:micronaut-flyway for Spring SQL init compatibility via Flyway"
                    )
                content = updated

            if "org.springframework.boot:spring-boot-starter-data-jpa" in original_content:
                for notation in (
                    "io.micronaut:micronaut-inject-java",
                    "io.micronaut.data:micronaut-data-processor",
                ):
                    updated = self._ensure_gradle_dependency(
                        content,
                        notation,
                        "annotationProcessor",
                    )
                    if updated != content:
                        changes[f"annotation-processor:{notation.split(':')[1]}"] = (
                            f"Added {notation} for Micronaut compile-time repository generation"
                        )
                    content = updated

            for dependency in self.auditor.extract_direct_gradle_dependencies(gradle_path):
                if dependency.version or dependency.ga not in self.source_managed_runtime_preserve_allowlist:
                    continue
                source_managed_version = self._resolve_source_managed_version(
                    dependency_ga=dependency.ga,
                    source_path=gradle_path,
                    build_tool="gradle",
                )
                if not source_managed_version:
                    continue
                updated = self._replace_gradle_dependency_version(
                    content,
                    dependency.ga,
                    source_managed_version,
                )
                if updated != content:
                    changes[dependency.artifact_id] = (
                        f"Preserved {dependency.ga} with source-managed version {source_managed_version}"
                    )
                content = updated

            if "org.springframework.boot:spring-boot-devtools" in content:
                content = self._remove_gradle_dependency(content, "org.springframework.boot:spring-boot-devtools")
                changes["spring-boot-devtools"] = "Removed"

            if self._should_drop_direct_jcache_api_dependency(project_path, original_content):
                updated = self._remove_gradle_dependency(content, "javax.cache:cache-api")
                if updated != content:
                    changes["javax.cache:cache-api"] = (
                        "Removed direct javax.cache API dependency after migrating cache configuration to Micronaut-native cache manager usage"
                    )
                content = updated

            content = self._apply_gradle_catalog_mappings(content, gradle_path, changes)
            content = self._align_gradle_dependencies_to_target_platform(content, changes)

            # Drop explicit versions for common Micronaut dependencies when the BOM is present.
            content = re.sub(
                r'((?:implementation|api|runtimeOnly|testImplementation)\s*\(?\s*["\']io\.micronaut(?:\.[^:"\']+)?:[^:"\']+):[^"\']+(["\']\)?)',
                r"\1\2",
                content,
            )
            content = self._apply_gradle_legacy_transitive_exclusions(
                content,
                changes,
                exclusion_plan,
            )
            content = self._apply_gradle_target_platform_override_plan(
                content,
                changes,
                self._build_target_platform_override_plan(audit_report),
            )
            if not is_kts:
                content, removed_java_apply = self._remove_redundant_gradle_java_apply(content)
                if removed_java_apply:
                    changes["gradle.java.plugin"] = "Removed redundant apply plugin: 'java'"
                content = self._normalize_gradle_groovy_formatting(content)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            if self._write_gradle_properties(gradle_path, output_path):
                changes["gradle.properties"] = f"Set micronautVersion={self.micronaut_version}"
        except Exception as e:
            print(f"Error during Gradle migration: {e}")
            
        return changes

    def _resolve_source_managed_version(
        self,
        *,
        dependency_ga: str,
        source_path: str,
        build_tool: str,
    ) -> str:
        if build_tool == "maven":
            build_context = self.auditor.extract_maven_build_context(source_path)
            for coord in build_context.maven_managed_dependency_coords or ():
                if coord.startswith(f"{dependency_ga}:"):
                    return coord.split(":", 2)[2]
            if build_context.uses_spring_boot_maven_parent() and build_context.maven_parent_version:
                return self._resolve_version_from_spring_boot_bom(
                    spring_boot_version=build_context.maven_parent_version,
                    dependency_ga=dependency_ga,
                )
            return ""

        if build_tool == "gradle":
            spring_boot_version = self._extract_gradle_spring_boot_plugin_version(source_path)
            if spring_boot_version:
                return self._resolve_version_from_spring_boot_bom(
                    spring_boot_version=spring_boot_version,
                    dependency_ga=dependency_ga,
                )
        return ""

    def _extract_gradle_spring_boot_plugin_version(self, build_file_path: str) -> str:
        try:
            with open(build_file_path, "r", encoding="utf-8") as handle:
                content = handle.read()
        except OSError:
            return ""

        patterns = (
            r"""id\s+['"]org\.springframework\.boot['"]\s+version\s+['"]([^'"]+)['"]""",
            r"""id\(\s*['"]org\.springframework\.boot['"]\s*\)\s*version\s*['"]([^'"]+)['"]""",
        )
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1).strip()
        return ""

    def _resolve_version_from_spring_boot_bom(
        self,
        *,
        spring_boot_version: str,
        dependency_ga: str,
    ) -> str:
        descriptor = self.auditor.maven_central.fetch_pom_text(
            "org.springframework.boot:spring-boot-dependencies",
            spring_boot_version,
        )
        pom_text = str(descriptor.get("pom_text") or "")
        if not pom_text:
            return ""
        return self._extract_managed_version_from_bom_text(pom_text, dependency_ga)

    def _extract_managed_version_from_bom_text(self, pom_text: str, dependency_ga: str) -> str:
        if not pom_text.strip() or ":" not in dependency_ga:
            return ""
        try:
            root = ET.fromstring(pom_text)
        except ET.ParseError:
            return ""

        ns = {"maven": "http://maven.apache.org/POM/4.0.0"}
        properties = self._extract_maven_properties(root, ns)
        target_group, target_artifact = dependency_ga.split(":", 1)
        managed_node = root.find("maven:dependencyManagement/maven:dependencies", ns)
        if managed_node is None:
            return ""

        for dep in managed_node.findall("maven:dependency", ns):
            group = (dep.findtext("maven:groupId", default="", namespaces=ns) or "").strip()
            artifact = (dep.findtext("maven:artifactId", default="", namespaces=ns) or "").strip()
            version = (dep.findtext("maven:version", default="", namespaces=ns) or "").strip()
            if group == target_group and artifact == target_artifact:
                return self._resolve_maven_property_value(version, properties).strip()
        return ""

    def _extract_maven_properties(self, root: ET.Element, ns: Dict[str, str]) -> Dict[str, str]:
        properties: Dict[str, str] = {}
        properties_node = root.find("maven:properties", ns)
        if properties_node is None:
            return properties

        for child in list(properties_node):
            tag_name = child.tag.split("}", 1)[-1]
            properties[tag_name] = (child.text or "").strip()

        for _ in range(8):
            changed = False
            for key, value in list(properties.items()):
                resolved = self._resolve_maven_property_value(value, properties).strip()
                if resolved != value:
                    changed = True
                properties[key] = resolved
            if not changed:
                break
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

    def _project_uses_spring_sql_init(self, project_path: Optional[str]) -> bool:
        if not project_path or not os.path.isdir(project_path):
            return False

        resources_root = os.path.join(project_path, "src", "main", "resources")
        if not os.path.isdir(resources_root):
            return False

        for root_dir, _, files in os.walk(resources_root):
            for filename in files:
                if not filename.startswith("application"):
                    continue
                if not filename.endswith((".properties", ".yml", ".yaml")):
                    continue
                source_path = os.path.join(root_dir, filename)
                try:
                    with open(source_path, "r", encoding="utf-8") as handle:
                        content = handle.read()
                except OSError:
                    continue
                if "spring.sql.init.mode" in content:
                    return True
                if "spring.sql.init.schema-locations" in content:
                    return True
                if "spring.sql.init.data-locations" in content:
                    return True
                if re.search(r"(?m)^\s*sql:\s*$", content) and "init:" in content:
                    return True
        return False
