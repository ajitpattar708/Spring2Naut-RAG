import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class VerificationFinding:
    severity: str
    file: str
    rule: str
    message: str
    evidence: Optional[str] = None


@dataclass
class VerificationReport:
    source_root: str
    target_root: str
    compared_files: int
    matched_files: int
    missing_target_files: List[str] = field(default_factory=list)
    findings: List[VerificationFinding] = field(default_factory=list)
    severity_counts: Dict[str, int] = field(default_factory=dict)
    trusted_ready: bool = False

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload["findings"] = [asdict(item) for item in self.findings]
        return payload


class VerificationAgent:
    """
    Performs file-by-file verification of migrated output.
    This is a structural confidence pass, not a proof of runtime correctness.
    """

    JAVA_PATTERNS = (
        ("blocking", "markdown_fence", re.compile(r"```"), "Generated Java file still contains markdown code fences."),
        ("blocking", "invalid_manual_placeholder", re.compile(r"^\s*(?:import\s+manual\b|@manual\b)", re.MULTILINE), "Generated Java file still contains an invalid manual-review placeholder token."),
        ("blocking", "spring_import", re.compile(r"^\s*import\s+org\.springframework\.", re.MULTILINE), "Spring import still present after migration."),
        ("blocking", "spring_test_slice", re.compile(r"@WebMvcTest\b|@DataJpaTest\b|@AutoConfigureTestDatabase\b"), "Spring test-slice annotation still present."),
        ("blocking", "mockmvc", re.compile(r"\bMockMvc\b|MockMvcRequestBuilders|MockMvcResultMatchers"), "Spring MockMvc API still present."),
        ("blocking", "local_server_port", re.compile(r"@LocalServerPort\b|\bLocalServerPort\b"), "Spring test port injection still present."),
        ("review", "spring_web_environment", re.compile(r"\bWebEnvironment\b"), "Spring Boot test web-environment enum still present or unresolved."),
        ("review", "rest_template_builder", re.compile(r"\bRestTemplateBuilder\b"), "RestTemplateBuilder still present and needs deterministic Micronaut client migration."),
        ("review", "spring_model_api", re.compile(r"\bModelAndView\b|\bModelMap\b|\bModel\b"), "Spring MVC model type still present."),
        ("review", "spring_paging_api", re.compile(r"\bPage<|\bPageable\b|\bPageImpl<"), "Spring Data paging API still present."),
        ("review", "binding_result", re.compile(r"\bBindingResult\b"), "Spring BindingResult flow still present and needs Micronaut validation redesign."),
        ("review", "web_data_binder", re.compile(r"\bWebDataBinder\b"), "Spring WebDataBinder customization still present."),
        ("review", "jcache_customizer", re.compile(r"\bJCacheManagerCustomizer\b"), "Spring cache customization API still present."),
    )

    BUILD_PATTERNS = (
        (
            "blocking",
            "spring_dependency",
            re.compile(
                r"""['"](?:org\.springframework(?:\.boot)?:[^'"]+|[^'"]*spring-boot-starter[^'"]*)['"]"""
            ),
            "Spring dependency still present in migrated build file.",
        ),
        ("review", "spring_devtools", re.compile(r"spring-boot-devtools"), "Spring devtools dependency still present in migrated build file."),
    )

    def verify_project(
        self,
        source_root: str,
        target_root: str,
        build_file_relative_path: Optional[str] = None,
        report_path: Optional[str] = None,
    ) -> VerificationReport:
        source_files = self._collect_java_files(source_root)
        findings: List[VerificationFinding] = []
        missing_target_files: List[str] = []
        matched_files = 0

        for source_file in source_files:
            relative_path = os.path.relpath(source_file, source_root)
            target_file = os.path.join(target_root, relative_path)
            if not os.path.exists(target_file):
                missing_target_files.append(relative_path)
                findings.append(
                    VerificationFinding(
                        severity="blocking",
                        file=relative_path,
                        rule="missing_target_file",
                        message="Source file has no migrated target counterpart.",
                    )
                )
                continue

            matched_files += 1
            findings.extend(self._scan_java_file(target_file, relative_path))

        if build_file_relative_path:
            build_file = os.path.join(target_root, build_file_relative_path)
            if os.path.exists(build_file):
                findings.extend(self._scan_build_file(build_file, build_file_relative_path))

        severity_counts = {"blocking": 0, "review": 0, "info": 0}
        for finding in findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1

        report = VerificationReport(
            source_root=source_root,
            target_root=target_root,
            compared_files=len(source_files),
            matched_files=matched_files,
            missing_target_files=missing_target_files,
            findings=findings,
            severity_counts=severity_counts,
            trusted_ready=severity_counts.get("blocking", 0) == 0,
        )

        if report_path:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(report.to_dict(), handle, indent=2)

        return report

    def _collect_java_files(self, root: str) -> List[str]:
        java_files: List[str] = []
        ignored_dirs = {".git", ".idea", "target", "__pycache__", ".gradle", ".mvn", "build"}
        for current_root, dirs, files in os.walk(root):
            dirs[:] = [item for item in dirs if item not in ignored_dirs]
            for file_name in files:
                if file_name.endswith(".java"):
                    java_files.append(os.path.join(current_root, file_name))
        return sorted(java_files)

    def _scan_java_file(self, file_path: str, relative_path: str) -> List[VerificationFinding]:
        with open(file_path, "r", encoding="utf-8") as handle:
            content = handle.read()

        findings: List[VerificationFinding] = []
        for severity, rule, pattern, message in self.JAVA_PATTERNS:
            match = pattern.search(content)
            if match:
                if rule == "spring_import" and not self._is_framework_spring_import(match.group(0)):
                    continue
                if rule == "spring_model_api" and self._is_supported_model_usage(content, match.group(0)):
                    continue
                if rule == "spring_paging_api" and self._is_supported_paging_usage(content, match.group(0)):
                    continue
                findings.append(
                    VerificationFinding(
                        severity=severity,
                        file=relative_path,
                        rule=rule,
                        message=message,
                        evidence=match.group(0),
                    )
                )
        return findings

    def _is_supported_model_usage(self, content: str, evidence: str) -> bool:
        if "import org.springframework.ui." in content or "import org.springframework.web.servlet.ModelAndView;" in content:
            return False
        if evidence == "ModelAndView":
            return "import io.micronaut.views.ModelAndView;" in content
        if evidence in {"Model", "ModelMap"}:
            return "Map<String, Object>" in content
        return False

    def _is_supported_paging_usage(self, content: str, evidence: str) -> bool:
        if "import org.springframework.data.domain." in content:
            return False
        if evidence in {"Page<", "Pageable", "PageImpl<"}:
            return (
                "import io.micronaut.data.model.Page;" in content
                or "import io.micronaut.data.model.Pageable;" in content
                or "Pageable.from(" in content
            )
        return False

    def _is_framework_spring_import(self, evidence: str) -> bool:
        stripped = str(evidence or "").strip()
        framework_prefixes = (
            "import org.springframework.aot.",
            "import org.springframework.beans.",
            "import org.springframework.boot.",
            "import org.springframework.cache.",
            "import org.springframework.context.",
            "import org.springframework.core.",
            "import org.springframework.dao.",
            "import org.springframework.data.",
            "import org.springframework.format.",
            "import org.springframework.http.",
            "import org.springframework.jdbc.",
            "import org.springframework.orm.",
            "import org.springframework.scheduling.",
            "import org.springframework.stereotype.",
            "import org.springframework.test.",
            "import org.springframework.transaction.",
            "import org.springframework.ui.",
            "import org.springframework.util.",
            "import org.springframework.validation.",
            "import org.springframework.web.",
        )
        return stripped.startswith(framework_prefixes)

    def _scan_build_file(self, file_path: str, relative_path: str) -> List[VerificationFinding]:
        with open(file_path, "r", encoding="utf-8") as handle:
            content = handle.read()

        findings: List[VerificationFinding] = []
        for severity, rule, pattern, message in self.BUILD_PATTERNS:
            for match in pattern.finditer(content):
                findings.append(
                    VerificationFinding(
                        severity=severity,
                        file=relative_path,
                        rule=rule,
                        message=message,
                        evidence=match.group(0),
                    )
                )
        return findings
