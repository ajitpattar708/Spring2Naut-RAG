import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional
from src.agent.core.models import ProjectStructure, MigrationReport
from src.agent.rag.knowledge_base import LocalMigrationKnowledgeBase
from src.agent.core.config import MigrationConfig
from src.agent.core.llm_provider import get_llm_provider
from src.agent.core.versioning import normalize_major_minor, validate_migration_target_versions
from src.agent.agents.dependency_agent import DependencyAgent
from src.agent.agents.code_transform_agent import CodeTransformAgent
from src.agent.agents.validation_agent import ValidationAgent
from src.agent.agents.verification_agent import VerificationAgent
from src.agent.agents.config_agent import ConfigAgent

# Modern Terminal Colors
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
WHITE = "\033[97m"
BOLD = "\033[1m"
RESET = "\033[0m"
DIM = "\033[2m"

class MigrationOrchestrator:
    """
    Central controller for the migration process.
    Coordinates specialized agents to transform a project from Spring Boot to Micronaut.
    """
    
    def __init__(self, spring_version: str, micronaut_version: str, build_tool_override: Optional[str] = None):
        validate_migration_target_versions(spring_version, micronaut_version)
        self.spring_version = spring_version
        self.micronaut_version = micronaut_version
        self.build_tool_override = (build_tool_override or "").strip().lower() or None
        
        # Initialize internal services
        self.kb = LocalMigrationKnowledgeBase()
        self._configure_target_platform_snapshot()
        # Note: In a real scenario, we would also call initialize_knowledge_base
        # but for this test, we assume the VDB is pre-populated or handled by load_dataset
        
        self.llm = get_llm_provider()
        self.llm_available = self.llm.is_available()
        self._print_runtime_banner()
        if not self.llm_available:
            print(f"{YELLOW}[WARN]{RESET} LLM Provider ({MigrationConfig.LLM_PROVIDER}) is not reachable. "
                  "Migration will proceed with local rules only, but refinement may fail or be very slow.")
        
        # Initialize specialized agents
        self.dependency_agent = DependencyAgent(self.kb, spring_version, micronaut_version)
        self.code_agent = CodeTransformAgent(self.kb, self.llm, spring_version, micronaut_version)
        self.config_agent = ConfigAgent(self.kb, spring_version, micronaut_version)
        self.verification_agent = VerificationAgent()
        self.validation_agent = None # Initialized after structure discovery

    def _print_runtime_banner(self) -> None:
        kb_counts = self._kb_collection_counts()
        trusted_rules = sum(kb_counts.values())
        llm_provider_name = type(self.llm).__name__
        llm_model = getattr(self.llm, "model", MigrationConfig.LLM_MODEL)
        llm_base_url = getattr(self.llm, "base_url", "")
        llm_status = f"{GREEN}reachable{RESET}" if self.llm_available else f"{RED}unreachable{RESET}"
        kb_status = f"{GREEN}ready{RESET}" if getattr(self.kb, "available", False) else f"{YELLOW}deterministic-only{RESET}"

        print(f"{BOLD}{CYAN}[Runtime]{RESET}")
        print(
            f"  {MAGENTA}[LLM]{RESET} provider={llm_provider_name} "
            f"configured={MigrationConfig.LLM_PROVIDER} model={llm_model} status={llm_status}"
        )
        if llm_base_url:
            print(f"  {MAGENTA}[LLM]{RESET} endpoint={llm_base_url}")
        print(
            f"  {BLUE}[VDB]{RESET} engine={getattr(self.kb, 'vector_backend_name', MigrationConfig.VECTOR_DB_TYPE)} status={kb_status} "
            f"path={getattr(self.kb, 'db_path', MigrationConfig.VECTOR_DB_PATH)}"
        )
        if not getattr(self.kb, "available", False):
            print(
                f"  {YELLOW}[VDB]{RESET} vector retrieval inactive; agent will use deterministic rules first "
                f"and only then optional LLM refinement"
            )
        print(
            f"  {BLUE}[VDB]{RESET} embedding_model={MigrationConfig.EMBEDDING_MODEL} "
            f"dimension={getattr(self.kb, 'embedding_dimension', MigrationConfig.EMBEDDING_DIMENSION)} "
            f"trusted_rules={trusted_rules}"
        )
        manifest = self._load_kb_manifest()
        if manifest:
            init_spring = manifest.get("spring_version")
            init_micronaut = manifest.get("micronaut_version")
            init_mode = manifest.get("mode")
            compatible_rules = manifest.get("compatible_rule_count")
            init_spring_line = manifest.get("spring_line") or normalize_major_minor(str(init_spring or ""))
            init_micronaut_line = manifest.get("micronaut_line") or normalize_major_minor(str(init_micronaut or ""))
            print(
                f"  {BLUE}[VDB]{RESET} initialized_for=Spring {init_spring} -> Micronaut {init_micronaut} "
                f"mode={init_mode} compatible_rules={compatible_rules}"
            )
            target_profile = manifest.get("target_profile") if isinstance(manifest.get("target_profile"), dict) else None
            if target_profile:
                print(
                    f"  {BLUE}[VDB]{RESET} target_profile={target_profile.get('compatibility_mode', 'unknown')} "
                    f"spring_line={target_profile.get('spring_line', init_spring_line)} "
                    f"micronaut_line={target_profile.get('micronaut_line', init_micronaut_line)} "
                    f"pair_line_specific={target_profile.get('pair_line_specific_rule_count', 0)}"
                )
            target_platform_snapshot = manifest.get("target_platform_snapshot")
            if isinstance(target_platform_snapshot, dict):
                snapshot_summary = (
                    target_platform_snapshot.get("target_platform_summary")
                    if isinstance(target_platform_snapshot.get("target_platform_summary"), dict)
                    else {}
                )
                snapshot_path = str(target_platform_snapshot.get("snapshot_path", "") or "").strip()
                snapshot_version = str(
                    snapshot_summary.get("target_platform_version")
                    or init_micronaut
                    or self.micronaut_version
                ).strip()
                snapshot_channel = str(
                    snapshot_summary.get("target_platform_resolution_channel")
                    or snapshot_summary.get("resolution_channel")
                    or "unknown"
                ).strip()
                managed_count = int(target_platform_snapshot.get("managed_dependency_count", 0) or 0)
                snapshot_matches_current_target = (
                    str(init_spring or "").strip() == self.spring_version
                    and str(init_micronaut or "").strip() == self.micronaut_version
                )
                if snapshot_matches_current_target:
                    print(
                        f"  {BLUE}[VDB]{RESET} using init snapshot for Micronaut {snapshot_version}"
                    )
                    print(f"  {BLUE}[VDB]{RESET} managed deps loaded={managed_count}")
                    print(f"  {BLUE}[VDB]{RESET} snapshot channel={snapshot_channel}")
                print(
                    f"  {BLUE}[VDB]{RESET} target_platform_snapshot="
                    f"{managed_count} managed path={snapshot_path}"
                )
            if init_spring != self.spring_version or init_micronaut != self.micronaut_version:
                current_spring_line = normalize_major_minor(self.spring_version)
                current_micronaut_line = normalize_major_minor(self.micronaut_version)
                same_line = (
                    current_spring_line == str(init_spring_line)
                    and current_micronaut_line == str(init_micronaut_line)
                )
                note = (
                    "runtime filtering will still use the current target pair"
                    if same_line
                    else "re-running init for this target line is recommended for the strongest release evidence"
                )
                print(
                    f"  {YELLOW}[VDB]{RESET} current_target=Spring {self.spring_version} -> Micronaut {self.micronaut_version} "
                    f"({note})"
                )
        if trusted_rules:
            counts_text = ", ".join(f"{name}={count}" for name, count in kb_counts.items() if count)
            if counts_text:
                print(f"  {BLUE}[VDB]{RESET} collections: {counts_text}")
        build_tool_override = getattr(self, "build_tool_override", None)
        if build_tool_override:
            print(f"  {BLUE}[Build]{RESET} forced_build_tool={build_tool_override}")

    def _load_kb_manifest(self) -> Optional[Dict[str, object]]:
        db_path = Path(getattr(self.kb, "db_path", MigrationConfig.VECTOR_DB_PATH))
        manifest_path = db_path / "kb_manifest.json"
        if not manifest_path.exists():
            return None
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _configure_target_platform_snapshot(self) -> None:
        manifest = self._load_kb_manifest()
        snapshot_path = ""
        if manifest:
            init_spring = str(manifest.get("spring_version") or "").strip()
            init_micronaut = str(manifest.get("micronaut_version") or "").strip()
            target_platform_snapshot = manifest.get("target_platform_snapshot")
            if (
                isinstance(target_platform_snapshot, dict)
                and init_spring == self.spring_version
                and init_micronaut == self.micronaut_version
            ):
                snapshot_path = str(target_platform_snapshot.get("snapshot_path") or "").strip()
        MigrationConfig.TARGET_PLATFORM_MANAGED_FILE = snapshot_path

    def _kb_collection_counts(self) -> Dict[str, int]:
        collection_names = ("annotations", "dependencies", "configurations", "code_patterns", "imports", "types")
        counts: Dict[str, int] = {}
        for name in collection_names:
            try:
                stats = self.kb.get_collection_stats(name)
                counts[name] = int(stats.get("count", 0))
            except Exception:
                counts[name] = 0
        return counts

    def migrate_project(self, input_dir: str, output_dir: str) -> MigrationReport:
        """
        Executes the full migration workflow.
        """
        print(f"Starting migration from {input_dir} to {output_dir}")
        started_at = datetime.now(timezone.utc).isoformat()
        
        # Phase 1: Project Analysis & Discovery
        print(f"\n{BOLD}{BLUE}[Phase 1/3] Project Analysis & Discovery{RESET}")
        structure = self._analyze_structure(input_dir)
        if not hasattr(self, "verification_agent") or self.verification_agent is None:
            self.verification_agent = VerificationAgent()
        project_input_root = structure.project_root or input_dir
        relative_project_root = structure.relative_project_root or "."
        project_output_root = (
            output_dir
            if relative_project_root in {"", "."}
            else os.path.join(output_dir, relative_project_root)
        )
        self.validation_agent = ValidationAgent(structure.build_tool)
        
        print(f"  {GREEN}[OK]{RESET} Detected Build Tool: {BOLD}{structure.build_tool.capitalize()}{RESET}", flush=True)
        if structure.build_tool_forced:
            print(f"  {BLUE}[INFO]{RESET} Build tool forced by CLI override", flush=True)
        print(f"  {GREEN}[OK]{RESET} Found {BOLD}{len(structure.source_files)}{RESET} Java source files", flush=True)
        print(f"  {GREEN}[OK]{RESET} Identified {BOLD}{len(structure.config_files)}{RESET} configuration files", flush=True)
        if relative_project_root not in {"", "."}:
            print(
                f"  {BLUE}[INFO]{RESET} Using nested project root: {BOLD}{relative_project_root}{RESET}",
                flush=True,
            )

        self._prepare_output_directory(input_dir, output_dir)
        
        report = MigrationReport(
            total_files=len(structure.source_files) + (1 if structure.dependency_file else 0),
            migrated_files=0,
            failed_files=[],
            warnings=[],
            dependency_changes={},
            config_changes={},
            input_dir=input_dir,
            output_dir=output_dir,
            build_tool=structure.build_tool,
            spring_version=self.spring_version,
            micronaut_version=self.micronaut_version,
            status="in_progress",
            started_at=started_at,
        )

        if structure.dependency_file:
            input_dependency_file = os.path.join(input_dir, structure.dependency_file)
            audit_mode = (
                "deterministic local-only"
                if not MigrationConfig.BUILD_METADATA_ENABLED
                else "resolved build-metadata inspection"
            )
            print(
                f"  {BLUE}[INFO]{RESET} Running source dependency audit ({audit_mode})...",
                flush=True,
            )
            report.dependency_audit_report_path = self._dependency_audit_report_path(
                output_dir,
                "source_dependency_audit_report.json",
            )
            report.dependency_audit = self.dependency_agent.audit_project_dependencies(
                input_dependency_file,
                project_path=project_input_root,
                report_path=report.dependency_audit_report_path,
            )
            report.dependency_inventory_report_path = self._resolved_dependency_inventory_report_path(
                output_dir,
                "source_resolved_dependency_inventory.json",
            )
            self._write_resolved_dependency_inventory(
                report.dependency_audit,
                report.dependency_inventory_report_path,
            )
            self._print_resolved_evidence_paths(
                report.dependency_audit,
                heading=f"Source {structure.build_tool.capitalize()} evidence",
            )
            severity_counts = report.dependency_audit.get("severity_counts", {})
            print(
                f"  {GREEN}[OK]{RESET} Dependency audit found "
                f"{BOLD}{report.dependency_audit.get('direct_dependency_count', 0)}{RESET} direct and "
                f"{BOLD}{report.dependency_audit.get('transitive_dependency_count', 0)}{RESET} transitive dependencies",
                flush=True,
            )
            resolved_scope_counts = report.dependency_audit.get("resolved_dependency_scope_counts", {})
            if any(int(count) > 0 for count in resolved_scope_counts.values()):
                print(
                    f"  {BLUE}[INFO]{RESET} {structure.build_tool.capitalize()} resolved inventory: "
                    f"compile={resolved_scope_counts.get('compile', 0)}, "
                    f"runtime={resolved_scope_counts.get('runtime', 0)}, "
                    f"effective-direct={report.dependency_audit.get('resolved_direct_dependency_count', 0)} "
                    f"({report.dependency_audit.get('evidence_quality', 'unknown')})",
                    flush=True,
                )
            print(
                f"  {YELLOW}>>{RESET} Dependency risk summary: "
                f"blocking={severity_counts.get('blocking', 0)}, "
                f"review={severity_counts.get('review', 0)}, "
                f"info={severity_counts.get('info', 0)}",
                flush=True,
            )
            self._print_dependency_graph_summary(report.dependency_audit)
            self._print_dependency_findings(report.dependency_audit, heading="Source dependency findings")
            report.warnings.extend(self._summarize_dependency_audit(report.dependency_audit))
        
        # Phase 2: Transformation
        print(f"\n{BOLD}{BLUE}[Phase 2/3] Executing Transformations{RESET}")
        
        # 2a: Migrate Build Configuration
        if structure.dependency_file:
            print(f"  {YELLOW}>>{RESET} Transforming Build Config: {BOLD}{structure.dependency_file}{RESET}", flush=True)
            input_pom = os.path.join(input_dir, structure.dependency_file)
            output_pom = os.path.join(output_dir, structure.dependency_file)
            os.makedirs(os.path.dirname(output_pom), exist_ok=True)
            report.dependency_changes = self.dependency_agent.migrate_project_config(
                input_pom,
                output_pom,
                audit_report=report.dependency_audit,
                project_path=project_input_root,
            )
            
            # Detailed Logging of POM changes for GA transparency
            for original, mapped in report.dependency_changes.items():
                print(f"    {GREEN}|--{RESET} {original} {YELLOW}->{RESET} {BLUE}{mapped}{RESET}", flush=True)
                
            report.migrated_files += 1

        # 2b: Migrate Source Code Files
        print(f"  - Transforming {len(structure.source_files)} Source Files...")
        for source_file in structure.source_files:
            try:
                relative_path = os.path.relpath(source_file, input_dir)
                target_path = os.path.join(output_dir, relative_path)
                
                print(f"  {YELLOW}>>{RESET} [{report.migrated_files + 1}/{len(structure.source_files)}] Migrating: {BOLD}{os.path.basename(source_file)}{RESET}", flush=True)
                self.code_agent.transform_file(source_file, target_path)
                report.migrated_files += 1
            except Exception as e:
                report.failed_files.append(source_file)
                print(f"  {RED}[ERROR]{RESET} Failed to migrate {os.path.basename(source_file)}: {e}")
        
        # 2c: Migrate Configuration Files
        if structure.config_files:
            print(f"  {YELLOW}>>{RESET} Transforming {len(structure.config_files)} Configuration Files...")
            for config_file in structure.config_files:
                relative_path = os.path.relpath(config_file, input_dir)
                target_path = os.path.join(output_dir, relative_path)
                
                print(f"    {YELLOW}>>{RESET} Migrating: {BOLD}{os.path.basename(config_file)}{RESET}", flush=True)
                file_changes = self.config_agent.migrate_config(config_file, target_path)
                report.config_changes.update(file_changes)
                
                # Detailed logging of property-level changes
                for old_prop, new_prop in file_changes.items():
                    print(f"      {GREEN}|--{RESET} {old_prop} {YELLOW}->{RESET} {BLUE}{new_prop}{RESET}", flush=True)

        copied_resources = self._copy_supporting_resources(
            project_input_root,
            project_output_root,
            structure.config_files,
        )
        sql_init_materializer = getattr(self.config_agent, "materialize_sql_init_migrations", None)
        generated_sql_migrations = 0
        if callable(sql_init_materializer):
            generated_sql_migrations = sql_init_materializer(
                project_input_root,
                project_output_root,
                structure.config_files,
            )
        if copied_resources:
            print(
                f"  {BLUE}[INFO]{RESET} Copied {BOLD}{copied_resources}{RESET} non-config resource files "
                f"(templates, DB scripts, messages, static assets, test resources)",
                flush=True,
            )
        if generated_sql_migrations:
            print(
                f"  {BLUE}[INFO]{RESET} Generated {BOLD}{generated_sql_migrations}{RESET} Flyway SQL migration resource(s) "
                f"from Spring SQL init scripts",
                flush=True,
            )
        template_normalization_findings = self._find_template_normalization_issues(project_output_root)
        if template_normalization_findings:
            print(
                f"  {YELLOW}[REVIEW]{RESET} Found {BOLD}{len(template_normalization_findings)}{RESET} template normalization risk(s)",
                flush=True,
            )
            for finding in template_normalization_findings[:5]:
                print(f"    - {finding}", flush=True)
            if len(template_normalization_findings) > 5:
                print(
                    f"    - ... and {len(template_normalization_findings) - 5} more template normalization risk(s)",
                    flush=True,
                )
            report.warnings.extend(template_normalization_findings[:10])

        if structure.dependency_file:
            migrated_dependency_file = os.path.join(output_dir, structure.dependency_file)
            audit_mode = (
                "deterministic local-only"
                if not MigrationConfig.BUILD_METADATA_ENABLED
                else "resolved build-metadata inspection"
            )
            print(
                f"  {BLUE}[INFO]{RESET} Running migrated dependency audit ({audit_mode})...",
                flush=True,
            )
            report.migrated_dependency_audit_report_path = self._dependency_audit_report_path(
                output_dir,
                "migrated_dependency_audit_report.json",
            )
            report.migrated_dependency_audit = self.dependency_agent.audit_project_dependencies(
                migrated_dependency_file,
                project_path=project_output_root,
                report_path=report.migrated_dependency_audit_report_path,
            )
            report.migrated_dependency_inventory_report_path = self._resolved_dependency_inventory_report_path(
                output_dir,
                "migrated_resolved_dependency_inventory.json",
            )
            self._write_resolved_dependency_inventory(
                report.migrated_dependency_audit,
                report.migrated_dependency_inventory_report_path,
            )
            self._print_resolved_evidence_paths(
                report.migrated_dependency_audit,
                heading=f"Migrated {structure.build_tool.capitalize()} evidence",
            )
            migrated_counts = report.migrated_dependency_audit.get("severity_counts", {})
            print(
                f"  {YELLOW}>>{RESET} Migrated dependency risk summary: "
                f"blocking={migrated_counts.get('blocking', 0)}, "
                f"review={migrated_counts.get('review', 0)}, "
                f"info={migrated_counts.get('info', 0)}",
                flush=True,
            )
            self._print_dependency_graph_summary(report.migrated_dependency_audit)
            resolved_scope_counts = report.migrated_dependency_audit.get("resolved_dependency_scope_counts", {})
            if any(int(count) > 0 for count in resolved_scope_counts.values()):
                print(
                    f"  {BLUE}[INFO]{RESET} Migrated {structure.build_tool.capitalize()} resolved inventory: "
                    f"compile={resolved_scope_counts.get('compile', 0)}, "
                    f"runtime={resolved_scope_counts.get('runtime', 0)}, "
                    f"effective-direct={report.migrated_dependency_audit.get('resolved_direct_dependency_count', 0)} "
                    f"({report.migrated_dependency_audit.get('evidence_quality', 'unknown')})",
                    flush=True,
                )
            self._print_dependency_findings(
                report.migrated_dependency_audit,
                heading="Migrated dependency findings",
            )
            report.warnings.extend(self._summarize_dependency_audit(report.migrated_dependency_audit))

        report.verification_report_path = self._verification_report_path(output_dir)
        verification_report = self.verification_agent.verify_project(
            source_root=project_input_root,
            target_root=project_output_root,
            build_file_relative_path=os.path.basename(structure.dependency_file) if structure.dependency_file else None,
            report_path=report.verification_report_path,
        )
        report.verification_summary = {
            "compared_files": verification_report.compared_files,
            "matched_files": verification_report.matched_files,
            "severity_counts": verification_report.severity_counts,
            "trusted_ready": verification_report.trusted_ready,
        }
        for item in verification_report.findings[:10]:
            report.warnings.append(
                f"[VERIFICATION {item.severity.upper()}] {item.file}: {item.message}"
            )
        verification_counts = verification_report.severity_counts
        print(
            f"\n{BOLD}{CYAN}[Verification]{RESET} File-by-file migration audit",
            flush=True,
        )
        print(
            f"  {GREEN}[OK]{RESET} Compared {BOLD}{verification_report.compared_files}{RESET} source Java files, "
            f"matched {BOLD}{verification_report.matched_files}{RESET} migrated files",
            flush=True,
        )
        print(
            f"  {YELLOW}>>{RESET} Verification findings: "
            f"blocking={verification_counts.get('blocking', 0)}, "
            f"review={verification_counts.get('review', 0)}, "
            f"info={verification_counts.get('info', 0)}",
            flush=True,
        )
        if verification_report.missing_target_files:
            for item in verification_report.missing_target_files[:5]:
                print(f"    {RED}[MISSING]{RESET} {item}", flush=True)
        if verification_report.findings:
            for item in verification_report.findings[:8]:
                color = RED if item.severity == "blocking" else YELLOW if item.severity == "review" else BLUE
                evidence = f" {DIM}evidence={item.evidence}{RESET}" if item.evidence else ""
                print(
                    f"    {color}[{item.severity.upper()}]{RESET} {item.file}: {item.message}.{evidence}",
                    flush=True,
                )

        # Phase 3: Validation & Self-Refinement (Try-Compile-Fix)
        print(f"\n{BOLD}{MAGENTA}[Phase 3/3] Build Validation & Self-Refinement{RESET}")
        max_retries = 3
        
        for attempt in range(max_retries):
            print(f"  {YELLOW}>>{RESET} Build validation attempt {BOLD}{attempt + 1}{RESET} of {max_retries}...")
            report.validation_attempts = attempt + 1
            success, errors = self.validation_agent.validate(project_output_root)
            
            if success:
                report.validation_success = True
                report.validation_status = "passed"
                print(f"  {GREEN}[OK]{RESET} Build successful! No further refinement needed.")
                break
            
            if not errors:
                failure_kind = getattr(self.validation_agent, "last_failure_kind", "")
                if failure_kind == "environment":
                    report.validation_status = "blocked_environment"
                    print(
                        f"  {YELLOW}[WARN]{RESET} Compilation passed, but runtime validation was blocked by the local environment."
                    )
                elif failure_kind == "test":
                    report.validation_status = "failed_runtime_tests"
                    print(
                        f"  {YELLOW}[WARN]{RESET} Compilation passed, but runtime test execution failed."
                    )
                else:
                    report.validation_status = "failed_compile_or_build"
                    print(f"  {YELLOW}[WARN]{RESET} Build failed but no specific Java compilation errors recognized.")
                
                # Show some of the raw output if available
                if hasattr(self.validation_agent, 'last_output'):
                    print(f"  {RED}Raw Output Excerpt:{RESET}")
                    lines = self._build_validation_excerpt(self.validation_agent.last_output)
                    for line in lines:
                        if line.strip(): print(f"    {line}")
                
                if failure_kind == "environment":
                    print(
                        f"  {BOLD}Suggestion:{RESET} Run the generated project locally outside restricted sandboxing to confirm runtime tests."
                    )
                elif failure_kind == "test":
                    print(
                        f"  {BOLD}Suggestion:{RESET} Review migrated test/runtime dependencies or failing Micronaut runtime wiring."
                    )
                else:
                    print(f"  {BOLD}Suggestion:{RESET} Check for environmental issues, missing dependencies, or parent POM availability.")
                report.validation_success = False
                break
                
            grouped_errors = self._group_validation_errors_by_file(errors)
            print(
                f"Build failed with {len(errors)} errors across {len(grouped_errors)} file(s). Attempting self-fix..."
            )

            for file_path, file_errors in grouped_errors.items():
                if os.path.exists(file_path):
                    print(f"  Attempting to fix: {os.path.basename(file_path)}")
                    for detail in file_errors[:3]:
                        print(f"    - {detail}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    fixed_content = self.code_agent.self_fix(content, file_errors, source_path=file_path)

                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                else:
                    print(f"  Warning: File path in error log does not exist locally: {file_path}")
                    for detail in file_errors[:2]:
                        print(f"    - {detail}")
            
            if attempt == max_retries - 1:
                print("Reached maximum retries. Remaining errors documented in report.")
                report.warnings.extend(errors)
                report.validation_success = False
                report.validation_status = "failed_compile_or_build"

        if report.validation_success is None:
            report.validation_success = False
        report.finished_at = datetime.now(timezone.utc).isoformat()
        report.status = self._derive_migration_status(report)
        report.migration_report_path = self._migration_report_path(output_dir)
        self._write_migration_report(report)

        self._print_completion_summary(report, output_dir)
        
        # Expert Insights Section for GA
        print(f"\n{BOLD}{YELLOW}[EXPERT INSIGHTS & POTENTIAL CONCERNS]{RESET}")
        
        # Concern 1: Architectural Shifts
        if any("gateway" in str(c).lower() for c in report.dependency_changes.values()):
             print(f"  {RED}![IMPORTANT]{RESET} {BOLD}Spring Cloud Gateway detected.{RESET}")
             print("    - Migrated to micronaut-gateway. Verify route definitions and filter implementations manually.")
             
        # Concern 2: Redis/Cache
        if any("redis" in str(c).lower() for c in report.dependency_changes.values()):
             print(f"  {YELLOW}![NOTICE]{RESET} {BOLD}Redis detected.{RESET}")
             print("    - Migrated from Jedis to Lettuce. Check connection strings in micronaut-redis-lettuce format.")
        
        # Concern 3: Manual Action Item
        print(f"  {BLUE}![ACTION]{RESET} {BOLD}Review {report.migrated_files} transformed Java files.{RESET}")
        print("    - RAG applied base mappings. LLM refined complex logic. Verify final imports.")

        if report.dependency_audit:
            severity_counts = report.dependency_audit.get("severity_counts", {})
            print(f"  {BLUE}![ACTION]{RESET} {BOLD}Dependency compatibility audit summary.{RESET}")
            print(
                "    - "
                f"blocking={severity_counts.get('blocking', 0)}, "
                f"review={severity_counts.get('review', 0)}, "
                f"info={severity_counts.get('info', 0)}"
            )
        if report.migrated_dependency_audit:
            migrated_counts = report.migrated_dependency_audit.get("severity_counts", {})
            print(f"  {BLUE}![ACTION]{RESET} {BOLD}Post-migration dependency audit summary.{RESET}")
            print(
                "    - "
                f"blocking={migrated_counts.get('blocking', 0)}, "
                f"review={migrated_counts.get('review', 0)}, "
                f"info={migrated_counts.get('info', 0)}"
            )
        dependency_change_counts = self._dependency_change_counts(report.dependency_changes)
        if dependency_change_counts["transitive_align"] > 0:
            print(f"  {GREEN}![AUTO-FIX]{RESET} {BOLD}Transitive Micronaut version drift pinned automatically.{RESET}")
            print(
                "    - "
                f"Applied {dependency_change_counts['transitive_align']} target-managed transitive override"
                f"{'s' if dependency_change_counts['transitive_align'] != 1 else ''}."
            )
        if dependency_change_counts["transitive_exclude"] > 0:
            print(f"  {GREEN}![AUTO-FIX]{RESET} {BOLD}Legacy transitive `javax` carryover excluded automatically.{RESET}")
            print(
                "    - "
                f"Applied {dependency_change_counts['transitive_exclude']} high-confidence transitive exclusion"
                f"{'s' if dependency_change_counts['transitive_exclude'] != 1 else ''}."
            )
        self._print_platform_evidence_notice(report.dependency_audit, dependency_change_counts)

        print("\n" + "-" * 50)
        print(f"{BOLD}MIGRATION COMPLETED.{RESET}")
        
        return report

    def _print_platform_evidence_notice(self, audit_report: Optional[Dict], dependency_change_counts: Dict[str, int]) -> None:
        if dependency_change_counts.get("platform_align", 0) != 0 or not audit_report:
            return

        platform = dict(audit_report.get("target_platform_summary", {}) or {})
        evidence_level = str(platform.get("target_platform_evidence_level") or "")
        if evidence_level == "none":
            print(f"  {RED}![RISK]{RESET} {BOLD}Exact Micronaut target BOM could not be resolved during audit.{RESET}")
            print("    - Automatic version alignment was limited. Review dependency reports carefully before enterprise sign-off.")
        elif evidence_level == "configured_target_line":
            print(f"  {YELLOW}![REVIEW]{RESET} {BOLD}Target Micronaut line is configured locally but full managed-module evidence was not fetched.{RESET}")
            print("    - Exact platform-managed alignment was limited in this runtime. Review dependency reports before enterprise sign-off.")

    def _platform_evidence_summary(self, audit_report: Optional[Dict]) -> str:
        if not audit_report:
            return ""

        platform = dict(audit_report.get("target_platform_summary", {}) or {})
        evidence_level = str(platform.get("target_platform_evidence_level") or "")
        version = str(platform.get("target_platform_version") or self.micronaut_version)

        if evidence_level == "exact_resolved":
            return f"{GREEN}EXACT RESOLVED{RESET} (Micronaut {version})"
        if evidence_level == "configured_target_line":
            return f"{YELLOW}CONFIGURED LOCALLY{RESET} (Micronaut {version})"
        if evidence_level == "none":
            return f"{RED}UNPROVEN{RESET} (Micronaut {version})"
        return ""

    def _prepare_output_directory(self, input_dir: str, output_dir: str) -> None:
        real_input = os.path.realpath(input_dir)
        real_output = os.path.realpath(output_dir)

        if real_input == real_output:
            raise ValueError("Output directory must be different from the input directory.")

        try:
            output_contains_input = os.path.commonpath([real_input, real_output]) == real_output
        except ValueError:
            output_contains_input = False
        if output_contains_input:
            raise ValueError("Output directory cannot be the same as or a parent of the input directory.")

        if os.path.isdir(output_dir):
            existing_entries = sorted(os.listdir(output_dir))
            if existing_entries:
                print(
                    f"  {YELLOW}[WARN]{RESET} Target output directory already exists and will be cleared: {BOLD}{output_dir}{RESET}",
                    flush=True,
                )
                shutil.rmtree(output_dir)
                print(
                    f"  {BLUE}[INFO]{RESET} Cleared stale migration output from {BOLD}{output_dir}{RESET}",
                    flush=True,
                )
        elif os.path.exists(output_dir):
            raise ValueError("Output path exists but is not a directory.")

        os.makedirs(output_dir, exist_ok=True)

    def _group_validation_errors_by_file(self, errors: List[str]) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}
        current_file: Optional[str] = None

        for raw_error in errors:
            error = str(raw_error).strip()
            if not error:
                continue

            file_path = self._extract_java_path_from_error(error)
            if file_path:
                current_file = file_path
                grouped.setdefault(file_path, [])
                if error not in grouped[file_path]:
                    grouped[file_path].append(error)
                continue

            if current_file:
                if error not in grouped[current_file]:
                    grouped[current_file].append(error)

        return grouped

    def _extract_java_path_from_error(self, error: str) -> Optional[str]:
        match = re.search(r'(([a-zA-Z]:\\|/)[^\s:]+\.java):', error)
        if match:
            return match.group(1)
        return None

    def _print_completion_summary(self, report: MigrationReport, output_dir: str) -> None:
        elapsed = self._format_elapsed(report.started_at, report.finished_at)
        if report.validation_success:
            validation_label = f"{GREEN}PASSED{RESET}"
        elif report.validation_status == "blocked_environment":
            validation_label = f"{YELLOW}BLOCKED BY ENVIRONMENT{RESET}"
        else:
            validation_label = f"{RED}FAILED{RESET}"

        print(f"\n{BOLD}{CYAN}{'=' * 58}{RESET}")
        print(f"{BOLD}{WHITE}Migration Summary{RESET}")
        print(f"{BOLD}{CYAN}{'=' * 58}{RESET}")
        print(f"{BOLD}Total Files Processed:{RESET} {report.total_files}")
        print(f"{BOLD}Successfully Migrated:{RESET} {GREEN}{report.migrated_files}{RESET}")
        print(f"{BOLD}Failed Files:{RESET} {RED}{len(report.failed_files)}{RESET}")
        print(f"{BOLD}Elapsed Time:{RESET} {elapsed}")
        dependency_change_counts = self._dependency_change_counts(report.dependency_changes)
        if any(dependency_change_counts.values()):
            print(
                f"{BOLD}Dependency Actions:{RESET} "
                f"platform-align={dependency_change_counts['platform_align']} "
                f"transitive-pin={dependency_change_counts['transitive_align']} "
                f"transitive-exclude={dependency_change_counts['transitive_exclude']} "
                f"other={dependency_change_counts['other']}"
            )
        platform_evidence = self._platform_evidence_summary(report.dependency_audit)
        if platform_evidence:
            print(f"{BOLD}Platform Evidence:{RESET} {platform_evidence}")
        if report.verification_summary:
            verification_counts = report.verification_summary.get("severity_counts", {})
            trusted_ready = report.verification_summary.get("trusted_ready", False)
            trust_label = f"{GREEN}READY{RESET}" if trusted_ready else f"{YELLOW}NEEDS REVIEW{RESET}"
            print(
                f"{BOLD}Verification:{RESET} {trust_label} "
                f"(blocking={verification_counts.get('blocking', 0)}, "
                f"review={verification_counts.get('review', 0)}, "
                f"info={verification_counts.get('info', 0)})"
            )
        print(
            f"{BOLD}Build Validation:{RESET} {validation_label}"
            f" ({report.validation_attempts} attempt{'s' if report.validation_attempts != 1 else ''})"
        )
        print(f"{BOLD}Output Root:{RESET} {output_dir}")
        if report.verification_report_path:
            print(f"{BOLD}Verification Report:{RESET} {report.verification_report_path}")
        if report.migration_report_path:
            print(f"{BOLD}Machine Report:{RESET} {report.migration_report_path}")
        print(f"{BOLD}{CYAN}{'-' * 58}{RESET}")

    def _dependency_change_counts(self, changes: Dict[str, str]) -> Dict[str, int]:
        counts = {
            "platform_align": 0,
            "transitive_align": 0,
            "transitive_exclude": 0,
            "other": 0,
        }
        for key in changes.keys():
            if str(key).startswith("platform-align:"):
                counts["platform_align"] += 1
            elif str(key).startswith("transitive-align:"):
                counts["transitive_align"] += 1
            elif str(key).startswith("transitive-exclude:"):
                counts["transitive_exclude"] += 1
            else:
                counts["other"] += 1
        return counts

    def _format_elapsed(self, started_at: Optional[str], finished_at: Optional[str]) -> str:
        if not started_at or not finished_at:
            return "unknown"
        try:
            started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
            elapsed_seconds = max(0.0, (finished - started).total_seconds())
        except ValueError:
            return "unknown"

        if elapsed_seconds < 1:
            return f"{elapsed_seconds:.2f}s"

        total_seconds = int(elapsed_seconds)
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    def _summarize_dependency_audit(self, audit_report: Dict) -> List[str]:
        findings = audit_report.get("findings", [])
        summarized: List[str] = []
        for item in findings[:5]:
            severity = item.get("severity", "review").upper()
            dependency = item.get("dependency", "unknown")
            message = item.get("message", "")
            summarized.append(f"[DEPENDENCY {severity}] {dependency}: {message}")
        if len(findings) > 5:
            summarized.append(
                f"[DEPENDENCY INFO] Additional dependency findings suppressed: {len(findings) - 5}"
            )
        return summarized

    def _print_dependency_findings(self, audit_report: Dict, heading: str) -> None:
        findings = sorted(
            list(audit_report.get("findings", [])),
            key=lambda item: (
                0 if str(item.get("severity", "review")).lower() == "blocking" else 1 if str(item.get("severity", "review")).lower() == "review" else 2,
                -int(item.get("depth", 0) or 0),
            ),
        )[:5]
        if not findings:
            return

        print(f"  {BLUE}>>{RESET} {BOLD}{heading}:{RESET}", flush=True)
        for item in findings:
            severity = str(item.get("severity", "review")).lower()
            dependency = item.get("dependency", "unknown")
            message = item.get("message", "")
            metadata = dict(item.get("metadata", {}) or {})
            verification = metadata.get("repository_verification")
            repository_descriptor = metadata.get("repository_dependency_intelligence")
            recommended_upgrade_version = metadata.get("recommended_upgrade_version")
            bom_version = metadata.get("bom_compatible_recommended_version")
            latest_repository_version = metadata.get("latest_repository_version")
            version_management = metadata.get("replacement_version_management")
            platform_reference_version = metadata.get("platform_reference_version")
            if severity == "blocking":
                color = RED
                badge = "BLOCKING"
            elif severity == "info":
                color = BLUE
                badge = "INFO"
            else:
                color = YELLOW
                badge = "REVIEW"
            depth = int(item.get("depth", 0) or 0)
            source = str(item.get("source", "") or "")
            depth_text = f" depth={depth}" if depth else ""
            source_text = f" source={source}" if source else ""
            print(f"    {color}[{badge}]{RESET} {dependency}:{depth_text}{source_text} {message}", flush=True)
            related = list(item.get("related_dependencies", []) or [])
            if related:
                chain = " -> ".join(related[:3])
                print(f"      {CYAN}[PATH]{RESET} {chain}", flush=True)
            if bom_version:
                print(
                    f"      {GREEN}[BOM]{RESET} compatible recommended version={bom_version}",
                    flush=True,
                )
            elif version_management == "platform_managed" and platform_reference_version:
                print(
                    f"      {GREEN}[BOM]{RESET} version should stay platform-managed by Micronaut target={platform_reference_version}",
                    flush=True,
                )
            if isinstance(verification, dict) and verification.get("checked"):
                if latest_repository_version:
                    print(
                        f"      {BLUE}[CENTRAL]{RESET} verified in Maven Central, latest={latest_repository_version}",
                        flush=True,
                    )
            if isinstance(repository_descriptor, dict) and repository_descriptor.get("pom_available") is True:
                print(
                    f"      {BLUE}[POM]{RESET} declared deps compile={repository_descriptor.get('compile_dependency_count', 0)} "
                    f"runtime={repository_descriptor.get('runtime_dependency_count', 0)} "
                    f"child-inspected={repository_descriptor.get('child_dependency_inspected_count', 0)}",
                    flush=True,
                )
            elif recommended_upgrade_version and not bom_version:
                print(
                    f"      {BLUE}[RECOMMENDED]{RESET} recommended upgrade version={recommended_upgrade_version}",
                    flush=True,
                )

    def _print_dependency_graph_summary(self, audit_report: Dict) -> None:
        graph = dict(audit_report.get("dependency_graph_summary", {}) or {})
        platform = dict(audit_report.get("target_platform_summary", {}) or {})
        if not graph and not platform:
            return

        print(f"  {BLUE}>>{RESET} {BOLD}Dependency graph intelligence:{RESET}", flush=True)
        if platform:
            evidence_level = str(platform.get("target_platform_evidence_level") or "")
            platform_ready = bool(platform.get("target_platform_resolved"))
            if platform_ready:
                platform_color = GREEN
            elif evidence_level == "configured_target_line":
                platform_color = YELLOW
            else:
                platform_color = RED
            print(
                f"    {platform_color}[PLATFORM]{RESET} target={platform.get('target_platform_ga', 'io.micronaut.platform:micronaut-platform')}:"
                f"{platform.get('target_platform_version', self.micronaut_version)} managed={platform.get('target_platform_managed_dependency_count', 0)} "
                f"source={platform.get('target_platform_source', 'unresolved')} "
                f"channel={platform.get('target_platform_resolution_channel', 'unknown')}",
                flush=True,
            )
            print(
                f"    {platform_color}[PLATFORM]{RESET} imported-boms={platform.get('target_platform_imported_bom_count', 0)} "
                f"visited-boms={platform.get('target_platform_visited_bom_count', 0)} "
                f"unresolved-placeholders={platform.get('target_platform_unresolved_placeholder_count', 0)}",
                flush=True,
            )

        max_depth = int(graph.get("max_transitive_depth", 0) or 0)
        deep_count = int(graph.get("deep_transitive_dependency_count", 0) or 0)
        very_deep_count = int(graph.get("very_deep_transitive_dependency_count", 0) or 0)
        drift_count = int(graph.get("micronaut_version_drift_count", 0) or 0)
        deep_spring = int(graph.get("deep_transitive_spring_dependency_count", 0) or 0)
        deep_javax = int(graph.get("deep_transitive_javax_dependency_count", 0) or 0)
        managed_hits = int(graph.get("target_platform_managed_dependency_hits", 0) or 0)

        depth_color = RED if very_deep_count > 0 else YELLOW if deep_count > 0 else GREEN
        spring_color = RED if deep_spring > 0 else GREEN
        drift_color = RED if drift_count > 0 else GREEN
        javax_color = YELLOW if deep_javax > 0 else GREEN

        print(
            f"    {depth_color}[DEPTH]{RESET} max={max_depth} deep(>=4)={deep_count} very-deep(>=6)={very_deep_count}",
            flush=True,
        )
        print(
            f"    {spring_color}[RISK]{RESET} deep-spring={deep_spring}  "
            f"{javax_color}deep-javax={deep_javax}{RESET}  "
            f"{drift_color}micronaut-drift={drift_count}{RESET}  "
            f"{GREEN}platform-managed-hits={managed_hits}{RESET}",
            flush=True,
        )

    def _print_resolved_evidence_paths(self, audit_report: Dict, heading: str) -> None:
        evidence_paths = dict(audit_report.get("resolved_evidence_paths", {}))
        if not evidence_paths:
            return

        print(f"  {BLUE}>>{RESET} {BOLD}{heading}:{RESET}", flush=True)
        for key, path in evidence_paths.items():
            print(f"    {BLUE}[EVIDENCE]{RESET} {key} -> {path}", flush=True)

    def _dependency_audit_report_path(self, output_dir: str, filename: str) -> str:
        reports_dir = os.path.join(output_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        return os.path.join(reports_dir, filename)

    def _migration_report_path(self, output_dir: str) -> str:
        reports_dir = os.path.join(output_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        return os.path.join(reports_dir, "migration_report.json")

    def _verification_report_path(self, output_dir: str) -> str:
        reports_dir = os.path.join(output_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        return os.path.join(reports_dir, "verification_report.json")

    def _resolved_dependency_inventory_report_path(self, output_dir: str, filename: str) -> str:
        reports_dir = os.path.join(output_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        return os.path.join(reports_dir, filename)

    def _derive_migration_status(self, report: MigrationReport) -> str:
        if not report.validation_success:
            return "failed_validation"
        if report.failed_files or report.warnings:
            return "completed_with_warnings"
        return "completed"

    def _write_migration_report(self, report: MigrationReport) -> None:
        if not report.migration_report_path:
            return
        with open(report.migration_report_path, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2)

    def _write_resolved_dependency_inventory(
        self,
        audit_report: Dict,
        report_path: Optional[str],
    ) -> None:
        if not report_path:
            return

        payload = {
            "evidence_quality": audit_report.get("evidence_quality", "raw_build_only"),
            "effective_pom_source": audit_report.get("effective_pom_source", "unavailable"),
            "dependency_tree_source": audit_report.get("dependency_tree_source", "unavailable"),
            "resolved_direct_dependency_count": audit_report.get("resolved_direct_dependency_count", 0),
            "resolved_dependency_scope_counts": audit_report.get(
                "resolved_dependency_scope_counts",
                {"compile": 0, "runtime": 0},
            ),
            "resolved_direct_dependencies": audit_report.get("resolved_direct_dependencies", []),
            "resolved_dependency_scopes": audit_report.get(
                "resolved_dependency_scopes",
                {"compile": [], "runtime": []},
            ),
            "target_platform_summary": audit_report.get("target_platform_summary", {}),
            "repository_intelligence_summary": audit_report.get("repository_intelligence_summary", {}),
        }
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _copy_supporting_resources(
        self,
        input_root: str,
        output_root: str,
        migrated_config_files: List[str],
    ) -> int:
        migrated_config_paths = {
            os.path.normpath(os.path.abspath(path))
            for path in migrated_config_files
        }
        copied = 0
        for relative_dir in ("src/main/resources", "src/test/resources"):
            source_dir = os.path.join(input_root, relative_dir)
            if not os.path.isdir(source_dir):
                continue
            destination_dir = os.path.join(output_root, relative_dir)
            for root_dir, _, files in os.walk(source_dir):
                for filename in files:
                    source_path = os.path.join(root_dir, filename)
                    if os.path.normpath(os.path.abspath(source_path)) in migrated_config_paths:
                        continue
                    relative_path = os.path.relpath(source_path, source_dir)
                    if relative_dir == "src/main/resources" and relative_path.startswith(f"templates{os.sep}"):
                        relative_path = os.path.join("views", os.path.relpath(relative_path, "templates"))
                    destination_path = os.path.join(destination_dir, relative_path)
                    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                    if source_path.endswith(".html") and f"{os.sep}templates{os.sep}" in source_path:
                        with open(source_path, "r", encoding="utf-8") as handle:
                            template_content = handle.read()
                        with open(destination_path, "w", encoding="utf-8") as handle:
                            handle.write(self._normalize_template_resource(template_content))
                    elif source_path.endswith(".sql") and f"{os.sep}db{os.sep}h2{os.sep}" in source_path:
                        with open(source_path, "r", encoding="utf-8") as handle:
                            sql_content = handle.read()
                        with open(destination_path, "w", encoding="utf-8") as handle:
                            handle.write(self._normalize_h2_sql_resource(sql_content))
                    else:
                        shutil.copy2(source_path, destination_path)
                    copied += 1
        if self._source_tests_use_mockito(input_root):
            if self._ensure_mockito_subclass_extension(output_root):
                copied += 1
        return copied

    def _source_tests_use_mockito(self, input_root: str) -> bool:
        test_root = os.path.join(input_root, "src", "test", "java")
        if not os.path.isdir(test_root):
            return False

        markers = (
            "org.mockito",
            "@MockBean",
            "MockitoExtension",
            "Mockito.",
            "BDDMockito",
        )
        for root_dir, _, files in os.walk(test_root):
            for filename in files:
                if not filename.endswith(".java"):
                    continue
                source_path = os.path.join(root_dir, filename)
                try:
                    with open(source_path, "r", encoding="utf-8") as handle:
                        content = handle.read()
                except OSError:
                    continue
                if any(marker in content for marker in markers):
                    return True
        return False

    def _ensure_mockito_subclass_extension(self, output_root: str) -> bool:
        extension_path = os.path.join(
            output_root,
            "src",
            "test",
            "resources",
            "mockito-extensions",
            "org.mockito.plugins.MockMaker",
        )
        if os.path.exists(extension_path):
            return False
        os.makedirs(os.path.dirname(extension_path), exist_ok=True)
        with open(extension_path, "w", encoding="utf-8") as handle:
            handle.write("mock-maker-subclass\n")
        return True

    def _normalize_template_resource(self, content: str) -> str:
        normalized = str(content or "")
        normalized = re.sub(
            r"(?P<target>[A-Za-z_][A-Za-z0-9_]*)\?\.(?P<field>[A-Za-z_][A-Za-z0-9_]*)",
            lambda match: (
                f"({match.group('target')} != null && {match.group('target')}.{match.group('field')} != null"
                f" ? {match.group('target')}.{match.group('field')} : '')"
            ),
            normalized,
        )
        replacements = {
            'th:href="@{/owners/__${owner.id}__}"': 'th:href="@{\'/owners/\' + ${owner.id}}"',
            'th:href="@{__${owner.id}__/edit}"': 'th:href="@{\'/owners/\' + ${owner.id} + \'/edit\'}"',
            'th:href="@{__${owner.id}__/pets/new}"': 'th:href="@{\'/owners/\' + ${owner.id} + \'/pets/new\'}"',
            'th:href="@{__${owner.id}__/pets/__${pet.id}__/edit}"': 'th:href="@{\'/owners/\' + ${owner.id} + \'/pets/\' + ${pet.id} + \'/edit\'}"',
            'th:href="@{__${owner.id}__/pets/__${pet.id}__/visits/new}"': 'th:href="@{\'/owners/\' + ${owner.id} + \'/pets/\' + ${pet.id} + \'/visits/new\'}"',
            'th:href="@{__${link}__}"': 'th:href="${link}"',
        }
        for source, target in replacements.items():
            normalized = normalized.replace(source, target)

        normalized = re.sub(
            r'th:href="@\{\'(/owners\?page=)__\$\{([^}]+)\}__\'\}"',
            lambda match: f'th:href="@{{\'{match.group(1)}\' + ${{{match.group(2)}}}}}"',
            normalized,
        )
        normalized = re.sub(
            r'th:href="@\{\'(/vets\.html\?page=)__\$\{([^}]+)\}__\'\}"',
            lambda match: f'th:href="@{{\'{match.group(1)}\' + ${{{match.group(2)}}}}}"',
            normalized,
        )
        normalized = re.sub(
            r'th:(href|src|action)="@\{__\$\{([^}]+)\}__\}"',
            lambda match: f'th:{match.group(1)}="${{{match.group(2)}}}"',
            normalized,
        )
        normalized = normalized.replace(
            "valid=${!#fields.hasErrors(name)}",
            "valid=${_fieldErrors == null || !_fieldErrors.containsKey(name)}",
        )
        normalized = re.sub(
            r'th:errors="\*\{(?:__)?\$\{([^}]+)\}(?:__)?\}"',
            lambda match: f'th:text="${{_fieldErrors != null ? #messages.msg(_fieldErrors[{match.group(1)}]) : \'\'}}"',
            normalized,
        )
        normalized = normalized.replace(
            'th:if="${#fields.hasAnyErrors()}"',
            'th:if="${_fieldErrors != null && !_fieldErrors.isEmpty()}"',
        )
        normalized = normalized.replace(
            'th:each="err : ${#fields.allErrors()}" th:text="${err}"',
            'th:each="err : ${_fieldErrors.values()}" th:text="${#messages.msg(err)}"',
        )
        normalized = re.sub(
            r'\$\{(?P<target>[A-Za-z_][A-Za-z0-9_]*)\.(?P<first>[A-Za-z_][A-Za-z0-9_]*)\s*\+\s*\' \' \+\s*(?P=target)\.(?P<second>[A-Za-z_][A-Za-z0-9_]*)\}',
            lambda match: (
                "${"
                f"(({match.group('target')}.{match.group('first')} != null ? {match.group('target')}.{match.group('first')} : '')"
                f" + ' ' + "
                f"({match.group('target')}.{match.group('second')} != null ? {match.group('target')}.{match.group('second')} : ''))"
                "}"
            ),
            normalized,
        )
        normalized = re.sub(
            r'\*\{(?P<first>[A-Za-z_][A-Za-z0-9_]*)\s*\+\s*\' \' \+\s*(?P<second>[A-Za-z_][A-Za-z0-9_]*)\}',
            lambda match: (
                "*{"
                f"(({match.group('first')} != null ? {match.group('first')} : '')"
                f" + ' ' + "
                f"({match.group('second')} != null ? {match.group('second')} : ''))"
                "}"
            ),
            normalized,
        )
        normalized = re.sub(r"__\$\{([^}]+)\}__", r"${\1}", normalized)
        normalized = re.sub(
            r'th:field="\*\{\$\{([^}]+)\}\}"',
            lambda match: f'th:field="*{{__${{{match.group(1)}}}__}}"',
            normalized,
        )
        return normalized

    def _normalize_h2_sql_resource(self, content: str) -> str:
        normalized = str(content or "")
        normalized = re.sub(
            r"(?im)^\s*DROP\s+TABLE\s+([A-Za-z_][A-Za-z0-9_]*)\s+IF\s+EXISTS\s*;",
            lambda match: f"DROP TABLE IF EXISTS {match.group(1)};",
            normalized,
        )
        return normalized

    def _find_template_normalization_issues(self, output_root: str) -> List[str]:
        findings: List[str] = []
        for templates_root in (
            os.path.join(output_root, "src", "main", "resources", "views"),
            os.path.join(output_root, "src", "main", "resources", "templates"),
        ):
            if not os.path.isdir(templates_root):
                continue

            for root_dir, _, files in os.walk(templates_root):
                for filename in files:
                    if not filename.endswith(".html"):
                        continue
                    template_path = os.path.join(root_dir, filename)
                    try:
                        with open(template_path, "r", encoding="utf-8") as handle:
                            content = handle.read()
                    except OSError:
                        continue

                    markers = []
                    placeholder_scan_content = re.sub(
                        r'th:field="\*\{__\$\{[^}]+\}__\}"',
                        "",
                        content,
                    )
                    if "__${" in placeholder_scan_content:
                        markers.append("Spring Thymeleaf preprocessed placeholder")
                    if "?." in content:
                        markers.append("safe-navigation operator")
                    if markers:
                        rel_path = os.path.relpath(template_path, output_root)
                        findings.append(
                            f"Template normalization review: {rel_path} still contains "
                            + ", ".join(markers)
                        )
        return findings

    def _build_validation_excerpt(self, output: str) -> List[str]:
        lines = [line for line in str(output or "").splitlines() if line.strip()]
        if not lines:
            return []

        prioritized_patterns = (
            r"\[ERROR\]",
            r"Caused by:",
            r"Exception",
            r"Unable to start Micronaut server",
            r"Operation not permitted",
            r"SocketException",
            r"HttpClientResponseException",
            r"SQL Error",
            r"ERROR:",
            r"WARN:",
        )
        selected: List[str] = []
        for line in lines:
            if any(re.search(pattern, line) for pattern in prioritized_patterns):
                selected.append(line)

        if selected:
            return selected[:12]
        return lines[-10:]

    def _analyze_structure(self, input_dir: str) -> ProjectStructure:
        """
        Identifies source files, configuration files, and the build tool.
        """
        all_source_files = []
        all_config_files = []
        build_files = []
        ignored_dirs = {".git", ".idea", "target", "__pycache__", ".gradle", ".mvn"}
        
        for root, _, files in os.walk(input_dir):
            path_parts = set(os.path.normpath(root).split(os.sep))
            if path_parts & ignored_dirs:
                continue

            for file in files:
                full_path = os.path.join(root, file)
                if file == "pom.xml":
                    build_files.append((full_path, "maven"))
                elif file in {"build.gradle", "build.gradle.kts"}:
                    build_files.append((full_path, "gradle"))
                elif file.endswith(".java"):
                    all_source_files.append(full_path)
                elif file.startswith("application") and file.endswith((".properties", ".yml", ".yaml")):
                    all_config_files.append(full_path)

        if build_files:
            build_tool_override = getattr(self, "build_tool_override", None)
            build_tool_forced = False
            if build_tool_override in {"maven", "gradle"}:
                filtered_build_files = [item for item in build_files if item[1] == build_tool_override]
                if filtered_build_files:
                    build_files = filtered_build_files
                    build_tool_forced = True

            def _candidate_score(item):
                build_path, build_tool = item
                candidate_root = os.path.dirname(build_path)
                source_count = sum(
                    1 for path in all_source_files
                    if os.path.commonpath([candidate_root, path]) == candidate_root
                )
                config_count = sum(
                    1 for path in all_config_files
                    if os.path.commonpath([candidate_root, path]) == candidate_root
                )
                build_priority = 1 if build_tool == "gradle" else 0
                return (source_count + config_count > 0, source_count + config_count, build_priority, len(candidate_root))

            selected_build_path, build_tool = max(build_files, key=_candidate_score)
            project_root = os.path.dirname(selected_build_path)
            source_files = [
                path for path in all_source_files
                if os.path.commonpath([project_root, path]) == project_root
            ]
            config_files = [
                path for path in all_config_files
                if os.path.commonpath([project_root, path]) == project_root
            ]
            dependency_file = os.path.relpath(selected_build_path, input_dir)
            relative_project_root = os.path.relpath(project_root, input_dir)
        else:
            dependency_file = None
            build_tool = "maven"
            project_root = input_dir
            relative_project_root = "."
            source_files = all_source_files
            config_files = all_config_files
            build_tool_forced = False

        return ProjectStructure(
            root_path=input_dir,
            source_files=source_files,
            config_files=config_files,
            dependency_file=dependency_file,
            build_tool=build_tool,
            project_root=project_root,
            relative_project_root=relative_project_root,
            build_tool_forced=build_tool_forced,
        )
