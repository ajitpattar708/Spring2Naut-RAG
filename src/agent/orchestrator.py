import os
from typing import List, Dict
from src.agent.core.models import ProjectStructure, MigrationReport
from src.agent.rag.knowledge_base import LocalMigrationKnowledgeBase
from src.agent.core.config import MigrationConfig
from src.agent.core.llm_provider import get_llm_provider
from src.agent.agents.dependency_agent import DependencyAgent
from src.agent.agents.code_transform_agent import CodeTransformAgent
from src.agent.agents.validation_agent import ValidationAgent
from src.agent.agents.config_agent import ConfigAgent

# Modern Terminal Colors
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

class MigrationOrchestrator:
    """
    Central controller for the migration process.
    Coordinates specialized agents to transform a project from Spring Boot to Micronaut.
    """
    
    def __init__(self, spring_version: str, micronaut_version: str):
        self.spring_version = spring_version
        self.micronaut_version = micronaut_version
        
        # Initialize internal services
        self.kb = LocalMigrationKnowledgeBase()
        # Note: In a real scenario, we would also call initialize_knowledge_base
        # but for this test, we assume the VDB is pre-populated or handled by load_dataset
        
        self.llm = get_llm_provider()
        if not self.llm.is_available():
            print(f"{YELLOW}[WARN]{RESET} LLM Provider ({MigrationConfig.LLM_PROVIDER}) is not reachable. "
                  "Migration will proceed with local rules only, but refinement may fail or be very slow.")
        
        # Initialize specialized agents
        self.dependency_agent = DependencyAgent(self.kb, spring_version, micronaut_version)
        self.code_agent = CodeTransformAgent(self.kb, self.llm)
        self.config_agent = ConfigAgent(self.kb)
        self.validation_agent = None # Initialized after structure discovery

    def migrate_project(self, input_dir: str, output_dir: str) -> MigrationReport:
        """
        Executes the full migration workflow.
        """
        print(f"Starting migration from {input_dir} to {output_dir}")
        
        # Phase 1: Project Analysis & Discovery
        print(f"\n{BOLD}{BLUE}[Phase 1/3] Project Analysis & Discovery{RESET}")
        structure = self._analyze_structure(input_dir)
        self.validation_agent = ValidationAgent(structure.build_tool)
        
        print(f"  {GREEN}[OK]{RESET} Detected Build Tool: {BOLD}{structure.build_tool.capitalize()}{RESET}", flush=True)
        print(f"  {GREEN}[OK]{RESET} Found {BOLD}{len(structure.source_files)}{RESET} Java source files", flush=True)
        print(f"  {GREEN}[OK]{RESET} Identified {BOLD}{len(structure.config_files)}{RESET} configuration files", flush=True)
        
        report = MigrationReport(
            total_files=len(structure.source_files) + (1 if structure.dependency_file else 0),
            migrated_files=0,
            failed_files=[],
            warnings=[],
            dependency_changes={},
            config_changes={}
        )
        
        # Phase 2: Transformation
        print(f"\n{BOLD}{BLUE}[Phase 2/3] Executing Transformations{RESET}")
        
        # 2a: Migrate Build Configuration
        if structure.dependency_file:
            print(f"  {YELLOW}>>{RESET} Transforming Build Config: {BOLD}{structure.dependency_file}{RESET}", flush=True)
            input_pom = os.path.join(input_dir, structure.dependency_file)
            output_pom = os.path.join(output_dir, structure.dependency_file)
            report.dependency_changes = self.dependency_agent.migrate_project_config(input_pom, output_pom)
            
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

        # Phase 3: Validation & Self-Refinement (Try-Compile-Fix)
        print(f"\n{BOLD}{BLUE}[Phase 3/3] Validation & Self-Refinement{RESET}")
        max_retries = 3
        
        import re # Ensure re is available
        
        for attempt in range(max_retries):
            print(f"  {YELLOW}>>{RESET} Build validation attempt {BOLD}{attempt + 1}{RESET} of {max_retries}...")
            success, errors = self.validation_agent.validate(output_dir)
            
            if success:
                print(f"  {GREEN}[OK]{RESET} Build successful! No further refinement needed.")
                break
            
            if not errors:
                print(f"  {YELLOW}[WARN]{RESET} Build failed but no specific Java compilation errors recognized.")
                
                # Show some of the raw output if available
                if hasattr(self.validation_agent, 'last_output'):
                    print(f"  {RED}Raw Output Excerpt:{RESET}")
                    lines = self.validation_agent.last_output.split('\n')
                    for line in lines[-10:]: # Show last 10 lines
                        if line.strip(): print(f"    {line}")
                
                print(f"  {BOLD}Suggestion:{RESET} Check for environmental issues, missing dependencies, or parent POM availability.")
                break
                
            print(f"Build failed with {len(errors)} errors. Attempting self-fix...")
            
            # Map errors back to files and apply fixes
            for error in errors:
                # Improved regex to handle both Unix and Windows paths
                # Looks for something like F:\path\to\File.java: or /path/to/File.java:
                match = re.search(r'(([a-zA-Z]:\\|/)[^\s:]+\.java):', error)
                if match:
                    file_path = match.group(1)
                    if os.path.exists(file_path):
                        print(f"  Attempting to fix: {os.path.basename(file_path)}")
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        fixed_content = self.code_agent.self_fix(content, [error])
                        
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(fixed_content)
                    else:
                        print(f"  Warning: File path in error log does not exist locally: {file_path}")
            
            if attempt == max_retries - 1:
                print("Reached maximum retries. Remaining errors documented in report.")
                report.warnings.extend(errors)
        
        print("Migration process completed.")
        print("\n" + "=" * 50)
        print(f"{BOLD}MIGRATION SUMMARY{RESET}")
        print("=" * 50)
        print(f"Total Files Processed: {BOLD}{report.total_files}{RESET}")
        print(f"Successfully Migrated: {GREEN}{report.migrated_files}{RESET}")
        print(f"Failed Files:          {RED}{len(report.failed_files)}{RESET}")
        print("-" * 50)
        
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
        
        print("\n" + "-" * 50)
        print(f"{BOLD}MIGRATION COMPLETED.{RESET}")
        
        return report

    def _analyze_structure(self, input_dir: str) -> ProjectStructure:
        """
        Identifies source files, configuration files, and the build tool.
        """
        source_files = []
        config_files = []
        dependency_file = None
        build_tool = "maven"
        
        for root, _, files in os.walk(input_dir):
            for file in files:
                full_path = os.path.join(root, file)
                if file == "pom.xml":
                    dependency_file = file
                    build_tool = "maven"
                elif file == "build.gradle":
                    dependency_file = file
                    build_tool = "gradle"
                elif file.endswith(".java"):
                    source_files.append(full_path)
                elif file.startswith("application."):
                    config_files.append(full_path)
                    
        return ProjectStructure(
            root_path=input_dir,
            source_files=source_files,
            config_files=config_files,
            dependency_file=dependency_file,
            build_tool=build_tool
        )
