import os
from typing import List, Dict
from src.agent.core.models import ProjectStructure, MigrationReport
from src.agent.rag.knowledge_base import LocalMigrationKnowledgeBase
from src.agent.core.llm_provider import get_llm_provider
from src.agent.agents.dependency_agent import DependencyAgent
from src.agent.agents.code_transform_agent import CodeTransformAgent
from src.agent.agents.validation_agent import ValidationAgent

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
        # Initialize knowledge base with datasets
        self.kb.load_dataset() 
        # Note: In a real scenario, we would also call initialize_knowledge_base
        # but for this test, we assume the VDB is pre-populated or handled by load_dataset
        
        self.llm = get_llm_provider()
        
        # Initialize specialized agents
        self.dependency_agent = DependencyAgent(self.kb, spring_version, micronaut_version)
        self.code_agent = CodeTransformAgent(self.kb, self.llm)
        self.validation_agent = None # Initialized after structure analysis

    def migrate_project(self, input_dir: str, output_dir: str) -> MigrationReport:
        """
        Executes the full migration workflow.
        """
        print(f"Starting migration from {input_dir} to {output_dir}")
        
        # Step 1: Analyze Project Structure
        structure = self._analyze_structure(input_dir)
        self.validation_agent = ValidationAgent(structure.build_tool)
        
        report = MigrationReport(
            total_files=len(structure.source_files) + 1,
            migrated_files=0,
            failed_files=[],
            warnings=[],
            dependency_changes={},
            config_changes={}
        )
        
        # Step 2: Migrate Build Configuration
        if structure.dependency_file:
            input_pom = os.path.join(input_dir, structure.dependency_file)
            output_pom = os.path.join(output_dir, structure.dependency_file)
            report.dependency_changes = self.dependency_agent.migrate_project_config(input_pom, output_pom)
            report.migrated_files += 1

        # Step 3: Migrate Source Code Files
        for source_file in structure.source_files:
            try:
                relative_path = os.path.relpath(source_file, input_dir)
                target_path = os.path.join(output_dir, relative_path)
                
                self.code_agent.transform_file(source_file, target_path)
                report.migrated_files += 1
            except Exception as e:
                report.failed_files.append(source_file)
                print(f"Failed to migrate {source_file}: {e}")
        
        # Step 4: Final Validation and Self-Refinement (Try-Compile-Fix)
        print("Starting Validation and Self-Refinement Loop...")
        max_retries = 3
        
        import re # Ensure re is available
        
        for attempt in range(max_retries):
            print(f"Build validation attempt {attempt + 1} of {max_retries}...")
            success, errors = self.validation_agent.validate(output_dir)
            
            if success:
                print("Build successful! No further refinement needed.")
                break
            
            if not errors:
                print("Build failed but no specific error patterns recognized. Check logs.")
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
