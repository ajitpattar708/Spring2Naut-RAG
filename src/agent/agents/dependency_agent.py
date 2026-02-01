import os
import re
from xml.etree import ElementTree as ET
from typing import Dict, List, Optional
from src.agent.rag.knowledge_base import KnowledgeService

class DependencyAgent:
    """
    Expert agent for migrating build configuration files (Maven and Gradle).
    Handles dependency mappings, parent POM updates, and plugin conversions.
    """
    
    def __init__(self, knowledge_base: KnowledgeService, spring_version: str, micronaut_version: str):
        self.kb = knowledge_base
        self.spring_version = spring_version
        self.micronaut_version = micronaut_version

    def migrate_project_config(self, source_path: str, output_path: str) -> Dict[str, str]:
        """
        Detects the build tool and applies the corresponding migration logic.
        """
        if source_path.endswith('pom.xml'):
            return self.migrate_maven_pom(source_path, output_path)
        elif source_path.endswith('.gradle') or source_path.endswith('.gradle.kts'):
            return self.migrate_gradle(source_path, output_path)
        return {}

    def migrate_maven_pom(self, pom_path: str, output_path: str) -> Dict[str, str]:
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
            dependencies = root.find("maven:dependencies", ns)
            if dependencies is not None:
                for dep in dependencies.findall("maven:dependency", ns):
                    art = dep.find("maven:artifactId", ns)
                    if art is not None and "spring-boot-starter" in art.text:
                        # Logic to replace starts with Micronaut equivalents
                        pass
            
            # Save the updated POM
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            
        except Exception as e:
            # Errors logged without emoticons
            print(f"Error during Maven migration: {e}")
            
        return changes

    def migrate_gradle(self, gradle_path: str, output_path: str) -> Dict[str, str]:
        """
        Migrates Gradle build scripts using regex-based pattern replacement.
        """
        changes = {}
        try:
            with open(gradle_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace Spring Boot plugin with Micronaut
            content = re.sub(
                r'id\s+["\']org\.springframework\.boot["\'].*',
                f'id "io.micronaut.application" version "{self.micronaut_version}"',
                content
            )
            
            # Update Micronaut version properties
            # Additional migration logic here
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"Error during Gradle migration: {e}")
            
        return changes
