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
            dependencies_node = root.find("maven:dependencies", ns)
            if dependencies_node is not None:
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
                
                # Step 3: Replace Spring plugins with Micronaut
                build_node = root.find("maven:build", ns)
                if build_node is not None:
                    plugins = build_node.find("maven:plugins", ns)
                    if plugins is not None:
                        for plugin in plugins.findall("maven:plugin", ns):
                            art = plugin.find("maven:artifactId", ns)
                            if art is not None and "spring-boot-maven-plugin" in art.text:
                                group = plugin.find("maven:groupId", ns)
                                if group is not None: group.text = "io.micronaut.maven"
                                art.text = "micronaut-maven-plugin"
                                changes["spring-boot-maven-plugin"] = "micronaut-maven-plugin"

                # Step 4: Individual Dependency Replacement (RAG-based)
                # We'll collect nodes to remove to avoid concurrent modification issues
                to_remove = []
                
                for dep in dependencies_node.findall("maven:dependency", ns):
                    group = dep.find("maven:groupId", ns)
                    artifact = dep.find("maven:artifactId", ns)
                    version = dep.find("maven:version", ns)
                    
                    if artifact is not None:
                        artifact_id = artifact.text
                        group_id = group.text if group is not None else ""
                        
                        # Search for Micronaut equivalent
                        rules = self.kb.search_dependency(artifact_id)
                        
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
