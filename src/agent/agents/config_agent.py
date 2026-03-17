import os
import yaml
from typing import Dict, List, Optional
from src.agent.rag.knowledge_base import KnowledgeService

class ConfigAgent:
    """
    Expert agent for migrating application configuration files (Properties, YAML).
    Handles mapping of Spring properties to Micronaut equivalents.
    """
    
    def __init__(self, knowledge_base: KnowledgeService):
        self.kb = knowledge_base

    def migrate_config(self, source_path: str, output_path: str) -> Dict[str, str]:
        """
        Determines the format of the configuration file and migrates accordingly.
        """
        if source_path.endswith('.properties'):
            return self._migrate_properties(source_path, output_path)
        elif source_path.endswith('.yml') or source_path.endswith('.yaml'):
            return self._migrate_yaml(source_path, output_path)
        return {}

    def _migrate_properties(self, source_path: str, output_path: str) -> Dict[str, str]:
        changes = {}
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            migrated_lines = []
            for line in lines:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    # RAG Search for property replacement
                    rules = self.kb.search_configuration(key)
                    if rules:
                        new_key = rules[0].micronaut_pattern
                        migrated_lines.append(f"{new_key}={value}")
                        changes[key] = new_key
                    else:
                        # Fallback: simple prefix replacement if no specific rule
                        if key.startswith('spring.'):
                            new_key = key.replace('spring.', 'micronaut.', 1)
                            migrated_lines.append(f"{new_key}={value}")
                            changes[key] = new_key
                        else:
                            migrated_lines.append(line)
                else:
                    migrated_lines.append(line)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.writelines(migrated_lines)
        except Exception as e:
            print(f"Error migrating properties file {source_path}: {e}")
            
        return changes

    def _migrate_yaml(self, source_path: str, output_path: str) -> Dict[str, str]:
        changes = {}
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                # Use safe_load to avoid security issues (GA best practice)
                config_data = yaml.safe_load(f)
            
            if not config_data:
                return {}
            
            # YAML migration is more complex due to nesting. 
            # For GA, we'll implement a flattened key-to-key mapper.
            migrated_data = self._transform_dict(config_data, changes)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(migrated_data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            print(f"Error migrating YAML file {source_path}: {e}")
            
        return changes

    def _transform_dict(self, data: Dict, changes: Dict, prefix: str = "") -> Dict:
        """
        Recursively transforms nested dictionaries (YAML) using RAG rules.
        """
        if not isinstance(data, dict):
            return data
            
        new_dict = {}
        for k, v in data.items():
            full_key = f"{prefix}{k}"
            # Expert RAG search for the specific property path
            rules = self.kb.search_configuration(full_key)
            
            if rules:
                new_key_path = rules[0].micronaut_pattern
                # Use the last part of the dot-separated path as the new key
                new_key = new_key_path.split('.')[-1]
                changes[full_key] = new_key_path
                print(f"      [VDB Found] {full_key} -> {new_key_path}", flush=True)
                new_dict[new_key] = self._transform_dict(v, changes, f"{full_key}.")
            elif k == "spring":
                # Standard Spring to Micronaut top-level migration
                new_dict["micronaut"] = self._transform_dict(v, changes, "spring.")
                changes["spring"] = "micronaut"
                print(f"      [Rule Match] spring -> micronaut (Local Fallback)", flush=True)
            elif full_key.startswith("spring."):
                # Deterministic fallback for spring properties
                new_full_key = full_key.replace("spring.", "micronaut.", 1)
                new_key = new_full_key.split('.')[-1]
                new_dict[new_key] = self._transform_dict(v, changes, f"{full_key}.")
                changes[full_key] = new_full_key
                print(f"      [Rule Match] {full_key} -> {new_full_key} (Local Fallback)", flush=True)
            else:
                new_dict[k] = self._transform_dict(v, changes, f"{full_key}.")
                
        return new_dict
