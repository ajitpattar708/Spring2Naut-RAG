import os
import re
import shutil
from typing import Dict, List, Optional, Tuple
from src.agent.core.interfaces import KnowledgeService

class ConfigAgent:
    """
    Expert agent for migrating application configuration files (Properties, YAML).
    Handles mapping of Spring properties to Micronaut equivalents.
    """
    
    def __init__(self, knowledge_base: KnowledgeService, spring_version: str = "3.x", micronaut_version: str = "4.x"):
        self.kb = knowledge_base
        self.spring_version = spring_version
        self.micronaut_version = micronaut_version
        self.explicit_property_mappings = {
            "spring.datasource.url": "datasources.default.url",
            "spring.datasource.username": "datasources.default.username",
            "spring.datasource.password": "datasources.default.password",
            "spring.datasource.driver-class-name": "datasources.default.driver-class-name",
            "spring.datasource.driverClassName": "datasources.default.driver-class-name",
            "spring.jpa.hibernate.ddl-auto": "jpa.default.properties.hibernate.hbm2ddl.auto",
        }
        self._embedded_database_tokens = ("h2", "hsqldb", "derby")

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
            migrated_keys = set()
            raw_values = {}
            for line in lines:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    raw_values[key] = value.strip()
                    if key.startswith("spring.sql.init."):
                        continue
                    if key == "management.endpoints.web.exposure.include":
                        management_changes, management_lines = self._build_management_endpoint_lines(value.strip())
                        migrated_lines.extend(management_lines)
                        changes.update(management_changes)
                        migrated_keys.update(
                            {
                                mapped_key
                                for mapped_key in management_changes.values()
                                if isinstance(mapped_key, str) and "." in mapped_key
                            }
                        )
                        continue
                    # RAG Search for property replacement
                    rules = self.kb.search_configuration(
                        key,
                        spring_version=self.spring_version,
                        micronaut_version=self.micronaut_version,
                    )
                    if rules:
                        new_key = rules[0].micronaut_pattern
                        migrated_lines.append(f"{new_key}={value}")
                        changes[key] = new_key
                        migrated_keys.add(new_key)
                    elif key in self.explicit_property_mappings:
                        new_key = self.explicit_property_mappings[key]
                        migrated_lines.append(f"{new_key}={value}")
                        changes[key] = new_key
                        migrated_keys.add(new_key)
                    else:
                        # Fallback: simple prefix replacement if no specific rule
                        if key.startswith('spring.'):
                            new_key = key.replace('spring.', 'micronaut.', 1)
                            migrated_lines.append(f"{new_key}={value}")
                            changes[key] = new_key
                            migrated_keys.add(new_key)
                        else:
                            migrated_lines.append(line)
                            migrated_keys.add(key)
                else:
                    migrated_lines.append(line)

            sql_init_changes, sql_init_lines = self._build_sql_init_property_lines(raw_values)
            if sql_init_lines:
                if migrated_lines and migrated_lines[-1].strip():
                    migrated_lines.append("\n")
                migrated_lines.extend(sql_init_lines)
                changes.update(sql_init_changes)
                migrated_keys.update(
                    {
                        value
                        for value in sql_init_changes.values()
                        if isinstance(value, str) and "." in value
                    }
                )

            if raw_values.get("database") == "h2" and "datasources.default.url" not in migrated_keys:
                datasource_url = (
                    "jdbc:h2:mem:petclinic;DB_CLOSE_DELAY=-1;DB_CLOSE_ON_EXIT=FALSE"
                    if sql_init_lines
                    else "jdbc:h2:mem:petclinic;DB_CLOSE_DELAY=-1;DB_CLOSE_ON_EXIT=FALSE;"
                    "INIT=RUNSCRIPT FROM 'classpath:/db/h2/schema.sql'\\\\;RUNSCRIPT FROM 'classpath:/db/h2/data.sql'"
                )
                migrated_lines.extend(
                    [
                        "\n",
                        "# Generated Micronaut embedded datasource defaults for H2\n",
                        f"datasources.default.url={datasource_url}\n",
                        "datasources.default.driver-class-name=org.h2.Driver\n",
                        "datasources.default.username=sa\n",
                        "datasources.default.password=\n",
                        "datasources.default.schema-generate=NONE\n",
                    ]
                )
                changes["database"] = "datasources.default.* (generated H2 defaults)"

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.writelines(migrated_lines)
        except Exception as e:
            print(f"Error migrating properties file {source_path}: {e}")
            
        return changes

    def collect_sql_init_resource_mappings(self, config_files: List[str]) -> List[Tuple[str, str]]:
        properties_files = [
            path
            for path in config_files
            if path.endswith(".properties")
            and f"{os.sep}src{os.sep}main{os.sep}resources{os.sep}" in path
            and os.path.basename(path).startswith("application")
        ]
        if not properties_files:
            return []

        file_properties: Dict[str, Dict[str, str]] = {}
        for path in properties_files:
            try:
                file_properties[path] = self._read_properties_file(path)
            except OSError:
                continue

        base_properties: Dict[str, str] = {}
        for path, props in file_properties.items():
            if os.path.basename(path) == "application.properties":
                base_properties = dict(props)
                break

        mappings: List[Tuple[str, str]] = []
        seen_pairs = set()
        for path, props in file_properties.items():
            merged = dict(base_properties)
            merged.update(props)
            version_counter = 1
            locations_by_kind = (
                ("schema", merged.get("spring.sql.init.schema-locations", "")),
                ("data", merged.get("spring.sql.init.data-locations", "")),
            )
            for kind, raw_location in locations_by_kind:
                for index, location in enumerate(self._split_sql_locations(raw_location), start=1):
                    resolved_location = self._resolve_property_placeholders(location, merged)
                    source_relative_path = self._classpath_sql_location_to_relative_path(resolved_location)
                    if not source_relative_path:
                        continue
                    migration_relative_path = self._migration_relative_path_for_sql(
                        source_relative_path,
                        kind,
                        version_counter,
                        index,
                    )
                    pair = (source_relative_path, migration_relative_path)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    mappings.append(pair)
                    version_counter += 1
        return mappings

    def materialize_sql_init_migrations(
        self,
        input_root: str,
        output_root: str,
        config_files: List[str],
    ) -> int:
        generated = 0
        for source_relative_path, migration_relative_path in self.collect_sql_init_resource_mappings(config_files):
            source_path = os.path.join(output_root, "src", "main", "resources", source_relative_path)
            if not os.path.exists(source_path):
                fallback_source_path = os.path.join(input_root, "src", "main", "resources", source_relative_path)
                if not os.path.exists(fallback_source_path):
                    continue
                os.makedirs(os.path.dirname(source_path), exist_ok=True)
                shutil.copy2(fallback_source_path, source_path)

            destination_path = os.path.join(output_root, "src", "main", "resources", migration_relative_path)
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            with open(source_path, "r", encoding="utf-8") as handle:
                sql_content = handle.read()
            if f"{os.sep}db{os.sep}h2{os.sep}" in source_path:
                sql_content = self._normalize_h2_sql_resource(sql_content)
            with open(destination_path, "w", encoding="utf-8") as handle:
                handle.write(sql_content)
            generated += 1
        return generated

    def _migrate_yaml(self, source_path: str, output_path: str) -> Dict[str, str]:
        changes = {}
        try:
            import yaml

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
            rules = self.kb.search_configuration(
                full_key,
                spring_version=self.spring_version,
                micronaut_version=self.micronaut_version,
            )
            
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

    def _build_sql_init_property_lines(self, raw_values: Dict[str, str]) -> Tuple[Dict[str, str], List[str]]:
        schema_locations = raw_values.get("spring.sql.init.schema-locations", "").strip()
        data_locations = raw_values.get("spring.sql.init.data-locations", "").strip()
        mode = raw_values.get("spring.sql.init.mode", "").strip()
        if not schema_locations and not data_locations and not mode:
            return {}, []

        changes: Dict[str, str] = {}
        generated_lines = [
            "# Generated Flyway compatibility mapping for Spring SQL initialization\n",
        ]

        enabled = self._resolve_sql_init_enabled(raw_values)
        generated_lines.append(
            f"flyway.datasources.default.enabled={'true' if enabled else 'false'}\n"
        )
        if mode:
            changes["spring.sql.init.mode"] = "flyway.datasources.default.enabled"

        location_mappings: List[str] = []
        for key in ("spring.sql.init.schema-locations", "spring.sql.init.data-locations"):
            raw_location = raw_values.get(key, "").strip()
            if not raw_location:
                continue
            location_mappings.extend(
                self._migration_locations_for_sql_init(raw_location)
            )
            changes[key] = "flyway.datasources.default.locations"

        deduped_locations = list(dict.fromkeys(location_mappings))
        if deduped_locations:
            generated_lines.append(
                f"flyway.datasources.default.locations={','.join(deduped_locations)}\n"
            )

        return changes, generated_lines

    def _build_management_endpoint_lines(self, raw_value: str) -> Tuple[Dict[str, str], List[str]]:
        includes = [item.strip() for item in str(raw_value or "").split(",") if item.strip()]
        if not includes:
            return {}, []

        changes: Dict[str, str] = {}
        generated_lines: List[str] = []
        if includes == ["*"]:
            generated_lines.append("endpoints.all.enabled=true\n")
            changes["management.endpoints.web.exposure.include"] = "endpoints.all.enabled"
            return changes, generated_lines

        for endpoint in includes:
            normalized = endpoint.replace("-", "")
            generated_lines.append(f"endpoints.{normalized}.enabled=true\n")
        changes["management.endpoints.web.exposure.include"] = "endpoints.<id>.enabled"
        return changes, generated_lines

    def _resolve_sql_init_enabled(self, raw_values: Dict[str, str]) -> bool:
        mode = raw_values.get("spring.sql.init.mode", "").strip().lower()
        if mode == "never":
            return False
        if mode == "always":
            return True
        if mode == "embedded":
            return self._looks_like_embedded_database(raw_values)
        if raw_values.get("spring.sql.init.schema-locations") or raw_values.get("spring.sql.init.data-locations"):
            return self._looks_like_embedded_database(raw_values)
        return False

    def _looks_like_embedded_database(self, raw_values: Dict[str, str]) -> bool:
        database = raw_values.get("database", "").strip().lower()
        if any(token in database for token in self._embedded_database_tokens):
            return True

        datasource_url = raw_values.get("spring.datasource.url", "").strip().lower()
        return any(f"jdbc:{token}:" in datasource_url for token in self._embedded_database_tokens)

    def _migration_locations_for_sql_init(self, raw_locations: str) -> List[str]:
        mapped_locations: List[str] = []
        for location in self._split_sql_locations(raw_locations):
            relative_path = self._classpath_sql_location_to_relative_path(location)
            if not relative_path:
                continue
            parent_dir = os.path.dirname(relative_path).replace("\\", "/").strip("/")
            if parent_dir.startswith("db/"):
                target_dir = f"db/migration/{parent_dir[len('db/'):]}"
            elif parent_dir:
                target_dir = f"db/migration/{parent_dir}"
            else:
                target_dir = "db/migration"
            mapped_locations.append(f"classpath:{target_dir}")
        return mapped_locations

    def _read_properties_file(self, path: str) -> Dict[str, str]:
        properties: Dict[str, str] = {}
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                properties[key.strip()] = value.strip()
        return properties

    def _split_sql_locations(self, raw_locations: str) -> List[str]:
        return [item.strip() for item in str(raw_locations or "").split(",") if item.strip()]

    def _resolve_property_placeholders(self, value: str, properties: Dict[str, str]) -> str:
        resolved = str(value or "")
        for _ in range(8):
            changed = False

            def replace(match: re.Match) -> str:
                nonlocal changed
                key = match.group(1)
                default_value = match.group(2) or ""
                replacement = properties.get(key, default_value)
                if replacement == match.group(0):
                    return replacement
                changed = True
                return replacement

            updated = re.sub(r"\$\{([^}:]+)(?::([^}]*))?\}", replace, resolved)
            resolved = updated
            if not changed:
                break
        return resolved

    def _classpath_sql_location_to_relative_path(self, location: str) -> str:
        normalized = str(location or "").strip()
        if not normalized:
            return ""
        normalized = normalized.replace("classpath*:", "").replace("classpath:", "").strip()
        normalized = normalized.lstrip("/")
        normalized = normalized.replace("\\", "/")
        normalized = normalized.replace("*", "")
        while "//" in normalized:
            normalized = normalized.replace("//", "/")
        if not normalized.endswith(".sql"):
            return ""
        return normalized

    def _migration_relative_path_for_sql(
        self,
        source_relative_path: str,
        kind: str,
        version: int,
        kind_index: int,
    ) -> str:
        normalized = source_relative_path.replace("\\", "/").strip("/")
        parent_dir = os.path.dirname(normalized).strip("/")
        if parent_dir.startswith("db/"):
            migration_parent = f"db/migration/{parent_dir[len('db/'):]}"
        elif parent_dir:
            migration_parent = f"db/migration/{parent_dir}"
        else:
            migration_parent = "db/migration"
        suffix = f"{kind}_{kind_index}" if kind_index > 1 else kind
        filename = f"V{version}__{suffix}.sql"
        return f"{migration_parent}/{filename}"

    def _normalize_h2_sql_resource(self, content: str) -> str:
        normalized = str(content or "")
        normalized = normalized.replace("AUTO_INCREMENT", "GENERATED BY DEFAULT AS IDENTITY")
        normalized = normalized.replace("auto_increment", "GENERATED BY DEFAULT AS IDENTITY")
        normalized = normalized.replace("ENGINE=InnoDB", "")
        normalized = normalized.replace("engine=InnoDB", "")
        return normalized
