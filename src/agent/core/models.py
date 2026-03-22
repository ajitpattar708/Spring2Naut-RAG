from dataclasses import asdict, dataclass
from typing import List, Dict, Any, Optional

from src.agent.core.versioning import includes_version, matches_version_spec, normalize_major_minor

@dataclass
class MigrationRule:
    """
    Represents a single migration rule or pattern.
    Contains both simple mapping and full code examples.
    """
    spring_pattern: str
    micronaut_pattern: str
    category: str
    description: str
    complexity: str
    example_spring: Optional[str] = None
    example_micronaut: Optional[str] = None
    id: Optional[str] = None
    migration_type: Optional[str] = None
    spring_code: Optional[str] = None
    micronaut_code: Optional[str] = None
    source_framework: Optional[str] = None
    target_framework: Optional[str] = None
    spring_version: Optional[str] = None
    micronaut_version: Optional[str] = None
    explanation: Optional[str] = None
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ProjectStructure:
    """
    Represents the analyzed structure of the source project.
    Used by the orchestrator to coordinate agents.
    """
    root_path: str
    source_files: List[str]
    config_files: List[str]
    dependency_file: Optional[str]
    build_tool: str # maven or gradle
    project_root: Optional[str] = None
    relative_project_root: str = "."
    build_tool_forced: bool = False

@dataclass
class MigrationReport:
    """
    Summary of the migration process and results.
    Detailed statistics on file conversions and changes made.
    """
    total_files: int
    migrated_files: int
    failed_files: List[str]
    warnings: List[str]
    dependency_changes: Dict[str, str]
    config_changes: Dict[str, str]
    dependency_audit: Optional[Dict[str, Any]] = None
    dependency_audit_report_path: Optional[str] = None
    dependency_inventory_report_path: Optional[str] = None
    migrated_dependency_audit: Optional[Dict[str, Any]] = None
    migrated_dependency_audit_report_path: Optional[str] = None
    migrated_dependency_inventory_report_path: Optional[str] = None
    input_dir: Optional[str] = None
    output_dir: Optional[str] = None
    build_tool: Optional[str] = None
    spring_version: Optional[str] = None
    micronaut_version: Optional[str] = None
    status: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    validation_success: Optional[bool] = None
    validation_status: Optional[str] = None
    validation_attempts: int = 0
    verification_summary: Optional[Dict[str, Any]] = None
    verification_report_path: Optional[str] = None
    migration_report_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class VersionCompatibilityMatrix:
    """
    Handles version-specific logic and compatibility matching.
    Ensures correct patterns are applied based on Spring and Micronaut versions.
    """
    # Matrix of known compatibility issues and patterns
    API_COMPATIBILITY = {
        ("3.x", "4.10"): {
            "deprecated_apis": [],
            "new_apis": ["@Requires", "@EachProperty"],
            "breaking_changes": ["@ConfigurationProperties review for Micronaut 4.10.x"],
            "version_specific_patterns": {
                "@ConfigurationProperties": {
                    "replacement": "@EachProperty",
                    "note": "Micronaut 4.10.x tightened configuration binding expectations. Review each @ConfigurationProperties usage and prefer @EachProperty for repeated/named child configurations instead of assuming a direct one-to-one mapping."
                }
            }
        },
        ("3.x", "4.5"): {
            "deprecated_apis": [],
            "new_apis": ["@Requires"],
            "breaking_changes": ["@ConfigurationProperties review for Micronaut 4.5.x"],
            "version_specific_patterns": {
                "@ConfigurationProperties": {
                    "replacement": "@ConfigurationProperties",
                    "note": "Micronaut 4.5.x configuration binding should be reviewed during migration to confirm prefix semantics and bean registration still match the source Spring design."
                }
            }
        },
    }

    @staticmethod
    def normalize_version(version: str) -> str:
        """
        Normalizes a version string to its major.minor components.
        Used for broader matching across patch versions.
        """
        return normalize_major_minor(version)

    @staticmethod
    def _window_from_metadata(rule: MigrationRule, prefix: str) -> Dict[str, Optional[str]]:
        metadata = rule.metadata if isinstance(rule.metadata, dict) else {}
        window = metadata.get(f"{prefix}_version_window")
        if isinstance(window, dict):
            return {
                "spec": window.get("spec"),
                "minimum": window.get("minimum"),
                "maximum": window.get("maximum"),
            }

        return {
            "spec": metadata.get(f"{prefix}_version_spec"),
            "minimum": metadata.get(f"{prefix}_version_minimum"),
            "maximum": metadata.get(f"{prefix}_version_maximum"),
        }

    @staticmethod
    def _apply_legacy_rule_version(
        window: Dict[str, Optional[str]],
        rule_version: Optional[str],
    ) -> Dict[str, Optional[str]]:
        if not rule_version:
            return window
        if window["spec"] or window["minimum"] or window["maximum"]:
            return window

        if any(token in rule_version.lower() for token in ("x", "*")):
            window["spec"] = rule_version
        else:
            window["minimum"] = rule_version
        return window

    @staticmethod
    def get_compatibility_info(spring_version: str, micronaut_version: str) -> dict:
        """
        Retrieves version-specific compatibility details.
        """
        spring_norm = VersionCompatibilityMatrix.normalize_version(spring_version)
        micronaut_norm = VersionCompatibilityMatrix.normalize_version(micronaut_version)
        
        # Try exact match first
        key = (spring_version, micronaut_version)
        if key in VersionCompatibilityMatrix.API_COMPATIBILITY:
            return VersionCompatibilityMatrix.API_COMPATIBILITY[key]
        
        # Fallback to normalized variations
        key_norm = (spring_norm, micronaut_norm)
        if key_norm in VersionCompatibilityMatrix.API_COMPATIBILITY:
            return VersionCompatibilityMatrix.API_COMPATIBILITY[key_norm]

        for (spring_spec, micronaut_spec), details in VersionCompatibilityMatrix.API_COMPATIBILITY.items():
            if matches_version_spec(spring_version, spring_spec) and matches_version_spec(micronaut_version, micronaut_spec):
                return details
            if matches_version_spec(spring_norm, spring_spec) and matches_version_spec(micronaut_norm, micronaut_spec):
                return details
        
        return {
            "deprecated_apis": [],
            "new_apis": [],
            "breaking_changes": [],
            "version_specific_patterns": {}
        }

    @staticmethod
    def is_version_compatible(rule: MigrationRule, spring_version: str, micronaut_version: str) -> bool:
        """
        Determines if a specific rule is applicable for the given project versions.
        """
        if not rule.spring_version and not rule.micronaut_version:
            spring_window = VersionCompatibilityMatrix._window_from_metadata(rule, "spring")
            micronaut_window = VersionCompatibilityMatrix._window_from_metadata(rule, "micronaut")
            if not any(spring_window.values()) and not any(micronaut_window.values()):
                return True
        else:
            spring_window = VersionCompatibilityMatrix._window_from_metadata(rule, "spring")
            micronaut_window = VersionCompatibilityMatrix._window_from_metadata(rule, "micronaut")

        spring_window = VersionCompatibilityMatrix._apply_legacy_rule_version(spring_window, rule.spring_version)
        micronaut_window = VersionCompatibilityMatrix._apply_legacy_rule_version(micronaut_window, rule.micronaut_version)

        return includes_version(
            spring_version,
            spec=spring_window["spec"],
            minimum=spring_window["minimum"],
            maximum=spring_window["maximum"],
        ) and includes_version(
            micronaut_version,
            spec=micronaut_window["spec"],
            minimum=micronaut_window["minimum"],
            maximum=micronaut_window["maximum"],
        )
