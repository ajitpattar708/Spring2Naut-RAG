from dataclasses import dataclass
from typing import List, Dict, Any, Optional

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
    dependency_file: str
    build_tool: str # maven or gradle

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

class VersionCompatibilityMatrix:
    """
    Handles version-specific logic and compatibility matching.
    Ensures correct patterns are applied based on Spring and Micronaut versions.
    """
    # Matrix of known compatibility issues and patterns
    API_COMPATIBILITY = {
        ("3.4.5", "4.10.1"): {
            "deprecated_apis": [],
            "new_apis": ["@Requires", "@EachProperty"],
            "breaking_changes": [],
            "version_specific_patterns": {
                "@ConfigurationProperties": {
                    "replacement": "@EachProperty",
                    "note": "In Micronaut 4.10.x, use @EachProperty for configuration properties"
                }
            }
        },
        # Add additional version pairs as needed for GA support
    }

    @staticmethod
    def normalize_version(version: str) -> str:
        """
        Normalizes a version string to its major.minor components.
        Used for broader matching across patch versions.
        """
        if not version or version in ["3.x", "4.x"]:
            return version
        
        parts = version.split('.')
        if len(parts) >= 2:
            return f"{parts[0]}.{parts[1]}"
        return version

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
            return True
        
        spring_norm = VersionCompatibilityMatrix.normalize_version(spring_version)
        micronaut_norm = VersionCompatibilityMatrix.normalize_version(micronaut_version)
        
        # Validation logic for versions
        # Logic ensures that we don't apply patterns from incompatible major versions
        # or patterns that require a newer target version than what is available.
        # (This is a placeholder for the more detailed logic from the original file)
        return True
