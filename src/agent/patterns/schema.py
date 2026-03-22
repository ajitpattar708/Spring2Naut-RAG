from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.agent.core.models import MigrationRule
from src.agent.core.versioning import compare_versions, matches_version_spec


class PatternType(str, Enum):
    ANNOTATION = "annotation"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    DEPENDENCY_INJECTION = "dependency_injection"
    TYPE = "type"
    APPLICATION = "application"
    CODE_PATTERN = "code_pattern"
    IMPORT = "import"


class SourceKind(str, Enum):
    OFFICIAL_DOC = "official_doc"
    GITHUB_REPO = "github_repo"
    MANUAL = "manual"
    GENERATED = "generated"
    TEST_FIXTURE = "test_fixture"


class ValidationStatus(str, Enum):
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"

@dataclass(frozen=True)
class VersionWindow:
    spec: Optional[str] = None
    minimum: Optional[str] = None
    maximum: Optional[str] = None

    def includes(self, version: str) -> bool:
        if self.spec and not matches_version_spec(version, self.spec):
            return False

        if self.minimum and compare_versions(version, self.minimum) < 0:
            return False

        if self.maximum and compare_versions(version, self.maximum) > 0:
            return False

        return True

    def validate(self) -> List[str]:
        errors: List[str] = []

        if self.minimum and self.maximum and compare_versions(self.minimum, self.maximum) > 0:
            errors.append("Version window minimum cannot be greater than maximum.")

        if self.minimum and any(token in self.minimum.lower() for token in ("x", "*")):
            errors.append("Version window minimum must be a concrete version.")

        if self.maximum and any(token in self.maximum.lower() for token in ("x", "*")):
            errors.append("Version window maximum must be a concrete version.")

        return errors


@dataclass(frozen=True)
class PatternEvidence:
    source_kind: SourceKind
    source_ref: str
    title: Optional[str] = None
    notes: Optional[str] = None
    retrieved_on: Optional[str] = None

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.source_ref.strip():
            errors.append("Evidence source_ref cannot be empty.")
        return errors


@dataclass(frozen=True)
class VersionedPattern:
    pattern_id: str
    pattern_type: PatternType
    spring_pattern: str
    micronaut_pattern: str
    description: str
    spring_versions: VersionWindow = field(default_factory=VersionWindow)
    micronaut_versions: VersionWindow = field(default_factory=VersionWindow)
    status: ValidationStatus = ValidationStatus.CANDIDATE
    confidence: float = 0.0
    complexity: str = "medium"
    category: str = "code_patterns"
    source_kind: SourceKind = SourceKind.MANUAL
    evidence: List[PatternEvidence] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        errors: List[str] = []

        if not self.pattern_id.strip():
            errors.append("pattern_id cannot be empty.")
        if not self.spring_pattern.strip():
            errors.append("spring_pattern cannot be empty.")
        if not self.micronaut_pattern.strip():
            errors.append("micronaut_pattern cannot be empty.")
        if self.spring_pattern.strip() == self.micronaut_pattern.strip():
            errors.append("spring_pattern and micronaut_pattern cannot be identical.")
        if not 0.0 <= self.confidence <= 1.0:
            errors.append("confidence must be between 0.0 and 1.0.")

        errors.extend(self.spring_versions.validate())
        errors.extend(self.micronaut_versions.validate())

        for item in self.evidence:
            errors.extend(item.validate())

        if self.status == ValidationStatus.VALIDATED and not self.evidence:
            errors.append("validated patterns must include at least one evidence record.")

        return errors

    def matches_versions(self, spring_version: str, micronaut_version: str) -> bool:
        return self.spring_versions.includes(spring_version) and self.micronaut_versions.includes(micronaut_version)

    def to_migration_rule(self) -> MigrationRule:
        return MigrationRule(
            id=self.pattern_id,
            migration_type=self.pattern_type.value,
            spring_pattern=self.spring_pattern,
            micronaut_pattern=self.micronaut_pattern,
            category=self.category,
            description=self.description,
            complexity=self.complexity,
            spring_version=self.spring_versions.spec or self.spring_versions.minimum,
            micronaut_version=self.micronaut_versions.spec or self.micronaut_versions.minimum,
            metadata={
                **self.metadata,
                "source_kind": self.source_kind.value,
                "status": self.status.value,
                "confidence": self.confidence,
                "evidence_count": len(self.evidence),
                "spring_version_window": asdict(self.spring_versions),
                "micronaut_version_window": asdict(self.micronaut_versions),
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["pattern_type"] = self.pattern_type.value
        payload["status"] = self.status.value
        payload["source_kind"] = self.source_kind.value
        payload["spring_versions"] = asdict(self.spring_versions)
        payload["micronaut_versions"] = asdict(self.micronaut_versions)
        payload["evidence"] = [
            {
                **asdict(item),
                "source_kind": item.source_kind.value,
            }
            for item in self.evidence
        ]
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VersionedPattern":
        evidence = [
            PatternEvidence(
                source_kind=SourceKind(item["source_kind"]),
                source_ref=item["source_ref"],
                title=item.get("title"),
                notes=item.get("notes"),
                retrieved_on=item.get("retrieved_on"),
            )
            for item in payload.get("evidence", [])
        ]

        return cls(
            pattern_id=payload["pattern_id"],
            pattern_type=PatternType(payload["pattern_type"]),
            spring_pattern=payload["spring_pattern"],
            micronaut_pattern=payload["micronaut_pattern"],
            description=payload["description"],
            spring_versions=VersionWindow(**payload.get("spring_versions", {})),
            micronaut_versions=VersionWindow(**payload.get("micronaut_versions", {})),
            status=ValidationStatus(payload.get("status", ValidationStatus.CANDIDATE.value)),
            confidence=float(payload.get("confidence", 0.0)),
            complexity=payload.get("complexity", "medium"),
            category=payload.get("category", "code_patterns"),
            source_kind=SourceKind(payload.get("source_kind", SourceKind.MANUAL.value)),
            evidence=evidence,
            examples=list(payload.get("examples", [])),
            metadata=dict(payload.get("metadata", {})),
        )
