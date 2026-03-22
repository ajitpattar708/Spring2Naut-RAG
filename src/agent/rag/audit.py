import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


VALID_CATEGORIES = {
    "annotations",
    "dependencies",
    "configurations",
    "code_patterns",
    "imports",
    "types",
}

REQUIRED_RULE_FIELDS = ("spring_pattern", "micronaut_pattern", "description")


@dataclass
class AuditIssue:
    severity: str
    message: str
    rule_id: Optional[str] = None


@dataclass
class AuditReport:
    total_rules: int = 0
    valid_rules: int = 0
    invalid_rules: int = 0
    category_counts: Dict[str, int] = field(default_factory=dict)
    duplicate_rule_ids: List[str] = field(default_factory=list)
    duplicate_source_patterns: List[str] = field(default_factory=list)
    issues: List[AuditIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.invalid_rules == 0 and not self.duplicate_rule_ids

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_rules": self.total_rules,
            "valid_rules": self.valid_rules,
            "invalid_rules": self.invalid_rules,
            "category_counts": self.category_counts,
            "duplicate_rule_ids": self.duplicate_rule_ids,
            "duplicate_source_patterns": self.duplicate_source_patterns,
            "issues": [issue.__dict__ for issue in self.issues],
            "is_valid": self.is_valid,
        }


def iter_rules(dataset) -> List[Dict[str, object]]:
    if dataset is None:
        return []

    if isinstance(dataset, list):
        return [rule for rule in dataset if isinstance(rule, dict)]

    if isinstance(dataset, dict):
        rules: List[Dict[str, object]] = []
        for category, values in dataset.items():
            if not isinstance(values, list):
                continue
            for rule in values:
                if not isinstance(rule, dict):
                    continue
                enriched = dict(rule)
                enriched.setdefault("category", category)
                rules.append(enriched)
        return rules

    return []


def _validate_rule(rule: Dict[str, object]) -> List[str]:
    errors: List[str] = []

    for field_name in REQUIRED_RULE_FIELDS:
        value = rule.get(field_name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"Missing required field: {field_name}")

    category = str(rule.get("category", "code_patterns")).strip()
    if category not in VALID_CATEGORIES:
        errors.append(f"Unknown category: {category}")

    spring_pattern = str(rule.get("spring_pattern", "")).strip()
    micronaut_pattern = str(rule.get("micronaut_pattern", "")).strip()
    if spring_pattern and micronaut_pattern and spring_pattern == micronaut_pattern:
        errors.append("Source and target patterns are identical")

    return errors


def audit_rules(rules: Iterable[Dict[str, object]]) -> AuditReport:
    materialized_rules = list(rules)
    report = AuditReport(total_rules=len(materialized_rules))

    id_counter: Counter[str] = Counter()
    pattern_counter: Counter[Tuple[str, str]] = Counter()
    category_counter: Counter[str] = Counter()

    for rule in materialized_rules:
        category = str(rule.get("category", "code_patterns")).strip() or "code_patterns"
        category_counter[category] += 1

        rule_id = str(rule.get("id", "")).strip()
        if rule_id:
            id_counter[rule_id] += 1

        spring_pattern = str(rule.get("spring_pattern", "")).strip()
        if spring_pattern:
            pattern_counter[(category, spring_pattern)] += 1

        errors = _validate_rule(rule)
        if errors:
            report.invalid_rules += 1
            for error in errors:
                report.issues.append(AuditIssue(severity="error", message=error, rule_id=rule_id or None))
        else:
            report.valid_rules += 1

    report.category_counts = dict(sorted(category_counter.items()))
    report.duplicate_rule_ids = sorted(rule_id for rule_id, count in id_counter.items() if count > 1)
    report.duplicate_source_patterns = sorted(
        f"{category}:{pattern}" for (category, pattern), count in pattern_counter.items() if count > 1
    )

    for duplicate_rule_id in report.duplicate_rule_ids:
        report.invalid_rules += 1
        report.valid_rules = max(0, report.valid_rules - 1)
        report.issues.append(
            AuditIssue(severity="error", message=f"Duplicate rule id: {duplicate_rule_id}", rule_id=duplicate_rule_id)
        )

    for duplicate_pattern in report.duplicate_source_patterns:
        report.issues.append(AuditIssue(severity="warning", message=f"Duplicate source pattern: {duplicate_pattern}"))

    return report


def audit_dataset(dataset) -> AuditReport:
    return audit_rules(iter_rules(dataset))


def audit_dataset_file(path: str) -> AuditReport:
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    return audit_dataset(dataset)
