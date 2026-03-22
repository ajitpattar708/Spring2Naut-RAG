import argparse
import base64
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from src.agent.core.config import SecurityConfig
from src.agent.rag.audit import iter_rules


@dataclass
class CleaningSummary:
    input_rules: int
    output_rules: int
    dropped_self_maps: int
    dropped_exact_duplicates: int
    category_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "input_rules": self.input_rules,
            "output_rules": self.output_rules,
            "dropped_self_maps": self.dropped_self_maps,
            "dropped_exact_duplicates": self.dropped_exact_duplicates,
            "category_counts": self.category_counts,
        }


def infer_category(rule: Dict[str, object]) -> str:
    migration_type = str(rule.get("migration_type", "")).strip().lower()
    spring_pattern = str(rule.get("spring_pattern", "")).strip()

    if migration_type == "annotation" or spring_pattern.startswith("@"):
        return "annotations"
    if migration_type == "configuration" or spring_pattern.startswith(("spring.", "server.", "management.", "logging.")):
        return "configurations"
    if migration_type == "type":
        return "types"
    if migration_type == "dependency":
        return "dependencies"
    if ":" in spring_pattern and "@" not in spring_pattern:
        return "dependencies"
    return "code_patterns"


def normalize_rule(rule: Dict[str, object]) -> Dict[str, object]:
    normalized = dict(rule)
    normalized["spring_pattern"] = str(rule.get("spring_pattern", "")).strip()
    normalized["micronaut_pattern"] = str(rule.get("micronaut_pattern", "")).strip()
    normalized["description"] = str(rule.get("description", "")).strip()
    normalized["category"] = infer_category(rule)
    return normalized


def _dedupe_key(rule: Dict[str, object]) -> Tuple[str, str, str, str, str]:
    return (
        str(rule.get("category", "code_patterns")).strip() or "code_patterns",
        str(rule.get("spring_pattern", "")).strip(),
        str(rule.get("micronaut_pattern", "")).strip(),
        str(rule.get("spring_version", "")).strip(),
        str(rule.get("micronaut_version", "")).strip(),
    )


def clean_rules(rules: Iterable[Dict[str, object]]) -> Tuple[List[Dict[str, object]], CleaningSummary]:
    cleaned_rules: List[Dict[str, object]] = []
    seen_keys = set()
    dropped_self_maps = 0
    dropped_exact_duplicates = 0
    category_counter: Counter[str] = Counter()

    materialized_rules = list(iter_rules(list(rules)))
    for raw_rule in materialized_rules:
        rule = normalize_rule(raw_rule)
        spring_pattern = rule["spring_pattern"]
        micronaut_pattern = rule["micronaut_pattern"]

        if not spring_pattern or not micronaut_pattern:
            dropped_self_maps += 1
            continue

        if spring_pattern == micronaut_pattern:
            dropped_self_maps += 1
            continue

        key = _dedupe_key(rule)
        if key in seen_keys:
            dropped_exact_duplicates += 1
            continue

        seen_keys.add(key)
        cleaned_rules.append(rule)
        category_counter[rule["category"]] += 1

    summary = CleaningSummary(
        input_rules=len(materialized_rules),
        output_rules=len(cleaned_rules),
        dropped_self_maps=dropped_self_maps,
        dropped_exact_duplicates=dropped_exact_duplicates,
        category_counts=dict(sorted(category_counter.items())),
    )
    return cleaned_rules, summary


def load_dataset(path: str):
    from src.agent.rag.knowledge_base import LocalMigrationKnowledgeBase

    loader = LocalMigrationKnowledgeBase.__new__(LocalMigrationKnowledgeBase)
    return loader.load_dataset(path)


def load_dataset_with_status(path: str):
    from src.agent.rag.knowledge_base import LocalMigrationKnowledgeBase

    loader = LocalMigrationKnowledgeBase.__new__(LocalMigrationKnowledgeBase)
    return loader.load_dataset_with_status(path)


def write_json_dataset(path: str, rules: List[Dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(rules, handle, indent=2)


def write_encrypted_dataset(path: str, rules: List[Dict[str, object]]) -> None:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    salt = b"spring2naut_rag_migration_2024"
    password = SecurityConfig.get_dataset_key().encode("utf-8")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend(),
    )
    key = base64.urlsafe_b64encode(kdf.derive(password))
    fernet = Fernet(key)
    payload = json.dumps(rules, ensure_ascii=True).encode("utf-8")

    with output_path.open("wb") as handle:
        handle.write(fernet.encrypt(payload))


def clean_dataset_file(input_path: str) -> Tuple[List[Dict[str, object]], CleaningSummary]:
    dataset = load_dataset(input_path)
    return clean_rules(dataset if isinstance(dataset, list) else iter_rules(dataset))


def build_default_output_path(input_path: str) -> str:
    path = Path(input_path)
    if path.suffix == ".dat":
        stem = path.name[:-4]
        return str(path.with_name(f"{stem}_cleaned.dat"))
    return str(path.with_name(f"{path.stem}_cleaned.json"))


def main():
    parser = argparse.ArgumentParser(description="Clean and regenerate migration datasets")
    parser.add_argument("--input", default="migration_dataset_enhanced.json.dat", help="Encrypted or plain dataset input")
    parser.add_argument("--output", default=None, help="Output dataset path")
    parser.add_argument("--format", choices=["json", "dat"], default="dat", help="Output format")
    parser.add_argument("--report", default=None, help="Optional JSON report output path")
    args = parser.parse_args()

    cleaned_rules, summary = clean_dataset_file(args.input)
    output_path = args.output or build_default_output_path(args.input)

    if args.format == "json":
        write_json_dataset(output_path, cleaned_rules)
    else:
        write_encrypted_dataset(output_path, cleaned_rules)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")

    print(json.dumps({"output_path": output_path, **summary.to_dict()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
