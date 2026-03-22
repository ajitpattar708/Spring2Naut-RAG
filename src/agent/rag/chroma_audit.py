import argparse
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from unittest.mock import patch

from src.agent.core.config import MigrationConfig
from src.agent.patterns.release import write_validated_release
from src.agent.rag.kb_release_smoke import _deterministic_model_init, _patched_runtime_dataset
from src.agent.rag.knowledge_base import LocalMigrationKnowledgeBase
from src.agent.rag.vector_store import create_persistent_client


AUDITED_COLLECTIONS = (
    "annotations",
    "dependencies",
    "configurations",
    "code_patterns",
    "imports",
    "types",
)
REQUIRED_METADATA_FIELDS = ("spring_pattern", "micronaut_pattern", "description", "category")
REQUIRED_ENTERPRISE_METADATA_FIELDS = (
    "source_kind",
    "status",
    "release_validation_status",
    "confidence",
    "evidence_count",
    "spring_version_spec",
    "micronaut_version_spec",
)
LOW_CONFIDENCE_THRESHOLD = 0.80


def _collection_payload(collection) -> Dict[str, object]:
    try:
        return collection.get(include=["metadatas", "documents"])
    except TypeError:
        return collection.get()


def _safe_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _safe_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = _safe_text(value)
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        text = _safe_text(value)
        return int(text) if text else None
    except (TypeError, ValueError):
        return None


def _derived_document_signature(metadata: Dict[str, object]) -> str:
    spring_pattern = _safe_text(metadata.get("spring_pattern"))
    description = _safe_text(metadata.get("description"))
    return f"{spring_pattern} {description}".strip()


def _is_generic_pattern(metadata: Dict[str, object]) -> bool:
    spring_pattern = _safe_text(metadata.get("spring_pattern")).lower()
    micronaut_pattern = _safe_text(metadata.get("micronaut_pattern")).lower()
    description = _safe_text(metadata.get("description")).lower()

    generic_patterns = {
        "spring",
        "spring boot",
        "spring service",
        "spring feature",
        "spring pattern",
        "micronaut",
        "micronaut framework",
        "micronaut service",
        "micronaut feature",
        "micronaut pattern",
    }
    generic_descriptions = {
        "generic migration",
        "general migration",
        "migration mapping",
        "framework migration",
        "basic migration",
    }

    return (
        spring_pattern in generic_patterns
        or micronaut_pattern in generic_patterns
        or description in generic_descriptions
    )


def _audit_collection(name: str, payload: Dict[str, object]) -> Dict[str, object]:
    ids = list(payload.get("ids") or [])
    metadatas = list(payload.get("metadatas") or [])
    documents = list(payload.get("documents") or [])

    total_records = len(ids)
    if not documents:
        documents = [None] * total_records

    missing_metadata_records: List[str] = []
    duplicate_mapping_pairs: List[str] = []
    duplicate_stored_documents: List[str] = []
    duplicate_derived_documents: List[str] = []
    conflicting_mappings: Dict[str, List[str]] = {}
    overly_generic_patterns: List[str] = []
    missing_enterprise_metadata_records: List[str] = []
    low_confidence_records: List[str] = []
    unvalidated_records: List[str] = []
    missing_version_window_records: List[str] = []
    weak_evidence_records: List[str] = []

    mapping_pair_counter: Counter[Tuple[str, str]] = Counter()
    stored_document_counter: Counter[str] = Counter()
    derived_document_counter: Counter[str] = Counter()
    targets_by_source: Dict[str, set[str]] = defaultdict(set)

    for index, rule_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
        document = documents[index] if index < len(documents) else None
        missing_fields = [field for field in REQUIRED_METADATA_FIELDS if not _safe_text(metadata.get(field))]
        if missing_fields:
            missing_metadata_records.append(f"{rule_id}:{','.join(missing_fields)}")
        missing_enterprise_fields = [
            field for field in REQUIRED_ENTERPRISE_METADATA_FIELDS if not _safe_text(metadata.get(field))
        ]
        if missing_enterprise_fields:
            missing_enterprise_metadata_records.append(f"{rule_id}:{','.join(missing_enterprise_fields)}")

        spring_pattern = _safe_text(metadata.get("spring_pattern"))
        micronaut_pattern = _safe_text(metadata.get("micronaut_pattern"))
        if spring_pattern and micronaut_pattern:
            mapping_pair_counter[(spring_pattern, micronaut_pattern)] += 1
            targets_by_source[spring_pattern].add(micronaut_pattern)

        if document is not None and _safe_text(document):
            stored_document_counter[_safe_text(document)] += 1

        derived_document_signature = _derived_document_signature(metadata)
        if derived_document_signature:
            derived_document_counter[derived_document_signature] += 1

        if _is_generic_pattern(metadata):
            overly_generic_patterns.append(str(rule_id))

        confidence = _safe_float(metadata.get("confidence"))
        if confidence is None or confidence < LOW_CONFIDENCE_THRESHOLD:
            low_confidence_records.append(str(rule_id))

        release_status = _safe_text(metadata.get("release_validation_status")) or _safe_text(metadata.get("status"))
        if release_status.lower() != "validated":
            unvalidated_records.append(str(rule_id))

        spring_version_spec = _safe_text(metadata.get("spring_version_spec"))
        spring_version_minimum = _safe_text(metadata.get("spring_version_minimum"))
        spring_version_maximum = _safe_text(metadata.get("spring_version_maximum"))
        micronaut_version_spec = _safe_text(metadata.get("micronaut_version_spec"))
        micronaut_version_minimum = _safe_text(metadata.get("micronaut_version_minimum"))
        micronaut_version_maximum = _safe_text(metadata.get("micronaut_version_maximum"))
        has_spring_window = any((spring_version_spec, spring_version_minimum, spring_version_maximum))
        has_micronaut_window = any((micronaut_version_spec, micronaut_version_minimum, micronaut_version_maximum))
        if not has_spring_window or not has_micronaut_window:
            missing_version_window_records.append(str(rule_id))

        evidence_count = _safe_int(metadata.get("evidence_count"))
        if evidence_count is None or evidence_count < 1:
            weak_evidence_records.append(str(rule_id))

    duplicate_mapping_pairs = sorted(
        f"{spring} -> {micronaut}"
        for (spring, micronaut), count in mapping_pair_counter.items()
        if count > 1
    )
    duplicate_stored_documents = sorted(doc for doc, count in stored_document_counter.items() if count > 1)
    duplicate_derived_documents = sorted(doc for doc, count in derived_document_counter.items() if count > 1)
    conflicting_mappings = {
        spring: sorted(targets)
        for spring, targets in sorted(targets_by_source.items())
        if len(targets) > 1
    }

    return {
        "collection": name,
        "record_count": total_records,
        "missing_metadata_count": len(missing_metadata_records),
        "missing_metadata_records": missing_metadata_records,
        "duplicate_mapping_pair_count": len(duplicate_mapping_pairs),
        "duplicate_mapping_pairs": duplicate_mapping_pairs,
        "duplicate_stored_document_count": len(duplicate_stored_documents),
        "duplicate_stored_documents": duplicate_stored_documents,
        "duplicate_derived_document_count": len(duplicate_derived_documents),
        "duplicate_derived_documents": duplicate_derived_documents,
        "conflicting_mapping_count": len(conflicting_mappings),
        "conflicting_mappings": conflicting_mappings,
        "generic_pattern_count": len(overly_generic_patterns),
        "generic_pattern_ids": sorted(overly_generic_patterns),
        "missing_enterprise_metadata_count": len(missing_enterprise_metadata_records),
        "missing_enterprise_metadata_records": missing_enterprise_metadata_records,
        "low_confidence_count": len(sorted(set(low_confidence_records))),
        "low_confidence_ids": sorted(set(low_confidence_records)),
        "unvalidated_record_count": len(sorted(set(unvalidated_records))),
        "unvalidated_record_ids": sorted(set(unvalidated_records)),
        "missing_version_window_count": len(sorted(set(missing_version_window_records))),
        "missing_version_window_ids": sorted(set(missing_version_window_records)),
        "weak_evidence_count": len(sorted(set(weak_evidence_records))),
        "weak_evidence_ids": sorted(set(weak_evidence_records)),
        "documents_persisted": any(document is not None for document in documents),
    }


def _calculate_trust_level(total_records: int, collection_reports: Iterable[Dict[str, object]]) -> str:
    if total_records == 0:
        return "empty"

    reports = list(collection_reports)
    has_blockers = any(
        report["missing_metadata_count"] > 0
        or report["conflicting_mapping_count"] > 0
        or report["missing_enterprise_metadata_count"] > 0
        or report["unvalidated_record_count"] > 0
        or report["missing_version_window_count"] > 0
        or report["low_confidence_count"] > 0
        for report in reports
    )
    has_warnings = any(
        report["duplicate_mapping_pair_count"] > 0
        or report["duplicate_stored_document_count"] > 0
        or report["duplicate_derived_document_count"] > 0
        or report["generic_pattern_count"] > 0
        or report["weak_evidence_count"] > 0
        for report in reports
    )

    if has_blockers:
        return "low"
    if has_warnings:
        return "medium"
    return "high"


def _calculate_distribution_readiness(
    total_records: int,
    collection_reports: Iterable[Dict[str, object]],
) -> Tuple[bool, List[str]]:
    if total_records == 0:
        return False, ["empty_database"]

    blocker_codes = [
        ("missing_metadata_count", "missing_metadata"),
        ("conflicting_mapping_count", "conflicting_mappings"),
        ("duplicate_mapping_pair_count", "duplicate_mapping_pairs"),
        ("duplicate_stored_document_count", "duplicate_stored_documents"),
        ("duplicate_derived_document_count", "duplicate_derived_documents"),
        ("generic_pattern_count", "generic_patterns"),
        ("missing_enterprise_metadata_count", "missing_enterprise_metadata"),
        ("unvalidated_record_count", "unvalidated_records"),
        ("missing_version_window_count", "missing_version_windows"),
        ("low_confidence_count", "low_confidence_records"),
        ("weak_evidence_count", "weak_evidence_records"),
    ]

    reasons = set()
    for report in collection_reports:
        for key, reason in blocker_codes:
            if int(report.get(key, 0)) > 0:
                reasons.add(reason)

    return len(reasons) == 0, sorted(reasons)


def audit_persisted_chroma_db(db_path: str, collection_names: Iterable[str] = AUDITED_COLLECTIONS) -> Dict[str, object]:
    client, backend_name, native_available = create_persistent_client(db_path)
    collection_reports: List[Dict[str, object]] = []
    collection_counts: Dict[str, int] = {}

    for name in collection_names:
        try:
            collection = client.get_collection(name)
            payload = _collection_payload(collection)
        except Exception:
            payload = {"ids": [], "metadatas": [], "documents": []}

        report = _audit_collection(name, payload)
        collection_reports.append(report)
        collection_counts[name] = report["record_count"]

    total_records = sum(collection_counts.values())
    trust_level = _calculate_trust_level(total_records, collection_reports)
    distribution_ready, distribution_blockers = _calculate_distribution_readiness(total_records, collection_reports)
    ok = trust_level in {"high", "medium"}

    return {
        "ok": ok,
        "db_path": db_path,
        "backend": backend_name,
        "native_chromadb_available": native_available,
        "total_record_count": total_records,
        "collection_counts": collection_counts,
        "collections": collection_reports,
        "trust_level": trust_level,
        "distribution_ready": distribution_ready,
        "enterprise_ready": distribution_ready,
        "distribution_blockers": distribution_blockers,
    }


def audit_validated_release_chroma(
    corpus_root: str = "corpus",
    spring_version: str = "3.4.5",
    micronaut_version: str = "4.10.1",
    runtime_dataset_path: Optional[str] = None,
    release_rule_count: Optional[int] = None,
) -> Dict[str, object]:
    if runtime_dataset_path is None:
        release_result = write_validated_release(corpus_root=corpus_root, runtime_format="json")
        runtime_dataset_path = str(release_result["runtime_dataset_path"])
        resolved_release_rule_count = int(release_result["release_rule_count"])
    else:
        resolved_release_rule_count = int(release_rule_count or 0)

    with tempfile.TemporaryDirectory(prefix="spring2naut-chroma-audit-") as db_tmpdir:
        with _patched_runtime_dataset(runtime_dataset_path):
            with patch.object(LocalMigrationKnowledgeBase, "_initialize_models", _deterministic_model_init):
                LocalMigrationKnowledgeBase(db_path=db_tmpdir)

        report = audit_persisted_chroma_db(db_tmpdir)

    report.update(
        {
            "runtime_dataset_path": runtime_dataset_path,
            "release_rule_count": resolved_release_rule_count,
            "spring_version": spring_version,
            "micronaut_version": micronaut_version,
        }
    )
    return report


def write_chroma_audit_report(
    corpus_root: str = "corpus",
    db_path: Optional[str] = None,
    release_mode: bool = True,
    runtime_dataset_path: Optional[str] = None,
    release_rule_count: Optional[int] = None,
    spring_version: str = "3.4.5",
    micronaut_version: str = "4.10.1",
) -> Dict[str, object]:
    if release_mode:
        report = audit_validated_release_chroma(
            corpus_root=corpus_root,
            spring_version=spring_version,
            micronaut_version=micronaut_version,
            runtime_dataset_path=runtime_dataset_path,
            release_rule_count=release_rule_count,
        )
        report_path = Path(corpus_root) / "validated_patterns" / "release" / "chroma_audit_report.json"
    else:
        resolved_db_path = db_path or MigrationConfig.VECTOR_DB_PATH
        report = audit_persisted_chroma_db(resolved_db_path)
        report_path = Path(corpus_root) / "validated_patterns" / "release" / "chroma_audit_report.json"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"report_path": str(report_path), **report}


def main():
    parser = argparse.ArgumentParser(description="Audit persisted Spring-to-Micronaut ChromaDB metadata and derive a retrieval trust level")
    parser.add_argument("--corpus-root", default="corpus", help="Corpus root directory")
    parser.add_argument("--db-path", default=None, help="Optional existing ChromaDB path to audit directly")
    parser.add_argument("--direct-db", action="store_true", help="Audit an existing db path instead of building a validated release db")
    parser.add_argument("--write", action="store_true", help="Write Chroma audit report")
    args = parser.parse_args()

    if args.write:
        print(
            json.dumps(
                write_chroma_audit_report(
                    corpus_root=args.corpus_root,
                    db_path=args.db_path,
                    release_mode=not args.direct_db,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return

    print(json.dumps({"message": "Use --write to materialize Chroma audit results."}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
