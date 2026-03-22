import argparse
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from unittest.mock import patch

from src.agent.core.config import MigrationConfig
from src.agent.patterns.release import write_validated_release
from src.agent.rag.kb_validator import DeterministicEmbeddingModel
from src.agent.rag.knowledge_base import LocalMigrationKnowledgeBase


DEFAULT_SAMPLE_QUERIES: List[Dict[str, str]] = [
    {
        "kind": "annotation",
        "query": "@RestController",
        "expected": "@Controller",
    },
    {
        "kind": "annotation",
        "query": "@GetMapping",
        "expected": "@Get",
    },
    {
        "kind": "annotation",
        "query": "@RequestBody",
        "expected": "@Body",
    },
    {
        "kind": "annotation",
        "query": "@RequestHeader",
        "expected": "@Header",
    },
    {
        "kind": "annotation",
        "query": "@Configuration",
        "expected": "@Factory",
    },
    {
        "kind": "annotation",
        "query": "@Repository",
        "expected": "@Singleton",
    },
    {
        "kind": "annotation",
        "query": "@Autowired",
        "expected": "jakarta.inject.Inject",
    },
    {
        "kind": "annotation",
        "query": "@Qualifier",
        "expected": "jakarta.inject.Named",
    },
    {
        "kind": "annotation",
        "query": "@ExceptionHandler",
        "expected": "io.micronaut.http.annotation.Error",
    },
    {
        "kind": "annotation",
        "query": "@ResponseStatus",
        "expected": "io.micronaut.http.annotation.Status",
    },
    {
        "kind": "annotation",
        "query": "@Cacheable",
        "expected": "io.micronaut.cache.annotation.Cacheable",
    },
    {
        "kind": "annotation",
        "query": "@Transactional",
        "expected": "jakarta.transaction.Transactional",
    },
    {
        "kind": "annotation",
        "query": "@Scheduled",
        "expected": "io.micronaut.scheduling.annotation.Scheduled",
    },
    {
        "kind": "configuration",
        "query": "spring.application.name",
        "expected": "micronaut.application.name",
    },
    {
        "kind": "types",
        "query": "ResponseEntity",
        "expected": "HttpResponse",
    },
    {
        "kind": "code_patterns",
        "query": "SpringApplication.run",
        "expected": "Micronaut.run",
    },
    {
        "kind": "code_patterns",
        "query": "Field injection",
        "expected": "Constructor injection",
    },
    {
        "kind": "dependencies",
        "query": "org.springframework.boot:spring-boot-starter-web",
        "expected": "io.micronaut:micronaut-http-server-netty",
    },
    {
        "kind": "dependencies",
        "query": "org.springframework.boot:spring-boot-starter-jdbc",
        "expected": "io.micronaut.sql:micronaut-jdbc-hikari",
    },
    {
        "kind": "dependencies",
        "query": "org.springframework.boot:spring-boot-starter-cache",
        "expected": "io.micronaut.cache:micronaut-cache-caffeine",
    },
    {
        "kind": "dependencies",
        "query": "org.ehcache:ehcache",
        "expected": "io.micronaut.cache:micronaut-cache-ehcache",
    },
    {
        "kind": "dependencies",
        "query": "org.springdoc:springdoc-openapi-ui",
        "expected": "io.micronaut.openapi:micronaut-openapi",
    },
]


class _EmbeddingList(list):
    def tolist(self):
        return list(self)


class CompatibleDeterministicEmbeddingModel(DeterministicEmbeddingModel):
    def encode(self, texts, batch_size: int = 32, show_progress_bar: bool = False):
        return _EmbeddingList(super().encode(texts, batch_size=batch_size, show_progress_bar=show_progress_bar))


def _deterministic_model_init(instance: LocalMigrationKnowledgeBase):
    instance.embedding_model = CompatibleDeterministicEmbeddingModel()
    instance.embedding_dimension = instance.embedding_model.dimensions


@contextmanager
def _patched_runtime_dataset(runtime_dataset_path: str):
    original_dataset = MigrationConfig.DATASET_FILE
    original_enhanced = MigrationConfig.ENHANCED_DATASET_FILE
    try:
        MigrationConfig.DATASET_FILE = runtime_dataset_path
        MigrationConfig.ENHANCED_DATASET_FILE = "__missing_runtime_dataset__.json"
        yield
    finally:
        MigrationConfig.DATASET_FILE = original_dataset
        MigrationConfig.ENHANCED_DATASET_FILE = original_enhanced


def _run_sample_queries(
    knowledge_base: LocalMigrationKnowledgeBase,
    sample_queries: Iterable[Dict[str, str]],
    spring_version: str,
    micronaut_version: str,
    collection_counts: Dict[str, int],
    top_k: int = 1000,
) -> Tuple[bool, List[Dict[str, object]]]:
    results: List[Dict[str, object]] = []

    for sample in sample_queries:
        kind = sample["kind"]
        query = sample["query"]
        expected = sample["expected"]
        collection_name = {
            "annotation": "annotations",
            "configuration": "configurations",
        }.get(kind, kind)

        if collection_counts.get(collection_name, 0) == 0:
            results.append(
                {
                    "kind": kind,
                    "query": query,
                    "expected_micronaut_pattern": expected,
                    "match_count": 0,
                    "top_match_micronaut_pattern": None,
                    "ok": True,
                    "status": "skipped_empty_collection",
                }
            )
            continue

        if kind == "configuration":
            matches = knowledge_base.search_configuration(
                query,
                top_k=top_k,
                spring_version=spring_version,
                micronaut_version=micronaut_version,
            )
        elif kind == "annotation":
            matches = knowledge_base.search_annotation(
                query,
                top_k=top_k,
                spring_version=spring_version,
                micronaut_version=micronaut_version,
            )
        elif kind == "dependencies":
            matches = knowledge_base.search_dependency(
                query,
                top_k=top_k,
                spring_version=spring_version,
                micronaut_version=micronaut_version,
            )
        else:
            matches = knowledge_base._search_collection(
                kind,
                query,
                top_k=top_k,
                spring_version=spring_version,
                micronaut_version=micronaut_version,
            )

        matched_rule = next((match for match in matches if match.micronaut_pattern == expected), None)
        top_match = matched_rule or (matches[0] if matches else None)
        matched = matched_rule is not None
        results.append(
            {
                "kind": kind,
                "query": query,
                "expected_micronaut_pattern": expected,
                "match_count": len(matches),
                "top_match_micronaut_pattern": top_match.micronaut_pattern if top_match else None,
                "ok": matched,
                "status": "matched" if matched else "mismatch",
            }
        )

    return all(item["ok"] for item in results), results


def validate_release_dataset_in_chroma(
    corpus_root: str = "corpus",
    spring_version: str = "3.4.5",
    micronaut_version: str = "4.10.1",
    sample_queries: Iterable[Dict[str, str]] = DEFAULT_SAMPLE_QUERIES,
    runtime_dataset_path: str | None = None,
    release_rule_count: int | None = None,
) -> Dict[str, object]:
    if runtime_dataset_path is None:
        release_result = write_validated_release(corpus_root=corpus_root, runtime_format="json")
        runtime_dataset_path = str(release_result["runtime_dataset_path"])
        resolved_release_rule_count = int(release_result["release_rule_count"])
    else:
        resolved_release_rule_count = int(release_rule_count or 0)

    with tempfile.TemporaryDirectory(prefix="spring2naut-release-kb-") as db_tmpdir:
        with _patched_runtime_dataset(runtime_dataset_path):
            with patch.object(LocalMigrationKnowledgeBase, "_initialize_models", _deterministic_model_init):
                knowledge_base = LocalMigrationKnowledgeBase(db_path=db_tmpdir)

        collection_counts = {name: collection.count() for name, collection in knowledge_base.collections.items()}
        samples_ok, sample_results = _run_sample_queries(
            knowledge_base,
            sample_queries=sample_queries,
            spring_version=spring_version,
            micronaut_version=micronaut_version,
            collection_counts=collection_counts,
            top_k=1000,
        )

    total_indexed = sum(collection_counts.values())
    return {
        "ok": bool(total_indexed > 0 and samples_ok),
        "runtime_dataset_path": runtime_dataset_path,
        "release_rule_count": resolved_release_rule_count,
        "vector_db_rule_count": total_indexed,
        "collection_counts": collection_counts,
        "spring_version": spring_version,
        "micronaut_version": micronaut_version,
        "sample_results": sample_results,
    }


def write_release_kb_smoke_report(
    corpus_root: str = "corpus",
    spring_version: str = "3.4.5",
    micronaut_version: str = "4.10.1",
    runtime_dataset_path: str | None = None,
    release_rule_count: int | None = None,
) -> Dict[str, object]:
    report = validate_release_dataset_in_chroma(
        corpus_root=corpus_root,
        spring_version=spring_version,
        micronaut_version=micronaut_version,
        runtime_dataset_path=runtime_dataset_path,
        release_rule_count=release_rule_count,
    )
    report_path = Path(corpus_root) / "validated_patterns" / "release" / "kb_smoke_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"report_path": str(report_path), **report}


def main():
    parser = argparse.ArgumentParser(description="Build the validated release dataset, load it into ChromaDB, and run sample retrieval smoke tests")
    parser.add_argument("--corpus-root", default="corpus", help="Corpus root directory")
    parser.add_argument("--spring-version", default="3.4.5", help="Spring version for compatibility-filtered sample queries")
    parser.add_argument("--micronaut-version", default="4.10.1", help="Micronaut version for compatibility-filtered sample queries")
    parser.add_argument("--write", action="store_true", help="Write KB smoke validation report")
    args = parser.parse_args()

    if args.write:
        print(
            json.dumps(
                write_release_kb_smoke_report(
                    corpus_root=args.corpus_root,
                    spring_version=args.spring_version,
                    micronaut_version=args.micronaut_version,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return

    print(json.dumps({"message": "Use --write to materialize release KB smoke results."}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
