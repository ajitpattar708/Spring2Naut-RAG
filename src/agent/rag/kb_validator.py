import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from src.agent.core.config import MigrationConfig
from src.agent.rag.audit import audit_dataset, iter_rules
from src.agent.rag.knowledge_base import LocalMigrationKnowledgeBase


class DeterministicEmbeddingModel:
    """
    Small local embedder for validation-only Chroma round trips.
    This avoids downloading production transformer weights just to
    validate dataset integrity and collection indexing.
    """

    def __init__(self, dimensions: int = MigrationConfig.EMBEDDING_DIMENSION):
        self.dimensions = dimensions

    def encode(self, texts, batch_size: int = 32, show_progress_bar: bool = False):
        vectors = []
        for text in texts:
            counts = [0.0] * self.dimensions
            for index, char in enumerate(text.encode("utf-8")):
                counts[index % self.dimensions] += float(char)
            total = sum(counts) or 1.0
            vectors.append([value / total for value in counts])
        return vectors


def _load_dataset(path: str):
    loader = LocalMigrationKnowledgeBase.__new__(LocalMigrationKnowledgeBase)
    return loader.load_dataset(path)


def validate_dataset_files(
    dataset_files: Optional[List[str]] = None,
    sample_size: int = 250,
) -> Dict[str, object]:
    loader = LocalMigrationKnowledgeBase.__new__(LocalMigrationKnowledgeBase)
    dataset_files = dataset_files or [
        MigrationConfig.DATASET_FILE,
        MigrationConfig.ENHANCED_DATASET_FILE,
    ]

    dataset_reports = {}
    raw_datasets = []

    for dataset_file in dataset_files:
        resolved_path = Path(dataset_file)
        if not resolved_path.exists() and not str(resolved_path).endswith(".dat"):
            dat_path = Path(f"{dataset_file}.dat")
            resolved_path = dat_path if dat_path.exists() else resolved_path

        dataset = _load_dataset(str(resolved_path))
        report = audit_dataset(dataset)
        dataset_reports[str(resolved_path)] = report.to_dict()
        raw_datasets.append(dataset)

    merged_rules = loader._merge_datasets(raw_datasets[0], raw_datasets[1] if len(raw_datasets) > 1 else None)
    sanitized_rules = loader._sanitize_rules(merged_rules)
    chroma_validation = validate_chroma_round_trip(sanitized_rules[:sample_size])

    return {
        "datasets": dataset_reports,
        "runtime_rules": {
            "merged_rule_count": len(merged_rules),
            "sanitized_rule_count": len(sanitized_rules),
            "dropped_rule_count": len(merged_rules) - len(sanitized_rules),
        },
        "sampled_rule_count": min(sample_size, len(sanitized_rules)),
        "chroma_validation": chroma_validation,
    }


def validate_chroma_round_trip(rules: List[Dict[str, object]]) -> Dict[str, object]:
    try:
        import chromadb
    except ImportError:
        return {
            "ok": False,
            "reason": "chromadb not installed",
        }

    if not rules:
        return {
            "ok": False,
            "reason": "no rules available for validation",
        }

    collections = {}
    embedder = DeterministicEmbeddingModel()

    with tempfile.TemporaryDirectory(prefix="spring2naut-kb-") as tmpdir:
        client = chromadb.PersistentClient(path=tmpdir)
        for name in ["annotations", "dependencies", "configurations", "code_patterns"]:
            collections[name] = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

        indexed = 0
        for offset, rule in enumerate(rules):
            category = str(rule.get("category", "code_patterns")).strip() or "code_patterns"
            collection = collections.get(category, collections["code_patterns"])
            text = f"{rule.get('spring_pattern', '')} {rule.get('description', '')}"
            collection.add(
                ids=[str(rule.get("id") or f"rule-{offset}")],
                embeddings=embedder.encode([text]),
                metadatas=[
                    {
                        "spring_pattern": str(rule.get("spring_pattern", "")),
                        "micronaut_pattern": str(rule.get("micronaut_pattern", "")),
                        "category": category,
                        "description": str(rule.get("description", "")),
                        "complexity": str(rule.get("complexity", "medium")),
                    }
                ],
            )
            indexed += 1

        sample_rule = rules[0]
        sample_text = f"{sample_rule.get('spring_pattern', '')} {sample_rule.get('description', '')}"
        query = collections.get(str(sample_rule.get("category", "code_patterns")), collections["code_patterns"]).query(
            query_embeddings=embedder.encode([sample_text]),
            n_results=1,
        )

    return {
        "ok": bool(query.get("ids") and query["ids"][0]),
        "indexed_rules": indexed,
        "query_result_count": len(query.get("ids", [[]])[0]) if query.get("ids") else 0,
    }


def main():
    report = validate_dataset_files()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
