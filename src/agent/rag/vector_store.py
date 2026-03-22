import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


class FallbackPersistentCollection:
    def __init__(self, root: Path, name: str, metadata: Optional[Dict[str, Any]] = None):
        self.root = root
        self.name = name
        self.path = self.root / f"{self.name}.json"
        self.metadata = dict(metadata or {})
        self._payload = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            payload = {"name": self.name, "metadata": self.metadata, "records": []}
            self._write(payload)
            return payload

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            payload = {"name": self.name, "metadata": self.metadata, "records": []}
            self._write(payload)
        payload.setdefault("name", self.name)
        payload.setdefault("metadata", dict(self.metadata))
        payload.setdefault("records", [])
        return payload

    def _write(self, payload: Optional[Dict[str, Any]] = None) -> None:
        materialized = payload or self._payload
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(materialized, indent=2), encoding="utf-8")

    def count(self) -> int:
        return len(self._payload.get("records", []))

    def add(
        self,
        *,
        ids: Iterable[str],
        embeddings: Iterable[Iterable[float]],
        metadatas: Iterable[Dict[str, Any]],
        documents: Optional[Iterable[str]] = None,
    ) -> None:
        ids_list = [str(value) for value in ids]
        embeddings_list = [[float(item) for item in list(vector)] for vector in embeddings]
        metadatas_list = [dict(value or {}) for value in metadatas]
        documents_list = list(documents or [])
        records = list(self._payload.get("records", []))
        by_id = {str(record.get("id")): index for index, record in enumerate(records)}

        for index, rule_id in enumerate(ids_list):
            record = {
                "id": rule_id,
                "embedding": embeddings_list[index],
                "metadata": metadatas_list[index],
                "document": documents_list[index] if index < len(documents_list) else "",
            }
            existing_index = by_id.get(record["id"])
            if existing_index is None:
                by_id[record["id"]] = len(records)
                records.append(record)
            else:
                records[existing_index] = record

        self._payload["records"] = records
        self._write()

    def get(self, where: Optional[Dict[str, Any]] = None, include: Optional[List[str]] = None) -> Dict[str, Any]:
        include = include or []
        records = list(self._payload.get("records", []))

        if where:
            filtered = []
            for record in records:
                metadata = record.get("metadata") or {}
                if all(metadata.get(key) == value for key, value in where.items()):
                    filtered.append(record)
            records = filtered

        result = {"ids": [record.get("id") for record in records]}
        if not include or "metadatas" in include:
            result["metadatas"] = [record.get("metadata") or {} for record in records]
        if "documents" in include:
            result["documents"] = [record.get("document") for record in records]
        if "embeddings" in include:
            result["embeddings"] = [record.get("embedding") or [] for record in records]
        return result

    def query(self, query_embeddings: Iterable[Iterable[float]], n_results: int = 3) -> Dict[str, Any]:
        query_vectors = list(query_embeddings or [])
        query_vector = list(query_vectors[0]) if query_vectors else []
        scored = []
        for record in self._payload.get("records", []):
            score = _cosine_similarity(query_vector, list(record.get("embedding") or []))
            scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = [record for _, record in scored[: max(0, n_results)]]
        return {
            "ids": [[record.get("id") for record in top]],
            "metadatas": [[record.get("metadata") or {} for record in top]],
            "documents": [[record.get("document") for record in top]],
        }


class FallbackPersistentClient:
    def __init__(self, path: str):
        self.root = Path(path)
        self.root.mkdir(parents=True, exist_ok=True)

    def get_collection(self, name: str):
        collection_path = self.root / f"{name}.json"
        if not collection_path.exists():
            raise KeyError(name)
        return FallbackPersistentCollection(self.root, name)

    def create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        return FallbackPersistentCollection(self.root, name, metadata=metadata)

    def get_or_create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        collection_path = self.root / f"{name}.json"
        if collection_path.exists():
            return FallbackPersistentCollection(self.root, name)
        return self.create_collection(name, metadata=metadata)

    def delete_collection(self, name: str) -> None:
        collection_path = self.root / f"{name}.json"
        if collection_path.exists():
            collection_path.unlink()


def create_persistent_client(path: str):
    try:
        import chromadb  # type: ignore

        return chromadb.PersistentClient(path=path), "chromadb", True
    except ImportError:
        return FallbackPersistentClient(path=path), "fallback-json", False
