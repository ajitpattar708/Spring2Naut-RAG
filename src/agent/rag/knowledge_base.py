import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.agent.core.config import MigrationConfig, SecurityConfig
from src.agent.core.interfaces import KnowledgeService
from src.agent.core.models import MigrationRule, VersionCompatibilityMatrix
from src.agent.rag.vector_store import create_persistent_client

YELLOW = "\033[93m"
RESET = "\033[0m"


class _EmbeddingList(list):
    def tolist(self):
        return list(self)


class DeterministicFallbackEmbeddingModel:
    """
    Small offline embedder used when production transformer weights are unavailable.
    It keeps the KB operational for exact-match and basic semantic flows in offline mode.
    """

    def __init__(self, dimensions: int = MigrationConfig.EMBEDDING_DIMENSION):
        self.dimensions = dimensions

    def encode(self, texts, batch_size: int = 32, show_progress_bar: bool = False):
        vectors = []
        for text in texts:
            counts = [0.0] * self.dimensions
            for index, char in enumerate(str(text).encode("utf-8")):
                counts[index % self.dimensions] += float(char)
            total = sum(counts) or 1.0
            vectors.append([value / total for value in counts])
        return _EmbeddingList(vectors)


class LocalMigrationKnowledgeBase(KnowledgeService):
    """
    Local implementation of the migration knowledge base using ChromaDB and CodeBERT.
    Includes logic for handling protected/encrypted datasets.
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or MigrationConfig.VECTOR_DB_PATH
        self.available = True
        self.vector_backend_name = MigrationConfig.VECTOR_DB_TYPE
        self.native_chromadb_available = False
        self._initialize_models()
        self._initialize_db()
        # Automatically initialize knowledge base with datasets
        self.initialize_knowledge_base()
        
    def _initialize_models(self):
        """
        Loads the embedding model used for semantic search across code patterns.
        """
        model_name = MigrationConfig.EMBEDDING_MODEL
        try:
            from sentence_transformers import SentenceTransformer
            # CodeBERT is preferred for technical accuracy in code transformation
            self.embedding_model = SentenceTransformer(model_name)
        except Exception:
            try:
                from sentence_transformers import SentenceTransformer
                # Fallback to a lighter model if preferred model is unavailable
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                print(
                    f"{YELLOW}[WARN]{RESET} sentence-transformers models are unavailable; "
                    "falling back to deterministic offline embeddings."
                )
                self.embedding_model = DeterministicFallbackEmbeddingModel()
            
        test_embedding = self.embedding_model.encode(["test"])
        self.embedding_dimension = len(test_embedding[0])

    def _initialize_db(self):
        """
        Sets up the vector database and initializes collections for different pattern categories.
        """
        self.client, backend_name, native_available = create_persistent_client(self.db_path)
        self.vector_backend_name = backend_name
        self.native_chromadb_available = native_available
        if not native_available:
            print(
                f"{YELLOW}[WARN]{RESET} chromadb is not installed; "
                "using persistent local fallback vector store."
            )
        self.collections = {
            "annotations": self._get_or_create_collection("annotations"),
            "dependencies": self._get_or_create_collection("dependencies"),
            "configurations": self._get_or_create_collection("configurations"),
            "code_patterns": self._get_or_create_collection("code_patterns"),
            "imports": self._get_or_create_collection("imports"),
            "types": self._get_or_create_collection("types")
        }

    def initialize_knowledge_base(self):
        """
        Populate knowledge base with migration rules from multiple sources.
        Supports the Community (JSON) and Pro (Encrypted) tiered model.
        """
        if not self.available:
            return

        # Check if DB is already populated to avoid redundant indexing
        if self._is_populated():
            counts = {name: col.count() for name, col in self.collections.items()}
            total = sum(counts.values())
            print(f"[INFO] Intelligence Engine loaded with {total} cached patterns.")
            return

        # 1. Load Community Dataset (Plain JSON)
        community_data = self.load_dataset(MigrationConfig.DATASET_FILE)
        
        # 2. Load Pro Dataset (Encrypted DAT)
        pro_data = self.load_dataset(MigrationConfig.ENHANCED_DATASET_FILE)
        
        # Merge datasets (Pro overrides/augments Community) with Auto-Categorization
        merged_rules = self._merge_datasets(community_data, pro_data)
        sanitized_rules = self._sanitize_rules(merged_rules)
        
        if sanitized_rules:
            # If we were previously ghost-partitioned, we need to clear everything first
            # to avoid duplicate IDs or mixed schemas
            for name, col in self.collections.items():
                try:
                    self.client.delete_collection(name)
                    self.collections[name] = self.client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
                except: pass
            self._index_rules(sanitized_rules)
        else:
            print("[WARN] Intelligence Engine running with limited internal patterns.")

    def _is_populated(self) -> bool:
        """
        Checks if the database is healthy and specialized collections are populated.
        Triggers a re-index if we detect 'Ghost Partitioning' (data only in generic collection).
        """
        if not self.available:
            return False
        try:
            counts = {name: col.count() for name, col in self.collections.items() if name != 'code_patterns'}
            total_specialized = sum(counts.values())
            # If specialized collections are empty but generic records exist, the DB is badly partitioned.
            if total_specialized == 0 and self.collections['code_patterns'].count() > 0:
                print(f"{YELLOW}[WARN]{RESET} Intelligence Engine detected as 'Ghost Partitioned' (0 specialized rules).")
                print("       Triggering structural re-alignment of indexed rules...")
                return False
            return total_specialized > 0 or self.collections['code_patterns'].count() > 0
        except Exception:
            return False

    def _merge_datasets(self, community: Optional[Dict], pro: Optional[Dict]) -> List[Dict]:
        """Merges Community and Pro datasets into a flat list of rules with auto-categorization."""
        all_rules = []
        
        def process_source(source_data):
            if not source_data: return
            if isinstance(source_data, list):
                rules_to_add = source_data
            elif isinstance(source_data, dict):
                rules_to_add = []
                for category, rules in source_data.items():
                    if isinstance(rules, list):
                        for r in rules:
                            if 'category' not in r: r['category'] = category
                            rules_to_add.append(r)
            else:
                return

            for r in rules_to_add:
                # Auto-categorization engine for GA-grade intelligence
                if 'category' not in r or r['category'] == 'code_patterns':
                    pattern = r.get('spring_pattern', '')
                    if pattern.startswith('@'):
                        r['category'] = 'annotations'
                    elif ':' in pattern and '@' not in pattern:
                        r['category'] = 'dependencies'
                    elif any(pattern.startswith(prefix) for prefix in ['spring.', 'server.', 'management.', 'logging.']):
                        r['category'] = 'configurations'
                    else:
                        r['category'] = 'code_patterns'
                all_rules.append(r)
        
        process_source(community)
        process_source(pro)
        return all_rules

    def _sanitize_rules(self, rules: List[Dict]) -> List[Dict]:
        """
        Removes rules that are unsafe or unhelpful for runtime retrieval.
        This protects indexing from low-quality synthetic self-maps and exact duplicates.
        """
        sanitized = []
        seen_rule_ids = set()
        seen_exact_rules = set()
        dropped_self_maps = 0
        dropped_duplicates = 0

        for rule in rules:
            spring_pattern = str(rule.get("spring_pattern", "")).strip()
            micronaut_pattern = str(rule.get("micronaut_pattern", "")).strip()

            if not spring_pattern or not micronaut_pattern:
                continue

            if spring_pattern == micronaut_pattern:
                dropped_self_maps += 1
                continue

            rule_id = str(rule.get("id", "")).strip()
            exact_key = (
                str(rule.get("category", "code_patterns")).strip() or "code_patterns",
                spring_pattern,
                micronaut_pattern,
                str(rule.get("spring_version", "")).strip(),
                str(rule.get("micronaut_version", "")).strip(),
            )

            if rule_id and rule_id in seen_rule_ids:
                dropped_duplicates += 1
                continue

            if exact_key in seen_exact_rules:
                dropped_duplicates += 1
                continue

            if rule_id:
                seen_rule_ids.add(rule_id)
            seen_exact_rules.add(exact_key)
            sanitized.append(rule)

        if dropped_self_maps or dropped_duplicates:
            print(
                f"[INFO] Sanitized rule corpus: removed {dropped_self_maps} self-maps and "
                f"{dropped_duplicates} duplicate rules before indexing."
            )

        return sanitized

    def _index_rules(self, rules: List[Dict]):
        """Populates ChromaDB collections with the rule data using batched processing.
        CodeBERT is intensive, so we use optimal batch sizes and provide detailed feedback.
        """
        import time
        start_time = time.time()
        # Increased batch size for better throughput with CodeBERT
        batch_size = 64 
        total_rules = len(rules)
        print(f"[INFO] Indexing {total_rules} patterns into Vector DB using {MigrationConfig.EMBEDDING_MODEL}...")
        print(f"[INFO] This is a one-time operation. Subsequent runs will be near-instant.")
        
        # Group rules by category
        categorized_rules = {}
        for rule in rules:
            cat = rule.get('category', 'code_patterns')
            if cat not in categorized_rules:
                categorized_rules[cat] = []
            categorized_rules[cat].append(rule)
            
        indexed_count = 0
        for category, cat_rules in categorized_rules.items():
            collection = self.collections.get(category, self.collections['code_patterns'])
            
            for i in range(0, len(cat_rules), batch_size):
                batch = cat_rules[i:i + batch_size]
                
                texts = [f"{r.get('spring_pattern', '')} {r.get('description', '')}" for r in batch]
                
                # Perform encoding in batches. Manual tolist() for maximum compatibility with Python 3.13+/Sentence-Transformers 3.x
                embeddings = self.embedding_model.encode(
                    texts, 
                    batch_size=batch_size, 
                    show_progress_bar=False
                ).tolist()
                
                ids = [r.get('id', os.urandom(8).hex()) for r in batch]
                metadatas = [self._rule_to_chroma_metadata(r, category) for r in batch]
                
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=texts,
                )
                
                indexed_count += len(batch)
                elapsed = time.time() - start_time
                # Estimate remaining time
                rate = indexed_count / elapsed
                remaining = (total_rules - indexed_count) / rate
                
                print(f"  > Progress: {indexed_count}/{total_rules} ({ (indexed_count/total_rules * 100):.1f}%) | Est. Remaining: {int(remaining/60)}m {int(remaining%60)}s ", end='\r')
            
        print(f"\n[OK] Intelligence Engine indexed with {total_rules} patterns in {int((time.time() - start_time)/60)}m {int((time.time() - start_time)%60)}s.")

    def _rule_to_chroma_metadata(self, rule: Dict[str, Any], category: str) -> Dict[str, Any]:
        metadata = rule.get("metadata") or {}
        spring_window = metadata.get("spring_version_window") or {}
        micronaut_window = metadata.get("micronaut_version_window") or {}

        spring_spec = spring_window.get("spec") or rule.get("spring_version") or ""
        micronaut_spec = micronaut_window.get("spec") or rule.get("micronaut_version") or ""

        source_kind = (
            metadata.get("source_kind")
            or metadata.get("release_source_kind")
            or ("generated" if str(metadata.get("source", "")).strip().lower() == "synthetic" else "")
        )
        status = (
            metadata.get("status")
            or metadata.get("release_validation_status")
            or ("validated" if metadata.get("validated") else "")
        )
        confidence = metadata.get("confidence")
        evidence_count = metadata.get("evidence_count")
        if evidence_count in (None, ""):
            support_count = metadata.get("support_count")
            merged_duplicate_count = metadata.get("merged_duplicate_count")
            if isinstance(support_count, int):
                evidence_count = support_count
            elif isinstance(merged_duplicate_count, int):
                evidence_count = max(1, merged_duplicate_count)
            else:
                evidence_count = 0

        return {
            "spring_pattern": rule.get("spring_pattern", ""),
            "micronaut_pattern": rule.get("micronaut_pattern", ""),
            "category": category,
            "description": rule.get("description", ""),
            "complexity": rule.get("complexity", "medium"),
            "spring_version": rule.get("spring_version", ""),
            "micronaut_version": rule.get("micronaut_version", ""),
            "spring_version_spec": spring_spec,
            "spring_version_minimum": spring_window.get("minimum") or "",
            "spring_version_maximum": spring_window.get("maximum") or "",
            "micronaut_version_spec": micronaut_spec,
            "micronaut_version_minimum": micronaut_window.get("minimum") or "",
            "micronaut_version_maximum": micronaut_window.get("maximum") or "",
            "source_kind": source_kind,
            "status": status,
            "release_validation_status": metadata.get("release_validation_status") or status,
            "confidence": float(confidence) if isinstance(confidence, (int, float)) else 0.0,
            "evidence_count": int(evidence_count) if isinstance(evidence_count, int) else 0,
            "source": metadata.get("source") or "",
            "seed_id": metadata.get("seed_id") or "",
        }

    def _get_or_create_collection(self, name: str):
        """
        Safely retrieves or creates a ChromaDB collection.
        Handles potential database corruption by recreating collections if necessary.
        """
        if not self.available or self.client is None:
            return None
        try:
            return self.client.get_collection(name)
        except Exception:
            return self.client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def get_collection_stats(self, collection_name: str) -> Dict[str, int]:
        collection = self.collections.get(collection_name)
        if not collection:
            return {"name": collection_name, "count": 0}
        try:
            return {"name": collection_name, "count": int(collection.count())}
        except Exception:
            return {"name": collection_name, "count": 0}

    def load_dataset(self, dataset_file: str):
        """
        Load a specific dataset file, auto-detecting if it needs decryption.
        """
        dataset, _ = self.load_dataset_with_status(dataset_file)
        return dataset

    def load_dataset_with_status(self, dataset_file: str):
        """
        Load a dataset and return structured status so init/reporting can explain
        whether encrypted legacy datasets were actually available.
        """
        file_path = Path(dataset_file)

        status = {
            "requested_path": str(dataset_file),
            "actual_path": str(file_path),
            "encrypted": False,
            "loaded": False,
            "rule_count": 0,
            "reason": "",
            "key_source": "",
        }

        # If the file ends in .dat or if a .dat alternative exists, use decryption
        is_encrypted = file_path.suffix == '.dat'
        actual_path = file_path

        if not is_encrypted and Path(str(file_path) + ".dat").exists():
            actual_path = Path(str(file_path) + ".dat")
            is_encrypted = True

        status["actual_path"] = str(actual_path)
        status["encrypted"] = is_encrypted

        if not actual_path.exists():
            status["reason"] = "missing"
            return None, status

        if is_encrypted:
            return self._load_encrypted_dataset(actual_path, status)

        try:
            with open(actual_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            status["loaded"] = True
            status["rule_count"] = self._dataset_rule_count(dataset)
            status["reason"] = "ok_json"
            return dataset, status
        except Exception as exc:
            status["reason"] = f"json_error:{type(exc).__name__}"
            return None, status

    def _dataset_rule_count(self, dataset) -> int:
        if isinstance(dataset, list):
            return len(dataset)
        if isinstance(dataset, dict):
            return sum(len(items) for items in dataset.values() if isinstance(items, list))
        return 0

    def _load_encrypted_dataset(self, path: Path, status: Optional[Dict[str, Any]] = None):
        """
        Decrypts a protected dataset. Main logic is kept internal for security.
        """
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            from cryptography.hazmat.backends import default_backend
            import base64
            salt = b'spring2naut_rag_migration_2024'

            with open(path, 'rb') as f:
                encrypted_data = f.read()

            attempts = []
            for label, candidate_password in SecurityConfig.get_dataset_key_candidates():
                try:
                    password = candidate_password.encode('utf-8')
                    kdf = PBKDF2HMAC(
                        algorithm=hashes.SHA256(),
                        length=32,
                        salt=salt,
                        iterations=100000,
                        backend=default_backend()
                    )
                    key = base64.urlsafe_b64encode(kdf.derive(password))
                    fernet = Fernet(key)
                    decrypted_data = fernet.decrypt(encrypted_data)
                    dataset = json.loads(decrypted_data.decode('utf-8'))
                    if status is not None:
                        status["loaded"] = True
                        status["rule_count"] = self._dataset_rule_count(dataset)
                        status["reason"] = "ok_encrypted"
                        status["key_source"] = label
                    return dataset, status
                except Exception as exc:
                    attempts.append(f"{label}:{type(exc).__name__}")

            if status is not None:
                status["reason"] = "decrypt_failed"
                status["attempts"] = attempts
            return None, status
        except ImportError:
            if status is not None:
                status["reason"] = "missing_cryptography"
            return None, status
        except Exception as exc:
            if status is not None:
                status["reason"] = f"decrypt_error:{type(exc).__name__}"
            return None, status

    def search_annotation(self, spring_annotation: str, **kwargs) -> List[MigrationRule]:
        """
        Performs vector search for annotation patterns.
        Falls back to general code patterns if no specific result is found.
        """
        rules = self._search_collection("annotations", spring_annotation, **kwargs)
        if not rules:
            rules = self._search_collection("code_patterns", spring_annotation, **kwargs)
        return rules

    def search_dependency(self, spring_dep: str, **kwargs) -> List[MigrationRule]:
        """
        Performs vector search for dependency patterns.
        Only searches the dependencies collection to avoid nonsensical matches with code.
        """
        return self._search_collection("dependencies", spring_dep, **kwargs)

    def search_configuration(self, spring_prop: str, **kwargs) -> List[MigrationRule]:
        """
        Performs vector search for configuration property patterns.
        Only searches the configurations collection; does NOT fall back to code patterns
        to avoid nonsensical mappings between config keys and code annotations.
        """
        return self._search_collection("configurations", spring_prop, **kwargs)

    def _search_collection(self, collection_name: str, query: str, top_k: int = 3, **kwargs) -> List[MigrationRule]:
        """
        Performs hybrid search: Semantic Proximity + Exact Identifier Filter.
        Ensures that 'hallucinated' neighbors are rejected in favor of logical matches.
        """
        collection = self.collections.get(collection_name)
        if not collection:
            return []

        spring_version = kwargs.get("spring_version")
        micronaut_version = kwargs.get("micronaut_version")

        exact_rules = self._exact_metadata_matches(
            collection=collection,
            query=query,
            spring_version=spring_version,
            micronaut_version=micronaut_version,
        )
        if exact_rules:
            return exact_rules
            
        # 1. Broad Semantic Query
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=top_k)
        
        raw_rules = []
        if results and results['metadatas'] and results['metadatas'][0]:
            for meta in results['metadatas'][0]:
                raw_rules.append(self._metadata_to_rule(meta))

        compatible_rules = self._filter_rules_by_version(raw_rules, spring_version, micronaut_version)
        
        # 2. Hard Keyword Re-Ranking & Filtering (GA-grade Precision)
        # We strip @ and packages for a fair comparison
        query_id = query.split('.')[-1].replace("@", "").lower()
        keyword_matches = []
        
        for rule in compatible_rules:
            pattern_id = rule.spring_pattern.split('.')[-1].replace("@", "").lower()
            # Expert check: Is this a logical match or just a semantic neighbor?
            if query_id == pattern_id or query_id in rule.spring_pattern.lower() or rule.spring_pattern.lower() in query.lower():
                keyword_matches.append(rule)
        
        # If we have keyword matches, they take absolute priority
        if keyword_matches:
            return keyword_matches
            
        # If no keyword matches, we return semantic neighbors ONLY for code patterns
        if collection_name == "code_patterns":
            return compatible_rules
            
        return []

    def _exact_metadata_matches(
        self,
        collection,
        query: str,
        spring_version: Optional[str] = None,
        micronaut_version: Optional[str] = None,
    ) -> List[MigrationRule]:
        seen = set()
        matches: List[MigrationRule] = []

        for variant in self._exact_query_variants(query):
            try:
                result = collection.get(where={"spring_pattern": variant}, include=["metadatas"])
            except Exception:
                continue

            metadatas = result.get("metadatas") if isinstance(result, dict) else None
            if not metadatas:
                continue

            for metadata in metadatas:
                rule = self._metadata_to_rule(metadata)
                rule_key = (
                    rule.spring_pattern,
                    rule.micronaut_pattern,
                    rule.category,
                    rule.spring_version or "",
                    rule.micronaut_version or "",
                )
                if rule_key in seen:
                    continue
                seen.add(rule_key)
                matches.append(rule)

        return self._filter_rules_by_version(matches, spring_version, micronaut_version)

    def _exact_query_variants(self, query: str) -> List[str]:
        cleaned = str(query or "").strip()
        if not cleaned:
            return []

        variants: List[str] = []

        def add(value: str):
            value = value.strip()
            if value and value not in variants:
                variants.append(value)

        add(cleaned)
        if cleaned.startswith("@"):
            add(cleaned[1:])
        else:
            add(f"@{cleaned}")

        if ":" not in cleaned and "." in cleaned:
            tail = cleaned.split(".")[-1].strip()
            add(tail)
            if tail and not tail.startswith("@") and tail[:1].isupper():
                add(f"@{tail}")

        return variants

    def _filter_rules_by_version(
        self,
        rules: List[MigrationRule],
        spring_version: Optional[str],
        micronaut_version: Optional[str],
    ) -> List[MigrationRule]:
        if not spring_version or not micronaut_version:
            return rules

        compatible = [
            rule
            for rule in rules
            if VersionCompatibilityMatrix.is_version_compatible(rule, spring_version, micronaut_version)
        ]
        return compatible

    def _metadata_to_rule(self, metadata: Dict) -> MigrationRule:
        """
        Converts database metadata back into a rich MigrationRule object.
        """
        rule_metadata = {
            key: value
            for key, value in metadata.items()
            if key.startswith("spring_version_")
            or key.startswith("micronaut_version_")
            or key in {"source_kind", "status", "release_validation_status", "confidence", "evidence_count", "source", "seed_id"}
        }
        return MigrationRule(
            spring_pattern=metadata.get('spring_pattern', ''),
            micronaut_pattern=metadata.get('micronaut_pattern', ''),
            category=metadata.get('category', ''),
            description=metadata.get('description', ''),
            complexity=metadata.get('complexity', 'low'),
            spring_version=metadata.get('spring_version'),
            micronaut_version=metadata.get('micronaut_version'),
            metadata=rule_metadata or None,
        )
