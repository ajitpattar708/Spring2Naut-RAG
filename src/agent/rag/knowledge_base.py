import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from src.agent.core.config import MigrationConfig, SecurityConfig
from src.agent.core.models import MigrationRule

class KnowledgeService:
    """
    Abstract interface for knowledge retrieval.
    Can be implemented locally (RAG) or remotely (API).
    """
    def search_annotation(self, spring_annotation: str, **kwargs) -> List[MigrationRule]:
        raise NotImplementedError
        
    def search_dependency(self, spring_dep: str, **kwargs) -> List[MigrationRule]:
        raise NotImplementedError

class LocalMigrationKnowledgeBase(KnowledgeService):
    """
    Local implementation of the migration knowledge base using ChromaDB and CodeBERT.
    Includes logic for handling protected/encrypted datasets.
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or MigrationConfig.VECTOR_DB_PATH
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
            # CodeBERT is preferred for technical accuracy in code transformation
            self.embedding_model = SentenceTransformer(model_name)
        except Exception:
            # Fallback to a lighter model if preferred model is unavailable
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            
        test_embedding = self.embedding_model.encode(["test"])
        self.embedding_dimension = len(test_embedding[0])

    def _initialize_db(self):
        """
        Sets up the vector database and initializes collections for different pattern categories.
        """
        self.client = chromadb.PersistentClient(path=self.db_path)
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
        # Check if DB is already populated to avoid redundant indexing
        if self._is_populated():
            return

        print("[INFO] Initializing Knowledge Base for first-time use...")
        
        # 1. Load Community Dataset (Plain JSON)
        community_data = self.load_dataset(MigrationConfig.DATASET_FILE)
        
        # 2. Load Pro Dataset (Encrypted DAT)
        pro_data = self.load_dataset(MigrationConfig.ENHANCED_DATASET_FILE)
        
        # Merge datasets (Pro overrides/augments Community)
        merged_rules = self._merge_datasets(community_data, pro_data)
        
        if merged_rules:
            self._index_rules(merged_rules)
            print(f"[OK] Knowledge Base indexed with {len(merged_rules)} patterns.")
        else:
            print("[WARN] No datasets found. Falling back to limited internal patterns.")

    def _is_populated(self) -> bool:
        """Checks if the collections have existing data."""
        try:
            return self.collections["annotations"].count() > 0
        except Exception:
            return False

    def _merge_datasets(self, community: Optional[Dict], pro: Optional[Dict]) -> List[Dict]:
        """Merges Community and Pro datasets into a flat list of rules."""
        all_rules = []
        
        def process_source(source_data):
            if not source_data: return
            if isinstance(source_data, list):
                all_rules.extend(source_data)
            elif isinstance(source_data, dict):
                for category, rules in source_data.items():
                    if isinstance(rules, list):
                        for r in rules:
                            if 'category' not in r: r['category'] = category
                            all_rules.append(r)
        
        process_source(community)
        process_source(pro)
        return all_rules

    def _index_rules(self, rules: List[Dict]):
        """Populates ChromaDB collections with the rule data."""
        for rule in rules:
            category = rule.get('category', 'code_patterns')
            collection = self.collections.get(category, self.collections['code_patterns'])
            
            # Prepare metadata for storage
            metadata = {
                "spring_pattern": rule.get('spring_pattern', ''),
                "micronaut_pattern": rule.get('micronaut_pattern', ''),
                "category": category,
                "description": rule.get('description', ''),
                "complexity": rule.get('complexity', 'medium'),
                "spring_version": rule.get('spring_version', ''),
                "micronaut_version": rule.get('micronaut_version', '')
            }
            
            # Simple content for embedding
            content = f"{rule.get('spring_pattern', '')} {rule.get('description', '')}"
            embedding = self.embedding_model.encode([content]).tolist()[0]
            
            collection.add(
                ids=[rule.get('id', os.urandom(8).hex())],
                embeddings=[embedding],
                metadatas=[metadata]
            )

    def _get_or_create_collection(self, name: str):
        """
        Safely retrieves or creates a ChromaDB collection.
        Handles potential database corruption by recreating collections if necessary.
        """
        try:
            return self.client.get_collection(name)
        except Exception:
            return self.client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def load_dataset(self, dataset_file: str):
        """
        Load a specific dataset file, auto-detecting if it needs decryption.
        """
        file_path = Path(dataset_file)
        
        # If the file ends in .dat or if a .dat alternative exists, use decryption
        is_encrypted = file_path.suffix == '.dat'
        actual_path = file_path
        
        if not is_encrypted and file_path.with_suffix('.dat').exists():
            actual_path = file_path.with_suffix('.dat')
            is_encrypted = True
            
        if not actual_path.exists():
            return None
            
        if is_encrypted:
            return self._load_encrypted_dataset(actual_path)
            
        try:
            with open(actual_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def _load_encrypted_dataset(self, path: Path):
        """
        Decrypts a protected dataset. Main logic is kept internal for security.
        """
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            from cryptography.hazmat.backends import default_backend
            import base64
            
            # Key derivation logic (Simplified for this snippet, typically involves more robust verification)
            salt = b'spring2naut_rag_migration_2024'
            password = SecurityConfig.get_dataset_key().encode('utf-8')
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            
            with open(path, 'rb') as f:
                encrypted_data = f.read()
            
            fernet = Fernet(key)
            decrypted_data = fernet.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode('utf-8'))
        except Exception:
            # Handle decryption failure gracefully
            return None

    def search_annotation(self, spring_annotation: str, **kwargs) -> List[MigrationRule]:
        """
        Performs vector search for annotation patterns.
        """
        return self._search_collection("annotations", spring_annotation, **kwargs)

    def _search_collection(self, collection_name: str, query: str, top_k: int = 1, **kwargs) -> List[MigrationRule]:
        """
        Common search logic for all collections.
        """
        collection = self.collections.get(collection_name)
        if not collection:
            return []
            
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=top_k)
        
        # Mapping search results back to MigrationRule objects
        rules = []
        if results and results['metadatas']:
            for metadata in results['metadatas'][0]:
                rules.append(self._metadata_to_rule(metadata))
        return rules

    def _metadata_to_rule(self, metadata: Dict) -> MigrationRule:
        """
        Converts database metadata back into a rich MigrationRule object.
        """
        return MigrationRule(
            spring_pattern=metadata.get('spring_pattern', ''),
            micronaut_pattern=metadata.get('micronaut_pattern', ''),
            category=metadata.get('category', ''),
            description=metadata.get('description', ''),
            complexity=metadata.get('complexity', 'low'),
            # Additional fields reconstructed from metadata
        )
