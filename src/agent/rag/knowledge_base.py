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
        
        if merged_rules:
            # If we were previously ghost-partitioned, we need to clear everything first
            # to avoid duplicate IDs or mixed schemas
            for name, col in self.collections.items():
                try:
                    self.client.delete_collection(name)
                    self.collections[name] = self.client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
                except: pass
            self._index_rules(merged_rules)
        else:
            print("[WARN] Intelligence Engine running with limited internal patterns.")

    def _is_populated(self) -> bool:
        """
        Checks if the database is healthy and specialized collections are populated.
        Triggers a re-index if we detect 'Ghost Partitioning' (data only in generic collection).
        """
        try:
            counts = {name: col.count() for name, col in self.collections.items() if name != 'code_patterns'}
            total_specialized = sum(counts.values())
            # If we have 10k patterns but 0 are specialized, our DB is 'badly partitioned'
            if total_specialized == 0 and self.collections['code_patterns'].count() > 0:
                print(f"{YELLOW}[WARN]{RESET} Intelligence Engine detected as 'Ghost Partitioned' (0 specialized rules).")
                print(f"       Triggers structural re-alignment of 10,000 patterns...")
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
                metadatas = [{
                    "spring_pattern": r.get('spring_pattern', ''),
                    "micronaut_pattern": r.get('micronaut_pattern', ''),
                    "category": category,
                    "description": r.get('description', ''),
                    "complexity": r.get('complexity', 'medium'),
                    "spring_version": r.get('spring_version', ''),
                    "micronaut_version": r.get('micronaut_version', '')
                } for r in batch]
                
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                
                indexed_count += len(batch)
                elapsed = time.time() - start_time
                # Estimate remaining time
                rate = indexed_count / elapsed
                remaining = (total_rules - indexed_count) / rate
                
                print(f"  > Progress: {indexed_count}/{total_rules} ({ (indexed_count/total_rules * 100):.1f}%) | Est. Remaining: {int(remaining/60)}m {int(remaining%60)}s ", end='\r')
            
        print(f"\n[OK] Intelligence Engine indexed with {total_rules} patterns in {int((time.time() - start_time)/60)}m {int((time.time() - start_time)%60)}s.")

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
        
        if not is_encrypted and Path(str(file_path) + ".dat").exists():
            actual_path = Path(str(file_path) + ".dat")
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
            
        # 1. Broad Semantic Query
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=top_k)
        
        raw_rules = []
        if results and results['metadatas'] and results['metadatas'][0]:
            for meta in results['metadatas'][0]:
                raw_rules.append(self._metadata_to_rule(meta))
        
        # 2. Hard Keyword Re-Ranking & Filtering (GA-grade Precision)
        # We strip @ and packages for a fair comparison
        query_id = query.split('.')[-1].replace("@", "").lower()
        keyword_matches = []
        
        for rule in raw_rules:
            pattern_id = rule.spring_pattern.split('.')[-1].replace("@", "").lower()
            # Expert check: Is this a logical match or just a semantic neighbor?
            if query_id == pattern_id or query_id in rule.spring_pattern.lower() or rule.spring_pattern.lower() in query.lower():
                keyword_matches.append(rule)
        
        # If we have keyword matches, they take absolute priority
        if keyword_matches:
            return keyword_matches
            
        # If no keyword matches, we return semantic neighbors ONLY for code patterns
        if collection_name == "code_patterns":
            return raw_rules
            
        return []

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
