import os
import requests
import hashlib
from pathlib import Path
from dataclasses import dataclass

class MigrationConfig:
    """
    Global configuration for migration.
    Supports environment variables for flexibility.
    """
    SPRING_BOOT_VERSION = os.getenv("SPRING_BOOT_VERSION", "3.x")
    MICRONAUT_VERSION = os.getenv("MICRONAUT_VERSION", "4.10.8")
    
    # Vector Database Configuration
    VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chromadb")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./migration_db")
    
    # Embedding Model Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "microsoft/codebert-base")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
    
    # LLM Provider Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
    LLM_MODEL = os.getenv("LLM_MODEL", "codellama:7b")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    
    # Dataset Configuration
    DATASET_FILE = os.getenv("DATASET_FILE", "./migration_dataset.json")
    ENHANCED_DATASET_FILE = os.getenv("ENHANCED_DATASET_FILE", "./migration_dataset_enhanced.json")
    
    # Performance Parameters
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

class SecurityConfig:
    """
    GA-Grade IP Protection.
    Uses a 'Split-Key' architecture:
    Key = SHA-256(Remote_Token + Local_Salt)
    This ensures neither the Code nor the Remote Gist contains the actual password.
    """
    
    # The URL where you store your 'Remote Token' (random characters)
    KEY_VAULT_URL = os.getenv("KEY_VAULT_URL", "https://raw.githubusercontent.com/ajitpattar708/Spring2Naut-RAG/main/.vault/token.txt")
    
    # A local hidden identifier to make the key unique to your codebase
    _LOCAL_SALT = "spring2naut_rag_ga_v1_secure_salt_7788"
    _cached_key = None

    @classmethod
    def get_dataset_key(cls) -> str:
        """
        Derives the decryption password using a split-key strategy.
        """
        # 1. Check Environment Variable (Manual Override)
        env_key = os.getenv("DATASET_ENCRYPTION_PASSWORD")
        if env_key:
            return env_key
            
        # 2. Check Memory Cache
        if cls._cached_key:
            return cls._cached_key
            
        # 3. Check Local Hidden Cache File
        cache_file = Path(".vdb_key_cache")
        if cache_file.exists():
            try:
                cls._cached_key = cache_file.read_text().strip()
                return cls._cached_key
            except:
                pass
        
        # 4. Fetch Remote Token and Derive Key
        try:
            print(f"[INFO] Authorizing knowledge base via Split-Key Vault...")
            response = requests.get(cls.KEY_VAULT_URL, timeout=5)
            if response.status_code == 200:
                remote_token = response.text.strip()
                
                # DERIVE KEY: Combine Remote Token + Local Salt
                # This ensures the actual password is never in the Gist or the Code
                combined = f"{remote_token}{cls._LOCAL_SALT}"
                derived_password = hashlib.sha256(combined.encode()).hexdigest()
                
                cls._cached_key = derived_password
                cache_file.write_text(derived_password)
                return derived_password
        except Exception:
            print("[WARN] Split-Key Vault unreachable. Using community mode.")
            
        return ""

    # Property for compatibility
    @property
    def DATASET_KEY(self):
        return self.get_dataset_key()

    # Obfuscation flag
    OBFUSCATE_ARTIFACTS = os.getenv("OBFUSCATE_ARTIFACTS", "false").lower() == "true"
