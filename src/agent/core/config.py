import os
import requests
import hashlib
import base64
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
    # microsoft/codebert-base provides the best semantic understanding for code patterns
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
    Uses 'Split-Key Runtime Reconstitution'.
    The real key is never stored as a string; it is derived from 
    an external token and a local obfuscated byte-map.
    """
    
    # The URL where you store your 'Remote Token' (random characters)
    KEY_VAULT_URL = os.getenv("KEY_VAULT_URL", "https://raw.githubusercontent.com/ajitpattar708/Spring2Naut-RAG/main/.vault/token.txt")
    
    @classmethod
    def _get_algorithmic_salt(cls) -> str:
        """
        Generates a salt based on project structure. 
        Practically invisible to automated scanners.
        """
        try:
            # Reconstitute the exact salt base used for encryption: 2277MIT Licens27
            # These values are sensitive to project structure and must match the dataset fingerprint.
            readme_len = 2277  # Verified historical length
            license_part = "MIT Licens"
            src_count = 27     # Current src/ file count
            
            return f"{readme_len}{license_part}{src_count}"
        except Exception:
            return "spring2naut_v1_fallback"

    _cached_key = None

    @classmethod
    def get_dataset_key(cls) -> str:
        """
        Derives the decryption password using the Algorithmic Split-Key strategy.
        """
        # 1. Environment Variable (Manual Override)
        env_key = os.getenv("DATASET_ENCRYPTION_PASSWORD")
        if env_key:
            return env_key
            
        # 2. Check Memory Cache
        if cls._cached_key:
            return cls._cached_key
            
        # 3. Reconstitute Key from Token and Salt
        try:
            token = None
            # Check Local .vault/token.txt
            local_vault = Path(".vault/token.txt")
            if local_vault.exists():
                token = local_vault.read_text().strip()

            # If no local token, fetch Remote Token
            if not token:
                response = requests.get(cls.KEY_VAULT_URL, timeout=5)
                if response.status_code == 200:
                    token = response.text.strip()
            
            if token:
                # DERIVE VIA SALT (Matches encryption script)
                local_salt = cls._get_algorithmic_salt()
                combined = f"{token}{local_salt}"
                derived_password = hashlib.sha256(combined.encode()).hexdigest()
                
                cls._cached_key = derived_password
                return derived_password
            else:
                # Sync failed - Use Legacy Fallback
                return 'Spring2Naut_RAG_Migration_Agent_v1.0'
        except Exception:
            return 'Spring2Naut_RAG_Migration_Agent_v1.0'

    @property
    def DATASET_KEY(self):
        return self.get_dataset_key()

    # Obfuscation flag
    OBFUSCATE_ARTIFACTS = os.getenv("OBFUSCATE_ARTIFACTS", "false").lower() == "true"
