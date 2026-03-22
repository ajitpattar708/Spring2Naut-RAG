import os
import hashlib
import shlex
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _default_runtime_search_roots(base_dir: str = ".") -> list[Path]:
    roots: list[Path] = []

    requested_root = Path(base_dir)
    roots.append(requested_root)

    repo_root = Path(__file__).resolve().parents[3]
    if repo_root not in roots:
        roots.append(repo_root)

    shared_root = Path(sys.prefix) / "share" / "spring2naut"
    if shared_root not in roots:
        roots.append(shared_root)

    return roots


def _resolve_default_artifact(candidates: tuple[str, ...], base_dir: str = ".") -> str:
    """
    Search common runtime locations for packaged artifacts.

    Search order:
    1. explicitly requested base directory
    2. repository root when running from source checkout
    3. installed shared data path for packaged CLI installs
    """
    for root in _default_runtime_search_roots(base_dir):
        for name in candidates:
            candidate = root / name
            if candidate.exists():
                if root == Path("."):
                    return f"./{name}"
                return str(candidate)
    return ""


def resolve_default_dataset_file(base_dir: str = ".") -> str:
    candidates = (
        "migration_dataset.json.dat",
        "migration_dataset.json",
    )
    resolved = _resolve_default_artifact(candidates, base_dir=base_dir)
    return resolved or "./migration_dataset.json"


def resolve_default_enhanced_dataset_file(base_dir: str = ".") -> str:
    """
    Prefer the cleaned enhanced dataset when present, then fall back to the
    existing enhanced/community artifacts. The returned path points to the
    real artifact on disk so runtime and release tooling use the same source.
    """
    candidates = (
        "migration_dataset_enhanced_cleaned.json.dat",
        "migration_dataset_enhanced_cleaned.json",
        "migration_dataset_enhanced.json.dat",
        "migration_dataset_enhanced.json",
        "migration_dataset.json.dat",
        "migration_dataset.json",
    )
    resolved = _resolve_default_artifact(candidates, base_dir=base_dir)
    return resolved or "./migration_dataset_enhanced.json"


def _expand_maven_repository_path(raw_path: str) -> str:
    expanded = str(raw_path or "").strip()
    if not expanded:
        return ""

    expanded = expanded.replace("${user.home}", str(Path.home()))
    for key, value in os.environ.items():
        expanded = expanded.replace(f"${{env.{key}}}", value)
    return os.path.expanduser(expanded)


def _extract_maven_repo_local_from_args(raw_args: str) -> str:
    try:
        tokens = shlex.split(str(raw_args or "").replace("\n", " "))
    except ValueError:
        tokens = str(raw_args or "").replace("\n", " ").split()

    for index, token in enumerate(tokens):
        if token.startswith("-Dmaven.repo.local="):
            return _expand_maven_repository_path(token.split("=", 1)[1])
        if token == "-Dmaven.repo.local" and index + 1 < len(tokens):
            return _expand_maven_repository_path(tokens[index + 1])
    return ""


def resolve_maven_local_repository(project_path: str = "") -> str:
    env_override = str(os.getenv("MAVEN_LOCAL_REPOSITORY", "") or "").strip()
    if env_override:
        return _expand_maven_repository_path(env_override)

    project_root = Path(os.path.expanduser(project_path)).resolve() if project_path else None
    if project_root:
        maven_config = project_root / ".mvn" / "maven.config"
        if maven_config.exists():
            try:
                configured_repo = _extract_maven_repo_local_from_args(
                    maven_config.read_text(encoding="utf-8")
                )
            except OSError:
                configured_repo = ""
            if configured_repo:
                return configured_repo

    maven_opts_repo = _extract_maven_repo_local_from_args(os.getenv("MAVEN_OPTS", ""))
    if maven_opts_repo:
        return maven_opts_repo

    settings_override = str(os.getenv("MAVEN_SETTINGS_FILE", "") or "").strip()
    settings_candidates = []
    if settings_override:
        settings_candidates.append(Path(os.path.expanduser(settings_override)))
    settings_candidates.append(Path.home() / ".m2" / "settings.xml")

    for settings_path in settings_candidates:
        if not settings_path.exists():
            continue
        try:
            root = ET.fromstring(settings_path.read_text(encoding="utf-8"))
        except (ET.ParseError, OSError):
            continue

        local_repository = (root.findtext("localRepository") or "").strip()
        if local_repository:
            expanded = _expand_maven_repository_path(local_repository)
            if expanded:
                return expanded

    return str(Path.home() / ".m2" / "repository")


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
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
    # Dataset Configuration
    DEFAULT_DATASET_FILE = resolve_default_dataset_file()
    DEFAULT_ENHANCED_DATASET_FILE = resolve_default_enhanced_dataset_file()
    DATASET_FILE = os.getenv("DATASET_FILE", DEFAULT_DATASET_FILE)
    ENHANCED_DATASET_FILE = os.getenv("ENHANCED_DATASET_FILE", DEFAULT_ENHANCED_DATASET_FILE)
    TARGET_PLATFORM_MANAGED_FILE = os.getenv("TARGET_PLATFORM_MANAGED_FILE", "")
    
    # Performance Parameters
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

    # Maven Central Verification
    MAVEN_CENTRAL_VERIFY = os.getenv("MAVEN_CENTRAL_VERIFY", "true").lower() == "true"
    MAVEN_CENTRAL_SEARCH_URL = os.getenv(
        "MAVEN_CENTRAL_SEARCH_URL",
        "https://search.maven.org/solrsearch/select",
    )
    MAVEN_CENTRAL_ARTIFACT_BASE_URL = os.getenv(
        "MAVEN_CENTRAL_ARTIFACT_BASE_URL",
        "https://repo1.maven.org/maven2",
    )
    MAVEN_CENTRAL_TIMEOUT = float(os.getenv("MAVEN_CENTRAL_TIMEOUT", "2.0"))
    MAVEN_LOCAL_REPOSITORY = resolve_maven_local_repository()
    BUILD_METADATA_ENABLED = os.getenv("BUILD_METADATA_ENABLED", "false").lower() == "true"
    BUILD_METADATA_COMMAND_TIMEOUT = float(os.getenv("BUILD_METADATA_COMMAND_TIMEOUT", "45"))
    BUILD_METADATA_TOTAL_BUDGET = float(os.getenv("BUILD_METADATA_TOTAL_BUDGET", "50"))
    BUILD_METADATA_OFFLINE_FIRST = os.getenv("BUILD_METADATA_OFFLINE_FIRST", "true").lower() == "true"
    BUILD_METADATA_ALLOW_ONLINE_FALLBACK = os.getenv("BUILD_METADATA_ALLOW_ONLINE_FALLBACK", "false").lower() == "true"

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
    _cached_token = None

    @classmethod
    def _get_legacy_algorithmic_salt_dynamic(cls) -> str:
        """
        Historical dynamic salt reconstruction used by older encrypted dataset flows.
        """
        try:
            repo_root = Path(__file__).resolve().parents[3]
            readme = repo_root / "README.md"
            license_file = repo_root / "LICENSE"
            src_dir = repo_root / "src"

            readme_len = len(readme.read_text(encoding="utf-8", errors="ignore")) if readme.exists() else 0
            license_part = license_file.read_text(encoding="utf-8", errors="ignore")[:10] if license_file.exists() else ""
            src_count = len([item for item in src_dir.rglob("*") if item.is_file()]) if src_dir.exists() else 0

            return f"{readme_len}{license_part}{src_count}"
        except Exception:
            return "spring2naut_v1_fallback"

    @classmethod
    def _load_dataset_token(cls) -> str:
        if cls._cached_token is not None:
            return cls._cached_token

        token = ""
        try:
            local_vault = Path(".vault/token.txt")
            if local_vault.exists():
                token = local_vault.read_text().strip()

            if not token:
                try:
                    import requests
                except ImportError:
                    requests = None
                if requests is not None:
                    response = requests.get(cls.KEY_VAULT_URL, timeout=5)
                    if response.status_code == 200:
                        token = response.text.strip()
        except Exception:
            token = ""

        cls._cached_token = token
        return token

    @classmethod
    def get_dataset_key_candidates(cls) -> list[tuple[str, str]]:
        """
        Returns ordered candidate passwords for automatic dataset decryption.
        Labels are safe to log; passwords are not.
        """
        candidates: list[tuple[str, str]] = []
        seen: set[str] = set()

        def add(label: str, value: str) -> None:
            token = str(value or "").strip()
            if not token or token in seen:
                return
            seen.add(token)
            candidates.append((label, token))

        env_key = os.getenv("DATASET_ENCRYPTION_PASSWORD")
        add("env_override", env_key)

        dataset_token = cls._load_dataset_token()
        if dataset_token:
            add(
                "token_hardcoded_salt",
                hashlib.sha256(f"{dataset_token}{cls._get_algorithmic_salt()}".encode()).hexdigest(),
            )
            add(
                "token_dynamic_salt",
                hashlib.sha256(f"{dataset_token}{cls._get_legacy_algorithmic_salt_dynamic()}".encode()).hexdigest(),
            )

        add("legacy_fallback", "Spring2Naut_RAG_Migration_Agent_v1.0")
        return candidates

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

        candidates = cls.get_dataset_key_candidates()
        if candidates:
            cls._cached_key = candidates[0][1]
            return cls._cached_key
        return "Spring2Naut_RAG_Migration_Agent_v1.0"

    @property
    def DATASET_KEY(self):
        return self.get_dataset_key()

    # Obfuscation flag
    OBFUSCATE_ARTIFACTS = os.getenv("OBFUSCATE_ARTIFACTS", "false").lower() == "true"
