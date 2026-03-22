from typing import List

from src.agent.core.models import MigrationRule


class KnowledgeService:
    """
    Lightweight contract for migration-rule lookup.
    Keeping this separate avoids importing heavy vector-db dependencies
    in modules that only need the interface.
    """

    def search_annotation(self, spring_annotation: str, **kwargs) -> List[MigrationRule]:
        raise NotImplementedError

    def search_dependency(self, spring_dep: str, **kwargs) -> List[MigrationRule]:
        raise NotImplementedError

    def search_configuration(self, spring_prop: str, **kwargs) -> List[MigrationRule]:
        raise NotImplementedError


class LLMProvider:
    """
    Lightweight contract for text/code generation providers.
    """

    def is_available(self) -> bool:
        raise NotImplementedError

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        raise NotImplementedError
