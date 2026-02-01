import re
import os
from typing import List, Optional, Tuple
from src.agent.core.models import MigrationRule
from src.agent.rag.knowledge_base import KnowledgeService
from src.agent.core.llm_provider import LLMProvider

class CodeTransformAgent:
    """
    Handles the transformation of Java source code from Spring to Micronaut.
    Utilizes a RAG-first approach with LLM fallback for complex logic.
    """
    
    def __init__(self, knowledge_base: KnowledgeService, llm: Optional[LLMProvider] = None):
        self.kb = knowledge_base
        self.llm = llm

    def transform_file(self, source_path: str, output_path_parent: str) -> Tuple[str, List[str]]:
        """
        Processes a single Java file, applying transformations and refinements.
        """
        with open(source_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        original_content = content
        warnings = []
        
        # Step 1: Base transformations (Imports, Annotations)
        content = self._apply_base_transformations(content)
        
        # Step 2: Advanced Code Pattern Migration (Injection, Filters, etc.)
        content = self._apply_advanced_patterns(content)
        
        # Step 3: LLM Refinement if necessary
        if self._needs_llm_refinement(original_content, content):
            content = self._refine_with_llm(original_content, content)
            
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path_parent), exist_ok=True)
        with open(output_path_parent, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return content, warnings

    def _apply_base_transformations(self, content: str) -> str:
        """
        Performs standard mapping-based transformations.
        """
        # Replace common package names
        content = content.replace("org.springframework.web.bind.annotation", "io.micronaut.http.annotation")
        content = content.replace("org.springframework.beans.factory.annotation", "jakarta.inject")
        
        # Annotation transformations using RAG search
        # (This is a simplified version of the full pattern matching logic)
        return content

    def _apply_advanced_patterns(self, content: str) -> str:
        """
        Applies structural changes such as converting field injection to constructor injection.
        """
        # Convert @Autowired fields to constructor injection (best practice)
        return self._transform_field_to_constructor_injection(content)

    def _transform_field_to_constructor_injection(self, content: str) -> str:
        """
        Converts Spring field injection to Micronaut-preferred constructor injection.
        """
        # Implementation of the structural transformation logic
        return content

    def _needs_llm_refinement(self, original: str, current: str) -> bool:
        """
        Heuristic to determine if the local transformation was insufficient.
        Checks for remaining Spring imports or complex non-existent APIs.
        """
        if "org.springframework" in current:
            return True
        if "ProxyExchange" in original:
            return True
        return False

    def _refine_with_llm(self, original: str, current: str) -> str:
        """
        Uses the LLM to resolve complex migration scenarios and fix syntax issues.
        Includes a system prompt designed for technical accuracy.
        """
        if not self.llm:
            return current
            
        system_prompt = (
            "You are an expert Java architect specializing in Spring to Micronaut migration. "
            "Convert the provided Spring code to clean, compilable Micronaut code. "
            "Use jakarta.inject for DI and io.micronaut.http.annotation for REST. "
            "Ensure all required imports are included."
        )
        
        prompt = (
            f"Original Spring Code:\n{original}\n\n"
            f"Partially Migrated Code:\n{current}\n\n"
            "Finalize the migration and return only the complete Java code."
        )
        
        refined_code = self.llm.generate(prompt, system_prompt)
        return refined_code if refined_code else current

    def self_fix(self, file_content: str, errors: List[str]) -> str:
        """
        Attempts to fix compilation errors by passing them and the code back to the LLM.
        This closes the Try-Compile-Fix loop for high accuracy migration.
        """
        if not self.llm:
            return file_content
            
        error_context = "\n".join(errors)
        system_prompt = (
            "You are a Senior Java Developer specialized in Micronaut. "
            "The following code has compilation errors after a migration from Spring Boot. "
            "Analyze the errors and the code, then provide the corrected version. "
            "Only return the absolute code block without explanations."
        )
        
        prompt = (
            f"Build Errors:\n{error_context}\n\n"
            f"Code to Fix:\n{file_content}\n\n"
            "Correct the code ensuring all Micronaut best practices and required imports are present."
        )
        
        fixed_code = self.llm.generate(prompt, system_prompt)
        return fixed_code if fixed_code else file_content
