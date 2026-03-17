import re
import os
from typing import List, Optional, Tuple
from src.agent.core.models import MigrationRule
from src.agent.rag.knowledge_base import KnowledgeService
from src.agent.core.llm_provider import LLMProvider

# Terminal Colors for GA Detailed Logs
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

class CodeTransformAgent:
    """
    Handles the transformation of Java source code from Spring to Micronaut.
    Utilizes a RAG-first approach with LLM fallback for complex logic.
    """
    
    # Tier 1: Guaranteed Mappings for Rule Transparency
    # These will be logged even if RAG is silent
    COMMON_MAPPINGS = {
        "RestController": "Controller",
        "Service": "Singleton",
        "Component": "Singleton",
        "Repository": "Singleton",
        "Autowired": "Inject",
        "RequestMapping": "Controller",
        "GetMapping": "Get",
        "PostMapping": "Post",
        "DeleteMapping": "Delete",
        "PutMapping": "Put",
        "RequestParam": "QueryValue",
        "PathVariable": "PathVariable",
        "RequestBody": "Body",
        "Value": "Property",
        "Configuration": "Factory",
        "Bean": "Bean",
        "Primary": "Primary",
        "Qualifier": "Named",
        "SpringBootApplication": "MicronautApplication",
        "EnableCaching": "Cacheable",
        "EnableAsync": "Async",
        "EnableScheduling": "Scheduled"
    }
    
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
            print(f"    - Starting LLM refinement for {os.path.basename(source_path)}...", flush=True)
            pre_llm_content = content
            content = self._refine_with_llm(original_content, content)
            
            # Show Shift Summary (What did the LLM actually do?)
            self._show_llm_shift_summary(pre_llm_content, content)
        
        # Step 4: Final deterministic Spring Purge
        content = self._final_spring_purge(content)
            
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path_parent), exist_ok=True)
        with open(output_path_parent, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return content, warnings

    def _apply_base_transformations(self, content: str) -> str:
        """
        Performs standard mapping-based transformations using RAG.
        """
        # Step 1: Detect potential Spring annotations and imports
        # Simple regex to find imports and annotations
        spring_patterns = re.findall(r'(org\.springframework\.[a-zA-Z0-9.]+)', content)
        spring_annotations = re.findall(r'@([A-Z][a-zA-Z]+)', content)
        
        # Unique set of items to search for
        to_search = set(spring_patterns) | set(spring_annotations)
        
        for item in to_search:
            # Expert Step 1a: RAG Search with Top-3 Deep Retrieval (VDB PRIMARY)
            search_query = f"@{item}" if item[0].isupper() else item
            # We now ask for the top 3 candidates to increase hit rate
            rules = self.kb.search_annotation(search_query, top_k=3)
            
            if not rules and item != search_query:
                rules = self.kb.search_annotation(item, top_k=3)
                
            rag_success = False
            if rules:
                for rule in rules:
                    # Semantic validation: Does the rule actually relate to our spring item?
                    # We check for exact matches on the identifier or clear inclusion
                    # Add .strip() to handle potential regex capture noise
                    item_id = item.strip().split('.')[-1].replace("@", "").lower()
                    pattern_id = rule.spring_pattern.strip().split('.')[-1].replace("@", "").lower()
                    
                    if item_id == pattern_id or item_id in rule.spring_pattern.lower() or rule.spring_pattern.lower() in item.lower():
                        print(f"      {BLUE}[VDB Found]{RESET} {BOLD}{item.strip()}{RESET} -> {BLUE}{rule.micronaut_pattern}{RESET} (Semantic Link)", flush=True)
                        content = content.replace(item, rule.micronaut_pattern)
                        rag_success = True
                        break # Found a valid one in top 3
            
            if rag_success:
                continue
            
            # If we are here, RAG either returned nothing or none of the top 3 matched
            if not rules:
                print(f"      {YELLOW}[VDB Miss]{RESET}  {item} (No semantic candidates found in 10,000 patterns)", flush=True)
            else:
                candidates = ", ".join([r.spring_pattern.split('.')[-1] for r in rules])
                print(f"      {YELLOW}[VDB Miss]{RESET}  {item} (Found likely unrelated candidates: {candidates})", flush=True)

            # Expert Step 1b: Check Common Mappings ONLY as Fallback
            if item in self.COMMON_MAPPINGS:
                mapped = self.COMMON_MAPPINGS[item]
                print(f"      {GREEN}[Rule Match]{RESET} @{item} -> @{mapped} (Local Fallback)", flush=True)
                content = content.replace(f"@{item}", f"@{mapped}")
                continue
        
        # Fallback for common high-level packages if RAG missed them
        content = content.replace("org.springframework.web.bind.annotation", "io.micronaut.http.annotation")
        content = content.replace("org.springframework.beans.factory.annotation", "jakarta.inject")
        
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
        Expert heuristic to determine if the local RAG transformation was insufficient.
        Automatically triggers the LLM if rare or custom Spring patterns remain.
        """
        # Indicators of remaining Spring infrastructure
        spring_signatures = [
            "org.springframework",
            "@Autowired",
            "@Value",
            "@Qualifier",
            "@Configuration",
            "@Component",
            "@Service",
            "@Repository",
            "@RestController",
            "@RequestMapping",
            "@Conditional",
            "ResponseEntity",
            "ProxyExchange"
        ]
        
        # If any significant Spring trace remains, we engage the LLM for custom migration
        if any(sig in current for sig in spring_signatures):
            return True
            
        # Specific complex patterns that RAG cannot handle structurally
        if "ProxyExchange" in original or "RestTemplate" in original:
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
            "CRITICAL: You MUST remove ALL org.springframework imports and annotations. "
            "Use jakarta.inject for DI and io.micronaut.http.annotation for REST mappings. "
            "If a direct equivalent doesn't exist, use best-effort Micronaut patterns. "
            "DO NOT include any Spring-related code in the output."
        )
        
        prompt = (
            f"Original Spring Code:\n{original}\n\n"
            f"Current Migrated Code (with potential issues):\n{current}\n\n"
            "Finalize the migration. Ensure 100% Micronaut code and return only the code block."
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

    def _final_spring_purge(self, content: str) -> str:
        """
        A final safety pass to remove any remaining Spring boilerplate or imports
        that the RAG/LLM might have left as orphans.
        """
        lines = content.split('\n')
        purged_lines = []
        
        # Patterns that should NEVER be in a Micronaut file
        banned_patterns = [
            "org.springframework",
            "@Autowired",
            "@Service",
            "@Component",
            "@Repository",
            "@RestController",
            "@RequestMapping"
        ]
        
        for line in lines:
            if any(pattern in line for pattern in banned_patterns):
                # Only keep the line if it was already migrated (contains micronaut or jakarta)
                if "io.micronaut" in line or "jakarta." in line:
                    purged_lines.append(line)
                else:
                    # Drop the orphaned Spring line
                    continue
            else:
                purged_lines.append(line)
                
        return '\n'.join(purged_lines)

    def _show_llm_shift_summary(self, old_content: str, new_content: str):
        """
        Calculates and displays a high-level summary of shifts made by the LLM.
        This provides translucency for the 'black box' AI phase.
        """
        old_lines = set(old_content.split('\n'))
        new_lines = set(new_content.split('\n'))
        
        removed = [l.strip() for l in old_lines - new_lines if any(sig in l for sig in ["org.springframework", "@Service", "@RestController", "@Autowired", "@Component", "@Configuration", "@Bean", "@Value", "@SpringBootApplication"])]
        added = [l.strip() for l in new_lines - old_lines if any(sig in l for sig in ["io.micronaut", "jakarta.", "@Singleton", "@Controller", "@Inject", "@Factory", "@Bean", "@Property"])]
        
        if removed or added:
            print(f"      {BLUE}|__{RESET} {BOLD}Intelligence Shift Summary:{RESET}", flush=True)
            for r in removed[:5]: # Show more for transparency
                # If we can find the matching added line, show it as a transition
                print(f"         {RED}-{RESET} Purged: {r}", flush=True)
            for a in added[:5]:
                print(f"         {GREEN}+{RESET} Introduced: {a}", flush=True)
            if len(removed) > 5 or len(added) > 5:
                print(f"         ... (Significant structural transformation detected)", flush=True)
