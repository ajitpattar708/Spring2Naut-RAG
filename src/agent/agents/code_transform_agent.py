import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

try:
    import javalang
except ImportError:  # pragma: no cover - optional dependency outside managed runtime
    javalang = None
from src.agent.core.models import MigrationRule, VersionCompatibilityMatrix
from src.agent.core.interfaces import KnowledgeService, LLMProvider

# Terminal Colors for GA Detailed Logs
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"


@dataclass(frozen=True)
class JavaAstField:
    name: str
    annotations: Tuple[str, ...]


@dataclass(frozen=True)
class JavaAstMethod:
    name: str
    annotations: Tuple[str, ...]
    parameter_annotations: Tuple[str, ...]


@dataclass(frozen=True)
class JavaAstClass:
    name: str
    annotations: Tuple[str, ...]
    fields: Tuple[JavaAstField, ...]
    methods: Tuple[JavaAstMethod, ...]
    constructor_count: int


@dataclass(frozen=True)
class JavaAstContext:
    parse_ok: bool
    classes: Tuple[JavaAstClass, ...]

    @property
    def primary_class(self) -> Optional[JavaAstClass]:
        return self.classes[0] if self.classes else None

    def has_annotation(self, annotation_name: str) -> bool:
        return any(annotation_name in item.annotations for item in self.classes)

    def has_method_annotation(self, annotation_name: str) -> bool:
        return any(
            annotation_name in method.annotations
            for item in self.classes
            for method in item.methods
        )

    def annotation_names(self) -> Tuple[str, ...]:
        names = []
        for item in self.classes:
            names.extend(item.annotations)
            for field in item.fields:
                names.extend(field.annotations)
            for method in item.methods:
                names.extend(method.annotations)
                names.extend(method.parameter_annotations)
        return tuple(dict.fromkeys(names))

    def injectable_field_names(self) -> Tuple[str, ...]:
        names = []
        for item in self.classes:
            for field in item.fields:
                if "Autowired" in field.annotations or "Inject" in field.annotations:
                    names.append(field.name)
        return tuple(names)


@dataclass
class FileMigrationStats:
    deterministic_hits: int = 0
    vdb_hits: int = 0
    vdb_misses: int = 0
    llm_used: bool = False
    llm_reason: str = ""

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
        "Controller": "Controller",
        "Repository": "Singleton",
        "Autowired": "Inject",
        "RequestMapping": "Controller",
        "GetMapping": "Get",
        "PostMapping": "Post",
        "PatchMapping": "Patch",
        "DeleteMapping": "Delete",
        "PutMapping": "Put",
        "RequestParam": "QueryValue",
        "PathVariable": "PathVariable",
        "RequestHeader": "Header",
        "RequestBody": "Body",
        "Value": "Property",
        "Configuration": "Factory",
        "Bean": "Bean",
        "Primary": "Primary",
        "Qualifier": "Named",
        "Secured": "Secured",
        "ControllerAdvice": "Singleton",
        "ExceptionHandler": "Error",
        "ResponseStatus": "Status",
        "ResponseBody": "",
        "ConfigurationProperties": "ConfigurationProperties",
        "Validated": "Validated",
        "Transactional": "Transactional",
        "FeignClient": "Client",
        "Cacheable": "Cacheable",
        "CachePut": "CachePut",
        "CacheEvict": "CacheInvalidate",
        "Scheduled": "Scheduled",
        "Async": "Async",
        "SpringBootTest": "MicronautTest",
        "MockBean": "MockBean",
        "SpringBootApplication": "",
        "EnableCaching": "",
        "EnableAsync": "",
        "EnableScheduling": ""
    }
    IMPORT_MAPPINGS = {
        "org.springframework.web.bind.annotation.RestController": "io.micronaut.http.annotation.Controller",
        "org.springframework.web.bind.annotation.RequestMapping": "io.micronaut.http.annotation.Controller",
        "org.springframework.web.bind.annotation.GetMapping": "io.micronaut.http.annotation.Get",
        "org.springframework.web.bind.annotation.PostMapping": "io.micronaut.http.annotation.Post",
        "org.springframework.web.bind.annotation.PatchMapping": "io.micronaut.http.annotation.Patch",
        "org.springframework.web.bind.annotation.DeleteMapping": "io.micronaut.http.annotation.Delete",
        "org.springframework.web.bind.annotation.PutMapping": "io.micronaut.http.annotation.Put",
        "org.springframework.web.bind.annotation.RequestParam": "io.micronaut.http.annotation.QueryValue",
        "org.springframework.web.bind.annotation.PathVariable": "io.micronaut.http.annotation.PathVariable",
        "org.springframework.web.bind.annotation.RequestHeader": "io.micronaut.http.annotation.Header",
        "org.springframework.web.bind.annotation.RequestBody": "io.micronaut.http.annotation.Body",
        "org.springframework.web.bind.annotation.ControllerAdvice": "jakarta.inject.Singleton",
        "org.springframework.web.bind.annotation.ExceptionHandler": "io.micronaut.http.annotation.Error",
        "org.springframework.web.bind.annotation.ResponseBody": "",
        "org.springframework.web.bind.annotation.ResponseStatus": "io.micronaut.http.annotation.Status",
        "org.springframework.web.bind.annotation.RequestMethod": "",
        "org.springframework.beans.factory.annotation.Autowired": "jakarta.inject.Inject",
        "org.springframework.beans.factory.annotation.Value": "io.micronaut.context.annotation.Property",
        "org.springframework.beans.factory.annotation.Qualifier": "jakarta.inject.Named",
        "org.springframework.cloud.openfeign.FeignClient": "io.micronaut.http.client.annotation.Client",
        "org.springframework.cloud.openfeign.EnableFeignClients": "",
        "org.springframework.security.access.annotation.Secured": "io.micronaut.security.annotation.Secured",
        "org.springframework.stereotype.Service": "jakarta.inject.Singleton",
        "org.springframework.stereotype.Controller": "io.micronaut.http.annotation.Controller",
        "org.springframework.stereotype.Component": "jakarta.inject.Singleton",
        "org.springframework.stereotype.Repository": "jakarta.inject.Singleton",
        "org.springframework.context.annotation.Configuration": "io.micronaut.context.annotation.Factory",
        "org.springframework.context.ApplicationContext": "io.micronaut.context.ApplicationContext",
        "org.springframework.context.annotation.Bean": "io.micronaut.context.annotation.Bean",
        "org.springframework.context.annotation.Primary": "io.micronaut.context.annotation.Primary",
        "org.springframework.boot.context.properties.ConfigurationProperties": "io.micronaut.context.annotation.ConfigurationProperties",
        "org.springframework.validation.annotation.Validated": "io.micronaut.validation.Validated",
        "org.springframework.transaction.annotation.Transactional": "jakarta.transaction.Transactional",
        "org.springframework.cache.annotation.Cacheable": "io.micronaut.cache.annotation.Cacheable",
        "org.springframework.cache.annotation.CachePut": "io.micronaut.cache.annotation.CachePut",
        "org.springframework.cache.annotation.CacheEvict": "io.micronaut.cache.annotation.CacheInvalidate",
        "org.springframework.cache.annotation.EnableCaching": "",
        "org.springframework.scheduling.annotation.Async": "io.micronaut.scheduling.annotation.Async",
        "org.springframework.scheduling.annotation.Scheduled": "io.micronaut.scheduling.annotation.Scheduled",
        "org.springframework.scheduling.annotation.EnableAsync": "",
        "org.springframework.scheduling.annotation.EnableScheduling": "",
        "org.springframework.http.ResponseEntity": "io.micronaut.http.HttpResponse",
        "org.springframework.http.RequestEntity": "io.micronaut.http.HttpRequest",
        "org.springframework.http.HttpStatus": "io.micronaut.http.HttpStatus",
        "org.springframework.http.MediaType": "io.micronaut.http.MediaType",
        "org.springframework.data.domain.Page": "io.micronaut.data.model.Page",
        "org.springframework.data.domain.Slice": "io.micronaut.data.model.Slice",
        "org.springframework.data.domain.Pageable": "io.micronaut.data.model.Pageable",
        "org.springframework.data.domain.Sort": "io.micronaut.data.model.Sort",
        "org.springframework.data.domain.PageRequest": "",
        "org.springframework.data.repository.CrudRepository": "io.micronaut.data.repository.CrudRepository",
        "org.springframework.data.repository.PagingAndSortingRepository": "io.micronaut.data.repository.PageableRepository",
        "org.springframework.data.jpa.repository.JpaRepository": "io.micronaut.data.jpa.repository.JpaRepository",
        "org.springframework.data.jpa.repository.JpaSpecificationExecutor": "io.micronaut.data.jpa.repository.JpaSpecificationExecutor",
        "org.springframework.data.jpa.domain.Specification": "io.micronaut.data.jpa.repository.criteria.Specification",
        "org.springframework.data.jpa.repository.EntityGraph": "io.micronaut.data.jpa.annotation.EntityGraph",
        "org.springframework.data.jpa.repository.EntityGraph.EntityGraphType": "io.micronaut.data.jpa.annotation.EntityGraph.Type",
        "org.springframework.data.jpa.repository.QueryHints": "io.micronaut.data.annotation.QueryHints",
        "jakarta.persistence.QueryHint": "io.micronaut.data.annotation.QueryHint",
        "javax.persistence.QueryHint": "io.micronaut.data.annotation.QueryHint",
        "org.springframework.data.jpa.repository.config.EnableJpaRepositories": "",
        "org.springframework.ui.Model": "java.util.Map",
        "org.springframework.ui.ModelMap": "java.util.Map",
        "org.springframework.web.servlet.ModelAndView": "io.micronaut.views.ModelAndView",
        "org.springframework.web.client.RestTemplate": "io.micronaut.http.client.HttpClient",
        "org.springframework.util.StringUtils": "io.micronaut.core.util.StringUtils",
        "org.springframework.boot.web.client.RestTemplateBuilder": "",
        "org.springframework.boot.test.context.SpringBootTest": "io.micronaut.test.extensions.junit5.annotation.MicronautTest",
        "org.springframework.boot.test.mock.mockito.MockBean": "io.micronaut.test.annotation.MockBean",
        "org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase": "",
        "org.springframework.boot.test.web.server.LocalServerPort": "",
        "org.springframework.boot.SpringApplication": "io.micronaut.runtime.Micronaut",
        "org.springframework.boot.autoconfigure.SpringBootApplication": "",
    }
    ANNOTATION_IMPORTS = {
        "Controller": "io.micronaut.http.annotation.Controller",
        "Get": "io.micronaut.http.annotation.Get",
        "Post": "io.micronaut.http.annotation.Post",
        "Put": "io.micronaut.http.annotation.Put",
        "Delete": "io.micronaut.http.annotation.Delete",
        "Patch": "io.micronaut.http.annotation.Patch",
        "Body": "io.micronaut.http.annotation.Body",
        "QueryValue": "io.micronaut.http.annotation.QueryValue",
        "PathVariable": "io.micronaut.http.annotation.PathVariable",
        "Header": "io.micronaut.http.annotation.Header",
        "Error": "io.micronaut.http.annotation.Error",
        "Status": "io.micronaut.http.annotation.Status",
        "MicronautTest": "io.micronaut.test.extensions.junit5.annotation.MicronautTest",
        "MockBean": "io.micronaut.test.annotation.MockBean",
        "Client": "io.micronaut.http.client.annotation.Client",
        "KafkaListener": "io.micronaut.configuration.kafka.annotation.KafkaListener",
        "Topic": "io.micronaut.configuration.kafka.annotation.Topic",
        "RabbitListener": "io.micronaut.rabbitmq.annotation.RabbitListener",
        "Queue": "io.micronaut.rabbitmq.annotation.Queue",
        "Disabled": "org.junit.jupiter.api.Disabled",
        "Property": "io.micronaut.context.annotation.Property",
        "ConfigurationProperties": "io.micronaut.context.annotation.ConfigurationProperties",
        "Factory": "io.micronaut.context.annotation.Factory",
        "Bean": "io.micronaut.context.annotation.Bean",
        "Primary": "io.micronaut.context.annotation.Primary",
        "Requires": "io.micronaut.context.annotation.Requires",
        "Singleton": "jakarta.inject.Singleton",
        "Inject": "jakarta.inject.Inject",
        "Named": "jakarta.inject.Named",
        "Secured": "io.micronaut.security.annotation.Secured",
        "Validated": "io.micronaut.validation.Validated",
        "Transactional": "jakarta.transaction.Transactional",
        "Cacheable": "io.micronaut.cache.annotation.Cacheable",
        "CachePut": "io.micronaut.cache.annotation.CachePut",
        "CacheInvalidate": "io.micronaut.cache.annotation.CacheInvalidate",
        "Scheduled": "io.micronaut.scheduling.annotation.Scheduled",
        "Async": "io.micronaut.scheduling.annotation.Async",
    }
    TYPE_MAPPINGS = {
        "ResponseEntity": "HttpResponse",
        "RequestEntity": "HttpRequest",
        "RestTemplate": "HttpClient",
        "SpringApplication": "Micronaut",
    }
    SILENT_SYMBOLS = {
        "Test",
        "BeforeEach",
        "AfterEach",
        "BeforeAll",
        "AfterAll",
        "Disabled",
        "ExtendWith",
        "Override",
    }
    DETERMINISTIC_SYMBOLS = {
        "SpringBootTest",
        "MockBean",
        "RestTemplate",
        "RequestEntity",
        "StringUtils",
        "MediaType",
        "org.springframework.util.StringUtils",
        "org.springframework.http.MediaType",
        "org.springframework.boot.test.context.SpringBootTest",
        "org.springframework.boot.test.mock.mockito.MockBean",
        "org.springframework.web.client.RestTemplate",
        "org.springframework.http.RequestEntity",
        "org.springframework.context.ApplicationContext",
    }
    STRUCTURAL_ONLY_SYMBOLS = {
        "WebMvcTest",
        "DataJpaTest",
        "AutoConfigureTestDatabase",
        "LocalServerPort",
        "WebEnvironment",
        "BindingResult",
        "WebDataBinder",
        "ModelAttribute",
        "InitBinder",
        "JpaRepository",
        "JpaSpecificationExecutor",
        "Specification",
        "EntityGraph",
        "EntityGraphType",
        "EnableJpaRepositories",
        "FeignClient",
        "EnableFeignClients",
        "KafkaListener",
        "RabbitListener",
        "MockMvc",
        "MockMvcRequestBuilders",
        "MockMvcResultMatchers",
        "ResultActions",
        "PageImpl",
        "MarshallingView",
        "org.springframework.web.servlet.view.xml.MarshallingView",
        "JCacheManagerCustomizer",
        "MutableConfiguration",
        "KafkaTemplate",
        "RabbitTemplate",
        "org.springframework.validation.BindingResult",
        "org.springframework.web.bind.WebDataBinder",
        "org.springframework.web.bind.annotation.ModelAttribute",
        "org.springframework.web.bind.annotation.InitBinder",
        "org.springframework.data.jpa.repository.JpaRepository",
        "org.springframework.data.jpa.repository.JpaSpecificationExecutor",
        "org.springframework.data.jpa.domain.Specification",
        "org.springframework.data.jpa.repository.EntityGraph",
        "org.springframework.data.jpa.repository.EntityGraph.EntityGraphType",
        "org.springframework.data.jpa.repository.config.EnableJpaRepositories",
        "org.springframework.cloud.openfeign.FeignClient",
        "org.springframework.cloud.openfeign.EnableFeignClients",
        "org.springframework.kafka.annotation.KafkaListener",
        "org.springframework.amqp.rabbit.annotation.RabbitListener",
        "org.springframework.test.web.servlet.MockMvc",
        "org.springframework.test.web.servlet.request.MockMvcRequestBuilders",
        "org.springframework.test.web.servlet.result.MockMvcResultMatchers",
        "org.springframework.test.web.servlet.ResultActions",
        "org.springframework.data.domain.PageImpl",
        "org.springframework.boot.autoconfigure.cache.JCacheManagerCustomizer",
        "javax.cache.configuration.MutableConfiguration",
        "org.springframework.kafka.core.KafkaTemplate",
        "org.springframework.amqp.rabbit.core.RabbitTemplate",
        "org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest",
        "org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest",
        "org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase",
        "org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase.Replace",
        "org.springframework.boot.test.web.server.LocalServerPort",
        "org.springframework.boot.test.context.SpringBootTest.WebEnvironment",
        "ConditionalOnProperty",
        "ConditionalOnBean",
        "ConditionalOnMissingBean",
        "ConditionalOnClass",
        "ConditionalOnMissingClass",
        "ConditionalOnExpression",
        "Profile",
        "org.springframework.boot.autoconfigure.condition.ConditionalOnProperty",
        "org.springframework.boot.autoconfigure.condition.ConditionalOnBean",
        "org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean",
        "org.springframework.boot.autoconfigure.condition.ConditionalOnClass",
        "org.springframework.boot.autoconfigure.condition.ConditionalOnMissingClass",
        "org.springframework.boot.autoconfigure.condition.ConditionalOnExpression",
        "org.springframework.context.annotation.Profile",
        "Query",
        "Param",
        "DateTimeFormat",
        "Formatter",
        "Errors",
        "Validator",
        "MutableSortDefinition",
        "PropertyComparator",
        "LocalValidatorFactoryBean",
        "LocaleContextHolder",
        "ImportRuntimeHints",
        "RuntimeHints",
        "RuntimeHintsRegistrar",
        "ComponentScan",
        "FilterType",
        "Assert",
        "ToStringCreator",
        "DataAccessException",
        "ObjectRetrievalFailureException",
        "SerializationUtils",
        "org.springframework.data.jpa.repository.Query",
        "org.springframework.data.repository.Repository",
        "org.springframework.data.repository.query.Param",
        "org.springframework.format.annotation.DateTimeFormat",
        "org.springframework.format.Formatter",
        "org.springframework.validation.Errors",
        "org.springframework.validation.Validator",
        "org.springframework.beans.support.MutableSortDefinition",
        "org.springframework.beans.support.PropertyComparator",
        "org.springframework.validation.beanvalidation.LocalValidatorFactoryBean",
        "org.springframework.context.i18n.LocaleContextHolder",
        "org.springframework.context.annotation.ImportRuntimeHints",
        "org.springframework.aot.hint.RuntimeHints",
        "org.springframework.aot.hint.RuntimeHintsRegistrar",
        "org.springframework.context.annotation.ComponentScan",
        "org.springframework.context.annotation.FilterType",
        "org.springframework.util.Assert",
        "org.springframework.core.style.ToStringCreator",
        "org.springframework.dao.DataAccessException",
        "org.springframework.orm.ObjectRetrievalFailureException",
        "org.springframework.util.SerializationUtils",
    }
    
    def __init__(
        self,
        knowledge_base: KnowledgeService,
        llm: Optional[LLMProvider] = None,
        spring_version: str = "3.x",
        micronaut_version: str = "4.x",
    ):
        self.kb = knowledge_base
        self.llm = llm
        self.llm_available = bool(llm and llm.is_available())
        self.spring_version = spring_version
        self.micronaut_version = micronaut_version
        self.compatibility_info = VersionCompatibilityMatrix.get_compatibility_info(
            spring_version,
            micronaut_version,
        )
        self._current_file_stats = FileMigrationStats()

    def transform_file(self, source_path: str, output_path_parent: str) -> Tuple[str, List[str]]:
        """
        Processes a single Java file, applying transformations and refinements.
        """
        self._current_file_stats = FileMigrationStats()
        with open(source_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        original_content = content
        warnings = []
        ast_context = self._build_ast_context(content)
        
        # Step 1: Base transformations (Imports, Annotations)
        content = self._apply_base_transformations(content, ast_context)
        
        # Step 2: Advanced Code Pattern Migration (Injection, Filters, etc.)
        content = self._apply_advanced_patterns(content, ast_context)
        
        # Step 3: LLM Refinement if necessary
        llm_reason = self._llm_refinement_reason(original_content, content, ast_context)
        if self.llm_available and llm_reason:
            self._current_file_stats.llm_used = True
            self._current_file_stats.llm_reason = llm_reason
            print(
                f"    {MAGENTA}[LLM Refinement]{RESET} {BOLD}{os.path.basename(source_path)}{RESET} "
                f"{MAGENTA}(reason: {llm_reason}){RESET}",
                flush=True,
            )
            pre_llm_content = content
            content = self._refine_with_llm(original_content, content, source_path)
            
            # Show Shift Summary (What did the LLM actually do?)
            self._show_llm_shift_summary(pre_llm_content, content)
        
        # Step 4: Final deterministic Spring Purge
        content = self._final_spring_purge(content)
        content = self._sanitize_llm_output(content)
        content = self._normalize_micronaut_output(content)
        content = self._ensure_constructor_for_required_final_fields(content)
        content = self._restore_safe_original_imports(content, original_content)
        content = self._sort_import_block(content)
        content = self._normalize_whitespace(content)
            
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path_parent), exist_ok=True)
        with open(output_path_parent, 'w', encoding='utf-8') as f:
            f.write(content)

        self._print_file_migration_summary(source_path)
            
        return content, warnings

    def _print_file_migration_summary(self, source_path: str) -> None:
        stats = self._current_file_stats
        llm_flag = f"{MAGENTA}yes{RESET}" if stats.llm_used else "no"
        llm_reason = f"{MAGENTA}{stats.llm_reason}{RESET}" if stats.llm_reason else "-"
        print(
            f"      {CYAN}[File Summary]{RESET} {os.path.basename(source_path)} | "
            f"deterministic_hits={stats.deterministic_hits} "
            f"vdb_hits={stats.vdb_hits} "
            f"vdb_misses={stats.vdb_misses} "
            f"llm_used={llm_flag} "
            f"llm_reason={llm_reason}",
            flush=True,
        )

    def _build_ast_context(self, content: str) -> JavaAstContext:
        if javalang is None:
            return JavaAstContext(parse_ok=False, classes=tuple())
        try:
            tree = javalang.parse.parse(content)
        except (javalang.parser.JavaSyntaxError, TypeError, IndexError, StopIteration):
            return JavaAstContext(parse_ok=False, classes=tuple())

        classes = []
        for _, node in tree.filter(javalang.tree.ClassDeclaration):
            class_annotations = tuple(annotation.name for annotation in node.annotations)
            fields = []
            for field in node.fields:
                field_annotations = tuple(annotation.name for annotation in field.annotations)
                for declarator in field.declarators:
                    fields.append(JavaAstField(name=declarator.name, annotations=field_annotations))
            methods = tuple(
                JavaAstMethod(
                    name=method.name,
                    annotations=tuple(annotation.name for annotation in method.annotations),
                    parameter_annotations=tuple(
                        annotation.name
                        for parameter in method.parameters
                        for annotation in parameter.annotations
                    ),
                )
                for method in node.methods
            )
            classes.append(
                JavaAstClass(
                    name=node.name,
                    annotations=class_annotations,
                    fields=tuple(fields),
                    methods=methods,
                    constructor_count=len(node.constructors),
                )
            )
        return JavaAstContext(parse_ok=True, classes=tuple(classes))

    def _apply_base_transformations(self, content: str, ast_context: Optional[JavaAstContext] = None) -> str:
        """
        Performs standard mapping-based transformations using RAG.
        """
        content = self._transform_request_mapping_annotations(content, ast_context)
        content = self._apply_deterministic_import_mappings(content)
        content = self._apply_deterministic_type_mappings(content)
        scan_content = self._strip_java_comments_for_symbol_scan(content)

        # Step 1: Detect potential Spring annotations and imports
        # Simple regex to find imports and annotations
        spring_patterns = re.findall(r'(org\.springframework\.[a-zA-Z0-9.]+)', scan_content)
        spring_annotations = re.findall(r'(?<![\w"])@([A-Z][a-zA-Z]+)\b', scan_content)
        spring_simple_names = {pattern.split('.')[-1] for pattern in spring_patterns}
        if ast_context and ast_context.parse_ok:
            known_annotations = set(ast_context.annotation_names())
            spring_annotations = [annotation for annotation in spring_annotations if annotation in known_annotations]
        spring_annotations = [
            annotation
            for annotation in spring_annotations
            if annotation in spring_simple_names or annotation in self.COMMON_MAPPINGS
        ]
        
        # Unique set of items to search for
        to_search = set(spring_patterns) | set(spring_annotations)
        
        for item in to_search:
            if self._should_ignore_symbol(item):
                continue

            if self._should_use_deterministic_rule(item):
                content = self._apply_direct_symbol_rule(content, item)
                continue

            # Expert Step 1a: RAG Search with Top-3 Deep Retrieval (VDB PRIMARY)
            search_query = f"@{item}" if item[0].isupper() else item
            # We now ask for the top 3 candidates to increase hit rate
            rules = self.kb.search_annotation(
                search_query,
                top_k=3,
                spring_version=self.spring_version,
                micronaut_version=self.micronaut_version,
            )
            
            if not rules and item != search_query:
                rules = self.kb.search_annotation(
                    item,
                    top_k=3,
                    spring_version=self.spring_version,
                    micronaut_version=self.micronaut_version,
                )
                
            rag_success = False
            if rules:
                normalized_candidates = []
                for rule in rules:
                    # Semantic validation: Does the rule actually relate to our spring item?
                    # We check for exact matches on the identifier or clear inclusion
                    # Add .strip() to handle potential regex capture noise
                    item_id = item.strip().split('.')[-1].replace("@", "").lower()
                    pattern_id = rule.spring_pattern.strip().split('.')[-1].replace("@", "").lower()
                    normalized_candidates.append((rule, item_id, pattern_id))

                exact_match = next((rule for rule, item_id, pattern_id in normalized_candidates if item_id == pattern_id), None)
                candidate_rules = [exact_match] if exact_match else [rule for rule, item_id, pattern_id in normalized_candidates if item_id in rule.spring_pattern.lower() or rule.spring_pattern.lower() in item.lower()]

                for rule in candidate_rules:
                    if rule:
                        if self._is_manual_review_rule(rule):
                            print(
                                f"      {CYAN}[Governed Review]{RESET} {BOLD}{item.strip()}{RESET} -> {CYAN}{rule.micronaut_pattern}{RESET}",
                                flush=True,
                            )
                            rag_success = True
                            break
                        normalized_replacement = self._normalize_retrieved_replacement(item, rule.micronaut_pattern)
                        print(
                            f"      {BLUE}[VDB Found]{RESET} {BOLD}{item.strip()}{RESET} -> {BLUE}{normalized_replacement}{RESET} (Semantic Link)",
                            flush=True,
                        )
                        self._current_file_stats.vdb_hits += 1
                        content = self._replace_spring_symbol(content, item, normalized_replacement)
                        rag_success = True
                        break # Found a valid one in top 3
            
            if rag_success:
                continue
            
            # If we are here, RAG either returned nothing or none of the top 3 matched
            if not rules:
                print(
                    f"      {YELLOW}[VDB Miss]{RESET}  {item} (No semantic candidates found in indexed KB)",
                    flush=True,
                )
                self._current_file_stats.vdb_misses += 1
            else:
                candidates = ", ".join([r.spring_pattern.split('.')[-1] for r in rules])
                print(f"      {YELLOW}[VDB Miss]{RESET}  {item} (Found likely unrelated candidates: {candidates})", flush=True)
                self._current_file_stats.vdb_misses += 1

            # Expert Step 1b: Check Common Mappings ONLY as Fallback
            if item in self.COMMON_MAPPINGS:
                mapped = self.COMMON_MAPPINGS[item]
                replacement = f"@{mapped}" if mapped else ""
                human_target = replacement if replacement else "(remove annotation)"
                print(f"      {GREEN}[Rule Match]{RESET} @{item} -> {human_target} (Local Fallback)", flush=True)
                self._current_file_stats.deterministic_hits += 1
                content = self._replace_spring_symbol(content, f"@{item}", replacement)
                continue
        
        # Fallback for common high-level packages if RAG missed them
        content = content.replace("org.springframework.web.bind.annotation", "io.micronaut.http.annotation")
        content = content.replace("org.springframework.beans.factory.annotation", "jakarta.inject")
        
        return content

    def _strip_java_comments_for_symbol_scan(self, content: str) -> str:
        stripped = re.sub(r"/\*.*?\*/", "", str(content or ""), flags=re.DOTALL)
        stripped = re.sub(r"(?m)^[ \t]*//.*$", "", stripped)
        return stripped

    def _should_ignore_symbol(self, item: str) -> bool:
        cleaned = str(item or "").strip()
        if not cleaned:
            return True
        if cleaned in self.SILENT_SYMBOLS:
            return True
        if cleaned.endswith("."):
            return True
        if cleaned.startswith("org.springframework.samples."):
            return True
        if cleaned.startswith("org.junit."):
            return True
        if cleaned.startswith("org.springframework.test.web.servlet.request.MockMvcRequestBuilders."):
            return True
        if cleaned.startswith("org.springframework.test.web.servlet.result.MockMvcResultMatchers."):
            return True
        return False

    def _should_use_deterministic_rule(self, item: str) -> bool:
        cleaned = str(item or "").strip()
        if cleaned in self.DETERMINISTIC_SYMBOLS:
            return True
        if cleaned in self.STRUCTURAL_ONLY_SYMBOLS:
            return True
        if cleaned in self.COMMON_MAPPINGS:
            return True
        if cleaned in self.TYPE_MAPPINGS:
            return True
        if cleaned in self.IMPORT_MAPPINGS:
            return True
        return False

    def _apply_direct_symbol_rule(self, content: str, item: str) -> str:
        cleaned = str(item or "").strip()
        replacement = ""

        if cleaned in self.STRUCTURAL_ONLY_SYMBOLS:
            print(
                f"      {GREEN}[Rule Match]{RESET} {cleaned} -> structured deterministic transform",
                flush=True,
            )
            self._current_file_stats.deterministic_hits += 1
            return content

        if cleaned in self.IMPORT_MAPPINGS:
            replacement = self.IMPORT_MAPPINGS[cleaned]
            if replacement:
                print(
                    f"      {GREEN}[Rule Match]{RESET} {cleaned} -> {replacement} (Deterministic)",
                    flush=True,
                )
            self._current_file_stats.deterministic_hits += 1
            content = self._replace_spring_symbol(content, cleaned, replacement)
            return content

        if cleaned in self.COMMON_MAPPINGS:
            mapped = self.COMMON_MAPPINGS[cleaned]
            replacement = f"@{mapped}" if mapped else ""
            human_target = replacement if replacement else "(remove annotation)"
            print(f"      {GREEN}[Rule Match]{RESET} @{cleaned} -> {human_target} (Deterministic)", flush=True)
            self._current_file_stats.deterministic_hits += 1
            return self._replace_spring_symbol(content, f"@{cleaned}", replacement)

        if cleaned in self.TYPE_MAPPINGS:
            replacement = self.TYPE_MAPPINGS[cleaned]
            print(
                f"      {GREEN}[Rule Match]{RESET} {cleaned} -> {replacement} (Deterministic)",
                flush=True,
            )
            self._current_file_stats.deterministic_hits += 1
            return self._replace_spring_symbol(content, cleaned, replacement)

        return content
    
    def _apply_advanced_patterns(self, content: str, ast_context: Optional[JavaAstContext] = None) -> str:
        """
        Applies structural changes such as converting field injection to constructor injection.
        """
        content = self._transform_exception_handler_patterns(content, ast_context)
        content = self._transform_conditional_on_property(content)
        content = self._transform_conditional_on_expression(content)
        content = self._transform_requires_condition_patterns(content)
        content = self._transform_runtime_hints_patterns(content)
        content = self._transform_factory_annotation_patterns(content)
        content = self._transform_micronaut_annotation_array_values(content)
        content = self._transform_transactional_patterns(content)
        content = self._transform_value_annotations(content)
        content = self._transform_response_entity_patterns(content)
        content = self._transform_feign_client_patterns(content)
        content = self._transform_kafka_listener_patterns(content)
        content = self._transform_rabbit_listener_patterns(content)
        content = self._transform_kafka_template_patterns(content)
        content = self._transform_rabbit_template_patterns(content)
        content = self._transform_string_utils_patterns(content)
        content = self._transform_test_validation_support_patterns(content)
        content = self._transform_entity_lookup_patterns(content)
        content = self._transform_serialization_support_patterns(content)
        content = self._transform_assert_patterns(content)
        content = self._transform_to_string_creator_patterns(content)
        content = self._transform_date_time_format_patterns(content)
        content = self._transform_formatter_patterns(content)
        content = self._transform_validator_patterns(content)
        content = self._transform_custom_init_binder_validator_patterns(content)
        content = self._transform_introspection_patterns(content)
        content = self._transform_xml_binding_patterns(content)
        content = self._transform_marshalling_view_patterns(content)
        content = self._transform_property_comparator_patterns(content)
        content = self._transform_paging_patterns(content)
        content = self._transform_page_api_usage_patterns(content)
        content = self._transform_model_patterns(content)
        content = self._transform_model_attribute_patterns(content)
        content = self._transform_init_binder_patterns(content)
        content = self._transform_binding_result_patterns(content)
        content = self._transform_custom_init_binder_validator_patterns(content)
        content = self._transform_controller_prepopulation_patterns(content)
        content = self._transform_cache_customizer_patterns(content)
        content = self._transform_jpa_repository_bootstrap_patterns(content)
        content = self._transform_jpa_specification_patterns(content)
        content = self._transform_repository_interface_patterns(content)
        content = self._transform_view_controller_runtime_patterns(content)
        content = self._transform_test_slice_patterns(content)
        content = self._transform_integration_http_test_patterns(content)
        content = self._transform_field_level_mockbean_patterns(content)
        content = self._transform_jakarta_namespace_patterns(content)
        content = self._transform_field_to_constructor_injection(content, ast_context)
        content = self._apply_version_specific_compatibility_patterns(content)
        return content

    def _transform_introspection_patterns(self, content: str) -> str:
        if "@Introspected" in content:
            return content
        if "@Entity" not in content and "@MappedSuperclass" not in content:
            return content

        class_match = re.search(
            r"(?m)^(?P<indent>[ \t]*)(?:(?:public|protected|private)\s+)?(?:abstract\s+|final\s+)?class\s+\w+[^{]*\{",
            content,
        )
        if not class_match:
            return content

        content = self._ensure_import(content, "io.micronaut.core.annotation.Introspected")
        class_match = re.search(
            r"(?m)^(?P<indent>[ \t]*)(?:(?:public|protected|private)\s+)?(?:abstract\s+|final\s+)?class\s+\w+[^{]*\{",
            content,
        )
        if not class_match:
            return content
        content = content[: class_match.start()] + f"{class_match.group('indent')}@Introspected\n" + content[class_match.start() :]
        return content

    def _apply_version_specific_compatibility_patterns(self, content: str) -> str:
        version_patterns = dict(self.compatibility_info.get("version_specific_patterns", {}) or {})
        if not version_patterns:
            return content

        config_properties_rule = dict(version_patterns.get("@ConfigurationProperties", {}) or {})
        if config_properties_rule and "@ConfigurationProperties" in content:
            note = str(
                config_properties_rule.get("note")
                or "Review this configuration-properties migration against the target Micronaut line."
            ).strip()
            content = self._ensure_manual_review_comment_near_first_annotation(
                content,
                annotation="@ConfigurationProperties",
                note=note,
            )
        return content

    def _ensure_manual_review_comment_near_first_annotation(
        self,
        content: str,
        *,
        annotation: str,
        note: str,
    ) -> str:
        review_comment = self._render_manual_review_comment(note)
        if review_comment in content:
            return content

        line_match = re.search(rf"(?m)^(?P<indent>[ \t]*){re.escape(annotation)}\b", content)
        if not line_match:
            return content

        insert_at = line_match.start()
        indent = line_match.group("indent")
        self._current_file_stats.deterministic_hits += 1
        print(
            f"      {GREEN}[Rule Match]{RESET} {annotation} -> target-version compatibility review marker",
            flush=True,
        )
        return content[:insert_at] + f"{indent}{review_comment}\n" + content[insert_at:]

    def _transform_runtime_hints_patterns(self, content: str) -> str:
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.context\.annotation\.ImportRuntimeHints;\n?",
            "",
            content,
        )
        content = re.sub(r"(?m)^[ \t]*@ImportRuntimeHints\([^)]*\)\s*\n?", "", content)

        has_runtime_hints = any(
            marker in content
            for marker in (
                "RuntimeHintsRegistrar",
                "RuntimeHints ",
                "registerHints(",
                "org.springframework.aot.hint.RuntimeHints",
            )
        )
        if not has_runtime_hints:
            return content

        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.aot\.hint\.RuntimeHints;\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.aot\.hint\.RuntimeHintsRegistrar;\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^(?P<indent>[ \t]*public\s+)(?P<modifiers>(?:(?:final|abstract)\s+)*)class\s+(?P<name>\w+)\s+implements\s+RuntimeHintsRegistrar\s*\{",
            r"\g<indent>final class \g<name> {",
            content,
        )
        content = re.sub(r"(?m)^[ \t]*@Override\s*\n?", "", content)
        content = self._remove_method_containing_text(content, "registerHints(")
        content = self._ensure_empty_runtime_hints_class_constructor(content)
        return content

    def _transform_factory_annotation_patterns(self, content: str) -> str:
        return re.sub(r"@Factory\s*\([^)]*\)", "@Factory", content)

    def _transform_micronaut_annotation_array_values(self, content: str) -> str:
        return re.sub(
            r"@(Controller|Get|Post|Put|Delete|Patch)\(\{\s*\"([^\"]+)\"\s*\}\)",
            r'@\1("\2")',
            content,
        )

    def _restore_safe_original_imports(self, content: str, original: str) -> str:
        current_package = self._extract_package_name(content) or ""
        existing_imports = set(re.findall(r"(?m)^import\s+([\w.]+);", content))
        existing_simple_names = {
            import_path.rsplit(".", 1)[-1]: import_path for import_path in existing_imports
        }
        imports_to_add = []

        for import_path in re.findall(r"(?m)^import\s+([\w.]+);", original):
            import_line = f"import {import_path};"
            if self._is_framework_spring_import(import_line):
                continue
            if import_path in existing_imports:
                continue
            import_package = import_path.rsplit(".", 1)[0] if "." in import_path else ""
            if import_package == current_package:
                continue
            simple_name = import_path.rsplit(".", 1)[-1]
            existing_for_simple_name = existing_simple_names.get(simple_name)
            if (
                import_path.startswith("javax.")
                and existing_for_simple_name == self._jakarta_equivalent_import(import_path)
            ):
                continue
            if (
                simple_name == "QueryHint"
                and existing_for_simple_name == "io.micronaut.data.annotation.QueryHint"
                and import_path in {"jakarta.persistence.QueryHint", "javax.persistence.QueryHint"}
            ):
                continue
            if not re.search(rf"\b{re.escape(simple_name)}\b", content):
                continue
            imports_to_add.append(import_path)

        if not imports_to_add:
            return content

        package_match = re.search(r"^(package\s+[\w.]+;\s*)", content, re.MULTILINE)
        package_end = package_match.end() if package_match else 0
        import_block = "".join(f"\nimport {item};" for item in sorted(set(imports_to_add)))
        return content[:package_end] + import_block + content[package_end:]

    def _jakarta_equivalent_import(self, import_path: str) -> Optional[str]:
        namespace_pairs = (
            ("javax.validation.", "jakarta.validation."),
            ("javax.persistence.", "jakarta.persistence."),
            ("javax.annotation.", "jakarta.annotation."),
            ("javax.xml.bind.", "jakarta.xml.bind."),
            ("javax.activation.", "jakarta.activation."),
        )
        for source_ns, target_ns in namespace_pairs:
            if import_path.startswith(source_ns):
                return import_path.replace(source_ns, target_ns, 1)
        return None

    def _transform_transactional_patterns(self, content: str) -> str:
        return re.sub(r"@Transactional\s*\([^)]*\)", "@Transactional", content)

    def _transform_assert_patterns(self, content: str) -> str:
        if "Assert.notNull(" not in content and "import org.springframework.util.Assert;" not in content:
            return content
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.util\.Assert;\n?",
            "import java.util.Objects;\n",
            content,
        )
        content = re.sub(r"\bAssert\.notNull\(", "Objects.requireNonNull(", content)
        return content

    def _transform_to_string_creator_patterns(self, content: str) -> str:
        if "ToStringCreator" not in content:
            return content
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.core\.style\.ToStringCreator;\n?",
            "",
            content,
        )
        content = re.sub(
            r"return\s+new\s+ToStringCreator\(this\)[\s\S]*?\.toString\(\);",
            'return getClass().getSimpleName() + "(id=" + getId() + ", new=" + isNew() + ")";',
            content,
            count=1,
        )
        return content

    def _transform_date_time_format_patterns(self, content: str) -> str:
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.format\.annotation\.DateTimeFormat;\n?",
            "",
            content,
        )
        content = re.sub(r"(?m)^[ \t]*@DateTimeFormat\([^)]*\)\s*\n?", "", content)
        return content

    def _transform_formatter_patterns(self, content: str) -> str:
        if "Formatter<" not in content and "import org.springframework.format.Formatter;" not in content:
            return content
        formatter_match = re.search(r"implements\s+Formatter<(?P<target>[^>{]+)>", content)
        formatter_target = self._normalize_java_type_name(formatter_match.group("target")) if formatter_match else None
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.format\.Formatter;\n?",
            "",
            content,
        )
        if formatter_target:
            content = re.sub(
                r"\s+implements\s+Formatter<[^>{]+>",
                f" implements io.micronaut.core.convert.TypeConverter<String, {formatter_target}>",
                content,
                count=1,
            )
        else:
            content = re.sub(r"\s+implements\s+Formatter<[^>{]+>", "", content)
        content = re.sub(r"(?m)^[ \t]*@Override\s*\n(?=[ \t]*(?:public|protected|private)\s+(?:String|[A-Z][A-Za-z0-9_<>?, ]+)\s+(?:print|parse)\s*\()", "", content)
        if formatter_target and "Optional<" not in content and " parse(" in content:
            content = self._ensure_import(content, "io.micronaut.core.convert.ConversionContext")
            content = self._ensure_import(content, "java.util.Optional")
            content = self._ensure_formatter_type_converter(content, formatter_target)
        return content

    def _ensure_formatter_type_converter(self, content: str, formatter_target: str) -> str:
        if "Optional<" in content and "ConversionContext" in content and " convert(" in content:
            return content
        method_block = (
            "\n\t@Override\n"
            f"\tpublic Optional<{formatter_target}> convert(String object, Class<{formatter_target}> targetType, ConversionContext context) {{\n"
            "\t\ttry {\n"
            "\t\t\treturn Optional.of(parse(object, java.util.Locale.getDefault()));\n"
            "\t\t}\n"
            "\t\tcatch (java.text.ParseException ex) {\n"
            "\t\t\tcontext.reject(object, ex);\n"
            "\t\t\treturn Optional.empty();\n"
            "\t\t}\n"
            "\t}\n"
        )
        last_brace = content.rfind("}")
        if last_brace == -1:
            return content + method_block
        return content[:last_brace] + method_block + content[last_brace:]

    def _transform_validator_patterns(self, content: str) -> str:
        if "Validator" not in content and "Errors" not in content:
            return content
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.validation\.Validator;\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.validation\.Errors;\n?",
            "",
            content,
        )
        validator_rejects_errors = "errors.rejectValue(" in content
        content = re.sub(r"\s+implements\s+Validator\b", "", content)
        if validator_rejects_errors:
            content = re.sub(
                r"public\s+void\s+validate\s*\(\s*Object\s+(\w+)\s*,\s*Errors\s+\w+\s*\)",
                r"public void validate(Object \1, Map<String, Object> model)",
                content,
            )
            content = re.sub(
                r'(?m)^(?P<indent>[ \t]*)errors\.rejectValue\(\s*"(?P<field>[^"]+)"\s*,\s*(?P<code>[^,)\s]+|\"[^\"]+\")[^;]*;\s*$',
                lambda match: (
                    f'{match.group("indent")}addFieldError(model, "{match.group("field")}", {match.group("code")});\n'
                ),
                content,
            )
            content = self._ensure_validator_model_error_helper(content)
            content = self._ensure_import(content, "java.util.LinkedHashMap")
            content = self._ensure_import(content, "java.util.Map")
        else:
            content = re.sub(
                r"public\s+void\s+validate\s*\(\s*Object\s+(\w+)\s*,\s*Errors\s+\w+\s*\)",
                r"public void validate(Object \1)",
                content,
            )
        content = re.sub(r"(?m)^[ \t]*@Override\s*\n(?=[ \t]*(?:public|protected|private)\s+(?:void|boolean)\s+(?:validate|supports)\s*\()", "", content)
        return content

    def _transform_custom_init_binder_validator_patterns(self, content: str) -> str:
        validator_types = list(dict.fromkeys(re.findall(r"setValidator\(\s*new\s+([A-Z][A-Za-z0-9_]*)\s*\(", content)))
        if not validator_types:
            return content

        for validator_type in validator_types:
            field_name = validator_type[:1].lower() + validator_type[1:]
            if not re.search(rf"\b{re.escape(validator_type)}\s+{re.escape(field_name)}\b", content):
                content = self._inject_field_into_class(
                    content,
                    f"private final {validator_type} {field_name} = new {validator_type}();",
                )
            content = self._insert_validator_calls_into_methods(content, field_name)
        return content

    def _ensure_validator_model_error_helper(self, content: str) -> str:
        if "private void addFieldError(Map<String, Object> model, String field, String code)" in content:
            return content
        helper = (
            "\n\tprivate void addFieldError(Map<String, Object> model, String field, String code) {\n"
            '\t\t@SuppressWarnings("unchecked") Map<String, String> fieldErrors = (Map<String, String>) model.computeIfAbsent("_fieldErrors", key -> new LinkedHashMap<String, String>());\n'
            "\t\tfieldErrors.put(field, code);\n"
            '\t\tmodel.put("_validationError", Boolean.TRUE);\n'
            "\t}\n"
        )
        last_brace = content.rfind("}")
        if last_brace == -1:
            return content
        return content[:last_brace] + helper + content[last_brace:]

    def _ensure_empty_runtime_hints_class_constructor(self, content: str) -> str:
        class_match = re.search(
            r"(?m)^(?P<indent>[ \t]*)(?P<visibility>public\s+)?final\s+class\s+(?P<name>\w+)\s*\{",
            content,
        )
        if not class_match:
            return content
        class_name = class_match.group("name")
        if re.search(rf"\b{re.escape(class_name)}\s*\(", content[class_match.end():]):
            return content
        body_start = content.find("{", class_match.start())
        body_end = self._find_matching_brace(content, body_start)
        if body_end == -1:
            return content
        body = content[body_start + 1 : body_end]
        if body.strip():
            return content
        indent = class_match.group("indent")
        child_indent = indent + ("\t" if "\t" in content else "    ")
        replacement_body = f"\n{child_indent}private {class_name}() {{\n{child_indent}}}\n{indent}"
        return content[: body_start + 1] + replacement_body + content[body_end:]

    def _ensure_import(self, content: str, import_path: str) -> str:
        import_line = f"import {import_path};"
        if import_line in content:
            return content
        package_match = re.search(r"(?m)^package [^;]+;\n", content)
        if not package_match:
            return import_line + "\n" + content
        return content[: package_match.end()] + "\n" + import_line + "\n" + content[package_match.end() :]

    def _inject_field_into_class(self, content: str, field_line: str) -> str:
        class_match = re.search(r"(?m)^(?P<indent>[ \t]*)(?:public\s+)?class\s+\w+[^{]*\{", content)
        if not class_match:
            return content
        indent = class_match.group("indent")
        child_indent = indent + ("\t" if "\t" in content else "    ")
        insertion = f"\n{child_indent}{field_line}\n"
        return content[: class_match.end()] + insertion + content[class_match.end() :]

    def _insert_validator_calls_into_methods(self, content: str, validator_field_name: str) -> str:
        method_pattern = re.compile(
            r"(?m)^(?P<header>[ \t]*(?:public|protected|private\s+)?[^{;=\n]+?\)\s*\{)"
        )
        updated_parts = []
        last_index = 0
        for match in method_pattern.finditer(content):
            header = match.group("header")
            params_start = header.find("(")
            params_end = header.rfind(")")
            if params_start == -1 or params_end == -1 or params_end <= params_start:
                continue
            params = header[params_start + 1 : params_end]
            model_match = re.search(
                r"(?:Map<String,\s*Object>|ModelMap|Model)\s+(?P<name>\w+)(?=\s*(?:,|$))",
                params,
            )
            if not model_match or not re.search(r"@(?:jakarta\.validation\.)?Valid\b", params):
                continue
            valid_match = re.search(
                r"@(?:jakarta\.validation\.)?Valid\s+[^,)]*?\s+(?P<name>\w+)(?=\s*(?:,|$))",
                params,
            )
            if not valid_match:
                continue
            attribute_var = valid_match.group("name")
            model_var = model_match.group("name")
            body_start = content.find("{", match.start())
            if body_start == -1:
                continue
            body_end = self._find_matching_brace(content, body_start)
            if body_end == -1:
                continue
            body = content[body_start + 1 : body_end]
            indent_match = re.search(r"\n([ \t]*)\S", body)
            inner_indent = indent_match.group(1) if indent_match else "\t\t"
            validator_call = f"\n{inner_indent}{validator_field_name}.validate({attribute_var}, {model_var});\n"
            if f"{validator_field_name}.validate({attribute_var}, {model_var});" in body:
                continue
            insertion_index = body.find("if (hasValidationErrors(")
            if insertion_index == -1:
                insertion_index = 0
            updated_parts.append(content[last_index : body_start + 1])
            updated_parts.append(body[:insertion_index] + validator_call + body[insertion_index:])
            last_index = body_end
        if not updated_parts:
            return content
        updated_parts.append(content[last_index:])
        return "".join(updated_parts)

    def _transform_controller_prepopulation_patterns(self, content: str) -> str:
        content = self._transform_pet_form_controller_prepopulation(content)
        content = self._transform_visit_form_controller_prepopulation(content)
        content = self._transform_model_attribute_loader_prepopulation(content)
        return content

    def _transform_model_attribute_loader_prepopulation(self, content: str) -> str:
        if "@Controller" not in content or "@RestController" in content:
            return content

        loaders = self._collect_model_attribute_loader_methods(content)
        if not loaders:
            return content

        methods = self._collect_string_returning_controller_methods(content)
        transformed = False

        for method in reversed(methods):
            annotations = method["annotations"]
            if not any(marker in annotations for marker in ("@Post", "@Put", "@Patch")):
                continue

            entity_param = self._extract_form_backing_parameter(method["params"])
            if not entity_param:
                continue

            loader = self._find_matching_loader_method(method["params"], entity_param["type"], loaders)
            if not loader:
                continue
            if f"{loader['name']}(" in method["body"] or "isBeanEffectivelyEmpty(" in method["body"]:
                continue

            existing_var = f"existing{entity_param['type'].split('.')[-1]}"
            prefix_lines = [
                f"{loader['return_type']} {existing_var} = {loader['name']}({', '.join(loader['call_args'])});",
                f"if (isBeanEffectivelyEmpty({entity_param['name']})) {{",
                f"    {entity_param['name']} = {existing_var};",
                "}",
            ]
            if method["model_name"]:
                prefix_lines.append(f'model.put("{entity_param["name"]}", {entity_param["name"]});')

            if all(line in method["body"] for line in prefix_lines):
                continue

            indent = self._infer_method_body_indent(method["body"])
            insertion = "".join(f"\n{indent}{line}" for line in prefix_lines)
            replacement = self._render_string_controller_method(
                method,
                return_type="String",
                body=insertion + method["body"],
            )
            content = content[: method["start"]] + replacement + content[method["end"] :]
            transformed = True

        if transformed:
            content = self._ensure_helper_method(
                content,
                method_name="isBeanEffectivelyEmpty",
                parameter_signature="Object bean",
                method_body=(
                    "if (bean == null) {\n"
                    "\t\t\treturn true;\n"
                    "\t\t}\n"
                    "\t\tClass<?> current = bean.getClass();\n"
                    "\t\twhile (current != null && current != Object.class) {\n"
                    "\t\t\tfor (java.lang.reflect.Field field : current.getDeclaredFields()) {\n"
                    "\t\t\t\tif (java.lang.reflect.Modifier.isStatic(field.getModifiers()) || field.isSynthetic()) {\n"
                    "\t\t\t\t\tcontinue;\n"
                    "\t\t\t\t}\n"
                    '\t\t\t\tif ("id".equals(field.getName())) {\n'
                    "\t\t\t\t\tcontinue;\n"
                    "\t\t\t\t}\n"
                    "\t\t\t\ttry {\n"
                    "\t\t\t\t\tfield.setAccessible(true);\n"
                    "\t\t\t\t\tObject value = field.get(bean);\n"
                    "\t\t\t\t\tif (value == null) {\n"
                    "\t\t\t\t\t\tcontinue;\n"
                    "\t\t\t\t\t}\n"
                    "\t\t\t\t\tif (value instanceof CharSequence sequence && sequence.length() == 0) {\n"
                    "\t\t\t\t\t\tcontinue;\n"
                    "\t\t\t\t\t}\n"
                    "\t\t\t\t\tif (value instanceof java.util.Collection<?> collection && collection.isEmpty()) {\n"
                    "\t\t\t\t\t\tcontinue;\n"
                    "\t\t\t\t\t}\n"
                    "\t\t\t\t\treturn false;\n"
                    "\t\t\t\t}\n"
                    "\t\t\t\tcatch (IllegalAccessException ignored) {\n"
                    "\t\t\t\t\treturn false;\n"
                    "\t\t\t\t}\n"
                    "\t\t\t}\n"
                    "\t\t\tcurrent = current.getSuperclass();\n"
                    "\t\t}\n"
                    "\t\treturn true;"
                ),
                return_type="boolean",
            )

        return content

    def _collect_model_attribute_loader_methods(self, content: str) -> List[dict]:
        signature_pattern = re.compile(
            r"(?m)^(?P<indent>[ \t]*)(?P<prefix>(?:(?:public|protected|private)\s+(?:static\s+|final\s+)*)?)"
            r"(?P<return_type>[A-Z][A-Za-z0-9_$.<>?, ]*)\s+(?P<name>\w+)\s*\("
        )

        loaders = []
        for match in signature_pattern.finditer(content):
            params_start = content.find("(", match.start("name"))
            if params_start == -1:
                continue
            params_end = self._find_matching_parenthesis(content, params_start)
            if params_end == -1:
                continue
            body_start = content.find("{", params_end)
            if body_start == -1:
                continue
            trailer = content[params_end + 1 : body_start]
            if ";" in trailer:
                continue
            body_end = self._find_matching_brace(content, body_start)
            if body_end == -1:
                continue
            annotations_start = self._expand_method_start_to_annotations(content, match.start())
            annotations = content[annotations_start:match.start()]
            if self._method_uses_http_mapping(annotations):
                continue
            params = content[params_start + 1 : params_end]
            path_params = []
            for part in self._split_signature_arguments(params):
                parsed = self._parse_parameter_signature(part)
                if parsed and "@PathVariable" in part:
                    path_params.append(parsed["name"])
            if not path_params:
                continue
            loaders.append(
                {
                    "name": match.group("name"),
                    "return_type": self._normalize_java_type_name(match.group("return_type")),
                    "path_params": tuple(path_params),
                }
            )
        return loaders

    def _find_matching_loader_method(self, method_params: str, entity_type: str, loaders: Sequence[dict]) -> Optional[dict]:
        available_params = {
            parsed["name"]
            for raw_part in self._split_signature_arguments(method_params)
            if (parsed := self._parse_parameter_signature(raw_part))
        }
        for loader in loaders:
            if loader["return_type"] != entity_type:
                continue
            if not set(loader["path_params"]).issubset(available_params):
                continue
            return {
                "name": loader["name"],
                "return_type": loader["return_type"],
                "call_args": loader["path_params"],
            }
        return None

    def _extract_form_backing_parameter(self, params: str) -> Optional[dict]:
        for raw_part in self._split_signature_arguments(params):
            part = raw_part.strip()
            if not part:
                continue
            if (
                "@PathVariable" in part
                or "@QueryValue" in part
                or "@Header" in part
                or "Map<String, Object>" in part
                or "Model " in part
                or "ModelMap " in part
            ):
                continue
            parsed = self._parse_parameter_signature(part)
            if not parsed:
                continue
            type_name = parsed["type"].split(".")[-1]
            if type_name in {
                "String",
                "int",
                "Integer",
                "long",
                "Long",
                "boolean",
                "Boolean",
                "double",
                "Double",
                "float",
                "Float",
                "Page",
                "Pageable",
                "List",
                "Set",
                "Collection",
                "Optional",
            }:
                continue
            return parsed
        return None

    def _parse_parameter_signature(self, raw_part: str) -> Optional[dict]:
        cleaned = re.sub(r"@\S+(?:\([^)]*\))?\s*", "", raw_part or "").strip()
        cleaned = re.sub(r"\bfinal\s+", "", cleaned).strip()
        if not cleaned:
            return None
        cleaned = re.sub(r"\s+", " ", cleaned)
        match = re.match(r"(?P<type>[A-Za-z0-9_$.<>?, \[\]]+)\s+(?P<name>\w+)$", cleaned)
        if not match:
            return None
        return {
            "type": self._normalize_java_type_name(match.group("type")),
            "name": match.group("name"),
        }

    def _normalize_java_type_name(self, raw_type: str) -> str:
        cleaned = " ".join((raw_type or "").split())
        cleaned = re.sub(r"<.*>", "", cleaned)
        return cleaned.strip()

    def _transform_pet_form_controller_prepopulation(self, content: str) -> str:
        if "class PetController" not in content or "findOwner(" not in content:
            return content
        content = re.sub(
            r"(?:public\s+)?String initCreationForm\(\s*Owner\s+owner\s*,\s*Map<String,\s*Object>\s+model\s*\)\s*\{",
            'public String initCreationForm(@PathVariable("ownerId") int ownerId, Map<String, Object> model) {',
            content,
        )
        content = re.sub(
            r"(?ms)(public String initCreationForm\(@PathVariable\(\"ownerId\"\) int ownerId, Map<String, Object> model\) \{\n)(?P<indent>[ \t]*)Pet pet = new Pet\(\);",
            lambda match: (
                match.group(1)
                + f'{match.group("indent")}Owner owner = findOwner(ownerId);\n'
                + f'{match.group("indent")}model.put("owner", owner);\n'
                + f'{match.group("indent")}model.put("types", populatePetTypes());\n'
                + f'{match.group("indent")}Pet pet = new Pet();'
            ),
            content,
            count=1,
        )
        content = re.sub(
            r"(?:public\s+)?String processCreationForm\(\s*Owner\s+owner\s*,\s*@(?:jakarta\.validation\.)?Valid\s+Pet\s+pet\s*,\s*Map<String,\s*Object>\s+model\s*\)\s*\{",
            'public String processCreationForm(@PathVariable("ownerId") int ownerId, @jakarta.validation.Valid Pet pet, Map<String, Object> model) {',
            content,
        )
        content = re.sub(
            r"(?ms)(public String processCreationForm\(@PathVariable\(\"ownerId\"\) int ownerId, @jakarta\.validation\.Valid Pet pet, Map<String, Object> model\) \{\n)(?P<indent>[ \t]*)if \(StringUtils\.isNotEmpty\(pet\.getName\(\)\)",
            lambda match: (
                match.group(1)
                + f'{match.group("indent")}Owner owner = findOwner(ownerId);\n'
                + f'{match.group("indent")}model.put("owner", owner);\n'
                + f'{match.group("indent")}model.put("types", populatePetTypes());\n'
                + f'{match.group("indent")}if (StringUtils.isNotEmpty(pet.getName())'
            ),
            content,
            count=1,
        )
        content = re.sub(
            r'(?:public\s+)?String initUpdateForm\(\s*Owner\s+owner\s*,\s*@PathVariable\("petId"\)\s+int\s+petId\s*,\s*Map<String,\s*Object>\s+model\s*\)\s*\{',
            'public String initUpdateForm(@PathVariable("ownerId") int ownerId, @PathVariable("petId") int petId, Map<String, Object> model) {',
            content,
        )
        content = re.sub(
            r"(?m)^(?P<indent>[ \t]*)Pet pet = owner\.getPet\(petId\);",
            lambda match: (
                f'{match.group("indent")}Owner owner = findOwner(ownerId);\n'
                + f'{match.group("indent")}Pet pet = findPet(ownerId, petId);\n'
                + f'{match.group("indent")}model.put("owner", owner);\n'
                + f'{match.group("indent")}model.put("types", populatePetTypes());'
            ),
            content,
            count=1,
        )
        content = re.sub(
            r"(?:public\s+)?String processUpdateForm\(\s*@(?:jakarta\.validation\.)?Valid\s+Pet\s+pet\s*,\s*Owner\s+owner\s*,\s*Map<String,\s*Object>\s+model\s*\)\s*\{",
            'public String processUpdateForm(@PathVariable("ownerId") int ownerId, @PathVariable("petId") int petId, @jakarta.validation.Valid Pet pet, Map<String, Object> model) {',
            content,
        )
        content = re.sub(
            r'(?ms)(public String processUpdateForm\(@PathVariable\("ownerId"\) int ownerId, @PathVariable\("petId"\) int petId, @jakarta\.validation\.Valid Pet pet, Map<String, Object> model\) \{\n)(?P<indent>[ \t]*)if \(hasValidationErrors\(model, "pet", pet\)\) \{',
            lambda match: (
                match.group(1)
                + f'{match.group("indent")}Owner owner = findOwner(ownerId);\n'
                + f'{match.group("indent")}Pet existingPet = findPet(ownerId, petId);\n'
                + f'{match.group("indent")}model.put("owner", owner);\n'
                + f'{match.group("indent")}model.put("types", populatePetTypes());\n'
                + f'{match.group("indent")}if (pet.getId() == null) {{\n'
                + f'{match.group("indent")}    pet.setId(existingPet.getId());\n'
                + f'{match.group("indent")}}}\n'
                + f'{match.group("indent")}if (hasValidationErrors(model, "pet", pet)) {{'
            ),
            content,
            count=1,
        )
        content = self._ensure_named_method_prefix(
            content,
            method_name="processCreationForm",
            required_header_fragment='@PathVariable("ownerId") int ownerId',
            missing_guard="Owner owner = findOwner(ownerId);",
            prefix_lines=(
                "Owner owner = findOwner(ownerId);",
                'model.put("owner", owner);',
                'model.put("types", populatePetTypes());',
            ),
        )
        content = self._ensure_named_method_prefix(
            content,
            method_name="processUpdateForm",
            required_header_fragment='@PathVariable("ownerId") int ownerId',
            missing_guard="Owner owner = findOwner(ownerId);",
            prefix_lines=(
                "Owner owner = findOwner(ownerId);",
                "Pet existingPet = findPet(ownerId, petId);",
                'model.put("owner", owner);',
                'model.put("types", populatePetTypes());',
                "if (pet.getId() == null) {",
                "    pet.setId(existingPet.getId());",
                "}",
            ),
        )
        return content

    def _transform_visit_form_controller_prepopulation(self, content: str) -> str:
        if "class VisitController" not in content or "loadPetWithVisit(" not in content:
            return content
        content = re.sub(
            r"(?:public\s+)?String initNewVisitForm\(\s*\)\s*\{",
            'public String initNewVisitForm(@PathVariable("ownerId") int ownerId, @PathVariable("petId") int petId, Map<String, Object> model) {',
            content,
        )
        content = re.sub(
            r'(?ms)(public String initNewVisitForm\(@PathVariable\("ownerId"\) int ownerId, @PathVariable\("petId"\) int petId, Map<String, Object> model\) \{\n)(?P<indent>[ \t]*)return "(?P<view>[^"]+)";',
            lambda match: (
                match.group(1)
                + f'{match.group("indent")}Visit visit = loadPetWithVisit(ownerId, petId, model);\n'
                + f'{match.group("indent")}model.put("visit", visit);\n'
                + f'{match.group("indent")}return "{match.group("view")}";'
            ),
            content,
            count=1,
        )
        content = re.sub(
            r"(?:public\s+)?String processNewVisitForm\(\s*Owner\s+owner\s*,\s*@PathVariable\s+int\s+petId\s*,\s*@(?:jakarta\.validation\.)?Valid\s+Visit\s+visit\s*,\s*Map<String,\s*Object>\s+model\s*\)\s*\{",
            'public String processNewVisitForm(@PathVariable("ownerId") int ownerId, @PathVariable int petId, @jakarta.validation.Valid Visit visit, Map<String, Object> model) {',
            content,
        )
        content = re.sub(
            r'(?ms)(public String processNewVisitForm\(@PathVariable\("ownerId"\) int ownerId, @PathVariable int petId, @jakarta\.validation\.Valid Visit visit, Map<String, Object> model\) \{\n)(?P<indent>[ \t]*)if \(hasValidationErrors\(model, "visit", visit\)\) \{',
            lambda match: (
                match.group(1)
                + f'{match.group("indent")}loadPetWithVisit(ownerId, petId, model);\n'
                + f'{match.group("indent")}Owner owner = (Owner) model.get("owner");\n'
                + f'{match.group("indent")}if (hasValidationErrors(model, "visit", visit)) {{'
            ),
            content,
            count=1,
        )
        return content

    def _transform_view_controller_runtime_patterns(self, content: str) -> str:
        if "@Controller" not in content or "@RestController" in content:
            return content

        content = self._transform_post_form_binding_signatures(content)
        content = self._transform_model_map_returning_methods(content)
        content = self._ensure_text_html_produces_for_view_routes(content)
        return content

    def _ensure_text_html_produces_for_view_routes(self, content: str) -> str:
        methods = self._collect_controller_methods(content, only_http_mapped=True)
        helper_methods = self._collect_controller_methods(content, only_http_mapped=False)
        view_helper_names = {
            method["name"]
            for method in helper_methods
            if "new ModelAndView" in method["body"] or "HttpResponse.seeOther(" in method["body"]
        }
        transformed = False

        for method in reversed(methods):
            annotations = method["annotations"] or ""
            if not self._method_uses_http_mapping(annotations) or "@Produces(" in annotations:
                continue

            return_type = " ".join(method["return_type"].split())
            body = method["body"]
            if not (
                return_type == "String"
                or "ModelAndView" in return_type
                or "new ModelAndView" in body
                or "HttpResponse.seeOther(" in body
                or any(re.search(rf"\b{re.escape(name)}\s*\(", body) for name in view_helper_names)
            ):
                continue

            annotation_lines = [line for line in annotations.splitlines() if line.strip()]
            annotation_lines.append(f"{method['indent']}@Produces(MediaType.TEXT_HTML)")
            updated_annotations = "\n".join(annotation_lines) + "\n"
            replacement = updated_annotations + content[method["signature_start"] : method["end"]]
            content = content[: method["start"]] + replacement + content[method["end"] :]
            transformed = True

        if not transformed:
            return content

        content = self._ensure_import(content, "io.micronaut.http.annotation.Produces")
        content = self._ensure_import(content, "io.micronaut.http.MediaType")
        return content

    def _transform_post_form_binding_signatures(self, content: str) -> str:
        methods = self._collect_string_returning_controller_methods(content)
        transformed = False

        for method in reversed(methods):
            annotations = method["annotations"]
            if "@Post" not in annotations:
                continue

            params = method["params"]
            rewritten_params, rewritten_body = self._normalize_post_form_parameters(params, method["body"])
            updated_annotations = annotations
            if "@Consumes(" not in updated_annotations:
                annotation_lines = [line for line in updated_annotations.splitlines() if line.strip()]
                insert_index = next(
                    (idx + 1 for idx, line in enumerate(annotation_lines) if "@Post" in line),
                    len(annotation_lines),
                )
                indent = method["indent"]
                annotation_lines.insert(insert_index, f"{indent}@Consumes(MediaType.APPLICATION_FORM_URLENCODED)")
                updated_annotations = "\n".join(annotation_lines) + "\n"

            if rewritten_params == params and updated_annotations == annotations and rewritten_body == method["body"]:
                continue

            replacement = self._render_string_controller_method(
                method,
                annotations=updated_annotations,
                params=rewritten_params,
                return_type="String",
                body=rewritten_body,
            )
            content = content[: method["start"]] + replacement + content[method["end"] :]
            transformed = True

        if not transformed:
            return content
        content = self._ensure_import(content, "io.micronaut.http.annotation.Consumes")
        content = self._ensure_import(content, "io.micronaut.http.MediaType")
        return self._ensure_form_binding_support(content)

    def _normalize_post_form_parameters(self, params: str, body: str) -> Tuple[str, str]:
        parts = self._split_signature_arguments(params)
        rewritten_parts = []
        form_param = self._extract_form_binding_target(parts)
        rewritten_body = body

        for index, raw_part in enumerate(parts):
            part = raw_part.strip()
            if not part:
                continue
            stripped_part = re.sub(r"(?<!\S)@Body\s+", "", part)
            if form_param and index == form_param["index"]:
                rewritten_parts.append("@Body @Nullable String formBody")
            else:
                rewritten_parts.append(stripped_part)

        if form_param:
            binding_line = (
                f'{form_param["type"]} {form_param["name"]} = '
                f'bindFormBean(formBody, {form_param["type"]}.class);'
            )
            if binding_line not in body:
                indent = self._infer_method_body_indent(body)
                rewritten_body = f"\n{indent}{binding_line}{body}"

        return ", ".join(rewritten_parts), rewritten_body

    def _extract_form_binding_target(self, parts: Sequence[str]) -> Optional[dict]:
        for index, raw_part in enumerate(parts):
            part = raw_part.strip()
            if not part:
                continue
            if (
                "@PathVariable" in part
                or "@QueryValue" in part
                or "@Header" in part
                or "@Body" in part
                or "HttpRequest<" in part
                or "HttpResponse<" in part
                or "Map<String, Object>" in part
                or "Model " in part
                or "ModelMap " in part
            ):
                continue
            parsed = self._parse_parameter_signature(part)
            if not parsed:
                continue
            type_name = parsed["type"].split(".")[-1]
            if type_name in {
                "String",
                "int",
                "Integer",
                "long",
                "Long",
                "boolean",
                "Boolean",
                "double",
                "Double",
                "float",
                "Float",
                "Page",
                "Pageable",
                "List",
                "Set",
                "Collection",
                "Optional",
            }:
                continue
            return {"index": index, "type": parsed["type"], "name": parsed["name"]}

        return None

    def _ensure_form_binding_support(self, content: str) -> str:
        if "bindFormBean(" not in content and "bindQueryBean(" not in content:
            return content

        content = self._ensure_import(content, "io.micronaut.core.convert.ConversionService")
        if "bindFormBean(" in content:
            content = self._ensure_import(content, "io.micronaut.http.annotation.Body")
            content = self._ensure_import(content, "io.micronaut.core.annotation.Nullable")
        if "bindQueryBean(" in content:
            content = self._ensure_import(content, "io.micronaut.http.HttpRequest")

        class_match = re.search(
            r"(?m)^(?P<indent>[ \t]*)(?:public|protected|private\s+)?(?:abstract\s+|final\s+)?class\s+(?P<class_name>\w+)[^{]*\{",
            content,
        )
        if not class_match:
            return content

        member_indent = class_match.group("indent") + "    "
        class_name = class_match.group("class_name")

        if "private final ConversionService conversionService;" not in content:
            insert_at = class_match.end()
            content = (
                content[:insert_at]
                + f"\n{member_indent}private final ConversionService conversionService;\n"
                + content[insert_at:]
            )

        constructor_pattern = re.compile(
            rf"(?ms)^(?P<indent>[ \t]*)(?:(?:public|protected|private)\s+)?{re.escape(class_name)}\((?P<params>[^)]*)\)\s*\{{(?P<body>.*?)^(?P=indent)\}}",
            re.MULTILINE,
        )
        constructor_match = constructor_pattern.search(content)
        if constructor_match and "this.conversionService = conversionService;" not in content:
            params = constructor_match.group("params").strip()
            new_params = (
                params + ", ConversionService conversionService"
                if params
                else "ConversionService conversionService"
            )
            body = constructor_match.group("body").rstrip()
            body = body + f"\n{constructor_match.group('indent')}    this.conversionService = conversionService;\n"
            replacement = (
                f"{constructor_match.group('indent')}public {class_name}({new_params}) {{"
                f"{body}"
                f"{constructor_match.group('indent')}}}"
            )
            content = content[: constructor_match.start()] + replacement + content[constructor_match.end() :]
        elif not constructor_match:
            insert_at = class_match.end()
            constructor_block = (
                f"\n{member_indent}public {class_name}(ConversionService conversionService) {{\n"
                f"{member_indent}    this.conversionService = conversionService;\n"
                f"{member_indent}}}\n"
            )
            content = content[:insert_at] + constructor_block + content[insert_at:]

        content = self._ensure_helper_method(
            content,
            method_name="bindFormBean",
            parameter_signature="String formBody, Class<T> type",
            method_body=(
                "T bean;\n"
                "\t\ttry {\n"
                "\t\t\tbean = type.getDeclaredConstructor().newInstance();\n"
                "\t\t\tjava.util.Map<String, String> formValues = readFormValues(formBody);\n"
                "\t\t\tfor (java.beans.PropertyDescriptor descriptor : java.beans.Introspector.getBeanInfo(type).getPropertyDescriptors()) {\n"
                "\t\t\t\tjava.lang.reflect.Method writeMethod = descriptor.getWriteMethod();\n"
                '\t\t\t\tif (writeMethod == null || "class".equals(descriptor.getName())) {\n'
                "\t\t\t\t\tcontinue;\n"
                "\t\t\t\t}\n"
                "\t\t\t\tif (!formValues.containsKey(descriptor.getName())) {\n"
                "\t\t\t\t\tcontinue;\n"
                "\t\t\t\t}\n"
                "\t\t\t\tString rawValue = formValues.get(descriptor.getName());\n"
                "\t\t\t\tObject convertedValue = descriptor.getPropertyType() == String.class\n"
                "\t\t\t\t\t? rawValue\n"
                "\t\t\t\t\t: conversionService.convert(rawValue, descriptor.getPropertyType()).orElse(null);\n"
                "\t\t\t\tif (convertedValue == null && descriptor.getPropertyType() != String.class) {\n"
                "\t\t\t\t\tcontinue;\n"
                "\t\t\t\t}\n"
                "\t\t\t\twriteMethod.setAccessible(true);\n"
                "\t\t\t\twriteMethod.invoke(bean, convertedValue);\n"
                "\t\t\t}\n"
                "\t\t}\n"
                "\t\tcatch (ReflectiveOperationException | java.beans.IntrospectionException e) {\n"
                '\t\t\tthrow new IllegalStateException("Failed to bind form bean for " + type.getName(), e);\n'
                "\t\t}\n"
                "\t\treturn bean;"
            ),
            return_type="<T> T",
        )
        content = self._ensure_helper_method(
            content,
            method_name="bindQueryBean",
            parameter_signature="HttpRequest<?> request, Class<T> type",
            method_body=(
                "T bean;\n"
                "\t\ttry {\n"
                "\t\t\tbean = type.getDeclaredConstructor().newInstance();\n"
                "\t\t\tjava.util.Map<String, String> queryValues = readQueryValues(request);\n"
                "\t\t\tfor (java.beans.PropertyDescriptor descriptor : java.beans.Introspector.getBeanInfo(type).getPropertyDescriptors()) {\n"
                "\t\t\t\tjava.lang.reflect.Method writeMethod = descriptor.getWriteMethod();\n"
                '\t\t\t\tif (writeMethod == null || "class".equals(descriptor.getName())) {\n'
                "\t\t\t\t\tcontinue;\n"
                "\t\t\t\t}\n"
                "\t\t\t\tif (!queryValues.containsKey(descriptor.getName())) {\n"
                "\t\t\t\t\tcontinue;\n"
                "\t\t\t\t}\n"
                "\t\t\t\tString rawValue = queryValues.get(descriptor.getName());\n"
                "\t\t\t\tObject convertedValue = descriptor.getPropertyType() == String.class\n"
                "\t\t\t\t\t? rawValue\n"
                "\t\t\t\t\t: conversionService.convert(rawValue, descriptor.getPropertyType()).orElse(null);\n"
                "\t\t\t\tif (convertedValue == null && descriptor.getPropertyType() != String.class) {\n"
                "\t\t\t\t\tcontinue;\n"
                "\t\t\t\t}\n"
                "\t\t\t\twriteMethod.setAccessible(true);\n"
                "\t\t\t\twriteMethod.invoke(bean, convertedValue);\n"
                "\t\t\t}\n"
                "\t\t}\n"
                "\t\tcatch (ReflectiveOperationException | java.beans.IntrospectionException e) {\n"
                '\t\t\tthrow new IllegalStateException("Failed to bind query bean for " + type.getName(), e);\n'
                "\t\t}\n"
                "\t\treturn bean;"
            ),
            return_type="<T> T",
        )
        content = self._ensure_helper_method(
            content,
            method_name="readFormValues",
            parameter_signature="String formBody",
            method_body=(
                "java.util.Map<String, String> values = new java.util.LinkedHashMap<>();\n"
                "\t\tif (formBody == null || formBody.isEmpty()) {\n"
                "\t\t\treturn values;\n"
                "\t\t}\n"
                '\t\tfor (String pair : formBody.split("&")) {\n'
                "\t\t\tif (pair.isEmpty()) {\n"
                "\t\t\t\tcontinue;\n"
                "\t\t\t}\n"
                "\t\t\tint separator = pair.indexOf('=');\n"
                "\t\t\tString key = separator >= 0 ? pair.substring(0, separator) : pair;\n"
                '\t\t\tString value = separator >= 0 ? pair.substring(separator + 1) : "";\n'
                "\t\t\tvalues.put(decodeFormComponent(key), decodeFormComponent(value));\n"
                "\t\t}\n"
                "\t\treturn values;"
            ),
            return_type="java.util.Map<String, String>",
        )
        content = self._ensure_helper_method(
            content,
            method_name="readQueryValues",
            parameter_signature="HttpRequest<?> request",
            method_body=(
                "java.util.Map<String, String> values = new java.util.LinkedHashMap<>();\n"
                "\t\tif (request == null) {\n"
                "\t\t\treturn values;\n"
                "\t\t}\n"
                "\t\tfor (java.util.Map.Entry<String, java.util.List<String>> entry : request.getParameters().asMap().entrySet()) {\n"
                "\t\t\tjava.util.List<String> rawValues = entry.getValue();\n"
                "\t\t\tif (rawValues == null || rawValues.isEmpty()) {\n"
                "\t\t\t\tcontinue;\n"
                "\t\t\t}\n"
                "\t\t\tvalues.put(entry.getKey(), rawValues.get(0));\n"
                "\t\t}\n"
                "\t\treturn values;"
            ),
            return_type="java.util.Map<String, String>",
        )
        content = self._ensure_helper_method(
            content,
            method_name="decodeFormComponent",
            parameter_signature="String value",
            method_body="return java.net.URLDecoder.decode(value, java.nio.charset.StandardCharsets.UTF_8);",
            return_type="String",
        )
        return content

    def _transform_model_map_returning_methods(self, content: str) -> str:
        if "String " not in content:
            return content

        methods = self._collect_string_returning_controller_methods(content)
        candidates = [
            method
            for method in methods
            if method["has_model_param"] or self._method_uses_http_mapping(method["annotations"])
        ]
        if not candidates:
            return content

        candidate_names = {method["name"] for method in candidates}
        uses_model_and_view = False
        uses_http_response = False

        for method in reversed(candidates):
            params = method["params"]
            model_name = method["model_name"] or "model"
            rewritten_params = ", ".join(
                part
                for part in self._split_signature_arguments(params)
                if "Map<String, Object>" not in part
            )
            rewritten_body, body_uses_model_and_view, body_uses_http_response = self._rewrite_string_controller_body(
                method["body"],
                model_name=model_name,
                params_without_model=rewritten_params,
                transformed_method_names=candidate_names,
            )
            if rewritten_body == method["body"] and rewritten_params == params and not method["has_model_param"]:
                continue

            replacement = self._render_string_controller_method(
                method,
                params=rewritten_params,
                return_type="Object",
                body=rewritten_body,
            )
            content = content[: method["start"]] + replacement + content[method["end"] :]
            uses_model_and_view = uses_model_and_view or body_uses_model_and_view
            uses_http_response = uses_http_response or body_uses_http_response

        if uses_model_and_view:
            content = self._ensure_import(content, "io.micronaut.views.ModelAndView")
            content = self._ensure_import(content, "java.util.LinkedHashMap")
            content = self._ensure_import(content, "java.util.Map")
        if uses_http_response:
            content = self._ensure_import(content, "io.micronaut.http.HttpResponse")
            content = self._ensure_import(content, "java.net.URI")
        return content

    def _render_redirect_target_expression(self, target: str, params: str) -> str:
        expression = f'"{target}"'
        param_names = {
            match.group(1)
            for match in re.finditer(r"\b([a-zA-Z_][A-Za-z0-9_]*)\b(?=\s*(?:,|$))", params)
        }
        for name in sorted(param_names, key=len, reverse=True):
            placeholder = "{" + name + "}"
            if placeholder in expression:
                expression = expression.replace(placeholder, f'" + {name} + "')
        expression = re.sub(r'"" \+ ', "", expression)
        expression = re.sub(r' \+ ""', "", expression)
        return expression

    def _collect_string_returning_controller_methods(self, content: str) -> List[dict]:
        signature_pattern = re.compile(
            r"(?m)^(?P<indent>[ \t]*)(?P<prefix>(?:public|protected|private)\s+(?:static\s+|final\s+)*)String\s+(?P<name>\w+)\s*\("
        )

        methods = []
        for match in signature_pattern.finditer(content):
            params_start = content.find("(", match.start("name"))
            if params_start == -1:
                continue
            params_end = self._find_matching_parenthesis(content, params_start)
            if params_end == -1:
                continue
            body_start = content.find("{", params_end)
            if body_start == -1:
                continue
            trailer = content[params_end + 1 : body_start]
            if ";" in trailer:
                continue
            body_end = self._find_matching_brace(content, body_start)
            if body_end == -1:
                continue
            annotations_start = self._expand_method_start_to_annotations(content, match.start())
            methods.append(
                {
                    "start": annotations_start,
                    "end": body_end + 1,
                    "annotations": content[annotations_start: match.start()],
                    "indent": match.group("indent"),
                    "prefix": match.group("prefix"),
                    "name": match.group("name"),
                    "params": content[params_start + 1 : params_end],
                    "tail": content[params_end + 1 : body_start],
                    "body": content[body_start + 1 : body_end],
                    "has_model_param": self._extract_model_parameter_name(content[params_start + 1 : params_end]) is not None,
                    "model_name": self._extract_model_parameter_name(content[params_start + 1 : params_end]),
                }
            )
        return methods

    def _collect_controller_methods(self, content: str, *, only_http_mapped: bool) -> List[dict]:
        signature_pattern = re.compile(
            r"(?m)^(?P<indent>[ \t]*)(?P<prefix>(?:public|protected|private)\s+(?:static\s+|final\s+)*)"
            r"(?P<return_type>[A-Za-z_][A-Za-z0-9_$.<>, ?\[\]]*)\s+(?P<name>\w+)\s*\("
        )

        methods = []
        for match in signature_pattern.finditer(content):
            params_start = content.find("(", match.start("name"))
            if params_start == -1:
                continue
            params_end = self._find_matching_parenthesis(content, params_start)
            if params_end == -1:
                continue
            body_start = content.find("{", params_end)
            if body_start == -1:
                continue
            trailer = content[params_end + 1 : body_start]
            if ";" in trailer:
                continue
            body_end = self._find_matching_brace(content, body_start)
            if body_end == -1:
                continue
            annotations_start = self._expand_method_start_to_annotations(content, match.start())
            annotations = content[annotations_start: match.start()]
            if only_http_mapped and not self._method_uses_http_mapping(annotations):
                continue
            methods.append(
                {
                    "start": annotations_start,
                    "signature_start": match.start(),
                    "end": body_end + 1,
                    "annotations": annotations,
                    "indent": match.group("indent"),
                    "name": match.group("name"),
                    "return_type": match.group("return_type"),
                    "body": content[body_start + 1 : body_end],
                }
            )
        return methods

    def _expand_method_start_to_annotations(self, content: str, method_start: int) -> int:
        start = method_start
        while start > 0:
            line_end = start - 1
            line_start = content.rfind("\n", 0, line_end)
            line_start = 0 if line_start == -1 else line_start + 1
            line = content[line_start:line_end + 1].strip()
            if not line.startswith("@"):
                break
            start = line_start
        return start

    def _method_uses_http_mapping(self, annotations: str) -> bool:
        return any(
            marker in (annotations or "")
            for marker in ("@Get", "@Post", "@Put", "@Patch", "@Delete", "@Head", "@Options")
        )

    def _render_string_controller_method(
        self,
        method: dict,
        *,
        annotations: Optional[str] = None,
        params: Optional[str] = None,
        return_type: str,
        body: str,
    ) -> str:
        rendered_annotations = method["annotations"] if annotations is None else annotations
        rendered_params = method["params"] if params is None else params
        return (
            f"{rendered_annotations}"
            f"{method['indent']}{method['prefix']}{return_type} {method['name']}({rendered_params})"
            f"{method['tail']}{{{body}}}"
        )

    def _rewrite_string_controller_body(
        self,
        body: str,
        *,
        model_name: str,
        params_without_model: str,
        transformed_method_names: set[str],
    ) -> Tuple[str, bool, bool]:
        rewritten = body
        uses_model_and_view = False
        uses_http_response = False

        def replace_return(match: re.Match) -> str:
            nonlocal uses_model_and_view, uses_http_response
            indent = match.group("indent")
            expr = self._remove_model_argument_from_transformed_helper_call(
                match.group("expr").strip(),
                transformed_method_names,
                model_name,
            )
            if self._is_redirect_return_expression(expr):
                uses_http_response = True
                target_expr = self._render_redirect_expression(expr, params_without_model)
                return f"{indent}return HttpResponse.seeOther(URI.create({target_expr}));"
            if self._is_transformed_helper_call(expr, transformed_method_names):
                return f"{indent}return {expr};"
            uses_model_and_view = True
            return f"{indent}return new ModelAndView<>({expr}, {model_name});"

        rewritten = re.sub(
            r"(?m)^(?P<indent>[ \t]*)return\s+(?P<expr>[^;]+);",
            replace_return,
            rewritten,
        )

        model_decl_pattern = re.compile(
            rf"\bMap<\s*String\s*,\s*Object\s*>\s+{re.escape(model_name)}\s*=\s*new\s+LinkedHashMap<>\(\);"
        )
        if re.search(rf"\b{re.escape(model_name)}\b", rewritten) and not model_decl_pattern.search(rewritten):
            inner_indent = self._infer_method_body_indent(rewritten)
            rewritten = f"\n{inner_indent}Map<String, Object> {model_name} = new LinkedHashMap<>();{rewritten}"
            uses_model_and_view = True

        return rewritten, uses_model_and_view, uses_http_response

    def _infer_method_body_indent(self, body: str) -> str:
        indent_match = re.search(r"\n([ \t]+)\S", body)
        if indent_match:
            return indent_match.group(1)
        return "    "

    def _is_redirect_return_expression(self, expr: str) -> bool:
        return expr.strip().startswith('"redirect:')

    def _render_redirect_expression(self, expr: str, params_without_model: str) -> str:
        cleaned = expr.strip()
        literal_match = re.fullmatch(r'"redirect:(?P<target>[^"]*)"', cleaned)
        if literal_match:
            return self._render_redirect_target_expression(literal_match.group("target"), params_without_model)
        if cleaned.startswith('"redirect:'):
            return cleaned.replace('"redirect:', '"', 1)
        return cleaned

    def _is_transformed_helper_call(self, expr: str, transformed_method_names: set[str]) -> bool:
        call_match = re.fullmatch(r"(?P<callee>[A-Za-z_][A-Za-z0-9_$.]*)\((?P<args>.*)\)", expr.strip(), re.DOTALL)
        if not call_match:
            return False
        return call_match.group("callee").split(".")[-1] in transformed_method_names

    def _remove_model_argument_from_transformed_helper_call(
        self,
        expr: str,
        transformed_method_names: set[str],
        model_name: str,
    ) -> str:
        call_match = re.fullmatch(r"(?P<callee>[A-Za-z_][A-Za-z0-9_$.]*)\((?P<args>.*)\)", expr.strip(), re.DOTALL)
        if not call_match:
            return expr
        if call_match.group("callee").split(".")[-1] not in transformed_method_names:
            return expr
        args = [
            arg
            for arg in self._split_signature_arguments(call_match.group("args"))
            if arg.strip() != model_name
        ]
        return f"{call_match.group('callee')}({', '.join(args)})"

    def _transform_xml_binding_patterns(self, content: str) -> str:
        content = re.sub(
            r"(?m)^[ \t]*import jakarta\.xml\.bind\.annotation\.XmlElement;\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import jakarta\.xml\.bind\.annotation\.XmlRootElement;\n?",
            "",
            content,
        )
        content = re.sub(r"(?m)^[ \t]*@XmlElement\s*\n?", "", content)
        content = re.sub(r"(?m)^[ \t]*@XmlRootElement\s*\n?", "", content)
        return content

    def _transform_marshalling_view_patterns(self, content: str) -> str:
        scan_content = self._strip_java_comments_for_symbol_scan(content)
        if "MarshallingView" not in scan_content and "org.springframework.web.servlet.view.xml.MarshallingView" not in scan_content:
            return content

        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.web\.servlet\.view\.xml\.MarshallingView;\n?",
            "import io.micronaut.views.ModelAndView;\n",
            content,
        )
        content = content.replace(
            "org.springframework.web.servlet.view.xml.MarshallingView",
            "io.micronaut.views.ModelAndView",
        )
        content = re.sub(r"\bMarshallingView\b", "ModelAndView", content)

        review_comment = self._render_manual_review_comment(
            "Spring MarshallingView has no direct Micronaut equivalent; review the migrated XML response/view flow."
        )
        if review_comment in content:
            return content

        line_match = re.search(r"(?m)^(?P<indent>[ \t]*)(?!import\b).*\bModelAndView\b", content)
        if not line_match:
            return content

        insert_at = line_match.start()
        indent = line_match.group("indent")
        return content[:insert_at] + f"{indent}{review_comment}\n" + content[insert_at:]

    def _transform_property_comparator_patterns(self, content: str) -> str:
        if "PropertyComparator.sort(" not in content and "MutableSortDefinition" not in content:
            return content
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.beans\.support\.MutableSortDefinition;\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.beans\.support\.PropertyComparator;\n?",
            "",
            content,
        )
        content = re.sub(
            r"PropertyComparator\.sort\(\s*(\w+)\s*,\s*new MutableSortDefinition\(\"name\",\s*true,\s*true\)\s*\);",
            r'\1.sort(java.util.Comparator.comparing(Specialty::getName, String.CASE_INSENSITIVE_ORDER));',
            content,
        )
        return content

    def _transform_repository_interface_patterns(self, content: str) -> str:
        has_repository_contract = any(
            marker in content
            for marker in (
                "extends Repository<",
                "extends CrudRepository<",
                "extends PagingAndSortingRepository<",
                "extends JpaSpecificationExecutor<",
            )
        )
        if not has_repository_contract:
            return content

        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.data\.repository\.Repository;\n?",
            "import io.micronaut.data.repository.GenericRepository;\n",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.data\.repository\.CrudRepository;\n?",
            "import io.micronaut.data.repository.CrudRepository;\n",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.data\.repository\.PagingAndSortingRepository;\n?",
            "import io.micronaut.data.repository.PageableRepository;\n",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.data\.jpa\.repository\.Query;\n?",
            "import io.micronaut.data.annotation.Query;\n",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.data\.jpa\.repository\.JpaSpecificationExecutor;\n?",
            "import io.micronaut.data.jpa.repository.JpaSpecificationExecutor;\n",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.data\.repository\.query\.Param;\n?",
            "",
            content,
        )
        if "import io.micronaut.data.annotation.Repository;" not in content:
            package_match = re.search(r"(?m)^package [^;]+;\n", content)
            if package_match:
                insert_at = package_match.end()
                content = (
                    content[:insert_at]
                    + "\nimport io.micronaut.data.annotation.Repository;\n"
                    + content[insert_at:]
                )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.dao\.DataAccessException;\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.data\.domain\.Sort;\n?",
            "import io.micronaut.data.model.Sort;\n",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.data\.domain\.Slice;\n?",
            "import io.micronaut.data.model.Slice;\n",
            content,
        )
        content = re.sub(r'@Param\(\s*"[^"]+"\s*\)\s*', "", content)
        content = re.sub(r"\s+throws\s+DataAccessException", "", content)
        content = re.sub(
            r"(?m)^(?P<indent>[ \t]*)(?:public\s+)?interface\s+(?P<name>\w+)\s+extends\s+Repository<(?P<entity>[^,>]+),\s*(?P<id>[^>]+)>\s*\{",
            r"@Repository\npublic interface \g<name> extends GenericRepository<\g<entity>, \g<id>> {",
            content,
            count=1,
        )
        content = re.sub(r"\bPagingAndSortingRepository\b", "PageableRepository", content)
        content = re.sub(
            r"(?m)^(?P<indent>[ \t]*)(?=(?:public\s+)?interface\s+\w+\s+extends\s+(?:CrudRepository|PageableRepository|JpaSpecificationExecutor)<)",
            r"\g<indent>@Repository\n",
            content,
        )
        content = re.sub(r"(?m)(^[ \t]*@Repository\s*$\n)(?=[ \t]*@Repository\s*$\n)", "", content)
        content = self._normalize_repository_query_wildcards(content)
        content = re.sub(
            r'@Query\("(?P<query>[^"]+)"\)(?P<between>\s*(?:@\w+(?:\([^)]*\))?\s*)*)(?P<signature>Page<[^>]+>\s+\w+\s*\([^;{}]*\)\s*;)',
            self._rewrite_page_query_annotation,
            content,
            flags=re.MULTILINE,
        )
        content = re.sub(
            r'@Query\(\s*value\s*=\s*"(?P<query>[^"]+)"(?P<attrs>(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*(?:"[^"]*"|true|false|[A-Za-z_][A-Za-z0-9_$.]*))*)\s*\)(?P<between>\s*(?:@\w+(?:\([^)]*\))?\s*)*)(?P<signature>Page<[^>]+>\s+\w+\s*\([^;{}]*\)\s*;)',
            self._rewrite_named_page_query_annotation,
            content,
            flags=re.MULTILINE,
        )
        return content

    def _normalize_repository_query_wildcards(self, content: str) -> str:
        def replace_simple_annotation(match: re.Match) -> str:
            query = self._normalize_query_wildcard_parameter_syntax(match.group("query"))
            return f'@Query("{query}")'

        def replace_named_annotation(match: re.Match) -> str:
            attrs = match.group("attrs") or ""
            if re.search(r"\bnativeQuery\s*=\s*true\b", attrs, flags=re.IGNORECASE):
                return match.group(0)
            query = self._normalize_query_wildcard_parameter_syntax(match.group("query"))
            normalized_attrs = self._normalize_query_annotation_attrs(attrs)
            return f'@Query(value = "{query}"{normalized_attrs})'

        content = re.sub(
            r'@Query\("(?P<query>[^"]+)"\)',
            replace_simple_annotation,
            content,
        )
        content = re.sub(
            r'@Query\(\s*value\s*=\s*"(?P<query>[^"]+)"(?P<attrs>(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*(?:"[^"]*"|true|false|[A-Za-z_][A-Za-z0-9_$.]*))*)\s*\)',
            replace_named_annotation,
            content,
        )
        return content

    def _normalize_query_annotation_attrs(self, attrs: str) -> str:
        if not attrs:
            return ""

        def replace_query_attr(match: re.Match) -> str:
            name = match.group("name")
            query = self._normalize_query_wildcard_parameter_syntax(match.group("query"))
            return f'{name} = "{query}"'

        return re.sub(
            r'(?P<name>countQuery)\s*=\s*"(?P<query>[^"]+)"',
            replace_query_attr,
            attrs,
        )

    def _normalize_query_wildcard_parameter_syntax(self, query: str) -> str:
        normalized = query or ""
        wildcard_patterns = (
            (
                r"%\s*:(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*%",
                lambda match: f"CONCAT('%', :{match.group('name')}, '%')",
            ),
            (
                r"%\s*:(?P<name>[A-Za-z_][A-Za-z0-9_]*)",
                lambda match: f"CONCAT('%', :{match.group('name')})",
            ),
            (
                r":(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*%",
                lambda match: f"CONCAT(:{match.group('name')}, '%')",
            ),
        )
        for pattern, replacement in wildcard_patterns:
            normalized = re.sub(pattern, replacement, normalized)
        return normalized

    def _transform_jpa_repository_bootstrap_patterns(self, content: str) -> str:
        has_jpa_repository = "JpaRepository" in content
        has_enable_jpa = "EnableJpaRepositories" in content
        if not has_jpa_repository and not has_enable_jpa:
            return content

        if has_jpa_repository:
            content = re.sub(
                r"(?m)^[ \t]*import org\.springframework\.data\.jpa\.repository\.JpaRepository;\n?",
                "import io.micronaut.data.jpa.repository.JpaRepository;\n",
                content,
            )

            if "import io.micronaut.data.annotation.Repository;" not in content:
                package_match = re.search(r"(?m)^package [^;]+;\n", content)
                if package_match:
                    insert_at = package_match.end()
                    content = (
                        content[:insert_at]
                        + "\nimport io.micronaut.data.annotation.Repository;\n"
                        + content[insert_at:]
                    )

            content = re.sub(
                r"(?m)^[ \t]*import org\.springframework\.stereotype\.Repository;\n?",
                "",
                content,
            )
            content = re.sub(
                r"(?m)^(?P<indent>[ \t]*)@Singleton\s*\n(?=(?P=indent)(?:public\s+)?interface\s+\w+\s+extends\s+JpaRepository<)",
                r"\g<indent>@Repository\n",
                content,
            )
            content = re.sub(
                r"(?m)^(?P<indent>[ \t]*)(?=(?:public\s+)?interface\s+\w+\s+extends\s+JpaRepository<)",
                r"\g<indent>@Repository\n",
                content,
                count=1,
            )
            content = re.sub(
                r"(?m)(@Repository\n)(?=\s*@Repository\n)",
                "",
                content,
            )

        if has_enable_jpa:
            content = re.sub(
                r"(?m)^[ \t]*import org\.springframework\.data\.jpa\.repository\.config\.EnableJpaRepositories;\n?",
                "",
                content,
            )
            content = re.sub(
                r"(?m)^[ \t]*@EnableJpaRepositories(?:\([^)]*\))?\s*\n?",
                "",
                content,
            )

            if "import io.micronaut.context.annotation.Requires;" not in content:
                package_match = re.search(r"(?m)^package [^;]+;\n", content)
                if package_match:
                    insert_at = package_match.end()
                    content = (
                        content[:insert_at]
                        + "\nimport io.micronaut.context.annotation.Requires;\n"
                        + "import jakarta.inject.Singleton;\n"
                        + "import jakarta.persistence.EntityManagerFactory;\n"
                        + content[insert_at:]
                    )

            class_match = re.search(
                r"(?m)^(?P<indent>[ \t]*)(?:public|protected|private)?(?:\s+final|\s+abstract)?\s*class\s+\w+[^{]*\{",
                content,
            )
            if class_match:
                insert_at = class_match.start()
                indent = class_match.group("indent")
                block = (
                    f"{indent}@Singleton\n"
                    f"{indent}@Requires(beans = EntityManagerFactory.class)\n"
                )
                if block not in content:
                    content = content[:insert_at] + block + content[insert_at:]

        return content

    def _transform_jpa_specification_patterns(self, content: str) -> str:
        has_specification_patterns = any(
            marker in content
            for marker in (
                "JpaSpecificationExecutor",
                "org.springframework.data.jpa.domain.Specification",
                "io.micronaut.data.jpa.repository.criteria.Specification",
                "Specification.where(",
                "@EntityGraph",
                "EntityGraphType",
                "org.springframework.data.jpa.repository.EntityGraph",
                "org.springframework.data.jpa.repository.QueryHints",
                "jakarta.persistence.QueryHint",
                "javax.persistence.QueryHint",
            )
        )
        if not has_specification_patterns:
            return content

        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.data\.jpa\.repository\.JpaSpecificationExecutor;\n?",
            "import io.micronaut.data.jpa.repository.JpaSpecificationExecutor;\n",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.data\.jpa\.domain\.Specification;\n?",
            "import io.micronaut.data.jpa.repository.criteria.Specification;\n",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.data\.jpa\.repository\.EntityGraph;\n?",
            "import io.micronaut.data.jpa.annotation.EntityGraph;\n",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.data\.jpa\.repository\.EntityGraph\.EntityGraphType;\n?",
            "import io.micronaut.data.jpa.annotation.EntityGraph.Type;\n",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.data\.jpa\.repository\.QueryHints;\n?",
            "import io.micronaut.data.annotation.QueryHints;\n",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import jakarta\.persistence\.QueryHint;\n?",
            "import io.micronaut.data.annotation.QueryHint;\n",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import javax\.persistence\.QueryHint;\n?",
            "import io.micronaut.data.annotation.QueryHint;\n",
            content,
        )
        content = re.sub(r"\bEntityGraphType\b", "Type", content)
        content = self._annotate_advanced_specification_review(content)
        return content

    def _annotate_advanced_specification_review(self, content: str) -> str:
        review_comment = self._render_manual_review_comment(
            "Spring Data Specification composition using Specification.where(...).and(...) or .or(...) "
            "was migrated at the import level only; review Micronaut criteria semantics for chained predicates manually."
        )
        if review_comment in content:
            return content
        if "Specification.where(" not in content:
            return content
        if ".and(" not in content and ".or(" not in content:
            return content

        line_match = re.search(r"(?m)^(?P<indent>[ \t]*).*\bSpecification\.where\(", content)
        if not line_match:
            return content

        insert_at = line_match.start()
        indent = line_match.group("indent")
        return content[:insert_at] + f"{indent}{review_comment}\n" + content[insert_at:]

    def _rewrite_page_query_annotation(self, match: re.Match) -> str:
        query = match.group("query")
        count_query = self._derive_micronaut_count_query(query)
        if not count_query:
            return self._render_paginated_query_manual_review(match.group(0))
        return (
            f'@Query(value = "{query}", countQuery = "{count_query}")'
            f'{match.group("between")}{match.group("signature")}'
        )

    def _rewrite_named_page_query_annotation(self, match: re.Match) -> str:
        query = match.group("query")
        attrs = match.group("attrs") or ""
        if re.search(r"\bcountQuery\s*=", attrs):
            return match.group(0)
        attr_review_reason = self._page_query_attr_manual_review_reason(attrs)
        if attr_review_reason:
            safe_attrs = self._strip_unsupported_page_query_attrs(attrs)
            safe_annotation = (
                f'@Query(value = "{query}"{safe_attrs})'
                f'{match.group("between")}{match.group("signature")}'
            )
            return self._render_paginated_query_manual_review(safe_annotation, attr_review_reason)
        count_query = self._derive_micronaut_count_query(query)
        if not count_query:
            return self._render_paginated_query_manual_review(match.group(0))
        attrs_suffix = attrs.rstrip()
        return (
            f'@Query(value = "{query}"{attrs_suffix}, countQuery = "{count_query}")'
            f'{match.group("between")}{match.group("signature")}'
        )

    def _derive_micronaut_count_query(self, query: str) -> Optional[str]:
        normalized = " ".join((query or "").split())
        if not normalized:
            return None
        if self._contains_top_level_query_keyword(normalized, "group by"):
            return None
        if self._contains_top_level_query_keyword(normalized, "having"):
            return None
        if self._contains_top_level_query_keyword(normalized, " union "):
            return None
        if self._contains_top_level_query_keyword(normalized, " intersect "):
            return None
        if self._contains_top_level_query_keyword(normalized, " except "):
            return None
        select_match = re.match(
            r"(?is)^select\s+(?P<distinct>distinct\s+)?(?P<select>.+?)\s+from\s+(?P<rest>.+)$",
            normalized,
        )
        if not select_match:
            return None

        select_expr = select_match.group("select").strip()
        if re.match(r"(?i)^new\s+", select_expr):
            return None
        if self._contains_top_level_comma(select_expr):
            return None
        rest = self._strip_top_level_order_by(select_match.group("rest").strip())
        rest = re.sub(r"(?i)\bleft\s+join\s+fetch\b", "left join", rest)
        rest = re.sub(r"(?i)\bright\s+join\s+fetch\b", "right join", rest)
        rest = re.sub(r"(?i)\bjoin\s+fetch\b", "join", rest)
        if not rest:
            return None
        distinct_prefix = "DISTINCT " if select_match.group("distinct") else ""
        return f"SELECT count({distinct_prefix}{select_expr}) FROM {rest}"

    def _page_query_attr_manual_review_reason(self, attrs: str) -> Optional[str]:
        normalized = attrs or ""
        if re.search(r"\bcountProjection\s*=", normalized):
            return (
                "paginated Spring @Query uses countProjection, which has no trusted direct Micronaut countQuery rewrite; "
                "review and add an explicit countQuery manually."
            )
        if re.search(r"\bcountName\s*=", normalized):
            return (
                "paginated Spring @Query uses countName, which depends on named-query conventions and needs a manual Micronaut countQuery review."
            )
        if re.search(r"\bname\s*=", normalized):
            return (
                "paginated Spring @Query uses a named query reference, which needs manual Micronaut review before adding a countQuery."
            )
        return None

    def _strip_unsupported_page_query_attrs(self, attrs: str) -> str:
        if not attrs:
            return ""
        stripped = attrs
        for attr_name in ("countProjection", "countName", "name"):
            stripped = re.sub(
                rf'\s*,\s*{attr_name}\s*=\s*(?:"[^"]*"|true|false|[A-Za-z_][A-Za-z0-9_$.]*)',
                "",
                stripped,
            )
        return stripped.rstrip()

    def _render_paginated_query_manual_review(self, original_annotation: str, reason: Optional[str] = None) -> str:
        indent_match = re.match(r"(?P<indent>[ \t]*)", original_annotation)
        indent = indent_match.group("indent") if indent_match else ""
        reason_text = (
            reason
            or "paginated Spring @Query could not be converted to a safe Micronaut countQuery automatically; add an explicit countQuery manually."
        )
        comment = (
            f"{indent}// TODO: manual review: {reason_text}\n"
        )
        return comment + original_annotation

    def _strip_top_level_order_by(self, query_fragment: str) -> str:
        lowered = query_fragment.lower()
        depth = 0
        index = 0
        while index < len(query_fragment):
            char = query_fragment[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth = max(0, depth - 1)
            elif depth == 0 and lowered.startswith("order by", index):
                before = lowered[index - 1] if index > 0 else " "
                after = lowered[index + len("order by")] if index + len("order by") < len(lowered) else " "
                if before.isspace() and after.isspace():
                    return query_fragment[:index].rstrip()
            index += 1
        return query_fragment.strip()

    def _contains_top_level_query_keyword(self, text: str, keyword: str) -> bool:
        lowered = text.lower()
        target = keyword.lower()
        depth = 0
        index = 0
        while index < len(text):
            char = text[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth = max(0, depth - 1)
            elif depth == 0 and lowered.startswith(target, index):
                return True
            index += 1
        return False

    def _contains_top_level_comma(self, text: str) -> bool:
        depth = 0
        for char in text:
            if char == "(":
                depth += 1
            elif char == ")":
                depth = max(0, depth - 1)
            elif char == "," and depth == 0:
                return True
        return False

    def _repository_placeholder_return(self, return_type: str) -> str:
        normalized = str(return_type or "").strip()
        if not normalized or normalized == "void":
            return ""
        primitive_defaults = {
            "boolean": "return false;",
            "byte": "return 0;",
            "short": "return 0;",
            "int": "return 0;",
            "long": "return 0L;",
            "float": "return 0.0f;",
            "double": "return 0.0d;",
            "char": "return '\\0';",
        }
        if normalized in primitive_defaults:
            return primitive_defaults[normalized]
        return "return null;"

    def _transform_string_utils_patterns(self, content: str) -> str:
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.util\.StringUtils;\n?",
            "import io.micronaut.core.util.StringUtils;\n",
            content,
        )
        content = re.sub(r"\bStringUtils\.hasLength\(", "StringUtils.isNotEmpty(", content)
        return content

    def _transform_jakarta_namespace_patterns(self, content: str) -> str:
        namespace_pairs = (
            ("javax.validation.", "jakarta.validation."),
            ("javax.persistence.", "jakarta.persistence."),
            ("javax.annotation.", "jakarta.annotation."),
            ("javax.xml.bind.", "jakarta.xml.bind."),
            ("javax.activation.", "jakarta.activation."),
        )
        updated = content
        for source_ns, target_ns in namespace_pairs:
            updated = updated.replace(source_ns, target_ns)
        return updated

    def _transform_test_validation_support_patterns(self, content: str) -> str:
        if "LocalValidatorFactoryBean" not in content and "LocaleContextHolder" not in content:
            return content

        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.validation\.beanvalidation\.LocalValidatorFactoryBean;\n?",
            "import jakarta.validation.Validation;\nimport org.hibernate.validator.messageinterpolation.ParameterMessageInterpolator;\n",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.context\.i18n\.LocaleContextHolder;\n?",
            "",
            content,
        )
        content = re.sub(r"\bLocaleContextHolder\.setLocale\(", "Locale.setDefault(", content)
        content = re.sub(
            r"LocalValidatorFactoryBean\s+\w+\s*=\s*new LocalValidatorFactoryBean\(\);\s*"
            r"\w+\.afterPropertiesSet\(\);\s*"
            r"return\s+\w+;",
            "return Validation.byDefaultProvider().configure().messageInterpolator(new ParameterMessageInterpolator()).buildValidatorFactory().getValidator();",
            content,
            flags=re.DOTALL,
        )
        return content

    def _transform_entity_lookup_patterns(self, content: str) -> str:
        if "ObjectRetrievalFailureException" not in content:
            return content

        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.orm\.ObjectRetrievalFailureException;\n?",
            "import java.util.NoSuchElementException;\n",
            content,
        )
        content = re.sub(r"\s+throws\s+ObjectRetrievalFailureException", "", content)
        content = re.sub(
            r"throw new ObjectRetrievalFailureException\(\s*entityClass\s*,\s*entityId\s*\);",
            'throw new NoSuchElementException(entityClass.getSimpleName() + " not found: " + entityId);',
            content,
        )
        return content

    def _transform_serialization_support_patterns(self, content: str) -> str:
        if "SerializationUtils" not in content:
            return content

        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.util\.SerializationUtils;\n?",
            "import java.io.ByteArrayInputStream;\n"
            "import java.io.ByteArrayOutputStream;\n"
            "import java.io.IOException;\n"
            "import java.io.ObjectInputStream;\n"
            "import java.io.ObjectOutputStream;\n"
            "import java.io.Serializable;\n",
            content,
        )
        content = re.sub(
            r"SerializationUtils\.deserialize\(\s*SerializationUtils\.serialize\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\)",
            r"roundTripSerialize(\1)",
            content,
        )
        if "roundTripSerialize(" in content and "private static <T extends Serializable> T roundTripSerialize" not in content:
            method_block = (
                "\n\t@SuppressWarnings(\"unchecked\")\n"
                "\tprivate static <T extends Serializable> T roundTripSerialize(T value) {\n"
                "\t\ttry {\n"
                "\t\t\tByteArrayOutputStream buffer = new ByteArrayOutputStream();\n"
                "\t\t\ttry (ObjectOutputStream output = new ObjectOutputStream(buffer)) {\n"
                "\t\t\t\toutput.writeObject(value);\n"
                "\t\t\t}\n"
                "\t\t\ttry (ObjectInputStream input = new ObjectInputStream(new ByteArrayInputStream(buffer.toByteArray()))) {\n"
                "\t\t\t\treturn (T) input.readObject();\n"
                "\t\t\t}\n"
                "\t\t} catch (IOException | ClassNotFoundException ex) {\n"
                "\t\t\tthrow new IllegalStateException(\"Java serialization round-trip failed\", ex);\n"
                "\t\t}\n"
                "\t}\n"
            )
            last_brace = content.rfind("}")
            if last_brace != -1:
                content = content[:last_brace] + method_block + content[last_brace:]
        return content

    def _transform_paging_patterns(self, content: str) -> str:
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.data\.domain\.Page;\n?",
            "import io.micronaut.data.model.Page;\n",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.data\.domain\.Pageable;\n?",
            "import io.micronaut.data.model.Pageable;\n",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.data\.domain\.PageRequest;\n?",
            "",
            content,
        )
        content = re.sub(r"\bPageRequest\.of\s*\(", "Pageable.from(", content)
        return content

    def _transform_page_api_usage_patterns(self, content: str) -> str:
        content = re.sub(r"\.toList\(\)", ".getContent()", content)
        page_variables = set(re.findall(r"\bPage<[^>]+>\s+(\w+)\b", content))
        for variable in page_variables:
            content = re.sub(rf"\b{re.escape(variable)}\.isEmpty\(\)", f"{variable}.getContent().isEmpty()", content)
            content = re.sub(
                rf"\b{re.escape(variable)}\.iterator\(\)\.next\(\)",
                f"{variable}.getContent().get(0)",
                content,
            )
            content = re.sub(
                rf"\b{re.escape(variable)}\.getTotalElements\(\)",
                f"{variable}.getTotalSize()",
                content,
            )
        return content

    def _transform_model_patterns(self, content: str) -> str:
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.ui\.Model;\n?",
            "import java.util.Map;\n",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.ui\.ModelMap;\n?",
            "import java.util.Map;\n",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.web\.servlet\.ModelAndView;\n?",
            "import io.micronaut.views.ModelAndView;\nimport java.util.LinkedHashMap;\n",
            content,
        )
        content = re.sub(r"\bModelMap\s+(\w+)\b", r"Map<String, Object> \1", content)
        content = re.sub(r"\bModel\s+(\w+)\b", r"Map<String, Object> \1", content)

        def replace_keyed_model_attribute(match):
            return f'{match.group("target")}.put("{match.group("key")}", {match.group("value").strip()});'

        content = re.sub(
            r"(?P<target>\b\w+\b)\.addAttribute\(\s*\"(?P<key>[^\"]+)\"\s*,\s*(?P<value>[^;]+)\);",
            replace_keyed_model_attribute,
            content,
        )

        def replace_single_model_attribute(match):
            target = match.group("target")
            value = match.group("value").strip()
            name = self._infer_model_attribute_name(value)
            return f'{target}.put("{name}", {value});'

        content = re.sub(
            r"(?P<target>\b\w+\b)\.addAttribute\(\s*(?P<value>[A-Za-z_][A-Za-z0-9_.$()<>]*)\s*\);",
            replace_single_model_attribute,
            content,
        )
        content = self._transform_model_and_view_declarations(content)
        return content

    def _transform_model_attribute_patterns(self, content: str) -> str:
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.web\.bind\.annotation\.ModelAttribute;\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import io\.micronaut\.http\.annotation\.ModelAttribute;\n?",
            "",
            content,
        )
        content = re.sub(r"(?m)^[ \t]*@ModelAttribute(?:\([^)]*\))?\s*\n", "", content)
        content = re.sub(r"@ModelAttribute(?:\([^)]*\))?\s+", "", content)
        content = re.sub(
            r'@PathVariable\s*\(\s*name\s*=\s*"([^"]+)"\s*,\s*required\s*=\s*(?:true|false)\s*\)',
            r'@PathVariable("\1")',
            content,
        )
        content = re.sub(
            r'@PathVariable\s*\(\s*value\s*=\s*"([^"]+)"\s*,\s*required\s*=\s*(?:true|false)\s*\)',
            r'@PathVariable("\1")',
            content,
        )
        content = re.sub(
            r'@PathVariable\s*\(\s*name\s*=\s*"([^"]+)"\s*\)',
            r'@PathVariable("\1")',
            content,
        )
        content = re.sub(
            r'@PathVariable\s*\(\s*value\s*=\s*"([^"]+)"\s*\)',
            r'@PathVariable("\1")',
            content,
        )
        return content

    def _transform_init_binder_patterns(self, content: str) -> str:
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.web\.bind\.annotation\.InitBinder;\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import io\.micronaut\.http\.annotation\.InitBinder;\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.web\.bind\.WebDataBinder;\n?",
            "",
            content,
        )
        content = self._remove_annotated_methods(content, ("InitBinder",))
        return content

    def _transform_binding_result_patterns(self, content: str) -> str:
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.validation\.BindingResult;\n?",
            "",
            content,
        )
        content = self._rewrite_binding_result_methods(content)
        if "HttpRequest<?> request" in content:
            content = self._ensure_import(content, "io.micronaut.http.HttpRequest")
        return content

    def _transform_cache_customizer_patterns(self, content: str) -> str:
        if "JCacheManagerCustomizer" not in content and "MutableConfiguration" not in content:
            return content

        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.boot\.autoconfigure\.cache\.JCacheManagerCustomizer;\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import javax\.cache\.configuration\.MutableConfiguration;\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import javax\.cache\.configuration\.Configuration;\n?",
            "",
            content,
        )
        cache_names = re.findall(r'createCache\(\s*"([^"]+)"', content)
        cache_names = list(dict.fromkeys(cache_names))

        content = re.sub(
            r"(?m)^[ \t]*import io\.micronaut\.context\.annotation\.Bean;\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import io\.micronaut\.context\.annotation\.Factory;\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.context\.annotation\.Bean;\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.context\.annotation\.Configuration;\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.cache\.annotation\.EnableCaching;\n?",
            "",
            content,
        )
        content = self._remove_method_containing_text(content, "JCacheManagerCustomizer")
        content = self._remove_method_containing_text(content, "cacheConfiguration(")
        content = self._remove_named_annotations(content, ("Factory", "Configuration", "EnableCaching"))

        constructor_lines = ['\t@Context', '\t@Singleton', '\tclass CacheConfiguration {', '', '\t\tCacheConfiguration(DynamicCacheManager<?> cacheManager) {']
        if cache_names:
            for cache_name in cache_names:
                constructor_lines.append(f'\t\t\tcacheManager.getCache("{cache_name}");')
        else:
            constructor_lines.append('\t\t\t// Micronaut creates named caches lazily; explicit warm-up retained for migration clarity.')
        constructor_lines.extend(['\t\t}', '\t}'])

        replacement = "\n".join(constructor_lines)
        class_pattern = re.compile(
            r"(?ms)(?:^[ \t]*@\w+(?:\([^)]*\))?\s*\n)*^[ \t]*class\s+CacheConfiguration\b[^{]*\{.*?^[ \t]*\}",
            re.MULTILINE,
        )
        if class_pattern.search(content):
            content = class_pattern.sub(replacement, content, count=1)
        else:
            content += "\n\n" + replacement + "\n"

        package_match = re.search(r"(?m)^package [^;]+;\n", content)
        if package_match:
            insert_at = package_match.end()
            additions = []
            if "import io.micronaut.context.annotation.Context;" not in content:
                additions.append("import io.micronaut.context.annotation.Context;")
            if "import io.micronaut.cache.DynamicCacheManager;" not in content:
                additions.append("import io.micronaut.cache.DynamicCacheManager;")
            if "import jakarta.inject.Singleton;" not in content:
                additions.append("import jakarta.inject.Singleton;")
            if additions:
                content = content[:insert_at] + "\n" + "\n".join(additions) + "\n" + content[insert_at:]
        return content

    def _transform_model_and_view_declarations(self, content: str) -> str:
        declaration_pattern = re.compile(
            r"(?m)^(?P<indent>[ \t]*)ModelAndView\s+(?P<name>\w+)\s*=\s*new\s+ModelAndView\s*\(\s*\"(?P<view>[^\"]+)\"\s*\)\s*;"
        )

        declarations = list(declaration_pattern.finditer(content))
        for match in reversed(declarations):
            indent = match.group("indent")
            name = match.group("name")
            view = match.group("view")
            replacement = (
                f"{indent}Map<String, Object> {name}Model = new LinkedHashMap<>();\n"
                f'{indent}ModelAndView<Map<String, Object>> {name} = new ModelAndView<>("{view}", {name}Model);'
            )
            content = content[: match.start()] + replacement + content[match.end() :]
            content = re.sub(
                rf"\b{name}\.addObject\(\s*\"(?P<key>[^\"]+)\"\s*,\s*(?P<value>[^;]+)\);",
                rf'{name}Model.put("\g<key>", \g<value>);',
                content,
            )

            def replace_single_add_object(object_match):
                value = object_match.group("value").strip()
                attribute_name = self._infer_model_attribute_name(value)
                return f'{name}Model.put("{attribute_name}", {value});'

            content = re.sub(
                rf"\b{name}\.addObject\(\s*(?P<value>[A-Za-z_][A-Za-z0-9_.$()<>]*)\s*\);",
                replace_single_add_object,
                content,
            )

        return content

    def _infer_model_attribute_name(self, expression: str) -> str:
        cleaned = str(expression or "").strip()
        if not cleaned:
            return "value"
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", cleaned):
            return cleaned
        constructor_match = re.fullmatch(r"new\s+([A-Z][A-Za-z0-9_]*)\s*\([^)]*\)", cleaned)
        if constructor_match:
            class_name = constructor_match.group(1)
            return class_name[:1].lower() + class_name[1:]
        tail = cleaned.split(".")[-1]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tail):
            return tail
        return "value"

    def _transform_test_slice_patterns(self, content: str) -> str:
        has_webmvc = "@WebMvcTest" in content or "WebMvcTest(" in content
        has_datajpa = "@DataJpaTest" in content or "DataJpaTest(" in content
        has_mockmvc = "MockMvc" in content or "MockMvcRequestBuilders" in content or "MockMvcResultMatchers" in content
        mockmvc_supported = False
        preserved_disabled_annotation = self._extract_preserved_disabled_annotation(content)

        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.boot\.test\.autoconfigure\.web\.servlet\.WebMvcTest;\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.boot\.test\.autoconfigure\.orm\.jpa\.DataJpaTest;\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.boot\.test\.autoconfigure\.jdbc\.AutoConfigureTestDatabase(?:\.[A-Za-z_]+)?;\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.context\.annotation\.ComponentScan;\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.context\.annotation\.FilterType;\n?",
            "",
            content,
        )

        disabled_reasons = []
        if has_webmvc:
            disabled_reasons.append("Spring MVC slice test requires manual Micronaut HTTP-test rewrite")

        if has_mockmvc:
            content = re.sub(
                r"(?m)^[ \t]*import static org\.springframework\.test\.web\.servlet\.[^\n]+;\n?",
                "",
                content,
            )
            content = re.sub(
                r"(?m)^[ \t]*import org\.springframework\.test\.web\.servlet\.[^\n]+;\n?",
                "",
                content,
            )
            content = re.sub(
                r"(?m)^(?:[ \t]*@(?:Inject|Autowired)\s*\n)?[ \t]*(?:private\s+|protected\s+|public\s+)?(?:final\s+)?MockMvc\s+\w+\s*;\n?",
                "",
                content,
            )
            content, mockmvc_supported = self._rewrite_mockmvc_http_test_patterns(content)

        if has_webmvc or has_datajpa:
            content = self._remove_named_annotations(
                content,
                ("WebMvcTest", "DataJpaTest", "AutoConfigureTestDatabase"),
            )
            annotation_parts = []
            if preserved_disabled_annotation:
                annotation_parts.append(preserved_disabled_annotation)
            annotation_parts.append("@MicronautTest")
            if has_webmvc and not mockmvc_supported:
                disabled_reason = "; ".join(disabled_reasons) or "Manual migration review required"
                annotation_parts = [f'@Disabled("{disabled_reason}")', "@MicronautTest"]
            annotation_block = "\n".join(annotation_parts)
            class_pattern = re.compile(r"(?m)^(?:[ \t]*@\w+(?:\([^)]*\))?\s*\n)*([ \t]*(?:public\s+)?class\s+\w+)")

            def replace_class(match):
                return f"{annotation_block}\n{match.group(1)}"

            content, replacements = class_pattern.subn(replace_class, content, count=1)
            if replacements == 0:
                content = annotation_block + "\n" + content
            if has_datajpa:
                content = re.sub(
                    r"(?m)^(?P<indent>[ \t]*)(?P<visibility>private|protected|public)\s+final\s+(?P<type>[A-Z][A-Za-z0-9_<>, ?]+)\s+(?P<name>\w+)\s*;",
                    r"\g<indent>@Inject\n\g<indent>\g<visibility> \g<type> \g<name>;",
                    content,
                )
                content = self._transform_field_to_constructor_injection(content, allow_test_classes=True)
                content = re.sub(
                    r"(?m)^(?P<indent>[ \t]*)Pageable\s+(?P<name>\w+)\s*;",
                    r"\g<indent>Pageable \g<name> = Pageable.from(0, 20);",
                    content,
                )

        if has_mockmvc and not mockmvc_supported:
            content = self._remove_annotated_fields(content, ("MockBean",))
            content = self._rewrite_annotated_methods(
                content,
                ("BeforeEach",),
                ["// Spring MVC slice setup removed during Micronaut placeholder migration."],
            )
            content = self._rewrite_annotated_methods(
                content,
                ("Test",),
                ['unsupportedSpringMvcTest("Spring MVC test interaction requires manual Micronaut rewrite");'],
            )
            content = self._ensure_helper_method(
                content,
                method_name="unsupportedSpringMvcTest",
                parameter_signature="String reason",
                method_body='throw new UnsupportedOperationException(reason);',
            )
            content = self._ensure_helper_method(
                content,
                method_name="unsupportedSpringMvcRequest",
                parameter_signature="Object... ignored",
                method_body='throw new UnsupportedOperationException("Spring MVC request builder requires manual Micronaut rewrite");',
                return_type="Object",
            )

        content = self._rewrite_pageimpl_test_usage(content)
        if "assertTrue(response.getStatus().getCode() >= 300 && response.getStatus().getCode() < 400)" in content:
            content = self._ensure_micronaut_test_property(
                content,
                "micronaut.http.client.follow-redirects",
                "false",
            )
        return content

    def _extract_preserved_disabled_annotation(self, content: str) -> Optional[str]:
        match = re.search(r"(?m)^[ \t]*(@Disabled(?:\([^)]*\))?)\s*$", content)
        if not match:
            return None
        return match.group(1)

    def _ensure_micronaut_test_property(self, content: str, name: str, value: str) -> str:
        property_line = f'@Property(name = "{name}", value = "{value}")'
        if property_line in content or "@MicronautTest" not in content:
            return content

        content = self._ensure_import(content, "io.micronaut.context.annotation.Property")
        return content.replace("@MicronautTest", f"@MicronautTest\n{property_line}", 1)

    def _ensure_named_method_prefix(
        self,
        content: str,
        *,
        method_name: str,
        required_header_fragment: str,
        missing_guard: str,
        prefix_lines: Sequence[str],
    ) -> str:
        method_match = re.search(
            rf"(?m)^(?P<indent>[ \t]*)(?:(?:public|protected|private)\s+)?String\s+{re.escape(method_name)}\b[^\n]*\{{",
            content,
        )
        if not method_match:
            return content
        header = method_match.group(0)
        if required_header_fragment not in header:
            return content
        body_start = content.find("{", method_match.start())
        if body_start == -1:
            return content
        body_end = self._find_matching_brace(content, body_start)
        if body_end == -1:
            return content
        body = content[body_start + 1 : body_end]
        if missing_guard in body:
            return content
        indent_match = re.search(r"\n([ \t]*)\S", body)
        inner_indent = indent_match.group(1) if indent_match else method_match.group("indent") + "    "
        injected_lines = []
        for line in prefix_lines:
            if line == "}":
                injected_lines.append(f"{inner_indent[:-4] if inner_indent.endswith('    ') else inner_indent}{line}")
            else:
                injected_lines.append(f"{inner_indent}{line}")
        prefix = "\n" + "\n".join(injected_lines) + "\n"
        return content[: body_start + 1] + prefix + body + content[body_end:]

    def _rewrite_mockmvc_http_test_patterns(self, content: str) -> Tuple[str, bool]:
        supported = False
        search_from = 0
        while True:
            perform_index = content.find("mockMvc.perform(", search_from)
            if perform_index == -1:
                break

            statement_start = content.rfind("\n", 0, perform_index) + 1
            statement_end = self._find_statement_end(content, perform_index)
            if statement_end == -1:
                return content, False

            statement = content[statement_start : statement_end + 1]
            rewritten_statement = self._rewrite_single_mockmvc_statement(statement)
            if rewritten_statement is None:
                return content, False

            content = content[:statement_start] + rewritten_statement + content[statement_end + 1 :]
            search_from = statement_start + len(rewritten_statement)
            supported = True

        if not supported:
            return content, False

        content = self._rewrite_mockmvc_resultactions_expectations(content)
        content = self._ensure_test_http_client_field(content)
        return content, True

    def _rewrite_single_mockmvc_statement(self, statement: str) -> Optional[str]:
        perform_index = statement.find("mockMvc.perform(")
        request_open = statement.find("(", perform_index)
        request_close = self._find_matching_parenthesis(statement, request_open)
        if request_close == -1:
            return None

        request_expression = statement[request_open + 1 : request_close].strip()
        request_code = self._rewrite_mockmvc_request_expression(request_expression)
        if request_code is None:
            return None

        tail = statement[request_close + 1 :].strip()
        expectations = []
        while tail.startswith(".andExpect("):
            expect_open = tail.find("(", len(".andExpect"))
            expect_close = self._find_matching_parenthesis(tail, expect_open)
            if expect_close == -1:
                return None
            expectations.append(tail[expect_open + 1 : expect_close].strip())
            tail = tail[expect_close + 1 :].strip()

        if tail != ";":
            return None

        indent_match = re.match(r"^([ \t]*)", statement)
        indent = indent_match.group(1) if indent_match else ""
        assigned = "=" in statement.split("mockMvc.perform(", 1)[0]
        explicit_accept = ".accept(" in request_expression

        if not explicit_accept and self._mockmvc_expectations_prefer_html(expectations):
            request_code += ".accept(io.micronaut.http.MediaType.TEXT_HTML_TYPE)"

        if not expectations and not assigned:
            return f"{indent}client.toBlocking().exchange({request_code}, String.class);\n"

        assertion_lines = self._rewrite_mockmvc_expectations(expectations, indent)
        if assertion_lines is None and expectations:
            return None

        lines = [f"{indent}HttpResponse<String> response = client.toBlocking().exchange({request_code}, String.class);"]
        if assertion_lines:
            lines.extend(assertion_lines)
        return "\n".join(lines) + "\n"

    def _rewrite_mockmvc_request_expression(self, expression: str) -> Optional[str]:
        cleaned = " ".join((expression or "").split())
        request_match = re.match(
            r"(?:MockMvcRequestBuilders\.)?(get|post|put|delete|patch)\s*\(",
            cleaned,
            flags=re.IGNORECASE,
        )
        if not request_match:
            return None

        method_name = request_match.group(1).upper()
        call_open = cleaned.find("(", request_match.end() - 1)
        call_close = self._find_matching_parenthesis(cleaned, call_open)
        if call_close == -1:
            return None

        base_args = self._split_top_level_arguments(cleaned[call_open + 1 : call_close])
        if not base_args:
            return None
        path_literal = base_args[0]
        path_args = base_args[1:]
        if not path_literal.startswith('"'):
            return None

        params = []
        accept_value = None
        tail = cleaned[call_close + 1 :].strip()
        while tail.startswith("."):
            if tail.startswith(".param("):
                param_open = tail.find("(", len(".param"))
                param_close = self._find_matching_parenthesis(tail, param_open)
                if param_close == -1:
                    return None
                param_args = self._split_top_level_arguments(tail[param_open + 1 : param_close])
                if len(param_args) != 2:
                    return None
                params.append((param_args[0], param_args[1]))
                tail = tail[param_close + 1 :].strip()
                continue
            if tail.startswith(".accept("):
                accept_open = tail.find("(", len(".accept"))
                accept_close = self._find_matching_parenthesis(tail, accept_open)
                if accept_close == -1:
                    return None
                accept_args = self._split_top_level_arguments(tail[accept_open + 1 : accept_close])
                if len(accept_args) != 1:
                    return None
                accept_value = accept_args[0]
                tail = tail[accept_close + 1 :].strip()
                continue
            return None

        if tail.strip():
            return None

        path_expression = self._build_mockmvc_path_expression(path_literal, path_args)
        if not path_expression:
            return None

        if method_name in {"GET", "DELETE"}:
            if params:
                query_expression = self._build_mockmvc_query_expression(params)
                separator = '"&"' if "?" in path_literal else '"?"'
                path_expression = f"{path_expression} + {separator} + {query_expression}"
            request_code = f"HttpRequest.{method_name}({path_expression})"
        else:
            if params:
                body_expression = self._build_mockmvc_query_expression(params)
                request_code = (
                    f"HttpRequest.{method_name}({path_expression}, {body_expression})"
                    f".contentType(io.micronaut.http.MediaType.APPLICATION_FORM_URLENCODED_TYPE)"
                )
            else:
                request_code = (
                    f"HttpRequest.{method_name}({path_expression}, \"\")"
                    f".contentType(io.micronaut.http.MediaType.APPLICATION_FORM_URLENCODED_TYPE)"
                )

        if accept_value:
            accept_literal = accept_value.replace(
                "MediaType.APPLICATION_JSON",
                "io.micronaut.http.MediaType.APPLICATION_JSON_TYPE",
            )
            request_code += f".accept({accept_literal})"
        return request_code

    def _rewrite_mockmvc_expectations(self, expectations: Sequence[str], indent: str) -> Optional[List[str]]:
        lines = []
        status_map = {
            "isOk": "OK",
            "isCreated": "CREATED",
            "isAccepted": "ACCEPTED",
            "isNoContent": "NO_CONTENT",
            "isBadRequest": "BAD_REQUEST",
            "isUnauthorized": "UNAUTHORIZED",
            "isForbidden": "FORBIDDEN",
            "isNotFound": "NOT_FOUND",
            "isConflict": "CONFLICT",
            "isInternalServerError": "INTERNAL_SERVER_ERROR",
        }

        for expectation in expectations:
            cleaned = " ".join((expectation or "").split())
            status_match = re.fullmatch(r"status\(\)\.(\w+)\(\)", cleaned)
            if status_match:
                if status_match.group(1) == "is3xxRedirection":
                    lines.append(
                        f"{indent}org.junit.jupiter.api.Assertions.assertTrue(response.getStatus().getCode() >= 300 && response.getStatus().getCode() < 400);"
                    )
                    continue
                status_name = status_map.get(status_match.group(1))
                if not status_name:
                    lines.append(
                        f'{indent}// TODO: review unsupported MockMvc status expectation after migration: {cleaned}'
                    )
                    continue
                lines.append(
                    f"{indent}org.junit.jupiter.api.Assertions.assertEquals(HttpStatus.{status_name}, response.getStatus());"
                )
                continue

            content_type_match = re.fullmatch(
                r"content\(\)\.(?:contentType|contentTypeCompatibleWith)\(\s*([A-Za-z0-9_\.]+)\s*\)",
                cleaned,
            )
            if content_type_match:
                expected_content_type = content_type_match.group(1)
                if (
                    expected_content_type.startswith("MediaType.APPLICATION_")
                    and not expected_content_type.endswith("_TYPE")
                ):
                    expected_content_type += "_TYPE"
                lines.append(
                    f"{indent}org.junit.jupiter.api.Assertions.assertEquals({expected_content_type}, response.getContentType().orElse(null));"
                )
                continue

            string_match = re.fullmatch(r"content\(\)\.string\(\s*(.+)\s*\)", cleaned)
            if string_match:
                lines.append(
                    f"{indent}org.junit.jupiter.api.Assertions.assertEquals({string_match.group(1)}, response.body());"
                )
                continue

            if cleaned.startswith("jsonPath("):
                lines.append(
                    f'{indent}// TODO: review JSON payload expectation after migration: {cleaned}'
                )
                continue
            if cleaned.startswith("view().") or cleaned.startswith("model().") or cleaned.startswith("forwardedUrl(") or cleaned.startswith("redirectedUrl("):
                lines.append(
                    f'{indent}// TODO: review migrated view/model expectation manually: {self._summarize_mockmvc_expectation(cleaned)}'
                )
                continue

            return None

        return lines

    def _mockmvc_expectations_prefer_html(self, expectations: Sequence[str]) -> bool:
        if not expectations:
            return False

        for expectation in expectations:
            cleaned = " ".join((expectation or "").split())
            if (
                cleaned.startswith("view().")
                or cleaned.startswith("model().")
                or cleaned == "status().is3xxRedirection()"
            ):
                return True
        return False

    def _summarize_mockmvc_expectation(self, expectation: str) -> str:
        cleaned = " ".join((expectation or "").split())
        if cleaned.startswith("model().attribute("):
            return "model().attribute(...)"
        if cleaned.startswith("model().attributeExists("):
            return "model().attributeExists(...)"
        if cleaned.startswith("model().attributeHasErrors("):
            return "model().attributeHasErrors(...)"
        if cleaned.startswith("model().attributeHasNoErrors("):
            return "model().attributeHasNoErrors(...)"
        if cleaned.startswith("model().attributeHasFieldErrors("):
            return "model().attributeHasFieldErrors(...)"
        if cleaned.startswith("model().attributeHasFieldErrorCode("):
            return "model().attributeHasFieldErrorCode(...)"
        if cleaned.startswith("view().name("):
            return "view().name(...)"
        return cleaned

    def _split_top_level_arguments(self, text: str) -> List[str]:
        if text is None:
            return []
        parts = []
        current = []
        paren_depth = 0
        brace_depth = 0
        angle_depth = 0
        in_string = False
        string_delimiter = ""
        escaped = False
        for char in text:
            if in_string:
                current.append(char)
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == string_delimiter:
                    in_string = False
                continue
            if char in {'"', "'"}:
                in_string = True
                string_delimiter = char
                current.append(char)
                continue
            if char == "(":
                paren_depth += 1
            elif char == ")" and paren_depth > 0:
                paren_depth -= 1
            elif char == "{":
                brace_depth += 1
            elif char == "}" and brace_depth > 0:
                brace_depth -= 1
            elif char == "<":
                angle_depth += 1
            elif char == ">" and angle_depth > 0:
                angle_depth -= 1
            elif char == "," and paren_depth == 0 and brace_depth == 0 and angle_depth == 0:
                piece = "".join(current).strip()
                if piece:
                    parts.append(piece)
                current = []
                continue
            current.append(char)
        final_piece = "".join(current).strip()
        if final_piece:
            parts.append(final_piece)
        return parts

    def _build_mockmvc_path_expression(self, path_literal: str, path_args: Sequence[str]) -> Optional[str]:
        if not path_args:
            return path_literal
        if not (path_literal.startswith('"') and path_literal.endswith('"')):
            return None
        format_literal = re.sub(r"\{[^}]+\}", "%s", path_literal[1:-1])
        return f'String.format("{format_literal}", {", ".join(path_args)})'

    def _build_mockmvc_query_expression(self, params: Sequence[Tuple[str, str]]) -> str:
        encoded_parts = []
        for key, value in params:
            encoded_parts.append(
                f'{key} + "=" + java.net.URLEncoder.encode(String.valueOf({value}), java.nio.charset.StandardCharsets.UTF_8)'
            )
        return " + \"&\" + ".join(encoded_parts) if encoded_parts else '""'

    def _ensure_test_http_client_field(self, content: str) -> str:
        if "private HttpClient client;" in content:
            return content

        class_match = re.search(
            r"(?m)^(?P<indent>[ \t]*)(?:public\s+)?class\s+\w+[^{]*\{",
            content,
        )
        if not class_match:
            return content

        field_indent = class_match.group("indent") + "    "
        insertion = (
            f"\n{field_indent}@Inject\n"
            f'{field_indent}@Client("/")\n'
            f"{field_indent}private HttpClient client;\n"
        )
        return content[: class_match.end()] + insertion + content[class_match.end() :]

    def _rewrite_mockmvc_resultactions_expectations(self, content: str) -> str:
        search_from = 0
        while True:
            match = re.search(r"(?m)^[ \t]*\w+\.andExpect\(", content[search_from:])
            if not match:
                return content
            statement_start = search_from + match.start()
            statement_end = self._find_statement_end(content, statement_start)
            if statement_end == -1:
                return content

            statement = content[statement_start : statement_end + 1]
            rewritten = self._rewrite_resultactions_expectation_statement(statement)
            if rewritten is None:
                search_from = statement_end + 1
                continue

            content = content[:statement_start] + rewritten + content[statement_end + 1 :]
            search_from = statement_start + len(rewritten)

    def _rewrite_resultactions_expectation_statement(self, statement: str) -> Optional[str]:
        prefix_match = re.match(r"^([ \t]*)\w+\.andExpect\(", statement)
        if not prefix_match:
            return None
        indent = prefix_match.group(1)
        tail = statement[len(prefix_match.group(0)) - 1 :]
        expectations = []
        while tail.startswith("(") or tail.startswith(".andExpect("):
            if tail.startswith("("):
                expect_open = 0
            else:
                expect_open = tail.find("(", len(".andExpect"))
            expect_close = self._find_matching_parenthesis(tail, expect_open)
            if expect_close == -1:
                return None
            expectations.append(tail[expect_open + 1 : expect_close].strip())
            tail = tail[expect_close + 1 :].strip()
            if not tail.startswith(".andExpect("):
                break

        if tail != ";":
            return None

        assertion_lines = self._rewrite_mockmvc_expectations(expectations, indent)
        if assertion_lines is None:
            return None
        return "\n".join(assertion_lines) + "\n"

    def _rewrite_pageimpl_test_usage(self, content: str) -> str:
        if "PageImpl<" not in content:
            return content

        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.data\.domain\.PageImpl;\n?",
            "",
            content,
        )
        search_from = 0
        needle = "new PageImpl<"
        while True:
            start = content.find(needle, search_from)
            if start == -1:
                return content
            generic_close = content.find(">", start + len(needle))
            if generic_close == -1:
                return content
            args_open = content.find("(", generic_close)
            if args_open == -1:
                return content
            args_close = self._find_matching_parenthesis(content, args_open)
            if args_close == -1:
                return content

            body = content[args_open + 1 : args_close].strip()
            replacement = f"Page.of({body}, Pageable.from(0, 10), (long) {body}.size())"
            content = content[:start] + replacement + content[args_close + 1 :]
            search_from = start + len(replacement)

    def _transform_integration_http_test_patterns(self, content: str) -> str:
        if "@MicronautTest" not in content and "@SpringBootTest" not in content:
            return content

        content = re.sub(r"@MicronautTest\s*\(\s*webEnvironment\s*=\s*WebEnvironment\.[A-Z_]+\s*\)", "@MicronautTest", content)
        content = re.sub(r"@SpringBootTest\s*\(\s*webEnvironment\s*=\s*WebEnvironment\.[A-Z_]+\s*\)", "@MicronautTest", content)
        content = re.sub(r"(?m)^[ \t]*import org\.springframework\.boot\.test\.context\.SpringBootTest\.WebEnvironment;\n?", "", content)
        content = re.sub(r"(?m)^[ \t]*@LocalServerPort\s*\n?", "", content)
        content = re.sub(r"(?m)^[ \t]*int port;\n?", "", content)
        content = re.sub(
            r'(?m)^(?:[ \t]*@(?:Inject|Autowired)\s*\n)?[ \t]*private(?:\s+final)?\s+RestTemplateBuilder\s+\w+\s*;\n?',
            '\t@Inject\n\t@Client("/")\n\tprivate HttpClient client;\n',
            content,
        )
        content = re.sub(r'HttpClient\s+(\w+)\s*=\s*\w+\.rootUri\("http://localhost:"\s*\+\s*port\)\.build\(\);', r"HttpClient \1 = client;", content)
        content = re.sub(r'\b\w+\.exchange\(HttpRequest\.get\(', "client.toBlocking().exchange(HttpRequest.GET(", content)
        content = re.sub(r'\b\w+\.exchange\(RequestEntity\.get\(', "client.toBlocking().exchange(HttpRequest.GET(", content)
        content = re.sub(r"HttpRequest\.get\(", "HttpRequest.GET(", content)
        content = re.sub(r"HttpRequest\.GET\(([^)]+)\)\.build\(\)", r"HttpRequest.GET(\1)", content)
        content = re.sub(r"\.getStatusCode\(\)", ".getStatus()", content)
        content = re.sub(
            r"assertThat\(\s*([A-Za-z_][A-Za-z0-9_\.]*)\.getStatus\(\)\s*\)\.isEqualTo\(\s*([^)]+?)\s*\);",
            r"org.junit.jupiter.api.Assertions.assertEquals(\2, \1.getStatus());",
            content,
        )
        return content

    def _transform_field_level_mockbean_patterns(self, content: str) -> str:
        if "@MockBean" not in content:
            return content

        pattern = re.compile(
            r"(?ms)^(?P<indent>[ \t]*)@MockBean(?:\([^)]*\))?\s*\n"
            r"(?:(?P=indent)@(?P<extra>[A-Za-z_][A-Za-z0-9_]*)(?:\([^)]*\))?\s*\n)*"
            r"(?P=indent)(?P<visibility>private|protected|public)\s+"
            r"(?P<modifiers>(?:(?:static|final|volatile|transient)\s+)*)"
            r"(?P<field_type>[A-Z][A-Za-z0-9_$.<>, ?\[\]]+)\s+"
            r"(?P<field_name>[a-zA-Z_][A-Za-z0-9_]*)\s*;",
        )

        mock_fields = []

        def replace(match: re.Match) -> str:
            field_type = " ".join(match.group("field_type").split())
            field_name = match.group("field_name")
            indent = match.group("indent")
            existing_annotations = re.findall(
                rf"(?m)^{re.escape(indent)}@([A-Za-z_][A-Za-z0-9_]*)(?:\([^)]*\))?\s*$",
                match.group(0),
            )
            kept_annotations = [
                annotation
                for annotation in existing_annotations
                if annotation not in {"MockBean", "Autowired", "Inject"}
            ]
            mock_fields.append((field_type, field_name, indent))
            lines = [f"{indent}@{annotation}" for annotation in kept_annotations]
            lines.append(f"{indent}@Inject")
            lines.append(f"{indent}{match.group('visibility')} {field_type} {field_name};")
            return "\n".join(lines)

        updated = pattern.sub(replace, content)
        if not mock_fields:
            return content

        updated = self._ensure_import(updated, "jakarta.inject.Inject")
        updated = self._ensure_import(updated, "org.mockito.Mockito")
        updated = self._append_mockbean_factory_methods(updated, mock_fields)
        return updated

    def _append_mockbean_factory_methods(
        self,
        content: str,
        mock_fields: Sequence[Tuple[str, str, str]],
    ) -> str:
        last_brace = content.rfind("}")
        if last_brace == -1:
            return content

        existing_factory_signatures = {
            (
                match.group("field_type").strip(),
                match.group("field_name").strip(),
            )
            for match in re.finditer(
                r'@MockBean\(\s*(?P<field_type>[A-Za-z_][A-Za-z0-9_$.<>, ?\[\]]+)\.class\s*\)\s*'
                r'(?:public|protected|private)?\s*(?P<field_type_repeat>[A-Za-z_][A-Za-z0-9_$.<>, ?\[\]]+)\s+'
                r'(?P<field_name>[a-zA-Z_][A-Za-z0-9_]*)\s*\(',
                content,
            )
            if match.group("field_type").strip() == match.group("field_type_repeat").strip()
        }

        blocks = []
        for field_type, field_name, indent in mock_fields:
            signature = (field_type.strip(), field_name.strip())
            if signature in existing_factory_signatures:
                continue
            child_indent = indent + ("\t" if "\t" in content else "    ")
            blocks.append(
                "\n"
                f"{indent}@MockBean({field_type}.class)\n"
                f"{indent}{field_type} {field_name}() {{\n"
                f"{child_indent}return Mockito.mock({field_type}.class);\n"
                f"{indent}}}\n"
            )

        if not blocks:
            return content
        return content[:last_brace] + "".join(blocks) + content[last_brace:]

    def _ensure_helper_method(
        self,
        content: str,
        *,
        method_name: str,
        parameter_signature: str,
        method_body: str,
        return_type: str = "void",
    ) -> str:
        if f"{method_name}(" in content and f"{return_type} {method_name}(" in content:
            return content

        method_block = (
            f"\n\tprivate {return_type} {method_name}({parameter_signature}) {{\n"
            f"\t\t{method_body}\n"
            f"\t}}\n"
        )

        last_brace = content.rfind("}")
        if last_brace == -1:
            return content + method_block
        return content[:last_brace] + method_block + content[last_brace:]

    def _transform_conditional_on_property(self, content: str) -> str:
        pattern = re.compile(r"@ConditionalOnProperty\s*\((?P<args>[\s\S]*?)\)")
        transformed = False

        def replace(match):
            nonlocal transformed
            replacement = self._rewrite_conditional_on_property_args(match.group("args"))
            if replacement is None:
                return match.group(0)
            transformed = True
            return replacement

        content = pattern.sub(replace, content)
        if transformed:
            content = content.replace(
                "import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;\n",
                "",
            )
            content = self._ensure_import_after_package(content, "io.micronaut.context.annotation.Requires")
            content = self._merge_adjacent_requires_annotations(content)
        return content

    def _transform_conditional_on_expression(self, content: str) -> str:
        pattern = re.compile(r"@ConditionalOnExpression\s*\((?P<args>[\s\S]*?)\)")
        transformed = False

        def replace(match):
            nonlocal transformed
            replacement = self._rewrite_conditional_on_expression_args(match.group("args"))
            transformed = True
            if replacement is None:
                expression = self._extract_annotation_string_argument(match.group("args"), "value")
                if expression is None:
                    expression = self._extract_shorthand_string_argument(match.group("args"))
                expression_text = expression.strip() if expression else "<unparsed expression>"
                return self._render_manual_review_comment(
                    "Spring @ConditionalOnExpression could not be mapped safely; review expression manually: "
                    f"{expression_text}"
                )
            return replacement

        content = pattern.sub(replace, content)
        if transformed:
            content = content.replace(
                "import org.springframework.boot.autoconfigure.condition.ConditionalOnExpression;\n",
                "",
            )
            content = self._ensure_import_after_package(content, "io.micronaut.context.annotation.Requires")
            content = self._merge_adjacent_requires_annotations(content)
        return content

    def _transform_requires_condition_patterns(self, content: str) -> str:
        transformed = False

        conditional_patterns = (
            (
                re.compile(r"@ConditionalOnBean\s*\((?P<args>[\s\S]*?)\)"),
                self._rewrite_conditional_on_bean_args,
                "import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;\n",
            ),
            (
                re.compile(r"@ConditionalOnMissingBean\s*\((?P<args>[\s\S]*?)\)"),
                self._rewrite_conditional_on_missing_bean_args,
                "import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;\n",
            ),
            (
                re.compile(r"@ConditionalOnClass\s*\((?P<args>[\s\S]*?)\)"),
                self._rewrite_conditional_on_class_args,
                "import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;\n",
            ),
            (
                re.compile(r"@ConditionalOnMissingClass\s*\((?P<args>[\s\S]*?)\)"),
                self._rewrite_conditional_on_missing_class_args,
                "import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingClass;\n",
            ),
            (
                re.compile(r"@Profile\s*\((?P<args>[\s\S]*?)\)"),
                self._rewrite_profile_args,
                "import org.springframework.context.annotation.Profile;\n",
            ),
        )

        for pattern, rewriter, removable_import in conditional_patterns:
            local_transformed = False

            def replace(match):
                nonlocal local_transformed
                replacement = rewriter(match.group("args"))
                if replacement is None:
                    return match.group(0)
                local_transformed = True
                return replacement

            content = pattern.sub(replace, content)
            if local_transformed:
                transformed = True
                content = content.replace(removable_import, "")

        if transformed:
            content = self._ensure_import_after_package(content, "io.micronaut.context.annotation.Requires")
            content = self._merge_adjacent_requires_annotations(content)
        return content

    def _rewrite_conditional_on_property_args(self, args: str) -> Optional[str]:
        prefix = self._extract_annotation_string_argument(args, "prefix")
        names = self._extract_annotation_string_arguments(args, "name")
        if not names:
            names = self._extract_annotation_string_arguments(args, "value")
        having_value = self._extract_annotation_string_argument(args, "havingValue")
        match_if_missing = self._extract_annotation_boolean_argument(args, "matchIfMissing")

        if not names or not having_value:
            return None

        requires_lines = []
        for name in names:
            property_name = name
            if prefix:
                prefix = prefix.strip(".")
                property_name = name if name.startswith(f"{prefix}.") else f"{prefix}.{name}"
            if match_if_missing is True:
                requires_lines.append(
                    f'@Requires(property = "{property_name}", value = "{having_value}", defaultValue = "{having_value}")'
                )
            else:
                requires_lines.append(
                    f'@Requires(property = "{property_name}", value = "{having_value}")'
                )

        return "\n".join(requires_lines)

    def _rewrite_conditional_on_expression_args(self, args: str) -> Optional[str]:
        expression = self._extract_annotation_string_argument(args, "value")
        if expression is None:
            expression = self._extract_shorthand_string_argument(args)
        if expression is None:
            return None
        expression = expression.strip().replace("\\'", "'").replace('\\"', '"')

        equals_match = re.fullmatch(
            r"'\$\{(?P<property>[\w.\-]+)(?::(?P<default>[^}]+))?\}'\s*==\s*'(?P<value>[^']+)'",
            expression,
        )
        if equals_match:
            return self._render_requires_property(
                property_name=equals_match.group("property"),
                value=equals_match.group("value").strip(),
                default_value=(equals_match.group("default") or "").strip() or None,
            )

        negated_placeholder_match = re.fullmatch(
            r"!\s*\$\{(?P<property>[\w.\-]+)(?::(?P<default>[^}]+))?\}",
            expression,
        )
        if negated_placeholder_match:
            default_value = (negated_placeholder_match.group("default") or "").strip() or None
            return self._render_requires_property(
                property_name=negated_placeholder_match.group("property"),
                value="false",
                default_value=default_value,
            )

        placeholder_match = re.fullmatch(
            r"\$\{(?P<property>[\w.\-]+)(?::(?P<default>[^}]+))?\}",
            expression,
        )
        if placeholder_match:
            default_value = (placeholder_match.group("default") or "").strip() or None
            return self._render_requires_property(
                property_name=placeholder_match.group("property"),
                value="true",
                default_value=default_value,
            )

        return None

    def _rewrite_conditional_on_bean_args(self, args: str) -> Optional[str]:
        bean_types = self._extract_annotation_class_arguments(args, "value")
        if not bean_types:
            bean_types = self._extract_annotation_class_arguments(args, "type")
        if not bean_types:
            bean_types = self._extract_annotation_class_name_strings(args, "type")
        bean_names = self._extract_annotation_string_arguments(args, "name")
        if not bean_types:
            if bean_names:
                return self._render_manual_review_comment(
                    "Spring @ConditionalOnBean(name = ...) has no direct Micronaut @Requires equivalent; review named-qualifier conditions manually."
                )
            bean_types = self._extract_shorthand_class_arguments(args)
        if not bean_types:
            return None
        rendered = self._render_requires_class_array("beans", bean_types)
        if bean_names:
            rendered += (
                "\n"
                + self._render_manual_review_comment(
                    "Spring @ConditionalOnBean(name = ...) was only partially migrated; verify named-qualifier conditions manually."
                )
            )
        return rendered

    def _rewrite_conditional_on_missing_bean_args(self, args: str) -> Optional[str]:
        bean_types = self._extract_annotation_class_arguments(args, "value")
        if not bean_types:
            bean_types = self._extract_annotation_class_arguments(args, "type")
        if not bean_types:
            bean_types = self._extract_annotation_class_name_strings(args, "type")
        bean_names = self._extract_annotation_string_arguments(args, "name")
        if not bean_types:
            if bean_names:
                return self._render_manual_review_comment(
                    "Spring @ConditionalOnMissingBean(name = ...) has no direct Micronaut @Requires equivalent; review named-qualifier conditions manually."
                )
            bean_types = self._extract_shorthand_class_arguments(args)
        if not bean_types:
            return None
        rendered = self._render_requires_class_array("missingBeans", bean_types)
        if bean_names:
            rendered += (
                "\n"
                + self._render_manual_review_comment(
                    "Spring @ConditionalOnMissingBean(name = ...) was only partially migrated; verify named-qualifier conditions manually."
                )
            )
        return rendered

    def _rewrite_conditional_on_class_args(self, args: str) -> Optional[str]:
        class_types = self._extract_annotation_class_arguments(args, "value")
        if not class_types:
            class_types = self._extract_annotation_class_arguments(args, "name")
        if not class_types:
            class_types = self._extract_annotation_class_name_strings(args, "name")
        if not class_types:
            class_types = self._extract_shorthand_class_arguments(args)
        if not class_types:
            return None
        return self._render_requires_class_array("classes", class_types)

    def _rewrite_conditional_on_missing_class_args(self, args: str) -> Optional[str]:
        class_types = self._extract_annotation_class_arguments(args, "value")
        if not class_types:
            class_types = self._extract_annotation_class_arguments(args, "name")
        if not class_types:
            class_types = self._extract_annotation_class_name_strings(args, "name")
        if not class_types:
            class_types = self._extract_shorthand_class_arguments(args)
        if not class_types:
            return None
        return self._render_requires_class_array("missing", class_types)

    def _rewrite_profile_args(self, args: str) -> Optional[str]:
        profile_values = self._extract_annotation_string_list(args)
        if not profile_values:
            return None

        negative_profiles = [value[1:] for value in profile_values if value.startswith("!")]
        positive_profiles = [value for value in profile_values if not value.startswith("!")]
        requires_lines = []
        if positive_profiles:
            requires_lines.append(self._render_requires_string_array("env", positive_profiles))
        if negative_profiles:
            requires_lines.append(self._render_requires_string_array("notEnv", negative_profiles))
        if not requires_lines:
            return None
        return "\n".join(requires_lines)

    def _extract_annotation_string_argument(self, args: str, name: str) -> Optional[str]:
        values = self._extract_annotation_string_arguments(args, name)
        return values[0] if values else None

    def _extract_shorthand_string_argument(self, args: str) -> Optional[str]:
        stripped = str(args or "").strip()
        if not stripped:
            return None
        if stripped.startswith('"') and stripped.endswith('"'):
            return stripped[1:-1]
        if "=" in stripped:
            return None
        return None

    def _extract_annotation_string_arguments(self, args: str, name: str) -> List[str]:
        string_match = re.search(rf"\b{name}\s*=\s*\"(?P<value>[^\"]+)\"", args)
        if string_match:
            return [string_match.group("value").strip()]

        array_match = re.search(
            rf"\b{name}\s*=\s*\{{(?P<value>[^}}]+)\}}",
            args,
            flags=re.DOTALL,
        )
        if array_match:
            return [value.strip() for value in re.findall(r'"([^"]+)"', array_match.group("value")) if value.strip()]

        return []

    def _extract_annotation_boolean_argument(self, args: str, name: str) -> Optional[bool]:
        match = re.search(rf"\b{name}\s*=\s*(true|false)", args, flags=re.IGNORECASE)
        if not match:
            return None
        return match.group(1).lower() == "true"

    def _extract_annotation_class_arguments(self, args: str, name: str) -> List[str]:
        array_match = re.search(
            rf"\b{name}\s*=\s*\{{(?P<value>[^}}]+)\}}",
            args,
            flags=re.DOTALL,
        )
        if array_match:
            return self._parse_class_literals(array_match.group("value"))

        single_match = re.search(
            rf"\b{name}\s*=\s*(?P<value>[A-Za-z_][A-Za-z0-9_$.]*)\.class",
            args,
        )
        if single_match:
            return [single_match.group("value").strip()]

        return []

    def _extract_annotation_class_name_strings(self, args: str, name: str) -> List[str]:
        values = self._extract_annotation_string_arguments(args, name)
        return [value.strip() for value in values if value.strip()]

    def _extract_shorthand_class_arguments(self, args: str) -> List[str]:
        stripped = str(args or "").strip()
        if not stripped or "=" in stripped:
            return []
        return self._parse_class_literals(stripped)

    def _parse_class_literals(self, text: str) -> List[str]:
        matches = re.findall(r"([A-Za-z_][A-Za-z0-9_$.]*)\.class", text or "")
        return [match.strip() for match in matches if match.strip()]

    def _extract_annotation_string_list(self, args: str) -> List[str]:
        values = []
        named_array_match = re.search(r"\b(?:value|profiles?)\s*=\s*\{(?P<value>[^}]+)\}", args, flags=re.DOTALL)
        named_single_match = re.search(r'\b(?:value|profiles?)\s*=\s*"(?P<value>[^"]+)"', args)

        if named_array_match:
            values.extend(re.findall(r'"([^"]+)"', named_array_match.group("value")))
        elif named_single_match:
            values.append(named_single_match.group("value"))
        else:
            stripped = str(args or "").strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                values.extend(re.findall(r'"([^"]+)"', stripped))
            elif stripped.startswith('"') and stripped.endswith('"'):
                values.append(stripped.strip('"'))

        return [value.strip() for value in values if value.strip()]

    def _render_requires_class_array(self, attribute: str, values: Sequence[str]) -> str:
        if len(values) == 1:
            return f"@Requires({attribute} = {values[0]}.class)"
        joined = ", ".join(f"{value}.class" for value in values)
        return f"@Requires({attribute} = {{{joined}}})"

    def _render_requires_string_array(self, attribute: str, values: Sequence[str]) -> str:
        if len(values) == 1:
            return f'@Requires({attribute} = "{values[0]}")'
        joined = ", ".join(f'"{value}"' for value in values)
        return f"@Requires({attribute} = {{{joined}}})"

    def _render_requires_property(
        self,
        *,
        property_name: str,
        value: str,
        default_value: Optional[str] = None,
    ) -> str:
        rendered = f'@Requires(property = "{property_name}", value = "{value}"'
        if default_value is not None and default_value != "":
            rendered += f', defaultValue = "{default_value}"'
        rendered += ")"
        return rendered

    def _render_manual_review_comment(self, message: str) -> str:
        return f"// TODO: manual review: {message}"

    def _merge_adjacent_requires_annotations(self, content: str) -> str:
        pattern = re.compile(r"(?P<block>(?:^[ \t]*@Requires\([^)]*\)\s*\n){2,})", re.MULTILINE)

        def replace(match: re.Match) -> str:
            block = match.group("block")
            requires_args = re.findall(r"@Requires\(([^)]*)\)", block)
            if len(requires_args) < 2:
                return block
            seen_attributes = set()
            for arg in requires_args:
                attribute_names = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=", arg)
                duplicates = seen_attributes.intersection(attribute_names)
                if duplicates:
                    return block
                seen_attributes.update(attribute_names)
            merged_args = ", ".join(arg.strip() for arg in requires_args if arg.strip())
            indent_match = re.match(r"^([ \t]*)@Requires", block)
            indent = indent_match.group(1) if indent_match else ""
            return f"{indent}@Requires({merged_args})\n"

        return pattern.sub(replace, content)

    def _transform_field_to_constructor_injection(
        self,
        content: str,
        ast_context: Optional[JavaAstContext] = None,
        *,
        allow_test_classes: bool = False,
    ) -> str:
        """
        Converts Spring field injection to Micronaut-preferred constructor injection.
        """
        if not allow_test_classes and any(marker in content for marker in ("@MicronautTest", "@SpringBootTest", "@WebMvcTest", "@DataJpaTest")):
            return content
        primary_class = ast_context.primary_class if ast_context else None
        if not allow_test_classes and primary_class and any(
            annotation in primary_class.annotations
            for annotation in ("SpringBootTest", "WebMvcTest", "DataJpaTest", "MicronautTest")
        ):
            return content
        if primary_class and primary_class.constructor_count > 0:
            return content
        ast_injectable_fields = set(ast_context.injectable_field_names()) if ast_context else set()

        class_pattern = re.compile(
            r"(?P<header>(?:public|protected|private)?\s*(?:abstract\s+|final\s+)?class\s+\w+[^{]*\{)(?P<body>.*?)(?P<footer>\n\})",
            re.DOTALL,
        )
        match = class_pattern.search(content)
        if not match:
            return content

        body = match.group("body")
        field_pattern = re.compile(
            r"(?P<indent>^[ \t]*)@(?:Autowired|Inject)\s*\n"
            r"(?:(?P=indent)@(?:Qualifier|Named)\((?P<qualifier>[^)]+)\)\s*\n)?"
            r"(?P=indent)(?P<mods>(?:private|protected|public)\s+)?(?P<type>[\w<>?, ]+)\s+(?P<name>\w+)\s*;",
            re.MULTILINE,
        )
        fields = list(field_pattern.finditer(body))
        if not fields:
            return content

        constructor_fields = []
        updated_body = body
        for field in reversed(fields):
            indent = field.group("indent")
            field_type = " ".join(field.group("type").split())
            field_name = field.group("name")
            if ast_injectable_fields and field_name not in ast_injectable_fields:
                continue
            qualifier = (field.group("qualifier") or "").strip()
            replacement = f"{indent}private final {field_type} {field_name};"
            updated_body = updated_body[: field.start()] + replacement + updated_body[field.end() :]
            constructor_fields.append((indent, field_type, field_name, qualifier))

        constructor_fields.reverse()
        constructor_pattern = re.compile(r"^[ \t]*(public|protected|private)?\s*\w+\s*\(", re.MULTILINE)
        if constructor_fields and not constructor_pattern.search(updated_body):
            indent = constructor_fields[0][0]
            class_name_match = re.search(r"class\s+(?P<name>\w+)", match.group("header"))
            if not class_name_match:
                return content
            class_name = class_name_match.group("name")
            params = ", ".join(
                f"{self._constructor_parameter_prefix(qualifier)}{field_type} {field_name}"
                for _, field_type, field_name, qualifier in constructor_fields
            )
            assignments = "\n".join(
                f"{indent}    this.{field_name} = {field_name};"
                for _, _, field_name, _ in constructor_fields
            )
            constructor_block = (
                f"\n\n{indent}public {class_name}({params}) {{\n"
                f"{assignments}\n"
                f"{indent}}}"
            )
            updated_body = constructor_block + updated_body

        return content[: match.start("body")] + updated_body + content[match.end("body") :]

    def _ensure_constructor_for_required_final_fields(self, content: str) -> str:
        class_pattern = re.compile(
            r"(?P<header>(?:public|protected|private)?\s*(?:abstract\s+|final\s+)?class\s+\w+[^{]*\{)(?P<body>.*?)(?P<footer>\n\})",
            re.DOTALL,
        )
        match = class_pattern.search(content)
        if not match:
            return content

        body = match.group("body")
        class_name_match = re.search(r"class\s+(?P<name>\w+)", match.group("header"))
        if not class_name_match:
            return content
        class_name = class_name_match.group("name")
        constructor_pattern = re.compile(
            rf"^[ \t]*(?:public|protected|private)?\s*{re.escape(class_name)}\s*\(",
            re.MULTILINE,
        )
        if constructor_pattern.search(body):
            return content

        field_pattern = re.compile(
            r"(?P<indent>^[ \t]*)"
            r"(?:@Named\((?P<qualifier>[^)]+)\)\s*\n(?P=indent))?"
            r"(?:private|protected|public)\s+final\s+"
            r"(?P<type>[\w<>?, .]+)\s+(?P<name>\w+)\s*;",
            re.MULTILINE,
        )
        fields = list(field_pattern.finditer(body))
        if not fields:
            return content

        indent = fields[0].group("indent")
        constructor_fields = [
            (
                " ".join(field.group("type").split()),
                field.group("name"),
                (field.group("qualifier") or "").strip(),
            )
            for field in fields
        ]
        params = ", ".join(
            f"{self._constructor_parameter_prefix(qualifier)}{field_type} {field_name}"
            for field_type, field_name, qualifier in constructor_fields
        )
        assignments = "\n".join(
            f"{indent}    this.{field_name} = {field_name};"
            for _, field_name, _ in constructor_fields
        )
        constructor_block = (
            f"\n\n{indent}public {class_name}({params}) {{\n"
            f"{assignments}\n"
            f"{indent}}}"
        )
        updated_body = constructor_block + body
        return content[: match.start("body")] + updated_body + content[match.end("body") :]

    def _constructor_parameter_prefix(self, qualifier: str) -> str:
        cleaned = qualifier.strip()
        if not cleaned:
            return ""
        return f"@Named({cleaned}) "

    def _transform_request_mapping_annotations(
        self,
        content: str,
        ast_context: Optional[JavaAstContext] = None,
    ) -> str:
        if ast_context and ast_context.parse_ok and not (
            ast_context.has_annotation("RequestMapping") or ast_context.has_method_annotation("RequestMapping")
        ):
            return content

        request_method_map = {
            "GET": "Get",
            "POST": "Post",
            "PUT": "Put",
            "DELETE": "Delete",
            "PATCH": "Patch",
        }

        pattern = re.compile(r"(?m)^(?P<indent>[ \t]*)@RequestMapping\s*\((?P<args>[\s\S]*?)\)")

        def replace(match):
            indent = match.group("indent")
            args = match.group("args")
            method_match = re.search(r"RequestMethod\.(GET|POST|PUT|DELETE|PATCH)", args)
            if not method_match:
                return match.group(0)
            annotation = request_method_map[method_match.group(1)]
            path_match = re.search(r'(?:value|path)\s*=\s*"(?P<path>[^"]+)"', args)
            if not path_match:
                path_match = re.search(r'"(?P<path>[^"]+)"', args)
            path = path_match.group("path") if path_match else ""
            if path:
                return f'{indent}@{annotation}("{path}")'
            return f"{indent}@{annotation}"

        return pattern.sub(replace, content)

    def _transform_exception_handler_patterns(
        self,
        content: str,
        ast_context: Optional[JavaAstContext] = None,
    ) -> str:
        if ast_context and not (
            ast_context.has_annotation("ControllerAdvice")
            or ast_context.has_method_annotation("ExceptionHandler")
            or ast_context.has_method_annotation("ResponseStatus")
        ):
            return content

        content = re.sub(r"(?m)^[ \t]*@ResponseBody\s*\n", "", content)
        content = re.sub(
            r"@(?:ExceptionHandler|Error)\(\s*(?:global\s*=\s*true\s*,\s*)?(?:exception\s*=\s*)?(?P<exception>[\w.$]+)\.class\s*\)",
            r"@Error(global = true, exception = \g<exception>.class)",
            content,
        )
        content = re.sub(
            r"@(?:ResponseStatus|Status)\(\s*(?:value\s*=\s*)?HttpStatus\.(?P<status>[A-Z_]+)\s*\)",
            r"@Status(HttpStatus.\g<status>)",
            content,
        )
        return content

    def _transform_value_annotations(self, content: str) -> str:
        pattern = re.compile(r'@(?:Value|Property)\(\s*"\$\{(?P<name>[^}:]+)(?::(?P<default>[^}]+))?\}"\s*\)')

        def replace(match):
            name = match.group("name").strip()
            default_value = (match.group("default") or "").strip()
            if default_value:
                return f'@Property(name = "{name}", defaultValue = "{default_value}")'
            return f'@Property(name = "{name}")'

        return pattern.sub(replace, content)

    def _transform_response_entity_patterns(self, content: str) -> str:
        replacements = (
            (r"\bResponseEntity\.ok\(", "HttpResponse.ok("),
            (r"\bResponseEntity\.status\(", "HttpResponse.status("),
            (r"\bResponseEntity\.badRequest\(\)", "HttpResponse.badRequest()"),
            (r"\bResponseEntity\.notFound\(\)", "HttpResponse.notFound()"),
            (r"\bResponseEntity\.noContent\(\)", "HttpResponse.noContent()"),
            (r"\bResponseEntity\.accepted\(\)", "HttpResponse.accepted()"),
            (r"\bResponseEntity\.created\(", "HttpResponse.created("),
            (r"\bResponseEntity::ok\b", "HttpResponse::ok"),
            (r"\bResponseEntity::badRequest\b", "HttpResponse::badRequest"),
            (r"\bResponseEntity::notFound\b", "HttpResponse::notFound"),
        )
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)
        content = re.sub(
            r"\bHttpResponse\.status\(([^)]+)\)\.body\(",
            r"HttpResponse.status(\1).body(",
            content,
        )
        content = re.sub(r"\bHttpResponse\.(ok|badRequest|notFound|noContent|accepted)\(\)\.build\(\)", r"HttpResponse.\1()", content)
        return content

    def _transform_feign_client_patterns(self, content: str) -> str:
        if "FeignClient" not in content and "EnableFeignClients" not in content:
            return content

        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.cloud\.openfeign\.EnableFeignClients;\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*@EnableFeignClients(?:\([^)]*\))?\s*\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.cloud\.openfeign\.FeignClient;\n?",
            "import io.micronaut.http.client.annotation.Client;\n",
            content,
        )

        pattern = re.compile(r"@FeignClient\s*\((?P<args>[\s\S]*?)\)")

        def replace(match: re.Match) -> str:
            args = match.group("args")
            client_target = self._extract_annotation_string_argument(args, "url")
            if not client_target:
                client_target = self._extract_annotation_string_argument(args, "name")
            if not client_target:
                client_target = self._extract_annotation_string_argument(args, "value")
            if not client_target:
                client_target = "/"
            return f'@Client("{client_target}")'

        content = pattern.sub(replace, content)

        # Feign interfaces frequently keep Spring MVC method annotations.
        # Normalize them explicitly here so the deterministic Feign path does not
        # depend on later generic annotation passes to become compilable.
        for spring_name, micronaut_name in (
            ("GetMapping", "Get"),
            ("PostMapping", "Post"),
            ("PutMapping", "Put"),
            ("DeleteMapping", "Delete"),
            ("PatchMapping", "Patch"),
        ):
            content = re.sub(
                rf"(?m)^[ \t]*import org\.springframework\.web\.bind\.annotation\.{spring_name};\n?",
                f"import io.micronaut.http.annotation.{micronaut_name};\n",
                content,
            )
            content = re.sub(rf"@{spring_name}\b", f"@{micronaut_name}", content)

        return content

    def _transform_kafka_listener_patterns(self, content: str) -> str:
        if "KafkaListener" not in content:
            return content

        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.kafka\.annotation\.KafkaListener;\n?",
            "",
            content,
        )

        transformed = False
        pattern = re.compile(
            r"(?m)^(?P<indent>[ \t]*)@KafkaListener\s*\((?P<args>[^)]*)\)\s*\n"
            r"(?P<signature>[ \t]*(?:public|protected|private)?[ \t\w<>\[\],?]+\((?P<params>[^)]*)\))"
        )

        def replace(match: re.Match) -> str:
            nonlocal transformed
            topic = self._extract_annotation_string_argument(match.group("args"), "topics")
            if not topic:
                topic = self._extract_annotation_string_argument(match.group("args"), "value")
            signature = match.group("signature")
            params = match.group("params")
            transformed = True
            return signature.replace(params, self._annotate_first_parameter(params, "Topic", topic), 1)

        content = pattern.sub(replace, content)
        if transformed:
            content = self._ensure_class_annotation(content, "@KafkaListener")
        return content

    def _transform_rabbit_listener_patterns(self, content: str) -> str:
        if "RabbitListener" not in content:
            return content

        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.amqp\.rabbit\.annotation\.RabbitListener;\n?",
            "",
            content,
        )

        transformed = False
        pattern = re.compile(
            r"(?m)^(?P<indent>[ \t]*)@RabbitListener\s*\((?P<args>[^)]*)\)\s*\n"
            r"(?P<signature>[ \t]*(?:public|protected|private)?[ \t\w<>\[\],?]+\((?P<params>[^)]*)\))"
        )

        def replace(match: re.Match) -> str:
            nonlocal transformed
            queue = self._extract_annotation_string_argument(match.group("args"), "queues")
            if not queue:
                queue = self._extract_annotation_string_argument(match.group("args"), "value")
            signature = match.group("signature")
            params = match.group("params")
            transformed = True
            return signature.replace(params, self._annotate_first_parameter(params, "Queue", queue), 1)

        content = pattern.sub(replace, content)
        if transformed:
            content = self._ensure_class_annotation(content, "@RabbitListener")
        return content

    def _transform_kafka_template_patterns(self, content: str) -> str:
        if "KafkaTemplate" not in content:
            return content

        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.kafka\.core\.KafkaTemplate;\n?",
            "",
            content,
        )

        send_calls = []
        pattern_with_key = re.compile(
            r"(?m)^(?P<indent>[ \t]*)(?P<target>\w+)\.send\(\s*\"(?P<topic>[^\"]+)\"\s*,\s*(?P<key>[^,\n;][^,\n;]*?)\s*,\s*(?P<payload>[^;\n]+?)\s*\)\s*;"
        )
        pattern_without_key = re.compile(
            r"(?m)^(?P<indent>[ \t]*)(?P<target>\w+)\.send\(\s*\"(?P<topic>[^\"]+)\"\s*,\s*(?P<payload>[^;\n]+?)\s*\)\s*;"
        )

        def replace_with_key(match: re.Match) -> str:
            target = match.group("target")
            topic = match.group("topic")
            method_name = self._messaging_method_name("send", topic, with_key=True)
            send_calls.append((target, topic, method_name, True))
            return f"{match.group('indent')}{target}.{method_name}({match.group('key').strip()}, {match.group('payload').strip()});"

        def replace_without_key(match: re.Match) -> str:
            target = match.group("target")
            topic = match.group("topic")
            method_name = self._messaging_method_name("send", topic, with_key=False)
            send_calls.append((target, topic, method_name, False))
            return f"{match.group('indent')}{target}.{method_name}({match.group('payload').strip()});"

        content = pattern_with_key.sub(replace_with_key, content)
        content = pattern_without_key.sub(replace_without_key, content)
        if not send_calls:
            return content

        has_key = any(with_key for _, _, _, with_key in send_calls)
        content = self._ensure_import_after_package(content, "io.micronaut.configuration.kafka.annotation.KafkaClient")
        content = self._ensure_import_after_package(content, "io.micronaut.configuration.kafka.annotation.Topic")
        if has_key:
            content = self._ensure_import_after_package(content, "io.micronaut.configuration.kafka.annotation.KafkaKey")

        grouped_calls = {}
        for target, topic, method_name, with_key in send_calls:
            grouped_calls.setdefault(target, {})
            grouped_calls[target][method_name] = (topic, with_key)

        for target, method_map in grouped_calls.items():
            client_name = self._generated_client_name(target, next(iter(method_map.values()))[0], "KafkaClient")
            content = re.sub(
                rf"\bKafkaTemplate(?:\s*<[^>]+>)?\s+{re.escape(target)}\b",
                f"{client_name} {target}",
                content,
            )
            interface_block = self._render_kafka_client_interface(client_name, method_map)
            content = self._ensure_nested_type(content, client_name, interface_block)

        return content

    def _transform_rabbit_template_patterns(self, content: str) -> str:
        if "RabbitTemplate" not in content:
            return content

        content = re.sub(
            r"(?m)^[ \t]*import org\.springframework\.amqp\.rabbit\.core\.RabbitTemplate;\n?",
            "",
            content,
        )

        send_calls = []
        pattern = re.compile(
            r"(?m)^(?P<indent>[ \t]*)(?P<target>\w+)\.convertAndSend\(\s*\"(?P<exchange>[^\"]+)\"\s*,\s*\"(?P<binding>[^\"]+)\"\s*,\s*(?P<payload>[^;\n]+?)\s*\)\s*;"
        )

        def replace(match: re.Match) -> str:
            target = match.group("target")
            exchange = match.group("exchange")
            binding = match.group("binding")
            method_name = self._messaging_method_name("send", binding, with_key=False)
            send_calls.append((target, exchange, binding, method_name))
            return f"{match.group('indent')}{target}.{method_name}({match.group('payload').strip()});"

        content = pattern.sub(replace, content)
        if not send_calls:
            return content

        content = self._ensure_import_after_package(content, "io.micronaut.rabbitmq.annotation.Binding")
        content = self._ensure_import_after_package(content, "io.micronaut.rabbitmq.annotation.RabbitClient")

        grouped_calls = {}
        for target, exchange, binding, method_name in send_calls:
            grouped_calls.setdefault(target, {"exchange": exchange, "methods": {}})
            grouped_calls[target]["methods"][method_name] = binding

        for target, details in grouped_calls.items():
            client_name = self._generated_client_name(target, details["exchange"], "RabbitClient")
            content = re.sub(
                rf"\bRabbitTemplate\s+{re.escape(target)}\b",
                f"{client_name} {target}",
                content,
            )
            interface_block = self._render_rabbit_client_interface(client_name, details["exchange"], details["methods"])
            content = self._ensure_nested_type(content, client_name, interface_block)

        return content

    def _annotate_first_parameter(self, params: str, annotation: str, value: Optional[str]) -> str:
        parts = self._split_signature_arguments(params)
        if not parts or not value:
            return params
        first_param = parts[0].strip()
        if not first_param or f"@{annotation}" in first_param:
            return params
        parts[0] = f'@{annotation}("{value}") {first_param}'
        return ", ".join(parts)

    def _ensure_class_annotation(self, content: str, annotation: str) -> str:
        class_annotation_pattern = re.compile(
            rf"(?m)^[ \t]*{re.escape(annotation)}\s*$\n(?=[ \t]*(?:public|protected|private)?(?:\s+final|\s+abstract)?\s*class\b)"
        )
        if class_annotation_pattern.search(content):
            return content

        class_match = re.search(
            r"(?m)^(?P<indent>[ \t]*)(?P<annotations>(?:@[^\n]+\n)*)"
            r"(?P<class_decl>(?:public|protected|private)?(?:\s+final|\s+abstract)?\s*class\s+\w+[^{]*\{)",
            content,
        )
        if not class_match:
            return content

        insert_at = class_match.start("class_decl")
        indent = class_match.group("indent")
        return content[:insert_at] + f"{indent}{annotation}\n" + content[insert_at:]

    def _ensure_import_after_package(self, content: str, import_name: str) -> str:
        import_line = f"import {import_name};\n"
        if import_line in content:
            return content
        package_match = re.search(r"(?m)^package [^;]+;\n", content)
        if not package_match:
            return import_line + content
        insert_at = package_match.end()
        return content[:insert_at] + "\n" + import_line + content[insert_at:]

    def _messaging_method_name(self, prefix: str, destination: str, *, with_key: bool) -> str:
        suffix = self._to_pascal_case(destination)
        method_name = prefix + suffix if suffix else prefix
        if with_key:
            method_name += "WithKey"
        return method_name

    def _generated_client_name(self, target_name: str, destination: str, suffix: str) -> str:
        base = target_name
        for removable in ("KafkaTemplate", "RabbitTemplate", "Template", "Client"):
            if base.endswith(removable):
                base = base[: -len(removable)]
                break
        if not base or base.lower() in {"kafka", "rabbit"}:
            base = destination
        return f"{self._to_pascal_case(base or destination)}{suffix}"

    def _to_pascal_case(self, value: str) -> str:
        parts = re.split(r"[^A-Za-z0-9]+", str(value or ""))
        return "".join(part[:1].upper() + part[1:] for part in parts if part)

    def _render_kafka_client_interface(self, client_name: str, method_map: dict) -> str:
        lines = [f"\n    @KafkaClient\n    interface {client_name} {{"]
        for method_name, (topic, with_key) in method_map.items():
            lines.append(f'        @Topic("{topic}")')
            if with_key:
                lines.append(f"        void {method_name}(@KafkaKey Object key, Object payload);")
            else:
                lines.append(f"        void {method_name}(Object payload);")
            lines.append("")
        if lines[-1] == "":
            lines.pop()
        lines.append("    }\n")
        return "\n".join(lines)

    def _render_rabbit_client_interface(self, client_name: str, exchange: str, method_map: dict) -> str:
        lines = [f'\n    @RabbitClient("{exchange}")', f"    interface {client_name} {{"]
        for method_name, binding in method_map.items():
            lines.append(f'        @Binding("{binding}")')
            lines.append(f"        void {method_name}(Object payload);")
            lines.append("")
        if lines[-1] == "":
            lines.pop()
        lines.append("    }\n")
        return "\n".join(lines)

    def _ensure_nested_type(self, content: str, type_name: str, type_block: str) -> str:
        if re.search(rf"(?m)\b(?:class|interface|enum|record)\s+{re.escape(type_name)}\b", content):
            return content
        last_brace = content.rfind("}")
        if last_brace == -1:
            return content + type_block
        return content[:last_brace] + type_block + content[last_brace:]

    def _apply_deterministic_import_mappings(self, content: str) -> str:
        for spring_import, micronaut_import in self.IMPORT_MAPPINGS.items():
            if micronaut_import:
                content = content.replace(
                    f"import {spring_import};",
                    f"import {micronaut_import};",
                )
            else:
                content = content.replace(f"import {spring_import};", "")
        return content

    def _apply_deterministic_type_mappings(self, content: str) -> str:
        for spring_type, micronaut_type in self.TYPE_MAPPINGS.items():
            content = re.sub(rf"\b{re.escape(spring_type)}\b", micronaut_type, content)
        return content

    def _replace_spring_symbol(self, content: str, spring_symbol: str, replacement: str) -> str:
        if spring_symbol.startswith("@") or replacement.startswith("@"):
            normalized_symbol = spring_symbol if spring_symbol.startswith("@") else f"@{spring_symbol}"
            return re.sub(
                rf"{re.escape(normalized_symbol)}(?=\s|\(|$)",
                replacement,
                content,
            )
        if "." in spring_symbol:
            return content.replace(spring_symbol, replacement)
        return re.sub(rf"\b{re.escape(spring_symbol)}\b", replacement, content)
        return content

    def _is_manual_review_rule(self, rule: MigrationRule) -> bool:
        metadata = rule.metadata if isinstance(rule.metadata, dict) else {}
        if metadata.get("target_status") == "manual_redesign":
            return True
        if metadata.get("automated_migration_supported") is False:
            return True
        return self._is_manual_review_text(rule.micronaut_pattern) or self._is_manual_review_text(rule.description)

    def _is_manual_review_text(self, value: Optional[str]) -> bool:
        cleaned = str(value or "").strip().lower()
        if not cleaned:
            return False
        markers = (
            "manual ",
            "manual-",
            "review ",
            "reviewed ",
            "placeholder",
            "rewrite",
            "redesign",
        )
        return any(marker in cleaned for marker in markers)

    def _normalize_retrieved_replacement(self, source_item: str, replacement: str) -> str:
        cleaned_source = str(source_item or "").strip()
        cleaned_replacement = str(replacement or "").strip()
        if not cleaned_replacement:
            return cleaned_replacement

        source_looks_like_annotation = cleaned_source.startswith("@") or (
            cleaned_source[:1].isupper() and "." not in cleaned_source
        )
        if not source_looks_like_annotation:
            return cleaned_replacement

        if cleaned_replacement.startswith("@"):
            return cleaned_replacement

        if "." in cleaned_replacement:
            simple_name = cleaned_replacement.split(".")[-1].strip()
            if simple_name and simple_name[:1].isupper():
                return f"@{simple_name}"

        return cleaned_replacement

    def _llm_refinement_reason(
        self,
        original: str,
        current: str,
        ast_context: Optional[JavaAstContext] = None,
    ) -> str:
        """
        Expert heuristic to determine if the local RAG transformation was insufficient.
        Automatically triggers the LLM if rare or custom Spring patterns remain.
        """
        current_annotation_lines = {
            match.group(1)
            for match in re.finditer(r"(?m)^[ \t]*@([A-Z][A-Za-z0-9_]*)", current)
        }
        remaining_spring_annotations = {
            "Autowired",
            "Value",
            "Qualifier",
            "ControllerAdvice",
            "ExceptionHandler",
            "ConditionalOnProperty",
            "Configuration",
            "Component",
            "Service",
            "Repository",
            "RestController",
            "RequestMapping",
            "ResponseStatus",
            "Conditional",
        }

        if ast_context and ast_context.parse_ok:
            known_annotations = set(ast_context.annotation_names())
            spring_annotations_present = any(
                annotation in current_annotation_lines
                for annotation in known_annotations
                if annotation in remaining_spring_annotations
                and not (
                    annotation == "Repository"
                    and "import io.micronaut.data.annotation.Repository;" in current
                )
            )
            if (
                not spring_annotations_present
                and not self._contains_framework_spring_imports(current)
                and "ResponseEntity" not in current
            ):
                if "ProxyExchange" in original and self._requires_complex_http_client_rewrite(current):
                    return "ProxyExchange requires complex HTTP-client rewrite"
                if "RestTemplate" in original and self._requires_complex_http_client_rewrite(current):
                    return "RestTemplate-based flow requires complex HTTP-client rewrite"
                return ""

        if any(
            annotation in current_annotation_lines
            for annotation in remaining_spring_annotations
            if not (
                annotation == "Repository"
                and "import io.micronaut.data.annotation.Repository;" in current
            )
        ):
            return "Spring annotations still remain after deterministic transforms"

        if "unsupportedSpringMvcTest(" in current:
            return ""

        # Indicators of remaining Spring infrastructure
        if self._contains_framework_spring_imports(current):
            return "Spring imports still remain after deterministic transforms"
        if any(signature in current for signature in ("ResponseEntity", "ProxyExchange")):
            return "Spring HTTP response/client patterns still remain after deterministic transforms"

        # Specific complex patterns that RAG cannot handle structurally
        if ("ProxyExchange" in original or "RestTemplate" in original) and self._requires_complex_http_client_rewrite(current):
            if "ProxyExchange" in original:
                return "ProxyExchange requires complex HTTP-client rewrite"
            return "RestTemplate-based flow requires complex HTTP-client rewrite"

        return ""

    def _requires_complex_http_client_rewrite(self, current: str) -> bool:
        unresolved_markers = (
            "RestTemplate",
            "RestTemplateBuilder",
            "RequestEntity",
            "ResponseEntity",
            "LocalServerPort",
            "WebEnvironment",
            "port).build()",
            "rootUri(",
            "getStatusCode()",
            ".exchange(HttpRequest.get(",
            ".exchange(RequestEntity.get(",
        )
        return any(marker in current for marker in unresolved_markers)

    def _needs_llm_refinement(
        self,
        original: str,
        current: str,
        ast_context: Optional[JavaAstContext] = None,
    ) -> bool:
        return bool(self._llm_refinement_reason(original, current, ast_context))

    def _refine_with_llm(self, original: str, current: str, source_path: Optional[str] = None) -> str:
        """
        Uses the LLM to resolve complex migration scenarios and fix syntax issues.
        Includes a system prompt designed for technical accuracy.
        """
        if not self.llm:
            return current
        if not self.llm_available:
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
        if not refined_code:
            return current
        return self._accept_llm_output(
            refined_code,
            fallback=current,
            reference=original,
            source_path=source_path,
            stage="refinement",
        )

    def self_fix(self, file_content: str, errors: List[str], source_path: Optional[str] = None) -> str:
        """
        Attempts to fix compilation errors by passing them and the code back to the LLM.
        This closes the Try-Compile-Fix loop for high accuracy migration.
        """
        error_context = "\n".join(errors)
        if "not initialized in the default constructor" in error_context:
            repaired = self._ensure_constructor_for_required_final_fields(file_content)
            if repaired != file_content:
                return repaired
        if not self.llm:
            return file_content
        if not self.llm_available:
            return file_content

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
        if not fixed_code:
            return file_content
        return self._accept_llm_output(
            fixed_code,
            fallback=file_content,
            reference=file_content,
            source_path=source_path,
            stage="self-fix",
        )

    def _accept_llm_output(
        self,
        content: Optional[str],
        *,
        fallback: str,
        reference: str,
        source_path: Optional[str],
        stage: str,
    ) -> str:
        cleaned = self._sanitize_llm_output(content)
        validation_error = self._validate_llm_output(cleaned, reference, source_path)
        if validation_error:
            display_name = os.path.basename(source_path) if source_path else "<unknown>"
            print(
                f"      {MAGENTA}[LLM Rejected]{RESET} {display_name}: {validation_error} during {stage}; keeping deterministic content.",
                flush=True,
            )
            return fallback
        return cleaned

    def _validate_llm_output(
        self,
        content: str,
        reference: str,
        source_path: Optional[str],
    ) -> Optional[str]:
        cleaned = str(content or "").strip()
        if not cleaned:
            return "empty response"

        if re.match(r"(?is)^(here(?:'s| is)\b|below is\b|the corrected version\b|corrected code\b|updated code\b)", cleaned):
            return "response contained prose instead of raw Java"

        expected_package = self._extract_package_name(reference)
        actual_package = self._extract_package_name(cleaned)
        if expected_package and not actual_package:
            return "package declaration was dropped"
        if expected_package and actual_package and actual_package != expected_package:
            return f"package changed from {expected_package} to {actual_package}"

        expected_type = self._expected_type_name(reference, source_path)
        actual_type = self._extract_primary_type_name(cleaned)
        if expected_type and not actual_type:
            return f"primary type {expected_type} is missing"
        if expected_type and actual_type and actual_type != expected_type:
            return f"primary type changed from {expected_type} to {actual_type}"

        if source_path:
            file_type = os.path.splitext(os.path.basename(source_path))[0]
            public_type = self._extract_public_type_name(cleaned)
            if public_type and public_type != file_type:
                return f"public type {public_type} does not match file name {file_type}"

        if self._has_persistent_model_annotation(reference):
            expected_fields = self._declared_field_names(reference)
            actual_fields = self._declared_field_names(cleaned)
            introduced_fields = sorted(actual_fields - expected_fields - {"serialVersionUID"})
            if introduced_fields:
                return (
                    "persistent model introduced unexpected field(s): "
                    + ", ".join(introduced_fields)
                )

        return None

    def _has_persistent_model_annotation(self, content: str) -> bool:
        return any(
            marker in str(content or "")
            for marker in ("@Entity", "@MappedSuperclass", "@Embeddable")
        )

    def _declared_field_names(self, content: str) -> set[str]:
        ast_context = self._build_ast_context(content)
        if ast_context.parse_ok:
            fields = {
                field.name
                for item in ast_context.classes
                for field in item.fields
            }
            if fields:
                return fields

        matches = re.findall(
            r"(?m)^[ \t]*(?:private|protected|public)\s+(?!static\b)(?:final\s+)?[\w<>\[\], ?.]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:=[^;]+)?;",
            str(content or ""),
        )
        return set(matches)

    def _contains_framework_spring_imports(self, content: str) -> bool:
        for line in str(content or "").splitlines():
            if self._is_framework_spring_import(line.strip()):
                return True
        return False

    def _extract_package_name(self, content: str) -> Optional[str]:
        match = re.search(r"(?m)^\s*package\s+([\w.]+)\s*;", str(content or ""))
        return match.group(1) if match else None

    def _extract_public_type_name(self, content: str) -> Optional[str]:
        match = re.search(
            r"(?m)^\s*public\s+(?:final\s+|abstract\s+)?(?:class|interface|enum|record)\s+([A-Za-z_][A-Za-z0-9_]*)\b",
            str(content or ""),
        )
        return match.group(1) if match else None

    def _extract_primary_type_name(self, content: str) -> Optional[str]:
        public_type = self._extract_public_type_name(content)
        if public_type:
            return public_type
        match = re.search(
            r"(?m)^\s*(?:final\s+|abstract\s+)?(?:class|interface|enum|record)\s+([A-Za-z_][A-Za-z0-9_]*)\b",
            str(content or ""),
        )
        return match.group(1) if match else None

    def _expected_type_name(self, reference: str, source_path: Optional[str]) -> Optional[str]:
        if source_path:
            file_name = os.path.splitext(os.path.basename(source_path))[0]
            if file_name:
                return file_name
        return self._extract_primary_type_name(reference)

    def _sanitize_llm_output(self, content: Optional[str]) -> str:
        cleaned = str(content or "").strip()
        if not cleaned:
            return ""

        fence_match = re.match(r"^```[a-zA-Z0-9_-]*\s*\n(?P<body>[\s\S]*?)\n```$", cleaned)
        if fence_match:
            cleaned = fence_match.group("body").strip()
        elif cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n", "", cleaned)
            if "\n```" in cleaned:
                cleaned = cleaned.split("\n```", 1)[0].rstrip()

        if "```" in cleaned:
            cleaned = cleaned.split("```", 1)[0].rstrip()

        if "Note:" in cleaned and re.search(r"\b(class|interface|enum|record)\b", cleaned):
            note_index = cleaned.find("Note:")
            last_brace = cleaned.rfind("}")
            if note_index != -1 and last_brace != -1 and last_brace < note_index:
                cleaned = cleaned[: last_brace + 1].rstrip()

        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
        cleaned = re.sub(r"(?m)^[ \t]*import manual [^\n]+;\n?", "", cleaned)
        cleaned = re.sub(r"(?m)^[ \t]*@manual [^\n]+\n?", "\t// TODO: manual review note removed from generated annotation.\n", cleaned)
        cleaned = re.sub(r"@manual [A-Za-z0-9 _\\-]+", "", cleaned)
        return cleaned.strip() + "\n"

    def _final_spring_purge(self, content: str) -> str:
        """
        A final safety pass to remove any remaining Spring boilerplate or imports
        that the RAG/LLM might have left as orphans.
        """
        lines = content.split('\n')
        purged_lines = []
        
        # Patterns that should NEVER be in a Micronaut file
        banned_patterns = [
            "@Autowired",
            "@Service",
            "@Component",
            "@Repository",
            "@RestController",
            "@RequestMapping",
            "@ResponseBody",
        ]
        
        for line in lines:
            stripped = line.strip()
            if stripped == "@Repository" and "import io.micronaut.data.annotation.Repository;" in content:
                purged_lines.append(line)
                continue
            if (
                self._is_framework_spring_import(stripped)
                or any(stripped.startswith(pattern) for pattern in banned_patterns if pattern.startswith("@"))
            ):
                # Only keep the line if it was already migrated (contains micronaut or jakarta)
                if "io.micronaut" in line or "jakarta." in line:
                    purged_lines.append(line)
                else:
                    # Drop the orphaned Spring line
                    continue
            else:
                purged_lines.append(line)
                
        return '\n'.join(purged_lines)

    def _is_framework_spring_import(self, stripped_line: str) -> bool:
        if not stripped_line.startswith("import org.springframework"):
            return False
        framework_prefixes = (
            "import org.springframework.aot.",
            "import org.springframework.amqp.",
            "import org.springframework.beans.",
            "import org.springframework.boot.",
            "import org.springframework.cache.",
            "import org.springframework.cloud.",
            "import org.springframework.context.",
            "import org.springframework.core.",
            "import org.springframework.dao.",
            "import org.springframework.data.",
            "import org.springframework.format.",
            "import org.springframework.http.",
            "import org.springframework.jdbc.",
            "import org.springframework.kafka.",
            "import org.springframework.orm.",
            "import org.springframework.security.",
            "import org.springframework.scheduling.",
            "import org.springframework.stereotype.",
            "import org.springframework.test.",
            "import org.springframework.transaction.",
            "import org.springframework.ui.",
            "import org.springframework.util.",
            "import org.springframework.validation.",
            "import org.springframework.web.",
        )
        return stripped_line.startswith(framework_prefixes)

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
            print(f"      {MAGENTA}|__{RESET} {BOLD}{MAGENTA}LLM Shift Summary:{RESET}", flush=True)
            for r in removed[:5]: # Show more for transparency
                # If we can find the matching added line, show it as a transition
                print(f"         {RED}-{RESET} Purged: {r}", flush=True)
            for a in added[:5]:
                print(f"         {GREEN}+{RESET} Introduced: {a}", flush=True)
            if len(removed) > 5 or len(added) > 5:
                print(f"         ... (Significant structural transformation detected)", flush=True)

    def _normalize_micronaut_output(self, content: str) -> str:
        content = self._sanitize_invalid_entity_data_annotations(content)
        content = self._normalize_import_layout(content)
        content = self._ensure_required_micronaut_imports(content)
        content = re.sub(r'(?m)^[ \t]*@Controller[ \t]*\n(?=[ \t]*@Controller\(")', "", content)
        content = re.sub(r'(?m)^([ \t]*@Inject\s*)\n(?:[ \t]*@Inject\s*\n)+', r'\1' + "\n", content)
        content = self._dedupe_imports(content)
        content = self._sort_import_block(content)
        content = self._normalize_whitespace(content)
        return content

    def _sanitize_invalid_entity_data_annotations(self, content: str) -> str:
        if "@Entity" not in content:
            return content

        # JPA entity classes should not carry Micronaut Data relation-mapping annotations
        # on fields/getters. Those belong to repository/query definitions, not entity models.
        content = re.sub(
            r"(?m)^[ \t]*import io\.micronaut\.data\.annotation\.(?:MappedCollection|Relation|Join);\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*@MappedCollection\b(?:\([^)]*\))?\s*\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*@Relation\b(?:\([^)]*\))?\s*\n?",
            "",
            content,
        )
        content = re.sub(
            r"(?m)^[ \t]*@Join\b(?:\([^)]*\))?\s*\n?",
            "",
            content,
        )
        return content

    def _ensure_required_micronaut_imports(self, content: str) -> str:
        package_match = re.search(r"^(package\s+[\w.]+;\s*)", content, re.MULTILINE)
        package_end = package_match.end() if package_match else 0

        existing_imports = set(re.findall(r"^import\s+([\w.]+);", content, re.MULTILINE))
        imports_to_add = []
        for annotation, import_path in self.ANNOTATION_IMPORTS.items():
            if f"@{annotation}" in content and import_path not in existing_imports:
                imports_to_add.append(import_path)
        if "HttpResponse" in content and "io.micronaut.http.HttpResponse" not in existing_imports:
            imports_to_add.append("io.micronaut.http.HttpResponse")
        if "HttpRequest" in content and "io.micronaut.http.HttpRequest" not in existing_imports:
            imports_to_add.append("io.micronaut.http.HttpRequest")
        if "HttpClient" in content and "io.micronaut.http.client.HttpClient" not in existing_imports:
            imports_to_add.append("io.micronaut.http.client.HttpClient")
        if "HttpStatus" in content and "io.micronaut.http.HttpStatus" not in existing_imports:
            imports_to_add.append("io.micronaut.http.HttpStatus")
        if "ModelAndView" in content and "io.micronaut.views.ModelAndView" not in existing_imports:
            imports_to_add.append("io.micronaut.views.ModelAndView")
        if "LinkedHashMap" in content and "java.util.LinkedHashMap" not in existing_imports:
            imports_to_add.append("java.util.LinkedHashMap")
        if "Map<String, Object>" in content and "java.util.Map" not in existing_imports:
            imports_to_add.append("java.util.Map")
        if ("Page<" in content or "Page.of(" in content) and "io.micronaut.data.model.Page" not in existing_imports:
            imports_to_add.append("io.micronaut.data.model.Page")
        if "Slice<" in content and "io.micronaut.data.model.Slice" not in existing_imports:
            imports_to_add.append("io.micronaut.data.model.Slice")
        if "Pageable" in content and "io.micronaut.data.model.Pageable" not in existing_imports:
            imports_to_add.append("io.micronaut.data.model.Pageable")

        if not imports_to_add:
            return content

        import_block = "".join(f"\nimport {item};" for item in sorted(set(imports_to_add)))
        return content[:package_end] + import_block + content[package_end:]

    def _dedupe_imports(self, content: str) -> str:
        lines = content.splitlines()
        normalized_imports = {
            spring_import: micronaut_import
            for spring_import, micronaut_import in self.IMPORT_MAPPINGS.items()
            if micronaut_import
        }
        selected_by_simple_name: dict[str, str] = {}
        seen_exact_imports = set()

        for line in lines:
            if not line.startswith("import "):
                continue
            import_name = line[len("import ") :].rstrip(";").strip()
            if not import_name:
                continue
            mapped_name = normalized_imports.get(import_name, import_name)
            simple_name = mapped_name.split(".")[-1]
            existing = selected_by_simple_name.get(simple_name)
            if existing is None:
                selected_by_simple_name[simple_name] = mapped_name
                continue
            if existing == mapped_name:
                continue
            if existing.startswith("org.springframework.") and not mapped_name.startswith("org.springframework."):
                selected_by_simple_name[simple_name] = mapped_name
                continue
            if import_name.startswith("org.springframework.") and not existing.startswith("org.springframework."):
                continue
            if existing.startswith("jakarta.") and mapped_name.startswith("io.micronaut."):
                selected_by_simple_name[simple_name] = mapped_name

        output = []
        for line in lines:
            if not line.startswith("import "):
                output.append(line)
                continue
            import_name = line[len("import ") :].rstrip(";").strip()
            if not import_name:
                continue
            mapped_name = normalized_imports.get(import_name, import_name)
            simple_name = mapped_name.split(".")[-1]
            if selected_by_simple_name.get(simple_name) != mapped_name:
                continue
            normalized_line = f"import {mapped_name};"
            if normalized_line in seen_exact_imports:
                continue
            seen_exact_imports.add(normalized_line)
            output.append(normalized_line)
        return "\n".join(output)

    def _normalize_import_layout(self, content: str) -> str:
        content = re.sub(r";(?=import(?: static)?\s)", ";\n", content)
        content = re.sub(r";[ \t]*(?=import(?: static)?\s)", ";\n", content)
        return content

    def _normalize_whitespace(self, content: str) -> str:
        lines = [line.rstrip() for line in content.splitlines()]
        normalized = "\n".join(lines)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        normalized = re.sub(r"(?m)^(package\s+[\w.]+;)\n(?!\n)", r"\1\n", normalized)
        normalized = re.sub(r"(?m)^(import [^;]+;)\n(?!(?:import|import static|$))", r"\1\n", normalized)
        return normalized.rstrip() + "\n"

    def _sort_import_block(self, content: str) -> str:
        package_match = re.search(r"(?m)^package\s+[\w.]+;\s*$", content)
        if not package_match:
            return content

        trailing = content[package_match.end() :]
        lines = trailing.splitlines(True)
        imports = []
        body_prefix = ""
        index = 0

        while index < len(lines) and not lines[index].strip():
            index += 1

        while index < len(lines):
            stripped = lines[index].strip()
            if stripped.startswith("import "):
                split_imports = list(re.finditer(r"import(?: static)? [^;]+;", stripped))
                if split_imports:
                    imports.extend(match.group(0) for match in split_imports)
                    tail = stripped[split_imports[-1].end() :].strip()
                    if tail:
                        body_prefix += tail + "\n"
                else:
                    imports.append(stripped)
                index += 1
                continue
            if not stripped and imports:
                index += 1
                continue
            break

        if not imports:
            return content

        body = (body_prefix + "".join(lines[index:])).lstrip("\n")
        normal_imports = sorted({item for item in imports if not item.startswith("import static ")})
        static_imports = sorted({item for item in imports if item.startswith("import static ")})
        import_lines = normal_imports[:]
        if static_imports:
            if import_lines:
                import_lines.append("")
            import_lines.extend(static_imports)
        import_block = "\n\n" + "\n".join(import_lines) + "\n\n"
        return content[: package_match.end()] + import_block + body

    def _replace_binding_result_block(self, match: re.Match) -> str:
        indent_match = re.match(r"(?m)^([ \t]*)if", match.group(0))
        indent = indent_match.group(1) if indent_match else ""
        comment = f"{indent}// TODO: migrate Spring validation error flow to Micronaut validation handling.\n"
        return comment

    def _rewrite_binding_result_methods(self, content: str) -> str:
        transformed = False
        binding_markers = list(re.finditer(r"\bBindingResult\s+(?P<result>\w+)\b", content))

        for marker in reversed(binding_markers):
            signature_open, signature_close = self._find_enclosing_method_signature(content, marker.start())
            if signature_open == -1:
                continue
            if signature_close == -1 or signature_close < marker.end():
                continue
            params = content[signature_open + 1 : signature_close]
            result_match = re.search(r"\bBindingResult\s+(?P<result>\w+)\b", params)
            if not result_match:
                continue
            result_name = result_match.group("result")
            brace_index = content.find("{", signature_close)
            if brace_index == -1:
                continue
            next_semicolon = content.find(";", signature_close, brace_index)
            if next_semicolon != -1:
                continue
            header_start = content.rfind("\n", 0, signature_open) + 1
            header = content[header_start : brace_index + 1]
            annotations = content[self._expand_method_start_to_annotations(content, header_start) : header_start]

            rewritten_param_parts = [
                part
                for part in self._split_signature_arguments(params)
                if "BindingResult" not in part
            ]
            query_binding_target = None
            if "@Get" in annotations:
                query_binding_target = self._extract_query_binding_target(rewritten_param_parts)
                if query_binding_target is not None:
                    rewritten_param_parts[query_binding_target["index"]] = "HttpRequest<?> request"
            rewritten_params = ", ".join(rewritten_param_parts)
            model_name = self._extract_model_parameter_name(rewritten_params)
            if not model_name:
                model_name = "model"
                rewritten_param_parts.append("Map<String, Object> model")
                rewritten_params = ", ".join(rewritten_param_parts)
            if query_binding_target is not None:
                attribute_name = query_binding_target["name"]
                attribute_var = query_binding_target["name"]
            else:
                attribute_name, attribute_var = self._infer_binding_attribute_name(rewritten_params)
            rewritten_header = header[: signature_open - header_start + 1] + rewritten_params + header[signature_close - header_start :]

            body_start = brace_index
            body_end = self._find_matching_brace(content, body_start)
            if body_end == -1:
                continue
            body = content[body_start + 1 : body_end]
            if query_binding_target is not None:
                binding_line = (
                    f'{query_binding_target["type"]} {query_binding_target["name"]} = '
                    f'bindQueryBean(request, {query_binding_target["type"]}.class);'
                )
                if binding_line not in body:
                    indent = self._infer_method_body_indent(body)
                    body = f"\n{indent}{binding_line}{body}"
            body = re.sub(
                rf"(?m)^(?P<indent>[ \t]*){re.escape(result_name)}\.rejectValue\(\s*\"(?P<field>[^\"]+)\"\s*,\s*\"(?P<code>[^\"]+)\"[^;]*;\s*$",
                rf'\g<indent>addFieldError({model_name}, "\g<field>", "\g<code>");' + "\n"
                + rf'{model_name}.put("{attribute_name}", {attribute_var});' + "\n",
                body,
            )
            body = re.sub(
                rf"(?ms)^(?P<indent>[ \t]*)if\s*\(\s*{re.escape(result_name)}\.hasErrors\(\)\s*\)\s*\{{\s*(?P<inner>.*?)^(?P=indent)\}}[ \t]*\n?",
                lambda m: self._manual_validation_block_replacement(
                    m,
                    model_name=model_name,
                    attribute_name=attribute_name,
                    attribute_var=attribute_var,
                ),
                body,
            )
            body = re.sub(
                rf"\b{re.escape(result_name)}\.hasErrors\(\)",
                f'hasValidationErrors({model_name}, "{attribute_name}", {attribute_var})',
                body,
            )
            body = re.sub(
                rf"\b{re.escape(result_name)}\b",
                attribute_var,
                body,
            )

            content = content[:header_start] + rewritten_header + body + content[body_end:]
            transformed = True

        if not transformed:
            return re.sub(r"\s*,\s*BindingResult\s+\w+", "", re.sub(r"BindingResult\s+\w+\s*,\s*", "", content))

        content = self._ensure_manual_validation_support(content)
        content = self._ensure_form_binding_support(content)
        return content

    def _extract_query_binding_target(self, parts: Sequence[str]) -> Optional[dict]:
        for index, raw_part in enumerate(parts):
            part = raw_part.strip()
            if not part:
                continue
            if (
                "@PathVariable" in part
                or "@QueryValue" in part
                or "@Header" in part
                or "@Body" in part
                or "HttpRequest<" in part
                or "HttpResponse<" in part
                or "Map<String, Object>" in part
                or "Model " in part
                or "ModelMap " in part
            ):
                continue
            parsed = self._parse_parameter_signature(part)
            if not parsed:
                continue
            type_name = parsed["type"].split(".")[-1]
            if type_name in {
                "String",
                "int",
                "Integer",
                "long",
                "Long",
                "boolean",
                "Boolean",
                "double",
                "Double",
                "float",
                "Float",
                "Page",
                "Pageable",
                "List",
                "Set",
                "Collection",
                "Optional",
            }:
                continue
            return {"index": index, "type": parsed["type"], "name": parsed["name"]}
        return None

    def _find_enclosing_method_signature(self, content: str, marker_start: int) -> Tuple[int, int]:
        best_open = -1
        best_close = -1
        candidate_open = content.rfind("(", 0, marker_start)

        while candidate_open != -1:
            candidate_close = self._find_matching_parenthesis(content, candidate_open)
            if candidate_close != -1 and candidate_close >= marker_start:
                brace_index = content.find("{", candidate_close)
                if brace_index != -1:
                    next_semicolon = content.find(";", candidate_close, brace_index)
                    if next_semicolon == -1 and candidate_close > best_close:
                        best_open = candidate_open
                        best_close = candidate_close
            candidate_open = content.rfind("(", 0, candidate_open)

        return best_open, best_close

    def _split_signature_arguments(self, params: str) -> List[str]:
        if not params.strip():
            return []

        parts = []
        current = []
        generic_depth = 0
        paren_depth = 0
        bracket_depth = 0
        in_string = False
        string_delimiter = ""
        escaped = False

        for char in params:
            if in_string:
                current.append(char)
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == string_delimiter:
                    in_string = False
                continue

            if char in {'"', "'"}:
                in_string = True
                string_delimiter = char
                current.append(char)
                continue
            if char == "<":
                generic_depth += 1
            elif char == ">" and generic_depth > 0:
                generic_depth -= 1
            elif char == "(":
                paren_depth += 1
            elif char == ")" and paren_depth > 0:
                paren_depth -= 1
            elif char == "[":
                bracket_depth += 1
            elif char == "]" and bracket_depth > 0:
                bracket_depth -= 1
            elif char == "," and generic_depth == 0 and paren_depth == 0 and bracket_depth == 0:
                parts.append("".join(current).strip())
                current = []
                continue
            current.append(char)

        tail = "".join(current).strip()
        if tail:
            parts.append(tail)
        return parts

    def _extract_model_parameter_name(self, params: str) -> Optional[str]:
        match = re.search(r"\bMap<\s*String\s*,\s*Object\s*>\s+(\w+)", params)
        if match:
            return match.group(1)
        match = re.search(r"\b(?:Model|ModelMap)\s+(\w+)", params)
        if match:
            return match.group(1)
        return None

    def _add_model_parameter_to_signature(self, params: str) -> str:
        cleaned = params.strip()
        if not cleaned:
            return "Map<String, Object> model"
        return cleaned + ", Map<String, Object> model"

    def _infer_binding_attribute_name(self, params: str) -> Tuple[str, str]:
        valid_match = re.search(
            r"(?:@(?:[\w.]+\.)?Valid)\s+([A-Z][A-Za-z0-9_<>, ?]*?)\s+(\w+)(?=\s*(?:,|$))",
            params,
        )
        if valid_match:
            return valid_match.group(2), valid_match.group(2)

        for raw_param in self._split_signature_arguments(params):
            raw_param = raw_param.strip()
            if not raw_param:
                continue
            if "Map<" in raw_param or "Model " in raw_param or "ModelMap " in raw_param:
                continue
            if "BindingResult" in raw_param:
                continue
            if (
                "@PathVariable" in raw_param
                or "@QueryValue" in raw_param
                or "@Header" in raw_param
                or "@Body" in raw_param
                or "@RequestParam" in raw_param
                or "@CookieValue" in raw_param
            ):
                continue
            cleaned_param = re.sub(r"@\S+(?:\([^)]*\))?\s*", "", raw_param).strip()
            cleaned_param = re.sub(r"\bfinal\s+", "", cleaned_param).strip()
            tokens = cleaned_param.split()
            if len(tokens) < 2:
                continue
            param_name = tokens[-1]
            type_name = tokens[-2]
            if type_name in {
                "int",
                "long",
                "boolean",
                "double",
                "float",
                "short",
                "byte",
                "char",
                "String",
                "Integer",
                "Long",
                "Boolean",
                "Byte",
                "Short",
                "Double",
                "Float",
                "Character",
            }:
                continue
            if param_name.isidentifier():
                return param_name, param_name
        return "command", "command"

    def _manual_validation_block_replacement(
        self,
        match: re.Match,
        *,
        model_name: str,
        attribute_name: str,
        attribute_var: str,
    ) -> str:
        indent = match.group("indent")
        inner = match.group("inner")
        return_line = ""
        return_match = re.search(r'return\s+([^;]+);', inner)
        if return_match:
            return_line = f"{indent}    return {return_match.group(1).strip()};\n"
        return (
            f'{indent}if (hasValidationErrors({model_name}, "{attribute_name}", {attribute_var})) {{\n'
            f"{return_line}"
            f"{indent}}}\n"
        )

    def _ensure_manual_validation_support(self, content: str) -> str:
        if "hasValidationErrors(" not in content and "addFieldError(" not in content:
            return content

        if "import jakarta.validation.Validator;" not in content:
            package_match = re.search(r"(?m)^package [^;]+;\n", content)
            if package_match:
                insert_at = package_match.end()
                content = content[:insert_at] + "\nimport jakarta.validation.Validator;\n" + content[insert_at:]

        class_match = re.search(
            r"(?m)^(?P<indent>[ \t]*)(?:public|protected|private\s+)?(?:abstract\s+|final\s+)?class\s+(?P<class_name>\w+)[^{]*\{",
            content,
        )
        if not class_match:
            return content

        member_indent = class_match.group("indent") + ("    ")
        class_name = class_match.group("class_name")

        if "private final Validator validator;" not in content:
            insert_at = class_match.end()
            content = (
                content[:insert_at]
                + f"\n{member_indent}private final Validator validator;\n"
                + content[insert_at:]
            )

        constructor_pattern = re.compile(
            rf"(?ms)^(?P<indent>[ \t]*)(?:(?:public|protected|private)\s+)?{re.escape(class_name)}\((?P<params>[^)]*)\)\s*\{{(?P<body>.*?)^(?P=indent)\}}",
            re.MULTILINE,
        )
        constructor_match = constructor_pattern.search(content)
        if constructor_match and "this.validator = validator;" not in content:
            params = constructor_match.group("params").strip()
            new_params = (params + ", Validator validator") if params else "Validator validator"
            body = constructor_match.group("body").rstrip()
            body = body + f"\n{constructor_match.group('indent')}    this.validator = validator;\n"
            replacement = (
                f"{constructor_match.group('indent')}public {class_name}({new_params}) {{"
                f"{body}"
                f"{constructor_match.group('indent')}}}"
            )
            content = content[: constructor_match.start()] + replacement + content[constructor_match.end() :]
        elif not constructor_match:
            insert_at = class_match.end()
            constructor_block = (
                f"\n{member_indent}public {class_name}(Validator validator) {{\n"
                f"{member_indent}    this.validator = validator;\n"
                f"{member_indent}}}\n"
            )
            content = content[:insert_at] + constructor_block + content[insert_at:]

        content = self._ensure_helper_method(
            content,
            method_name="hasValidationErrors",
            parameter_signature="Map<String, Object> model, String attributeName, T attribute",
            method_body=(
                "if (attribute == null) {\n"
                "\t\t\treturn Boolean.TRUE.equals(model.get(\"_validationError\"));\n"
                "\t\t}\n"
                "\t\tjava.util.Set<jakarta.validation.ConstraintViolation<T>> violations = validator.validate(attribute);\n"
                "\t\tif (violations.isEmpty() && !Boolean.TRUE.equals(model.get(\"_validationError\"))) {\n"
                "\t\t\treturn false;\n"
                "\t\t}\n"
                "\t\tmodel.put(attributeName, attribute);\n"
                "\t\tif (!violations.isEmpty()) {\n"
                "\t\t\tmodel.put(attributeName + \"Violations\", violations);\n"
                "\t\t}\n"
                "\t\treturn true;"
            ),
            return_type=" <T> boolean".strip(),
        )
        if "private <T> boolean hasValidationErrors(" not in content:
            content = content.replace("private boolean hasValidationErrors(", "private <T> boolean hasValidationErrors(")
        content = self._ensure_helper_method(
            content,
            method_name="addFieldError",
            parameter_signature="Map<String, Object> model, String field, String code",
            method_body=(
                "@SuppressWarnings(\"unchecked\") Map<String, String> fieldErrors = "
                "(Map<String, String>) model.computeIfAbsent(\"_fieldErrors\", key -> new LinkedHashMap<String, String>());\n"
                "\t\tfieldErrors.put(field, code);\n"
                "\t\tmodel.put(\"_validationError\", Boolean.TRUE);"
            ),
        )
        return content

    def _remove_named_annotations(self, content: str, annotations: Sequence[str]) -> str:
        if not annotations:
            return content

        pattern = re.compile(
            r"(?m)^[ \t]*@(" + "|".join(re.escape(item) for item in annotations) + r")\b"
        )
        matches = list(pattern.finditer(content))
        for match in reversed(matches):
            start = match.start()
            end = match.end()

            while end < len(content) and content[end] in " \t":
                end += 1

            if end < len(content) and content[end] == "(":
                paren_end = self._find_matching_parenthesis(content, end)
                if paren_end == -1:
                    continue
                end = paren_end + 1

            while end < len(content) and content[end] in " \t":
                end += 1
            while end < len(content) and content[end] in "\r\n":
                end += 1

            content = content[:start] + content[end:]

        return content

    def _remove_annotated_methods(self, content: str, annotations: Sequence[str]) -> str:
        for annotation in annotations:
            pattern = re.compile(rf"(?m)^[ \t]*@{annotation}\b[^\n]*\n?")
            matches = list(pattern.finditer(content))
            for match in reversed(matches):
                method_start = match.start()
                body_start = content.find("{", match.end())
                if body_start == -1:
                    continue
                body_end = self._find_matching_brace(content, body_start)
                if body_end == -1:
                    continue
                line_end = body_end + 1
                while line_end < len(content) and content[line_end] in "\r\n":
                    line_end += 1
                content = content[:method_start] + content[line_end:]
        return content

    def _remove_annotated_fields(self, content: str, annotations: Sequence[str]) -> str:
        for annotation in annotations:
            pattern = re.compile(rf"(?m)^[ \t]*@{annotation}\b[^\n]*\n?")
            matches = list(pattern.finditer(content))
            for match in reversed(matches):
                field_start = match.start()
                field_end = content.find(";", match.end())
                if field_end == -1:
                    continue
                line_end = field_end + 1
                while line_end < len(content) and content[line_end] in "\r\n":
                    line_end += 1
                content = content[:field_start] + content[line_end:]
        return content

    def _rewrite_annotated_methods(
        self,
        content: str,
        annotations: Sequence[str],
        body_lines: Sequence[str],
    ) -> str:
        pattern = re.compile(r"(?m)^[ \t]*@(" + "|".join(re.escape(item) for item in annotations) + r")\b[^\n]*\n?")
        matches = list(pattern.finditer(content))
        for match in reversed(matches):
            body_start = content.find("{", match.end())
            if body_start == -1:
                continue
            body_end = self._find_matching_brace(content, body_start)
            if body_end == -1:
                continue
            line_start = content.rfind("\n", 0, body_start) + 1
            indent = re.match(r"[ \t]*", content[line_start:body_start]).group(0)
            child_indent = indent + ("\t" if "\t" in indent or indent == "" else "    ")
            if body_lines:
                replacement_body = "\n" + "\n".join(f"{child_indent}{line}" for line in body_lines) + "\n" + indent
            else:
                replacement_body = "\n" + indent
            content = content[: body_start + 1] + replacement_body + content[body_end:]
        return content

    def _remove_method_containing_text(self, content: str, marker: str) -> str:
        index = content.find(marker)
        if index == -1:
            return content
        line_start = content.rfind("\n", 0, index) + 1
        method_start = line_start
        while method_start > 0:
            previous_newline = content.rfind("\n", 0, max(method_start - 1, 0))
            previous_line_start = 0 if previous_newline == -1 else previous_newline + 1
            previous_line = content[previous_line_start:method_start].strip()
            if previous_line.startswith("@"):
                method_start = previous_line_start
                continue
            break
        body_start = content.find("{", index)
        if body_start == -1:
            return content
        body_end = self._find_matching_brace(content, body_start)
        if body_end == -1:
            return content
        line_end = body_end + 1
        while line_end < len(content) and content[line_end] in "\r\n":
            line_end += 1
        return content[:method_start] + content[line_end:]

    def _find_matching_brace(self, content: str, open_brace_index: int) -> int:
        depth = 0
        in_string = False
        string_delimiter = ""
        escaped = False
        for index in range(open_brace_index, len(content)):
            char = content[index]
            if in_string:
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == string_delimiter:
                    in_string = False
                continue
            if char in {'"', "'"}:
                in_string = True
                string_delimiter = char
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return index
        return -1

    def _find_matching_parenthesis(self, content: str, open_paren_index: int) -> int:
        depth = 0
        in_string = False
        string_delimiter = ""
        escaped = False
        for index in range(open_paren_index, len(content)):
            char = content[index]
            if in_string:
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == string_delimiter:
                    in_string = False
                continue
            if char in {'"', "'"}:
                in_string = True
                string_delimiter = char
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return index
        return -1

    def _find_statement_end(self, content: str, start_index: int) -> int:
        in_string = False
        string_delimiter = ""
        escaped = False
        paren_depth = 0
        brace_depth = 0

        for index in range(start_index, len(content)):
            char = content[index]
            if in_string:
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == string_delimiter:
                    in_string = False
                continue
            if char in {'"', "'"}:
                in_string = True
                string_delimiter = char
                continue
            if char == "(":
                paren_depth += 1
            elif char == ")" and paren_depth > 0:
                paren_depth -= 1
            elif char == "{":
                brace_depth += 1
            elif char == "}" and brace_depth > 0:
                brace_depth -= 1
            elif char == ";" and paren_depth == 0 and brace_depth == 0:
                return index
        return -1
