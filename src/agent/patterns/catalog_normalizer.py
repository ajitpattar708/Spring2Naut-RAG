import argparse
import json
from pathlib import Path
from typing import Dict, List

from src.agent.agents.dependency_audit import DependencyCompatibilityAuditor
from src.agent.patterns.repository import PatternCorpusRepository
from src.agent.patterns.schema import (
    PatternEvidence,
    PatternType,
    SourceKind,
    ValidationStatus,
    VersionWindow,
    VersionedPattern,
)


_CATALOG_SOURCE_REF = "src/agent/agents/dependency_audit.py::_COMPATIBILITY_CATALOG"
_REVIEWED_CODE_SOURCE_REF = "src/agent/patterns/catalog_normalizer.py::_reviewed_code_patterns"


class _NoopKnowledgeBase:
    def search_annotation(self, spring_annotation: str, **kwargs):
        return []

    def search_dependency(self, spring_dep: str, **kwargs):
        return []

    def search_configuration(self, spring_prop: str, **kwargs):
        return []


def curated_catalog_patterns() -> List[VersionedPattern]:
    auditor = DependencyCompatibilityAuditor(_NoopKnowledgeBase(), "3.4.5", "4.10.8")
    patterns: List[VersionedPattern] = []

    for entry in auditor._COMPATIBILITY_CATALOG:
        replacement_notes = entry.notes or ""
        description = entry.rationale
        if replacement_notes:
            description = f"{description} {replacement_notes}"

        spring_patterns = [entry.ga, *entry.aliases]
        canonical_pattern_id = (
            "catalog.dependency."
            + entry.ga.replace(".", "_").replace(":", "_").replace("-", "_")
        )

        for index, spring_pattern in enumerate(spring_patterns):
            is_alias = spring_pattern != entry.ga
            pattern_id = canonical_pattern_id
            if is_alias:
                pattern_id = (
                    "catalog.dependency."
                    + spring_pattern.replace(".", "_").replace(":", "_").replace("-", "_")
                )

            metadata = {
                "catalog_entry": True,
                "target_status": entry.target_status,
                "automated_migration_supported": entry.automated_migration_supported,
                "replacement_version_management": entry.version_management,
                "replacement_version": entry.replacement_version,
                "aliases": list(entry.aliases),
            }
            if is_alias:
                metadata["canonical_catalog_pattern_id"] = canonical_pattern_id
                metadata["catalog_alias_for"] = entry.ga

            patterns.append(
                VersionedPattern(
                    pattern_id=pattern_id,
                    pattern_type=PatternType.DEPENDENCY,
                    spring_pattern=spring_pattern,
                    micronaut_pattern=entry.replacement,
                    description=description,
                    spring_versions=VersionWindow(spec=entry.spring_spec),
                    micronaut_versions=VersionWindow(spec=entry.micronaut_spec),
                    status=ValidationStatus.VALIDATED,
                    confidence=0.97 if entry.target_status == "manual_redesign" else 0.93,
                    complexity="medium" if entry.automated_migration_supported else "high",
                    category="dependencies",
                    source_kind=SourceKind.MANUAL,
                    evidence=[
                        PatternEvidence(
                            source_kind=SourceKind.MANUAL,
                            source_ref=_CATALOG_SOURCE_REF,
                            title="Curated compatibility catalog",
                            notes=(
                                f"target_status={entry.target_status}; "
                                f"version_management={entry.version_management}; "
                                f"alias_index={index}"
                            ),
                        )
                    ],
                    metadata=metadata,
                )
            )
    patterns.extend(_reviewed_code_patterns())
    return patterns


def _reviewed_code_patterns() -> List[VersionedPattern]:
    reviewed_entries = [
        (
            "catalog.code.jcache_manager_customizer",
            "org.springframework.boot.autoconfigure.cache.JCacheManagerCustomizer",
            "manual Micronaut cache provider bean configuration",
            "JCacheManagerCustomizer has no governed one-line Micronaut equivalent; migrate to explicit cache provider beans and keep runtime cache creation visible.",
        ),
        (
            "catalog.code.binding_result",
            "org.springframework.validation.BindingResult",
            "manual validation-error flow redesign",
            "BindingResult-heavy controller flows need explicit Micronaut validation and error-model handling instead of a blind type swap.",
        ),
        (
            "catalog.code.web_data_binder",
            "org.springframework.web.bind.WebDataBinder",
            "manual binder and validator registration",
            "WebDataBinder customization should be reviewed manually because Micronaut binding and validation hooks differ from Spring MVC binder callbacks.",
        ),
        (
            "catalog.code.model_attribute_import",
            "org.springframework.web.bind.annotation.ModelAttribute",
            "manual Micronaut view-model prepopulation review",
            "ModelAttribute-based controller prepopulation is not always a one-line Micronaut rewrite and should be reviewed with the target controller flow.",
        ),
        (
            "catalog.code.model_attribute_annotation",
            "ModelAttribute",
            "manual Micronaut view-model prepopulation review",
            "ModelAttribute-based controller prepopulation is not always a one-line Micronaut rewrite and should be reviewed with the target controller flow.",
        ),
        (
            "catalog.code.init_binder_import",
            "org.springframework.web.bind.annotation.InitBinder",
            "manual binder callback review",
            "InitBinder callbacks need manual migration because Micronaut binding and validator registration differ from Spring MVC binder hooks.",
        ),
        (
            "catalog.code.init_binder_annotation",
            "InitBinder",
            "manual binder callback review",
            "InitBinder callbacks need manual migration because Micronaut binding and validator registration differ from Spring MVC binder hooks.",
        ),
        (
            "catalog.code.webmvc_test_import",
            "org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest",
            "@Disabled + @MicronautTest placeholder",
            "WebMvcTest slice tests should be converted into explicit Micronaut HTTP tests or left as disabled placeholders until rewritten.",
        ),
        (
            "catalog.code.webmvc_test_annotation",
            "WebMvcTest",
            "@Disabled + @MicronautTest placeholder",
            "WebMvcTest slice tests should be converted into explicit Micronaut HTTP tests or left as disabled placeholders until rewritten.",
        ),
        (
            "catalog.code.data_jpa_test_import",
            "org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest",
            "@Disabled + @MicronautTest placeholder",
            "DataJpaTest slice tests need a Micronaut data-test rewrite instead of a blind annotation swap.",
        ),
        (
            "catalog.code.data_jpa_test_annotation",
            "DataJpaTest",
            "@Disabled + @MicronautTest placeholder",
            "DataJpaTest slice tests need a Micronaut data-test rewrite instead of a blind annotation swap.",
        ),
        (
            "catalog.code.auto_configure_test_database_import",
            "org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase",
            "remove annotation and review test datasource wiring",
            "AutoConfigureTestDatabase is Spring test-slice specific and should be removed while the Micronaut test datasource setup is reviewed.",
        ),
        (
            "catalog.code.auto_configure_test_database_annotation",
            "AutoConfigureTestDatabase",
            "remove annotation and review test datasource wiring",
            "AutoConfigureTestDatabase is Spring test-slice specific and should be removed while the Micronaut test datasource setup is reviewed.",
        ),
        (
            "catalog.code.spring_web_environment",
            "org.springframework.boot.test.context.SpringBootTest.WebEnvironment",
            "@MicronautTest",
            "SpringBootTest.WebEnvironment should be collapsed into MicronautTest plus explicit client or embedded-server test wiring.",
        ),
        (
            "catalog.code.local_server_port_import",
            "org.springframework.boot.test.web.server.LocalServerPort",
            "@Client or EmbeddedServer URI injection",
            "LocalServerPort should migrate to Micronaut client or embedded-server URI wiring rather than field port injection.",
        ),
        (
            "catalog.code.local_server_port_annotation",
            "LocalServerPort",
            "@Client or EmbeddedServer URI injection",
            "LocalServerPort should migrate to Micronaut client or embedded-server URI wiring rather than field port injection.",
        ),
        (
            "catalog.code.mockmvc_import",
            "org.springframework.test.web.servlet.MockMvc",
            "@Disabled + @MicronautTest placeholder",
            "MockMvc-based tests should migrate to Micronaut HTTP tests or remain explicit disabled placeholders until rewritten.",
        ),
        (
            "catalog.code.mockmvc_simple",
            "MockMvc",
            "@Disabled + @MicronautTest placeholder",
            "MockMvc-based tests should migrate to Micronaut HTTP tests or remain explicit disabled placeholders until rewritten.",
        ),
        (
            "catalog.code.mockmvc_request_builders",
            "org.springframework.test.web.servlet.request.MockMvcRequestBuilders",
            "manual Micronaut HTTP request test rewrite",
            "MockMvcRequestBuilders calls require a Micronaut HTTP client or embedded-server test rewrite.",
        ),
        (
            "catalog.code.mockmvc_result_matchers",
            "MockMvcResultMatchers",
            "manual Micronaut HTTP assertion rewrite",
            "MockMvcResultMatchers assertions should be rewritten using Micronaut HTTP test assertions.",
        ),
        (
            "catalog.code.result_actions",
            "ResultActions",
            "manual Micronaut HTTP assertion rewrite",
            "ResultActions chains are MockMvc-specific and need manual Micronaut HTTP assertion migration.",
        ),
        (
            "catalog.code.page_impl_import",
            "org.springframework.data.domain.PageImpl",
            "manual Micronaut Page fixture rewrite",
            "PageImpl fixtures should be rewritten to Micronaut paging factories or test-specific fixture helpers.",
        ),
        (
            "catalog.code.page_impl_simple",
            "PageImpl",
            "manual Micronaut Page fixture rewrite",
            "PageImpl fixtures should be rewritten to Micronaut paging factories or test-specific fixture helpers.",
        ),
        (
            "catalog.code.string_utils_import",
            "org.springframework.util.StringUtils",
            "io.micronaut.core.util.StringUtils",
            "Spring StringUtils helpers can usually migrate to Micronaut core StringUtils helpers for simple string checks.",
        ),
        (
            "catalog.code.string_utils_simple",
            "StringUtils",
            "io.micronaut.core.util.StringUtils",
            "Spring StringUtils helpers can usually migrate to Micronaut core StringUtils helpers for simple string checks.",
        ),
        (
            "catalog.code.media_type_import",
            "org.springframework.http.MediaType",
            "io.micronaut.http.MediaType",
            "Spring MediaType constants map to Micronaut HTTP MediaType constants.",
        ),
        (
            "catalog.code.media_type_simple",
            "MediaType",
            "io.micronaut.http.MediaType",
            "Spring MediaType constants map to Micronaut HTTP MediaType constants.",
        ),
        (
            "catalog.code.component_scan_import",
            "org.springframework.context.annotation.ComponentScan",
            "manual bean scanning and package registration review",
            "ComponentScan usage should be reviewed against Micronaut bean discovery and package scanning defaults.",
        ),
        (
            "catalog.code.filter_type_import",
            "org.springframework.context.annotation.FilterType",
            "manual bean scanning filter rewrite",
            "Spring FilterType scanning filters need manual review because Micronaut bean inclusion and exclusions differ.",
        ),
        (
            "catalog.code.application_context_import",
            "org.springframework.context.ApplicationContext",
            "manual Micronaut BeanContext or ApplicationContext rewrite",
            "Spring ApplicationContext usage should be rewritten against Micronaut BeanContext or ApplicationContext APIs case by case.",
        ),
        (
            "catalog.code.local_validator_factory_bean",
            "org.springframework.validation.beanvalidation.LocalValidatorFactoryBean",
            "manual Micronaut validator factory rewrite",
            "LocalValidatorFactoryBean wiring should be migrated to Micronaut validation configuration rather than directly ported.",
        ),
        (
            "catalog.code.locale_context_holder",
            "org.springframework.context.i18n.LocaleContextHolder",
            "manual Micronaut locale context rewrite",
            "LocaleContextHolder usage needs manual review against Micronaut request locale or locale resolver APIs.",
        ),
        (
            "catalog.code.runtime_hints",
            "org.springframework.aot.hint.RuntimeHints",
            "manual Micronaut native-image hint review",
            "Spring RuntimeHints APIs do not map one-to-one and should be reviewed against Micronaut AOT/native-image guidance.",
        ),
        (
            "catalog.code.runtime_hints_registrar",
            "org.springframework.aot.hint.RuntimeHintsRegistrar",
            "manual Micronaut native-image hint review",
            "RuntimeHintsRegistrar implementations require manual Micronaut native-image configuration review.",
        ),
        (
            "catalog.code.import_runtime_hints",
            "org.springframework.context.annotation.ImportRuntimeHints",
            "manual Micronaut native-image hint registration review",
            "ImportRuntimeHints should be reviewed against Micronaut AOT/native-image registration approaches.",
        ),
        (
            "catalog.code.import_runtime_hints_simple",
            "ImportRuntimeHints",
            "manual Micronaut native-image hint registration review",
            "ImportRuntimeHints should be reviewed against Micronaut AOT/native-image registration approaches.",
        ),
        (
            "catalog.code.date_time_format_import",
            "org.springframework.format.annotation.DateTimeFormat",
            "manual Micronaut format annotation review",
            "DateTimeFormat annotations should be reviewed against Micronaut conversion and formatting support.",
        ),
        (
            "catalog.code.date_time_format_simple",
            "DateTimeFormat",
            "manual Micronaut format annotation review",
            "DateTimeFormat annotations should be reviewed against Micronaut conversion and formatting support.",
        ),
        (
            "catalog.code.formatter_import",
            "org.springframework.format.Formatter",
            "manual Micronaut type converter rewrite",
            "Spring Formatter implementations should be migrated to Micronaut converters or formatting hooks explicitly.",
        ),
        (
            "catalog.code.validator_import",
            "org.springframework.validation.Validator",
            "manual Micronaut validator rewrite",
            "Spring Validator implementations should be rewritten against Micronaut validation or custom validator patterns.",
        ),
        (
            "catalog.code.errors_import",
            "org.springframework.validation.Errors",
            "manual Micronaut validation error model rewrite",
            "Spring Errors usage should be reviewed against Micronaut validation error handling.",
        ),
        (
            "catalog.code.data_access_exception",
            "org.springframework.dao.DataAccessException",
            "manual Micronaut data exception rewrite",
            "DataAccessException handling should be reviewed against Micronaut Data and persistence exceptions.",
        ),
        (
            "catalog.code_query_annotation",
            "org.springframework.data.jpa.repository.Query",
            "manual Micronaut repository query review",
            "Spring Data @Query repository methods should be reviewed against Micronaut Data query support.",
        ),
        (
            "catalog.code_param_annotation",
            "org.springframework.data.repository.query.Param",
            "manual Micronaut repository parameter binding review",
            "Spring Data @Param usage should be reviewed against Micronaut Data repository parameter binding.",
        ),
        (
            "catalog.code_paginated_query_group_by",
            'Page<T> repository method with @Query containing GROUP BY',
            "manual explicit Micronaut countQuery review",
            "Paginated Spring Data @Query methods with top-level GROUP BY need an explicit Micronaut countQuery review because automatic count derivation is not trusted.",
        ),
        (
            "catalog.code_paginated_query_having_or_union",
            'Page<T> repository method with @Query containing HAVING/UNION/INTERSECT/EXCEPT',
            "manual explicit Micronaut countQuery review",
            "Paginated Spring Data @Query methods using HAVING or set operators need a manual Micronaut countQuery because automatic count derivation is not trusted.",
        ),
        (
            "catalog.code_paginated_query_projection",
            'Page<T> repository method with @Query using constructor or multi-column projection',
            "manual explicit Micronaut countQuery review",
            "Paginated Spring Data @Query methods with constructor projections or multi-column select lists should not receive a guessed Micronaut countQuery and require manual review.",
        ),
        (
            "catalog.code_paginated_query_count_projection",
            'Page<T> repository method with @Query using countProjection',
            "manual explicit Micronaut countQuery review",
            "Paginated Spring Data @Query methods using Spring countProjection metadata need manual Micronaut countQuery review because that metadata is not promoted as a trusted direct rewrite.",
        ),
        (
            "catalog.code_paginated_query_count_name",
            'Page<T> repository method with @Query using countName',
            "manual explicit Micronaut countQuery review",
            "Paginated Spring Data @Query methods using Spring countName metadata need manual Micronaut countQuery review because named-query count conventions are not promoted as a trusted direct rewrite.",
        ),
        (
            "catalog.code_paginated_query_named_reference",
            'Page<T> repository method with @Query using name = "..."',
            "manual explicit Micronaut countQuery review",
            "Paginated Spring Data @Query methods using named-query references need manual Micronaut review before adding a trusted countQuery rewrite.",
        ),
        (
            "catalog.code_query_by_example_executor",
            "org.springframework.data.repository.query.QueryByExampleExecutor",
            "manual Micronaut repository query-by-example review",
            "Spring QueryByExampleExecutor usage should be reviewed manually against Micronaut Data query capabilities before promoting a direct rewrite.",
        ),
        (
            "catalog.code_query_by_example_example",
            "org.springframework.data.domain.Example",
            "manual Micronaut query-by-example review",
            "Spring Data Example usage should be reviewed manually before migrating repository query-by-example behavior.",
        ),
        (
            "catalog.code_query_by_example_matcher",
            "org.springframework.data.domain.ExampleMatcher",
            "manual Micronaut query-by-example review",
            "Spring Data ExampleMatcher usage should be reviewed manually before migrating repository query-by-example behavior.",
        ),
        (
            "catalog.code_assert_import",
            "org.springframework.util.Assert",
            "manual java.util.Objects or validation rewrite",
            "Spring Assert helpers should be rewritten to standard Java or project validation utilities.",
        ),
        (
            "catalog.code_serialization_utils",
            "org.springframework.util.SerializationUtils",
            "manual Java serialization helper review",
            "SerializationUtils usage should be reviewed against standard Java serialization or a project-specific serializer.",
        ),
        (
            "catalog.code_to_string_creator",
            "org.springframework.core.style.ToStringCreator",
            "manual toString rewrite",
            "ToStringCreator usage should be rewritten with plain Java toString logic or a project utility.",
        ),
        (
            "catalog.code_property_comparator",
            "org.springframework.beans.support.PropertyComparator",
            "manual sorting helper rewrite",
            "PropertyComparator usage should be rewritten with explicit Java comparators.",
        ),
        (
            "catalog.code_mutable_sort_definition",
            "org.springframework.beans.support.MutableSortDefinition",
            "manual sorting helper rewrite",
            "MutableSortDefinition usage should be rewritten with explicit Java comparators or sorting models.",
        ),
        (
            "catalog.code_marshalling_view",
            "org.springframework.web.servlet.view.xml.MarshallingView",
            "manual Micronaut XML view rewrite",
            "MarshallingView has no direct Micronaut equivalent and requires manual response/view redesign.",
        ),
        (
            "catalog.code_object_retrieval_failure_exception",
            "org.springframework.orm.ObjectRetrievalFailureException",
            "manual persistence exception rewrite",
            "ObjectRetrievalFailureException handling should be reviewed against Micronaut Data and ORM exception types.",
        ),
        (
            "catalog.code_bind_exception",
            "org.springframework.validation.BindException",
            "manual Micronaut binding error rewrite",
            "BindException handling should be reviewed against Micronaut binding and validation exception flows.",
        ),
        (
            "catalog.code_method_argument_not_valid_exception",
            "org.springframework.web.bind.MethodArgumentNotValidException",
            "manual Micronaut validation exception rewrite",
            "MethodArgumentNotValidException handling should be reviewed against Micronaut validation exception and error-body flows.",
        ),
        (
            "catalog.code_field_error",
            "org.springframework.validation.FieldError",
            "manual Micronaut validation field-error model rewrite",
            "FieldError usage should be reviewed against Micronaut constraint-violation and error response models.",
        ),
        (
            "catalog.code_object_error",
            "org.springframework.validation.ObjectError",
            "manual Micronaut validation object-error model rewrite",
            "ObjectError usage should be reviewed against Micronaut validation and structured error response handling.",
        ),
        (
            "catalog.code_redirect_attributes_import",
            "org.springframework.web.servlet.mvc.support.RedirectAttributes",
            "manual redirect attribute rewrite",
            "RedirectAttributes should be reviewed against Micronaut redirect response and flash/message-passing alternatives.",
        ),
        (
            "catalog.code_redirect_attributes_simple",
            "RedirectAttributes",
            "manual redirect attribute rewrite",
            "RedirectAttributes should be reviewed against Micronaut redirect response and flash/message-passing alternatives.",
        ),
        (
            "catalog.code_redirect_attributes_model_map",
            "org.springframework.web.servlet.mvc.support.RedirectAttributesModelMap",
            "manual redirect attribute model rewrite",
            "RedirectAttributesModelMap should be reviewed against Micronaut redirect response and model handling.",
        ),
        (
            "catalog.code_message_codes_resolver",
            "org.springframework.validation.MessageCodesResolver",
            "manual validation message code rewrite",
            "MessageCodesResolver customization should be reviewed against Micronaut validation message-resolution behavior.",
        ),
        (
            "catalog.code_default_message_codes_resolver",
            "org.springframework.validation.DefaultMessageCodesResolver",
            "manual validation message code rewrite",
            "DefaultMessageCodesResolver customization should be reviewed against Micronaut validation message-resolution behavior.",
        ),
        (
            "catalog.code_response_entity_exception_handler",
            "org.springframework.web.servlet.mvc.method.annotation.ResponseEntityExceptionHandler",
            "manual Micronaut global error handler rewrite",
            "ResponseEntityExceptionHandler subclasses should be reviewed against Micronaut @Error handlers or exception mappers.",
        ),
        (
            "catalog.code_method_argument_type_mismatch_exception",
            "org.springframework.web.method.annotation.MethodArgumentTypeMismatchException",
            "manual Micronaut type mismatch exception rewrite",
            "MethodArgumentTypeMismatchException handling should be reviewed against Micronaut conversion and binding failure flows.",
        ),
        (
            "catalog.code_mock_http_servlet_request_builder",
            "org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder",
            "manual Micronaut HTTP request builder rewrite",
            "MockHttpServletRequestBuilder usage should be reviewed against Micronaut HttpRequest creation and mutation APIs.",
        ),
        (
            "catalog.code_result_matcher",
            "org.springframework.test.web.servlet.ResultMatcher",
            "manual Micronaut HTTP assertion rewrite",
            "ResultMatcher implementations should be reviewed against direct JUnit assertions on Micronaut HttpResponse.",
        ),
        (
            "catalog.code_mvc_result",
            "org.springframework.test.web.servlet.MvcResult",
            "manual Micronaut HTTP response inspection rewrite",
            "MvcResult usage should be reviewed against Micronaut HttpResponse inspection and body decoding APIs.",
        ),
        (
            "catalog.code_mockmvc_result_matchers_status",
            "org.springframework.test.web.servlet.result.MockMvcResultMatchers.status",
            "manual Micronaut status assertion rewrite",
            "MockMvc status matcher chains should be rewritten against HttpResponse status assertions.",
        ),
        (
            "catalog.code_mockmvc_result_matchers_content",
            "org.springframework.test.web.servlet.result.MockMvcResultMatchers.content",
            "manual Micronaut content assertion rewrite",
            "MockMvc content matcher chains should be rewritten against HttpResponse body and content-type assertions.",
        ),
        (
            "catalog.code_mockmvc_result_matchers_jsonpath",
            "org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath",
            "manual Micronaut JSON assertion rewrite",
            "MockMvc jsonPath assertions should be reviewed against JSON parsing plus direct JUnit assertions.",
        ),
        (
            "catalog.code_mockmvc_result_matchers_model",
            "org.springframework.test.web.servlet.result.MockMvcResultMatchers.model",
            "manual Micronaut model assertion rewrite",
            "MockMvc model assertions are Spring MVC-specific and should be rewritten as HTTP or view-model assertions.",
        ),
        (
            "catalog.code_mockmvc_result_matchers_view",
            "org.springframework.test.web.servlet.result.MockMvcResultMatchers.view",
            "manual Micronaut view assertion rewrite",
            "MockMvc view assertions are Spring MVC-specific and should be rewritten for Micronaut views or endpoint contracts.",
        ),
        (
            "catalog.code_mockmvc_result_matchers_header",
            "org.springframework.test.web.servlet.result.MockMvcResultMatchers.header",
            "manual Micronaut header assertion rewrite",
            "MockMvc header assertions should be reviewed against direct Micronaut HttpResponse header assertions.",
        ),
        (
            "catalog.code_mockmvc_result_matchers_flash",
            "org.springframework.test.web.servlet.result.MockMvcResultMatchers.flash",
            "manual Micronaut flash attribute rewrite",
            "MockMvc flash assertions are Spring MVC-specific and should be reviewed against Micronaut redirect/message alternatives.",
        ),
        (
            "catalog.code_mockmvc_result_matchers_redirected_url",
            "org.springframework.test.web.servlet.result.MockMvcResultMatchers.redirectedUrl",
            "manual Micronaut redirect assertion rewrite",
            "MockMvc redirectedUrl assertions should be reviewed against direct Micronaut response status and Location-header assertions.",
        ),
        (
            "catalog.code_mockmvc_result_matchers_xpath",
            "org.springframework.test.web.servlet.result.MockMvcResultMatchers.xpath",
            "manual Micronaut XML assertion rewrite",
            "MockMvc XPath assertions should be reviewed against explicit XML parsing and assertions after migration.",
        ),
        (
            "catalog.code_mockmvc_result_matchers_request",
            "org.springframework.test.web.servlet.result.MockMvcResultMatchers.request",
            "manual Micronaut request assertion rewrite",
            "MockMvc request assertions are Spring MVC-specific and need explicit Micronaut test redesign.",
        ),
        (
            "catalog.code_mockmvc_result_matchers_handler",
            "org.springframework.test.web.servlet.result.MockMvcResultMatchers.handler",
            "manual Micronaut handler assertion rewrite",
            "MockMvc handler assertions are Spring MVC-specific and need explicit Micronaut test redesign.",
        ),
        (
            "catalog.code_security_mockmvc_request_post_processors",
            "org.springframework.security.test.web.servlet.request.SecurityMockMvcRequestPostProcessors",
            "manual Micronaut security test client rewrite",
            "Spring Security MockMvc request post-processors should be reviewed against Micronaut Security test support.",
        ),
        (
            "catalog.code_security_mockmvc_result_matchers",
            "org.springframework.security.test.web.servlet.response.SecurityMockMvcResultMatchers",
            "manual Micronaut security assertion rewrite",
            "Spring Security MockMvc result matchers should be reviewed against Micronaut Security test assertions.",
        ),
        (
            "catalog.code_response_status_exception",
            "org.springframework.web.server.ResponseStatusException",
            "manual Micronaut HttpStatusException or exception-handler rewrite",
            "ResponseStatusException handling should be reviewed against Micronaut HttpStatusException and @Error flows.",
        ),
        (
            "catalog.code_http_message_not_readable_exception",
            "org.springframework.http.converter.HttpMessageNotReadableException",
            "manual Micronaut request-body decode exception rewrite",
            "HttpMessageNotReadableException handling should be reviewed against Micronaut body-decoding and conversion failure flows.",
        ),
        (
            "catalog.code_missing_request_parameter_exception",
            "org.springframework.web.bind.MissingServletRequestParameterException",
            "manual Micronaut missing-parameter exception rewrite",
            "MissingServletRequestParameterException handling should be reviewed against Micronaut binding failure and validation flows.",
        ),
        (
            "catalog.code_conversion_service",
            "org.springframework.core.convert.ConversionService",
            "manual Micronaut conversion service rewrite",
            "Spring ConversionService usage should be reviewed against Micronaut conversion service APIs.",
        ),
        (
            "catalog.code_formatter_registry",
            "org.springframework.format.FormatterRegistry",
            "manual Micronaut converter registry rewrite",
            "FormatterRegistry customization should be reviewed against Micronaut conversion service registration.",
        ),
        (
            "catalog.code_formatting_conversion_service",
            "org.springframework.format.support.FormattingConversionService",
            "manual Micronaut conversion service rewrite",
            "FormattingConversionService wiring should be reviewed against Micronaut conversion service configuration.",
        ),
        (
            "catalog.code_converter",
            "org.springframework.core.convert.converter.Converter",
            "manual Micronaut TypeConverter rewrite",
            "Spring Converter implementations should be reviewed against Micronaut TypeConverter patterns.",
        ),
        (
            "catalog.code_generic_converter",
            "org.springframework.core.convert.converter.GenericConverter",
            "manual Micronaut TypeConverter rewrite",
            "GenericConverter implementations should be reviewed against Micronaut TypeConverter patterns.",
        ),
        (
            "catalog.code_converter_factory",
            "org.springframework.core.convert.converter.ConverterFactory",
            "manual Micronaut TypeConverter factory rewrite",
            "ConverterFactory implementations should be reviewed against Micronaut conversion registration patterns.",
        ),
        (
            "catalog.code_handler_method_argument_resolver",
            "org.springframework.web.method.support.HandlerMethodArgumentResolver",
            "manual Micronaut argument binder rewrite",
            "HandlerMethodArgumentResolver implementations should be rewritten as Micronaut request argument binders.",
        ),
        (
            "catalog.code_handler_method_return_value_handler",
            "org.springframework.web.method.support.HandlerMethodReturnValueHandler",
            "manual Micronaut response binding rewrite",
            "HandlerMethodReturnValueHandler implementations should be reviewed against Micronaut response body and route return handling.",
        ),
        (
            "catalog.code_handler_interceptor",
            "org.springframework.web.servlet.HandlerInterceptor",
            "manual Micronaut HttpServerFilter rewrite",
            "HandlerInterceptor implementations should be reviewed against Micronaut HttpServerFilter or route filters.",
        ),
        (
            "catalog.code_handler_interceptor_adapter",
            "org.springframework.web.servlet.handler.HandlerInterceptorAdapter",
            "manual Micronaut HttpServerFilter rewrite",
            "HandlerInterceptorAdapter usage should be reviewed against Micronaut server filters.",
        ),
        (
            "catalog.code_once_per_request_filter",
            "org.springframework.web.filter.OncePerRequestFilter",
            "manual Micronaut HttpServerFilter rewrite",
            "OncePerRequestFilter implementations should be reviewed against Micronaut HttpServerFilter patterns.",
        ),
        (
            "catalog.code_generic_filter_bean",
            "org.springframework.web.filter.GenericFilterBean",
            "manual Micronaut HttpServerFilter rewrite",
            "GenericFilterBean implementations should be reviewed against Micronaut HttpServerFilter patterns.",
        ),
        (
            "catalog.code_handler_exception_resolver",
            "org.springframework.web.servlet.HandlerExceptionResolver",
            "manual Micronaut exception-handler rewrite",
            "HandlerExceptionResolver implementations should be reviewed against Micronaut @Error handlers and exception mappers.",
        ),
        (
            "catalog.code_response_body_advice",
            "org.springframework.web.servlet.mvc.method.annotation.ResponseBodyAdvice",
            "manual Micronaut response-body advice rewrite",
            "ResponseBodyAdvice should be reviewed against Micronaut body writer customization or filters.",
        ),
        (
            "catalog.code_request_body_advice",
            "org.springframework.web.servlet.mvc.method.annotation.RequestBodyAdvice",
            "manual Micronaut request-body advice rewrite",
            "RequestBodyAdvice should be reviewed against Micronaut request-body readers, binders, or filters.",
        ),
        (
            "catalog.code_servlet_uri_components_builder",
            "org.springframework.web.servlet.support.ServletUriComponentsBuilder",
            "manual Micronaut URI builder rewrite",
            "ServletUriComponentsBuilder usage should be reviewed against Micronaut UriBuilder and request URI APIs.",
        ),
        (
            "catalog.code_uri_components_builder",
            "org.springframework.web.util.UriComponentsBuilder",
            "manual Micronaut UriBuilder rewrite",
            "UriComponentsBuilder usage should be reviewed against Micronaut UriBuilder APIs.",
        ),
        (
            "catalog.code_web_request",
            "org.springframework.web.context.request.WebRequest",
            "manual Micronaut HttpRequest rewrite",
            "WebRequest usage should be reviewed against Micronaut HttpRequest and request attribute APIs.",
        ),
        (
            "catalog.code_native_web_request",
            "org.springframework.web.context.request.NativeWebRequest",
            "manual Micronaut HttpRequest rewrite",
            "NativeWebRequest usage should be reviewed against Micronaut HttpRequest and request attribute APIs.",
        ),
        (
            "catalog.code_request_context_holder",
            "org.springframework.web.context.request.RequestContextHolder",
            "manual Micronaut request context rewrite",
            "RequestContextHolder usage should be reviewed against Micronaut request-scoped context access.",
        ),
        (
            "catalog.code_servlet_request_attributes",
            "org.springframework.web.context.request.ServletRequestAttributes",
            "manual Micronaut request attribute rewrite",
            "ServletRequestAttributes usage should be reviewed against Micronaut request attribute access patterns.",
        ),
        (
            "catalog.code_multipart_file",
            "org.springframework.web.multipart.MultipartFile",
            "manual Micronaut CompletedFileUpload rewrite",
            "MultipartFile usage should be reviewed against Micronaut CompletedFileUpload or StreamingFileUpload.",
        ),
        (
            "catalog.code_multipart_http_servlet_request",
            "org.springframework.web.multipart.MultipartHttpServletRequest",
            "manual Micronaut multipart request rewrite",
            "MultipartHttpServletRequest usage should be reviewed against Micronaut multipart request handling.",
        ),
        (
            "catalog.code_standard_multipart_http_servlet_request",
            "org.springframework.web.multipart.support.StandardMultipartHttpServletRequest",
            "manual Micronaut multipart request rewrite",
            "StandardMultipartHttpServletRequest usage should be reviewed against Micronaut multipart request handling.",
        ),
        (
            "catalog.code_pageable_handler_method_argument_resolver",
            "org.springframework.data.web.PageableHandlerMethodArgumentResolver",
            "manual Micronaut pageable binder rewrite",
            "PageableHandlerMethodArgumentResolver customization should be reviewed against Micronaut pageable binding behavior.",
        ),
        (
            "catalog.code_sort_handler_method_argument_resolver",
            "org.springframework.data.web.SortHandlerMethodArgumentResolver",
            "manual Micronaut sort binder rewrite",
            "SortHandlerMethodArgumentResolver customization should be reviewed against Micronaut sort binding behavior.",
        ),
        (
            "catalog.code_security_context_holder",
            "org.springframework.security.core.context.SecurityContextHolder",
            "manual Micronaut security context rewrite",
            "SecurityContextHolder usage should be reviewed against Micronaut Security context access patterns.",
        ),
        (
            "catalog.code_authentication",
            "org.springframework.security.core.Authentication",
            "manual Micronaut Authentication rewrite",
            "Spring Authentication usage should be reviewed against Micronaut Security Authentication APIs.",
        ),
        (
            "catalog.code_user_details",
            "org.springframework.security.core.userdetails.UserDetails",
            "manual Micronaut user principal rewrite",
            "UserDetails usage should be reviewed against Micronaut security principal or authentication models.",
        ),
        (
            "catalog.code_user_details_service",
            "org.springframework.security.core.userdetails.UserDetailsService",
            "manual Micronaut authentication provider rewrite",
            "UserDetailsService implementations should be reviewed against Micronaut authentication provider patterns.",
        ),
        (
            "catalog.code_password_encoder",
            "org.springframework.security.crypto.password.PasswordEncoder",
            "manual Micronaut password encoder rewrite",
            "PasswordEncoder usage should be reviewed against Micronaut Security password encoder support.",
        ),
        (
            "catalog.code_pre_authorize",
            "org.springframework.security.access.prepost.PreAuthorize",
            "manual Micronaut @Secured or rule-based authorization rewrite",
            "PreAuthorize expressions should be reviewed against Micronaut Security authorization semantics.",
        ),
        (
            "catalog.code_post_authorize",
            "org.springframework.security.access.prepost.PostAuthorize",
            "manual Micronaut authorization rewrite",
            "PostAuthorize expressions should be reviewed against Micronaut Security authorization semantics.",
        ),
        (
            "catalog.code_secured",
            "org.springframework.security.access.annotation.Secured",
            "manual Micronaut @Secured review",
            "Spring Secured usage should be reviewed against Micronaut @Secured semantics and role naming.",
        ),
        (
            "catalog.code_with_mock_user",
            "org.springframework.security.test.context.support.WithMockUser",
            "manual Micronaut security test user rewrite",
            "WithMockUser usage should be reviewed against Micronaut Security test support.",
        ),
        (
            "catalog.code_enable_method_security",
            "org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity",
            "manual Micronaut method security review",
            "EnableMethodSecurity should be reviewed against Micronaut Security method authorization configuration.",
        ),
        (
            "catalog.code_enable_global_method_security",
            "org.springframework.security.config.annotation.method.configuration.EnableGlobalMethodSecurity",
            "manual Micronaut method security review",
            "EnableGlobalMethodSecurity should be reviewed against Micronaut Security method authorization configuration.",
        ),
        (
            "catalog.code_enable_web_security",
            "org.springframework.security.config.annotation.web.configuration.EnableWebSecurity",
            "manual Micronaut security filter chain review",
            "EnableWebSecurity should be reviewed against Micronaut Security filter-chain and bean configuration.",
        ),
        (
            "catalog.code_security_filter_chain",
            "org.springframework.security.web.SecurityFilterChain",
            "manual Micronaut security rule chain rewrite",
            "SecurityFilterChain configuration should be reviewed against Micronaut Security rules and intercept URL map patterns.",
        ),
        (
            "catalog.code_web_security_customizer",
            "org.springframework.security.config.annotation.web.configuration.WebSecurityCustomizer",
            "manual Micronaut security ignore-path rewrite",
            "WebSecurityCustomizer should be reviewed against Micronaut security intercept URL exclusions or route config.",
        ),
        (
            "catalog.code_http_security",
            "org.springframework.security.config.annotation.web.builders.HttpSecurity",
            "manual Micronaut security configuration rewrite",
            "HttpSecurity DSL usage should be reviewed against Micronaut Security configuration and beans.",
        ),
        (
            "catalog.code_mvc_request_matcher",
            "org.springframework.security.web.servlet.util.matcher.MvcRequestMatcher",
            "manual Micronaut route security matcher rewrite",
            "MvcRequestMatcher usage should be reviewed against Micronaut route and security matcher configuration.",
        ),
        (
            "catalog.code_csrf_token",
            "org.springframework.security.web.csrf.CsrfToken",
            "manual Micronaut CSRF rewrite",
            "CsrfToken usage should be reviewed against Micronaut Security CSRF support.",
        ),
        (
            "catalog.code_application_event_publisher",
            "org.springframework.context.ApplicationEventPublisher",
            "manual Micronaut event publisher rewrite",
            "ApplicationEventPublisher usage should be reviewed against Micronaut event publisher APIs.",
        ),
        (
            "catalog.code_application_event",
            "org.springframework.context.ApplicationEvent",
            "manual Micronaut event model rewrite",
            "ApplicationEvent usage should be reviewed against Micronaut event model patterns.",
        ),
        (
            "catalog.code_smart_lifecycle",
            "org.springframework.context.SmartLifecycle",
            "manual Micronaut lifecycle hook rewrite",
            "SmartLifecycle implementations should be reviewed against Micronaut bean lifecycle events and startup hooks.",
        ),
        (
            "catalog.code_resource_loader",
            "org.springframework.core.io.ResourceLoader",
            "manual Micronaut resource resolver rewrite",
            "ResourceLoader usage should be reviewed against Micronaut ResourceResolver and resource loading APIs.",
        ),
        (
            "catalog.code_resource_pattern_resolver",
            "org.springframework.core.io.support.ResourcePatternResolver",
            "manual Micronaut resource pattern rewrite",
            "ResourcePatternResolver usage should be reviewed against Micronaut resource resolution patterns.",
        ),
        (
            "catalog.code_event_listener",
            "org.springframework.context.event.EventListener",
            "ApplicationEventListener review",
            "EventListener usage should be reviewed against Micronaut event listener semantics and threading behavior.",
        ),
        (
            "catalog.code_transactional_event_listener",
            "org.springframework.transaction.event.TransactionalEventListener",
            "manual Micronaut transactional event listener rewrite",
            "TransactionalEventListener usage should be reviewed against Micronaut transaction and event publication semantics.",
        ),
        (
            "catalog.code_application_listener",
            "org.springframework.context.ApplicationListener",
            "manual Micronaut ApplicationEventListener rewrite",
            "ApplicationListener implementations should be reviewed against Micronaut ApplicationEventListener patterns.",
        ),
        (
            "catalog.code_scheduled_annotation",
            "org.springframework.scheduling.annotation.Scheduled",
            "manual Micronaut @Scheduled review",
            "Spring Scheduled usage should be reviewed against Micronaut @Scheduled semantics for cron, fixed delay, and executor behavior.",
        ),
        (
            "catalog.code_async_annotation",
            "org.springframework.scheduling.annotation.Async",
            "manual Micronaut @Async review",
            "Spring Async usage should be reviewed against Micronaut @Async and executor configuration semantics.",
        ),
        (
            "catalog.code_async_task_executor",
            "org.springframework.core.task.AsyncTaskExecutor",
            "manual Micronaut ExecutorService rewrite",
            "AsyncTaskExecutor usage should be reviewed against Micronaut executor beans or named executors.",
        ),
        (
            "catalog.code_thread_pool_task_executor",
            "org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor",
            "manual Micronaut executor bean rewrite",
            "ThreadPoolTaskExecutor configuration should be reviewed against Micronaut executor bean configuration.",
        ),
        (
            "catalog.code_task_scheduler",
            "org.springframework.scheduling.TaskScheduler",
            "manual Micronaut scheduler rewrite",
            "TaskScheduler usage should be reviewed against Micronaut scheduling and executor configuration.",
        ),
        (
            "catalog.code_cache_manager",
            "org.springframework.cache.CacheManager",
            "manual Micronaut cache manager rewrite",
            "CacheManager usage should be reviewed against Micronaut cache manager/provider APIs.",
        ),
        (
            "catalog.code_cache",
            "org.springframework.cache.Cache",
            "manual Micronaut cache API rewrite",
            "Spring Cache API usage should be reviewed against Micronaut cache APIs and provider-specific behavior.",
        ),
        (
            "catalog.code_key_generator",
            "org.springframework.cache.interceptor.KeyGenerator",
            "manual Micronaut cache key rewrite",
            "Spring KeyGenerator implementations should be reviewed against Micronaut cache key and parameter semantics.",
        ),
        (
            "catalog.code_cache_resolver",
            "org.springframework.cache.interceptor.CacheResolver",
            "manual Micronaut cache resolution rewrite",
            "CacheResolver implementations should be reviewed against Micronaut cache resolution and provider configuration.",
        ),
        (
            "catalog.code_caffeine_cache_manager",
            "org.springframework.cache.caffeine.CaffeineCacheManager",
            "manual Micronaut Caffeine cache manager rewrite",
            "CaffeineCacheManager wiring should be reviewed against Micronaut cache-caffeine configuration.",
        ),
        (
            "catalog.code_enable_configuration_properties",
            "org.springframework.boot.context.properties.EnableConfigurationProperties",
            "manual Micronaut configuration properties registration review",
            "EnableConfigurationProperties should be reviewed against Micronaut configuration properties registration and scanning.",
        ),
        (
            "catalog.code_configuration_properties_scan",
            "org.springframework.boot.context.properties.ConfigurationPropertiesScan",
            "manual Micronaut configuration properties scanning review",
            "ConfigurationPropertiesScan should be reviewed against Micronaut configuration properties discovery semantics.",
        ),
        (
            "catalog.code_configuration_properties_binding",
            "org.springframework.boot.context.properties.ConfigurationPropertiesBinding",
            "manual Micronaut converter binding review",
            "ConfigurationPropertiesBinding converters should be reviewed against Micronaut configuration binding converters.",
        ),
        (
            "catalog.code_created_date",
            "org.springframework.data.annotation.CreatedDate",
            "manual Micronaut data auditing review",
            "CreatedDate auditing should be reviewed against Micronaut Data auditing support.",
        ),
        (
            "catalog.code_last_modified_date",
            "org.springframework.data.annotation.LastModifiedDate",
            "manual Micronaut data auditing review",
            "LastModifiedDate auditing should be reviewed against Micronaut Data auditing support.",
        ),
        (
            "catalog.code_created_by",
            "org.springframework.data.annotation.CreatedBy",
            "manual Micronaut data auditing review",
            "CreatedBy auditing should be reviewed against Micronaut Data auditing support.",
        ),
        (
            "catalog.code_last_modified_by",
            "org.springframework.data.annotation.LastModifiedBy",
            "manual Micronaut data auditing review",
            "LastModifiedBy auditing should be reviewed against Micronaut Data auditing support.",
        ),
        (
            "catalog.code_enable_jpa_auditing",
            "org.springframework.data.jpa.repository.config.EnableJpaAuditing",
            "manual Micronaut data auditing configuration review",
            "EnableJpaAuditing should be reviewed against Micronaut Data auditing configuration.",
        ),
        (
            "catalog.code_auditor_aware",
            "org.springframework.data.domain.AuditorAware",
            "manual Micronaut auditor provider rewrite",
            "AuditorAware implementations should be reviewed against Micronaut Data auditing providers.",
        ),
        (
            "catalog.code_pageable_default",
            "org.springframework.data.web.PageableDefault",
            "manual Micronaut pageable default binding review",
            "PageableDefault usage should be reviewed against Micronaut pageable binding and default value semantics.",
        ),
        (
            "catalog.code_sort_default",
            "org.springframework.data.web.SortDefault",
            "manual Micronaut sort default binding review",
            "SortDefault usage should be reviewed against Micronaut sort binding and default value semantics.",
        ),
        (
            "catalog.code_domain_class_converter",
            "org.springframework.data.repository.support.DomainClassConverter",
            "manual Micronaut entity binder rewrite",
            "DomainClassConverter usage should be reviewed against Micronaut entity lookup and request binder patterns.",
        ),
        (
            "catalog.code_rest_operations",
            "org.springframework.web.client.RestOperations",
            "manual Micronaut HttpClient contract rewrite",
            "RestOperations usage should be reviewed against Micronaut HttpClient APIs.",
        ),
        (
            "catalog.code_web_client",
            "org.springframework.web.reactive.function.client.WebClient",
            "manual Micronaut reactive HTTP client rewrite",
            "Spring WebClient usage should be reviewed against Micronaut HTTP client or reactive client APIs.",
        ),
        (
            "catalog.code_web_client_builder",
            "org.springframework.web.reactive.function.client.WebClient.Builder",
            "manual Micronaut reactive HTTP client builder rewrite",
            "Spring WebClient.Builder usage should be reviewed against Micronaut HTTP client configuration.",
        ),
        (
            "catalog.code_health_indicator",
            "org.springframework.boot.actuate.health.HealthIndicator",
            "manual Micronaut health indicator rewrite",
            "HealthIndicator implementations should be reviewed against Micronaut management/health indicator APIs.",
        ),
        (
            "catalog.code_info_contributor",
            "org.springframework.boot.actuate.info.InfoContributor",
            "manual Micronaut info endpoint rewrite",
            "InfoContributor implementations should be reviewed against Micronaut management/info endpoint patterns.",
        ),
        (
            "catalog.code_meter_registry",
            "io.micrometer.core.instrument.MeterRegistry",
            "manual Micronaut metrics registry review",
            "MeterRegistry usage should be reviewed against Micronaut Micrometer integration and metrics configuration.",
        ),
        (
            "catalog.code_kafka_listener",
            "org.springframework.kafka.annotation.KafkaListener",
            "manual Micronaut Kafka listener rewrite",
            "KafkaListener usage should be reviewed against Micronaut Kafka listener semantics, acking, and threading.",
        ),
        (
            "catalog.code_kafka_template",
            "org.springframework.kafka.core.KafkaTemplate",
            "manual Micronaut Kafka client rewrite",
            "KafkaTemplate usage should be reviewed against Micronaut Kafka client APIs.",
        ),
        (
            "catalog.code_rabbit_listener",
            "org.springframework.amqp.rabbit.annotation.RabbitListener",
            "manual Micronaut RabbitMQ listener rewrite",
            "RabbitListener usage should be reviewed against Micronaut RabbitMQ listener semantics and acking.",
        ),
        (
            "catalog.code_rabbit_template",
            "org.springframework.amqp.rabbit.core.RabbitTemplate",
            "manual Micronaut RabbitMQ client rewrite",
            "RabbitTemplate usage should be reviewed against Micronaut RabbitMQ client APIs.",
        ),
        (
            "catalog.code_jms_listener",
            "org.springframework.jms.annotation.JmsListener",
            "manual Micronaut JMS listener rewrite",
            "JmsListener usage should be reviewed against the target Micronaut messaging approach or JMS integration.",
        ),
        (
            "catalog.code_jms_template",
            "org.springframework.jms.core.JmsTemplate",
            "manual Micronaut JMS client rewrite",
            "JmsTemplate usage should be reviewed against the target Micronaut messaging approach or JMS integration.",
        ),
        (
            "catalog.code_enable_feign_clients",
            "org.springframework.cloud.openfeign.EnableFeignClients",
            "manual Micronaut declarative client review",
            "EnableFeignClients should be reviewed against Micronaut declarative HTTP client configuration and package scanning.",
        ),
        (
            "catalog.code_feign_client",
            "org.springframework.cloud.openfeign.FeignClient",
            "manual Micronaut declarative client rewrite",
            "FeignClient interfaces should be reviewed against Micronaut declarative @Client interfaces and error handling.",
        ),
        (
            "catalog.code_load_balanced",
            "org.springframework.cloud.client.loadbalancer.LoadBalanced",
            "manual Micronaut service discovery/load balancing rewrite",
            "LoadBalanced client wiring should be reviewed against Micronaut discovery client and load-balanced HTTP client configuration.",
        ),
        (
            "catalog.code_enable_retry",
            "org.springframework.retry.annotation.EnableRetry",
            "manual Micronaut retry configuration review",
            "EnableRetry should be reviewed against Micronaut retry support and interceptor configuration.",
        ),
        (
            "catalog.code_retryable",
            "org.springframework.retry.annotation.Retryable",
            "manual Micronaut @Retryable review",
            "Retryable usage should be reviewed against Micronaut retry semantics, includes/excludes, and backoff behavior.",
        ),
        (
            "catalog.code_recover",
            "org.springframework.retry.annotation.Recover",
            "manual Micronaut fallback/recovery rewrite",
            "Recover methods should be reviewed against Micronaut retry recovery and fallback patterns.",
        ),
        (
            "catalog.code_circuit_breaker",
            "io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker",
            "manual Micronaut circuit breaker review",
            "CircuitBreaker annotations should be reviewed against Micronaut retry/fault-tolerance or Resilience4j integration semantics.",
        ),
        (
            "catalog.code_bulkhead",
            "io.github.resilience4j.bulkhead.annotation.Bulkhead",
            "manual Micronaut bulkhead review",
            "Bulkhead annotations should be reviewed against Micronaut fault-tolerance and concurrency controls.",
        ),
        (
            "catalog.code_rate_limiter",
            "io.github.resilience4j.ratelimiter.annotation.RateLimiter",
            "manual Micronaut rate limiter review",
            "RateLimiter annotations should be reviewed against Micronaut or gateway-side rate limiting approaches.",
        ),
        (
            "catalog.code_time_limiter",
            "io.github.resilience4j.timelimiter.annotation.TimeLimiter",
            "manual Micronaut timeout review",
            "TimeLimiter usage should be reviewed against Micronaut timeout and reactive execution semantics.",
        ),
        (
            "catalog.code_enable_scheduling",
            "org.springframework.scheduling.annotation.EnableScheduling",
            "manual Micronaut scheduling enablement review",
            "EnableScheduling should be reviewed against Micronaut scheduler enablement and bean discovery semantics.",
        ),
        (
            "catalog.code_enable_async",
            "org.springframework.scheduling.annotation.EnableAsync",
            "manual Micronaut async enablement review",
            "EnableAsync should be reviewed against Micronaut async executor configuration and interception semantics.",
        ),
        (
            "catalog.code_job",
            "org.springframework.batch.core.Job",
            "manual Micronaut batch/job rewrite",
            "Spring Batch Job definitions should be reviewed against the target Micronaut batch or workflow approach.",
        ),
        (
            "catalog.code_step",
            "org.springframework.batch.core.Step",
            "manual Micronaut batch/step rewrite",
            "Spring Batch Step definitions should be reviewed against the target Micronaut batch or workflow approach.",
        ),
        (
            "catalog.code_job_launcher",
            "org.springframework.batch.core.launch.JobLauncher",
            "manual Micronaut batch launcher rewrite",
            "JobLauncher usage should be reviewed against the target Micronaut batch/job invocation approach.",
        ),
        (
            "catalog.code_job_parameters",
            "org.springframework.batch.core.JobParameters",
            "manual Micronaut batch parameter rewrite",
            "JobParameters usage should be reviewed against the target Micronaut batch/job parameter model.",
        ),
        (
            "catalog.code_job_execution_listener",
            "org.springframework.batch.core.JobExecutionListener",
            "manual Micronaut batch listener rewrite",
            "JobExecutionListener implementations should be reviewed against the target Micronaut batch/job lifecycle hooks.",
        ),
        (
            "catalog.code_step_execution_listener",
            "org.springframework.batch.core.StepExecutionListener",
            "manual Micronaut batch listener rewrite",
            "StepExecutionListener implementations should be reviewed against the target Micronaut batch/job lifecycle hooks.",
        ),
        (
            "catalog.code_item_reader",
            "org.springframework.batch.item.ItemReader",
            "manual Micronaut batch item reader rewrite",
            "ItemReader implementations should be reviewed against the target Micronaut batch processing approach.",
        ),
        (
            "catalog.code_item_processor",
            "org.springframework.batch.item.ItemProcessor",
            "manual Micronaut batch item processor rewrite",
            "ItemProcessor implementations should be reviewed against the target Micronaut batch processing approach.",
        ),
        (
            "catalog.code_item_writer",
            "org.springframework.batch.item.ItemWriter",
            "manual Micronaut batch item writer rewrite",
            "ItemWriter implementations should be reviewed against the target Micronaut batch processing approach.",
        ),
        (
            "catalog.code_enable_batch_processing",
            "org.springframework.batch.core.configuration.annotation.EnableBatchProcessing",
            "manual Micronaut batch enablement review",
            "EnableBatchProcessing should be reviewed against the target Micronaut batch/job framework configuration.",
        ),
        (
            "catalog.code_job_builder_factory",
            "org.springframework.batch.core.configuration.annotation.JobBuilderFactory",
            "manual Micronaut batch builder rewrite",
            "JobBuilderFactory usage should be reviewed against the target Micronaut batch/job construction approach.",
        ),
        (
            "catalog.code_step_builder_factory",
            "org.springframework.batch.core.configuration.annotation.StepBuilderFactory",
            "manual Micronaut batch builder rewrite",
            "StepBuilderFactory usage should be reviewed against the target Micronaut batch/job construction approach.",
        ),
        (
            "catalog.code_enable_transaction_management",
            "org.springframework.transaction.annotation.EnableTransactionManagement",
            "manual Micronaut transaction management review",
            "EnableTransactionManagement should be reviewed against Micronaut transaction management enablement and interception semantics.",
        ),
        (
            "catalog.code_transaction_template",
            "org.springframework.transaction.support.TransactionTemplate",
            "manual Micronaut transaction template rewrite",
            "TransactionTemplate usage should be reviewed against Micronaut transaction management or programmatic transaction APIs.",
        ),
        (
            "catalog.code_platform_transaction_manager",
            "org.springframework.transaction.PlatformTransactionManager",
            "manual Micronaut transaction manager rewrite",
            "PlatformTransactionManager usage should be reviewed against Micronaut transaction manager APIs.",
        ),
        (
            "catalog.code_oauth2_authorized_client_manager",
            "org.springframework.security.oauth2.client.OAuth2AuthorizedClientManager",
            "manual Micronaut OAuth2 client manager rewrite",
            "OAuth2AuthorizedClientManager usage should be reviewed against Micronaut Security OAuth2 client support.",
        ),
        (
            "catalog.code_oauth2_authorized_client_service",
            "org.springframework.security.oauth2.client.OAuth2AuthorizedClientService",
            "manual Micronaut OAuth2 client service rewrite",
            "OAuth2AuthorizedClientService usage should be reviewed against Micronaut Security OAuth2 client persistence and token management.",
        ),
        (
            "catalog.code_client_registration_repository",
            "org.springframework.security.oauth2.client.registration.ClientRegistrationRepository",
            "manual Micronaut OAuth2 client registration rewrite",
            "ClientRegistrationRepository usage should be reviewed against Micronaut Security OAuth2 client registration configuration.",
        ),
        (
            "catalog.code_enable_oauth2_client",
            "org.springframework.security.config.annotation.web.configuration.EnableOAuth2Client",
            "manual Micronaut OAuth2 client enablement review",
            "EnableOAuth2Client should be reviewed against Micronaut Security OAuth2 client configuration.",
        ),
        (
            "catalog.code_enable_oauth2_sso",
            "org.springframework.boot.autoconfigure.security.oauth2.client.EnableOAuth2Sso",
            "manual Micronaut OAuth2 login review",
            "EnableOAuth2Sso should be reviewed against Micronaut Security OAuth2 login configuration.",
        ),
        (
            "catalog.code_oauth2_login",
            "org.springframework.security.config.annotation.web.configurers.oauth2.client.OAuth2LoginConfigurer",
            "manual Micronaut OAuth2 login rewrite",
            "OAuth2LoginConfigurer usage should be reviewed against Micronaut Security OAuth2 login semantics.",
        ),
        (
            "catalog.code_oauth2_resource_server",
            "org.springframework.security.config.annotation.web.configurers.oauth2.server.resource.OAuth2ResourceServerConfigurer",
            "manual Micronaut OAuth2 resource server rewrite",
            "OAuth2 resource server configuration should be reviewed against Micronaut Security JWT/OAuth2 resource server support.",
        ),
        (
            "catalog.code_jwt_decoder",
            "org.springframework.security.oauth2.jwt.JwtDecoder",
            "manual Micronaut JWT decoder rewrite",
            "JwtDecoder usage should be reviewed against Micronaut Security JWT validation and signing configuration.",
        ),
        (
            "catalog.code_jwt_encoder",
            "org.springframework.security.oauth2.jwt.JwtEncoder",
            "manual Micronaut JWT encoder rewrite",
            "JwtEncoder usage should be reviewed against Micronaut Security token generation or external identity-provider integration.",
        ),
        (
            "catalog.code_jwt_claims_set",
            "org.springframework.security.oauth2.jwt.JwtClaimsSet",
            "manual Micronaut JWT claims rewrite",
            "JwtClaimsSet usage should be reviewed against Micronaut Security JWT claims handling.",
        ),
        (
            "catalog.code_bearer_token_authentication_filter",
            "org.springframework.security.oauth2.server.resource.web.BearerTokenAuthenticationFilter",
            "manual Micronaut bearer token filter rewrite",
            "BearerTokenAuthenticationFilter usage should be reviewed against Micronaut Security bearer token filter configuration.",
        ),
        (
            "catalog.code_specification_where_and",
            "Specification.where(...).and(...)",
            "manual Micronaut criteria composition review",
            "Spring Data JPA Specification.where(...).and(...) chains only migrate safely at the import level; review Micronaut criteria composition semantics manually.",
        ),
        (
            "catalog.code_specification_where_or",
            "Specification.where(...).or(...)",
            "manual Micronaut criteria composition review",
            "Spring Data JPA Specification.where(...).or(...) chains only migrate safely at the import level; review Micronaut criteria composition semantics manually.",
        ),
    ]

    deterministic_entries = [
        (
            "catalog.code.spring_boot_test_import",
            "org.springframework.boot.test.context.SpringBootTest",
            "@MicronautTest",
            "SpringBootTest generally maps to MicronautTest for Micronaut-managed integration tests.",
        ),
        (
            "catalog.code.spring_boot_test_annotation",
            "SpringBootTest",
            "@MicronautTest",
            "SpringBootTest generally maps to MicronautTest for Micronaut-managed integration tests.",
        ),
        (
            "catalog.code.web_environment_simple",
            "WebEnvironment",
            "@MicronautTest",
            "SpringBootTest.WebEnvironment should be collapsed into MicronautTest plus explicit client or embedded-server test wiring.",
        ),
        (
            "catalog.code.request_entity_import",
            "org.springframework.http.RequestEntity",
            "io.micronaut.http.HttpRequest",
            "Spring RequestEntity generally maps to Micronaut HttpRequest for direct request construction.",
        ),
        (
            "catalog.code.request_entity_simple",
            "RequestEntity",
            "HttpRequest",
            "Spring RequestEntity generally maps to Micronaut HttpRequest for direct request construction.",
        ),
        (
            "catalog.code.rest_template_import",
            "org.springframework.web.client.RestTemplate",
            "io.micronaut.http.client.HttpClient",
            "Spring RestTemplate generally maps to Micronaut HttpClient for direct HTTP client calls.",
        ),
        (
            "catalog.code.rest_template_simple",
            "RestTemplate",
            "HttpClient",
            "Spring RestTemplate generally maps to Micronaut HttpClient for direct HTTP client calls.",
        ),
        (
            "catalog.code.rest_template_builder_import",
            "org.springframework.boot.web.client.RestTemplateBuilder",
            "@Client + HttpClient injection",
            "RestTemplateBuilder-based tests and simple client wiring should migrate to Micronaut Client-driven HttpClient injection.",
        ),
        (
            "catalog.code.rest_template_builder_simple",
            "RestTemplateBuilder",
            "@Client + HttpClient injection",
            "RestTemplateBuilder-based tests and simple client wiring should migrate to Micronaut Client-driven HttpClient injection.",
        ),
        (
            "catalog.code_mockmvc_request_builders_get",
            "org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get",
            "HttpRequest.GET",
            "Simple MockMvc GET builders can migrate to Micronaut HttpRequest.GET in supported HTTP-test rewrites.",
        ),
        (
            "catalog.code_mockmvc_request_builders_post",
            "org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post",
            "HttpRequest.POST",
            "Simple MockMvc POST builders can migrate to Micronaut HttpRequest.POST in supported HTTP-test rewrites.",
        ),
        (
            "catalog.code_mockmvc_request_builders_put",
            "org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put",
            "HttpRequest.PUT",
            "Simple MockMvc PUT builders can migrate to Micronaut HttpRequest.PUT in supported HTTP-test rewrites.",
        ),
        (
            "catalog.code_mockmvc_request_builders_delete",
            "org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete",
            "HttpRequest.DELETE",
            "Simple MockMvc DELETE builders can migrate to Micronaut HttpRequest.DELETE in supported HTTP-test rewrites.",
        ),
        (
            "catalog.code_mockmvc_request_builders_patch",
            "org.springframework.test.web.servlet.request.MockMvcRequestBuilders.patch",
            "HttpRequest.PATCH",
            "Simple MockMvc PATCH builders can migrate to Micronaut HttpRequest.PATCH in supported HTTP-test rewrites.",
        ),
        (
            "catalog.code_mockmvc_request_builders_simple",
            "MockMvcRequestBuilders",
            "HttpRequest factory methods",
            "Simple MockMvcRequestBuilders usage can migrate to Micronaut HttpRequest factory methods in supported HTTP-test rewrites.",
        ),
        (
            "catalog.code_mockmvc_status_ok",
            "status().isOk()",
            "HttpStatus.OK assertion",
            "Supported MockMvc status assertions can migrate to direct Micronaut HttpResponse status checks.",
        ),
        (
            "catalog.code_mockmvc_status_created",
            "status().isCreated()",
            "HttpStatus.CREATED assertion",
            "Supported MockMvc status assertions can migrate to direct Micronaut HttpResponse status checks.",
        ),
        (
            "catalog.code_mockmvc_status_no_content",
            "status().isNoContent()",
            "HttpStatus.NO_CONTENT assertion",
            "Supported MockMvc status assertions can migrate to direct Micronaut HttpResponse status checks.",
        ),
        (
            "catalog.code_mockmvc_status_bad_request",
            "status().isBadRequest()",
            "HttpStatus.BAD_REQUEST assertion",
            "Supported MockMvc status assertions can migrate to direct Micronaut HttpResponse status checks.",
        ),
        (
            "catalog.code_mockmvc_status_unauthorized",
            "status().isUnauthorized()",
            "HttpStatus.UNAUTHORIZED assertion",
            "Supported MockMvc status assertions can migrate to direct Micronaut HttpResponse status checks.",
        ),
        (
            "catalog.code_mockmvc_status_forbidden",
            "status().isForbidden()",
            "HttpStatus.FORBIDDEN assertion",
            "Supported MockMvc status assertions can migrate to direct Micronaut HttpResponse status checks.",
        ),
        (
            "catalog.code_mockmvc_status_not_found",
            "status().isNotFound()",
            "HttpStatus.NOT_FOUND assertion",
            "Supported MockMvc status assertions can migrate to direct Micronaut HttpResponse status checks.",
        ),
        (
            "catalog.code_mockmvc_status_conflict",
            "status().isConflict()",
            "HttpStatus.CONFLICT assertion",
            "Supported MockMvc status assertions can migrate to direct Micronaut HttpResponse status checks.",
        ),
        (
            "catalog.code_mockmvc_status_internal_server_error",
            "status().isInternalServerError()",
            "HttpStatus.INTERNAL_SERVER_ERROR assertion",
            "Supported MockMvc status assertions can migrate to direct Micronaut HttpResponse status checks.",
        ),
        (
            "catalog.code_mockmvc_content_type",
            "content().contentType(...)",
            "HttpResponse contentType assertion",
            "Supported MockMvc content-type assertions can migrate to direct Micronaut HttpResponse content-type checks.",
        ),
        (
            "catalog.code_mockmvc_content_type_compatible",
            "content().contentTypeCompatibleWith(...)",
            "HttpResponse contentType assertion",
            "Supported MockMvc content-type assertions can migrate to direct Micronaut HttpResponse content-type checks.",
        ),
        (
            "catalog.code_mockmvc_content_string",
            "content().string(...)",
            "HttpResponse body assertion",
            "Supported MockMvc body-string assertions can migrate to direct Micronaut HttpResponse body checks.",
        ),
        (
            "catalog.code_redirect_view",
            "org.springframework.web.servlet.view.RedirectView",
            "HttpResponse.redirect(...)",
            "Simple RedirectView usage can often migrate to Micronaut redirect responses.",
        ),
        (
            "catalog.code_http_status_simple",
            "org.springframework.http.HttpStatus",
            "io.micronaut.http.HttpStatus",
            "Spring HttpStatus constants generally map directly to Micronaut HttpStatus constants.",
        ),
        (
            "catalog.code_response_entity_simple",
            "ResponseEntity",
            "HttpResponse",
            "Simple ResponseEntity usage generally maps to Micronaut HttpResponse.",
        ),
        (
            "catalog.code_request_param_map",
            "org.springframework.util.MultiValueMap",
            "io.micronaut.core.convert.value.MutableConvertibleValuesMap review",
            "Simple Spring MultiValueMap request parameter usage may migrate to Micronaut convertible values or request parameter maps depending on context.",
        ),
        (
            "catalog.code_completed_file_upload_target",
            "MultipartFile",
            "CompletedFileUpload",
            "MultipartFile arguments commonly migrate to Micronaut CompletedFileUpload for completed upload handling.",
        ),
        (
            "catalog.code_type_converter_target",
            "Converter",
            "TypeConverter",
            "Simple Spring Converter implementations commonly migrate to Micronaut TypeConverter.",
        ),
        (
            "catalog.code_http_server_filter_target",
            "HandlerInterceptor",
            "HttpServerFilter",
            "Simple request interception patterns commonly migrate from Spring HandlerInterceptor to Micronaut HttpServerFilter.",
        ),
        (
            "catalog.code_security_secured_target",
            "Secured",
            "@Secured review",
            "Spring security annotations often migrate to Micronaut @Secured with semantic review.",
        ),
        (
            "catalog.code_event_listener_target",
            "EventListener",
            "ApplicationEventListener",
            "Simple Spring event listeners commonly migrate to Micronaut ApplicationEventListener patterns.",
        ),
        (
            "catalog.code_application_event_publisher_target",
            "ApplicationEventPublisher",
            "ApplicationEventPublisher",
            "Simple Spring application event publication often maps to Micronaut event publication APIs with semantic review.",
        ),
        (
            "catalog.code_scheduled_target",
            "Scheduled",
            "@Scheduled review",
            "Simple Spring scheduled methods commonly migrate to Micronaut @Scheduled with semantic review.",
        ),
        (
            "catalog.code_async_target",
            "Async",
            "@Async review",
            "Simple Spring async methods commonly migrate to Micronaut @Async with semantic review.",
        ),
        (
            "catalog.code_cacheable_target",
            "Cacheable",
            "@Cacheable review",
            "Simple cache annotations commonly migrate to Micronaut cache annotations with semantic review.",
        ),
        (
            "catalog.code_configuration_properties_target",
            "ConfigurationProperties",
            "@ConfigurationProperties",
            "Spring configuration properties commonly migrate to Micronaut @ConfigurationProperties.",
        ),
        (
            "catalog.code_health_indicator_target",
            "HealthIndicator",
            "Micronaut health indicator review",
            "Simple Spring health indicators often migrate to Micronaut management health indicators with semantic review.",
        ),
        (
            "catalog.code_kafka_listener_target",
            "KafkaListener",
            "Micronaut @KafkaListener review",
            "Simple Spring Kafka listeners often migrate to Micronaut Kafka listeners with semantic review.",
        ),
        (
            "catalog.code_rabbit_listener_target",
            "RabbitListener",
            "Micronaut Rabbit listener review",
            "Simple Spring Rabbit listeners often migrate to Micronaut Rabbit listeners with semantic review.",
        ),
        (
            "catalog.code_retryable_target",
            "Retryable",
            "@Retryable review",
            "Simple Spring retry annotations often migrate to Micronaut retry support with semantic review.",
        ),
        (
            "catalog.code_feign_client_target",
            "FeignClient",
            "@Client review",
            "Simple Feign client interfaces often migrate to Micronaut declarative @Client interfaces with semantic review.",
        ),
        (
            "catalog.code_pre_authorize_target",
            "PreAuthorize",
            "@Secured/expression review",
            "Spring method-authorization annotations often migrate to Micronaut authorization rules with semantic review.",
        ),
        (
            "catalog.code_oauth2_resource_server_target",
            "OAuth2ResourceServerConfigurer",
            "Micronaut JWT/OAuth2 resource server review",
            "Spring OAuth2 resource server configuration often migrates to Micronaut JWT/OAuth2 support with semantic review.",
        ),
        (
            "catalog.code_jwt_decoder_target",
            "JwtDecoder",
            "Micronaut JWT validation review",
            "Spring JwtDecoder usage often migrates to Micronaut JWT validation support with semantic review.",
        ),
        (
            "catalog.code_transactional_target",
            "Transactional",
            "@Transactional",
            "Spring transactional annotations commonly migrate directly to Micronaut/Jakarta transactional annotations.",
        ),
        (
            "catalog.code_paginated_query_shorthand_page",
            'Page<T> repository method with @Query("...")',
            '@Query(value = "...", countQuery = "...") review',
            "Paginated Spring Data repository methods using shorthand @Query can migrate to Micronaut @Query with an explicit countQuery when the source query is safely countable.",
        ),
        (
            "catalog.code_paginated_query_order_by",
            'Page<T> repository method with @Query(value = "... ORDER BY ...")',
            '@Query(value = "...", countQuery = "...") with ORDER BY removed from countQuery',
            "Paginated Spring Data @Query methods with a top-level ORDER BY can migrate deterministically when the generated Micronaut countQuery removes the ORDER BY clause.",
        ),
        (
            "catalog.code_paginated_query_simple_native",
            'Page<T> repository method with @Query(value = "select * from ...", nativeQuery = true)',
            '@Query(value = "...", nativeQuery = true, countQuery = "SELECT count(*) FROM ...")',
            "Simple native paginated Spring Data @Query methods can migrate to Micronaut @Query when a direct count(*) query can be derived safely.",
        ),
        (
            "catalog.code_slice_repository_contract",
            "org.springframework.data.domain.Slice",
            "io.micronaut.data.model.Slice",
            "Spring Data Slice repository contracts can migrate directly to Micronaut Data Slice while preserving slice semantics without total-count synthesis.",
        ),
        (
            "catalog.code_query_hints_annotation",
            "org.springframework.data.jpa.repository.QueryHints",
            "io.micronaut.data.annotation.QueryHints",
            "Spring Data JPA QueryHints annotations can migrate directly to Micronaut Data QueryHints annotations.",
        ),
        (
            "catalog.code_jpa_specification_executor",
            "org.springframework.data.jpa.repository.JpaSpecificationExecutor",
            "io.micronaut.data.jpa.repository.JpaSpecificationExecutor",
            "Spring Data JPA JpaSpecificationExecutor imports and repository contracts can migrate directly to Micronaut Data JpaSpecificationExecutor, while complex specification semantics may still need review.",
        ),
        (
            "catalog.code_specification",
            "org.springframework.data.jpa.domain.Specification",
            "io.micronaut.data.jpa.repository.criteria.Specification",
            "Spring Data JPA Specification imports can migrate directly to Micronaut Data Specification, while complex criteria semantics may still need review.",
        ),
        (
            "catalog.code_entity_graph",
            "org.springframework.data.jpa.repository.EntityGraph",
            "io.micronaut.data.jpa.annotation.EntityGraph",
            "Spring Data JPA EntityGraph imports can migrate directly to Micronaut Data EntityGraph, while advanced fetch-plan semantics may still need review.",
        ),
        (
            "catalog.code_query_hint_import",
            "jakarta.persistence.QueryHint",
            "io.micronaut.data.annotation.QueryHint",
            "JPA QueryHint imports used in Spring Data repositories can migrate directly to Micronaut Data QueryHint annotations.",
        ),
        (
            "catalog.code_query_hint_import_javax",
            "javax.persistence.QueryHint",
            "io.micronaut.data.annotation.QueryHint",
            "Legacy javax.persistence.QueryHint imports used in Spring Data repositories can migrate directly to Micronaut Data QueryHint annotations.",
        ),
        (
            "catalog.code_entity_graph_type_import",
            "org.springframework.data.jpa.repository.EntityGraph.EntityGraphType",
            "io.micronaut.data.jpa.annotation.EntityGraph.Type",
            "Spring Data JPA EntityGraphType imports can migrate directly to Micronaut Data EntityGraph.Type imports.",
        ),
        (
            "catalog.code_entity_graph_attribute_paths",
            '@EntityGraph(attributePaths = {"..."}, type = EntityGraphType.FETCH)',
            '@EntityGraph(attributePaths = {"..."}, type = Type.FETCH)',
            "Spring Data JPA EntityGraph attributePaths declarations with explicit graph type can migrate directly to Micronaut Data EntityGraph syntax.",
        ),
        (
            "catalog.code_entity_graph_named_value",
            '@EntityGraph(value = "...", type = EntityGraphType.LOAD)',
            '@EntityGraph(value = "...", type = Type.LOAD)',
            "Spring Data JPA named EntityGraph declarations with explicit graph type can migrate directly to Micronaut Data EntityGraph syntax.",
        ),
        (
            "catalog.code_event_listener_annotation_target",
            "org.springframework.context.event.EventListener",
            "ApplicationEventListener review",
            "Spring event listener annotations often migrate to Micronaut application event listeners with semantic review.",
        ),
    ]

    patterns: List[VersionedPattern] = []
    for pattern_id, spring_pattern, micronaut_pattern, description in reviewed_entries:
        patterns.append(
            VersionedPattern(
                pattern_id=pattern_id,
                pattern_type=PatternType.CODE_PATTERN,
                spring_pattern=spring_pattern,
                micronaut_pattern=micronaut_pattern,
                description=description,
                spring_versions=VersionWindow(spec="3.x"),
                micronaut_versions=VersionWindow(spec="4.x"),
                status=ValidationStatus.VALIDATED,
                confidence=0.84,
                complexity="high",
                category="code_patterns",
                source_kind=SourceKind.MANUAL,
                evidence=[
                    PatternEvidence(
                        source_kind=SourceKind.MANUAL,
                        source_ref=_REVIEWED_CODE_SOURCE_REF,
                        title="Reviewed migration gap catalog",
                        notes="Manual-reviewed enterprise guardrail for unsupported or redesign-heavy Spring APIs.",
                    )
                ],
                metadata={
                    "catalog_entry": True,
                    "target_status": "manual_redesign",
                    "automated_migration_supported": False,
                    "replacement_version_management": "manual",
                },
            )
        )
    for pattern_id, spring_pattern, micronaut_pattern, description in deterministic_entries:
        patterns.append(
            VersionedPattern(
                pattern_id=pattern_id,
                pattern_type=PatternType.CODE_PATTERN,
                spring_pattern=spring_pattern,
                micronaut_pattern=micronaut_pattern,
                description=description,
                spring_versions=VersionWindow(spec="3.x"),
                micronaut_versions=VersionWindow(spec="4.x"),
                status=ValidationStatus.VALIDATED,
                confidence=0.9,
                complexity="medium",
                category="code_patterns",
                source_kind=SourceKind.MANUAL,
                evidence=[
                    PatternEvidence(
                        source_kind=SourceKind.MANUAL,
                        source_ref=_REVIEWED_CODE_SOURCE_REF,
                        title="Reviewed migration gap catalog",
                        notes="Manual-reviewed enterprise mapping for common deterministic Spring test/client symbols.",
                    )
                ],
                metadata={
                    "catalog_entry": True,
                    "target_status": "direct_rewrite",
                    "automated_migration_supported": True,
                    "replacement_version_management": "governed",
                },
            )
        )
    return patterns


def write_curated_catalog_patterns(corpus_root: str = "corpus") -> Dict[str, object]:
    repository = PatternCorpusRepository(root=corpus_root)
    repository.initialize_layout()

    patterns = curated_catalog_patterns()
    target_root = Path(corpus_root) / "validated_patterns" / "release" / "catalog"
    target_root.mkdir(parents=True, exist_ok=True)

    pattern_paths: List[str] = []
    for pattern in patterns:
        pattern_path = target_root / f"{pattern.pattern_id}.json"
        pattern_path.write_text(json.dumps(pattern.to_dict(), indent=2), encoding="utf-8")
        pattern_paths.append(str(pattern_path))

    index_payload = {
        "schema_version": 1,
        "catalog_type": "curated_dependency_catalog",
        "pattern_count": len(patterns),
        "patterns": [pattern.to_dict() for pattern in patterns],
    }
    index_path = target_root / "index.json"
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    return {
        "pattern_count": len(patterns),
        "index_path": str(index_path),
        "pattern_paths": pattern_paths,
    }


def main():
    parser = argparse.ArgumentParser(description="Materialize reviewed dependency catalog patterns into the validated release workspace")
    parser.add_argument("--corpus-root", default="corpus", help="Corpus root directory")
    parser.add_argument("--write", action="store_true", help="Write curated catalog patterns")
    args = parser.parse_args()

    if args.write:
        print(json.dumps(write_curated_catalog_patterns(corpus_root=args.corpus_root), indent=2, sort_keys=True))
        return

    print(json.dumps({"message": "Use --write to materialize curated catalog patterns."}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
