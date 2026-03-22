import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from src.agent.patterns.fixture_registry import write_fixture_registry
from src.agent.patterns.repository import PatternCorpusRepository


@dataclass(frozen=True)
class FixtureSourceFile:
    filename: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class FixturePack:
    pack_id: str
    title: str
    priority: str
    status: str
    covered_pattern_ids: List[str]
    goals: List[str]
    source_files: List[FixtureSourceFile]

    def to_dict(self) -> Dict[str, object]:
        return {
            "pack_id": self.pack_id,
            "title": self.title,
            "priority": self.priority,
            "status": self.status,
            "covered_pattern_ids": self.covered_pattern_ids,
            "goals": self.goals,
            "source_files": [source_file.to_dict() for source_file in self.source_files],
        }


DEFAULT_FIXTURE_PACKS: List[FixturePack] = [
    FixturePack(
        pack_id="controller_error_flow",
        title="Controller Error Flow",
        priority="high",
        status="seeded",
        covered_pattern_ids=["legacy_promoted.annotation.exceptionhandler"],
        goals=[
            "verify mapped exception handlers preserve HTTP status codes",
            "verify error payload shape remains stable after migration",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringControllerErrorFlow.java",
                content=(
                    "package fixtures.errorflow;\n\n"
                    "import org.springframework.http.HttpStatus;\n"
                    "import org.springframework.web.bind.annotation.ExceptionHandler;\n"
                    "import org.springframework.web.bind.annotation.GetMapping;\n"
                    "import org.springframework.web.bind.annotation.ResponseStatus;\n"
                    "import org.springframework.web.bind.annotation.RestController;\n\n"
                    "@RestController\n"
                    "class SpringControllerErrorFlow {\n"
                    "    @GetMapping(\"/orders/fail\")\n"
                    "    String fail() {\n"
                    "        throw new IllegalStateException(\"boom\");\n"
                    "    }\n\n"
                    "    @ExceptionHandler(IllegalStateException.class)\n"
                    "    @ResponseStatus(HttpStatus.BAD_REQUEST)\n"
                    "    String onIllegalState(IllegalStateException ex) {\n"
                    "        return \"bad-request:\" + ex.getMessage();\n"
                    "    }\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautControllerErrorFlow.java",
                content=(
                    "package fixtures.errorflow;\n\n"
                    "import io.micronaut.http.HttpRequest;\n"
                    "import io.micronaut.http.HttpResponse;\n"
                    "import io.micronaut.http.annotation.Controller;\n"
                    "import io.micronaut.http.annotation.Error;\n"
                    "import io.micronaut.http.annotation.Get;\n\n"
                    "@Controller(\"/orders\")\n"
                    "class MicronautControllerErrorFlow {\n"
                    "    @Get(\"/fail\")\n"
                    "    String fail() {\n"
                    "        throw new IllegalStateException(\"boom\");\n"
                    "    }\n\n"
                    "    @Error(exception = IllegalStateException.class)\n"
                    "    HttpResponse<String> onIllegalState(HttpRequest<?> request, IllegalStateException ex) {\n"
                    "        return HttpResponse.badRequest(\"bad-request:\" + ex.getMessage());\n"
                    "    }\n"
                    "}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="global_error_advice",
        title="Global Error Advice",
        priority="high",
        status="seeded",
        covered_pattern_ids=["legacy_promoted.annotation.controlleradvice"],
        goals=[
            "verify global advice is applied across controllers",
            "verify fallback error contract is preserved",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringGlobalAdvice.java",
                content=(
                    "package fixtures.advice;\n\n"
                    "import org.springframework.http.HttpStatus;\n"
                    "import org.springframework.web.bind.annotation.ControllerAdvice;\n"
                    "import org.springframework.web.bind.annotation.ExceptionHandler;\n"
                    "import org.springframework.web.bind.annotation.ResponseBody;\n"
                    "import org.springframework.web.bind.annotation.ResponseStatus;\n\n"
                    "@ControllerAdvice\n"
                    "class SpringGlobalAdvice {\n"
                    "    @ExceptionHandler(IllegalArgumentException.class)\n"
                    "    @ResponseStatus(HttpStatus.BAD_REQUEST)\n"
                    "    @ResponseBody\n"
                    "    String onBadArgument(IllegalArgumentException ex) {\n"
                    "        return \"global:\" + ex.getMessage();\n"
                    "    }\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautGlobalAdvice.java",
                content=(
                    "package fixtures.advice;\n\n"
                    "import io.micronaut.http.HttpRequest;\n"
                    "import io.micronaut.http.HttpResponse;\n"
                    "import io.micronaut.http.annotation.Error;\n"
                    "import jakarta.inject.Singleton;\n\n"
                    "@Singleton\n"
                    "class MicronautGlobalAdvice {\n"
                    "    @Error(global = true, exception = IllegalArgumentException.class)\n"
                    "    HttpResponse<String> onBadArgument(HttpRequest<?> request, IllegalArgumentException ex) {\n"
                    "        return HttpResponse.badRequest(\"global:\" + ex.getMessage());\n"
                    "    }\n"
                    "}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="server_filter_chain",
        title="Server Filter Chain",
        priority="high",
        status="seeded",
        covered_pattern_ids=["legacy_promoted.type.filterchain"],
        goals=[
            "verify filter order and request mutation are preserved",
            "verify response mutation survives migration to Micronaut filter API",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringFilterChainFixture.java",
                content=(
                    "package fixtures.filtering;\n\n"
                    "import jakarta.servlet.FilterChain;\n"
                    "import jakarta.servlet.ServletException;\n"
                    "import jakarta.servlet.http.HttpServletRequest;\n"
                    "import jakarta.servlet.http.HttpServletResponse;\n"
                    "import java.io.IOException;\n"
                    "import org.springframework.web.filter.OncePerRequestFilter;\n\n"
                    "class SpringFilterChainFixture extends OncePerRequestFilter {\n"
                    "    @Override\n"
                    "    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)\n"
                    "            throws ServletException, IOException {\n"
                    "        response.addHeader(\"X-Trace\", \"spring\");\n"
                    "        filterChain.doFilter(request, response);\n"
                    "    }\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautFilterChainFixture.java",
                content=(
                    "package fixtures.filtering;\n\n"
                    "import io.micronaut.http.HttpRequest;\n"
                    "import io.micronaut.http.MutableHttpResponse;\n"
                    "import io.micronaut.http.annotation.Filter;\n"
                    "import io.micronaut.http.filter.FilterChain;\n"
                    "import io.micronaut.http.filter.HttpServerFilter;\n"
                    "import org.reactivestreams.Publisher;\n"
                    "import reactor.core.publisher.Flux;\n\n"
                    "@Filter(\"/**\")\n"
                    "class MicronautFilterChainFixture implements HttpServerFilter {\n"
                    "    @Override\n"
                    "    public Publisher<MutableHttpResponse<?>> doFilter(HttpRequest<?> request, FilterChain chain) {\n"
                    "        return Flux.from(chain.proceed(request))\n"
                    "            .doOnNext(response -> response.getHeaders().add(\"X-Trace\", \"micronaut\"));\n"
                    "    }\n"
                    "}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="reactive_endpoint",
        title="Reactive Endpoint",
        priority="high",
        status="seeded",
        covered_pattern_ids=["legacy_promoted.type.mono_flux_webflux"],
        goals=[
            "verify publisher-based endpoint emits expected payloads",
            "verify completion semantics remain acceptable after migration",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringReactiveEndpointFixture.java",
                content=(
                    "package fixtures.reactive;\n\n"
                    "import org.springframework.web.bind.annotation.GetMapping;\n"
                    "import org.springframework.web.bind.annotation.RestController;\n"
                    "import reactor.core.publisher.Flux;\n"
                    "import reactor.core.publisher.Mono;\n\n"
                    "@RestController\n"
                    "class SpringReactiveEndpointFixture {\n"
                    "    @GetMapping(\"/reactive/one\")\n"
                    "    Mono<String> one() {\n"
                    "        return Mono.just(\"one\");\n"
                    "    }\n\n"
                    "    @GetMapping(\"/reactive/many\")\n"
                    "    Flux<String> many() {\n"
                    "        return Flux.just(\"one\", \"two\");\n"
                    "    }\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautReactiveEndpointFixture.java",
                content=(
                    "package fixtures.reactive;\n\n"
                    "import io.micronaut.http.annotation.Controller;\n"
                    "import io.micronaut.http.annotation.Get;\n"
                    "import org.reactivestreams.Publisher;\n"
                    "import reactor.core.publisher.Flux;\n"
                    "import reactor.core.publisher.Mono;\n\n"
                    "@Controller(\"/reactive\")\n"
                    "class MicronautReactiveEndpointFixture {\n"
                    "    @Get(\"/one\")\n"
                    "    Publisher<String> one() {\n"
                    "        return Mono.just(\"one\");\n"
                    "    }\n\n"
                    "    @Get(\"/many\")\n"
                    "    Publisher<String> many() {\n"
                    "        return Flux.just(\"one\", \"two\");\n"
                    "    }\n"
                    "}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="model_attribute_binding",
        title="Model Attribute Binding",
        priority="high",
        status="seeded",
        covered_pattern_ids=["legacy_promoted.annotation.modelattribute"],
        goals=[
            "verify request binding semantics are preserved for form-like payloads",
            "verify validation and binding errors remain visible after migration",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringModelAttributeFixture.java",
                content=(
                    "package fixtures.binding;\n\n"
                    "import org.springframework.web.bind.annotation.ModelAttribute;\n"
                    "import org.springframework.web.bind.annotation.PostMapping;\n"
                    "import org.springframework.web.bind.annotation.RestController;\n\n"
                    "@RestController\n"
                    "class SpringModelAttributeFixture {\n"
                    "    @PostMapping(\"/profiles\")\n"
                    "    String create(@ModelAttribute ProfileCommand command) {\n"
                    "        return command.name() + \":\" + command.region();\n"
                    "    }\n"
                    "}\n\n"
                    "record ProfileCommand(String name, String region) {}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautModelAttributeFixture.java",
                content=(
                    "package fixtures.binding;\n\n"
                    "import io.micronaut.http.annotation.Body;\n"
                    "import io.micronaut.http.annotation.Controller;\n"
                    "import io.micronaut.http.annotation.Post;\n\n"
                    "@Controller(\"/profiles\")\n"
                    "class MicronautModelAttributeFixture {\n"
                    "    @Post\n"
                    "    String create(@Body ProfileCommand command) {\n"
                    "        return command.name() + \":\" + command.region();\n"
                    "    }\n"
                    "}\n\n"
                    "record ProfileCommand(String name, String region) {}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="cache_enablement",
        title="Cache Enablement",
        priority="high",
        status="seeded",
        covered_pattern_ids=["legacy_promoted.annotation.enablecaching"],
        goals=[
            "verify cache-enabled methods are migrated with an explicit runtime requirement",
            "verify repeated calls use the same cached value semantics",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringCacheFixture.java",
                content=(
                    "package fixtures.cache;\n\n"
                    "import org.springframework.cache.annotation.Cacheable;\n"
                    "import org.springframework.cache.annotation.EnableCaching;\n"
                    "import org.springframework.context.annotation.Configuration;\n\n"
                    "@Configuration\n"
                    "@EnableCaching\n"
                    "class SpringCacheFixture {\n"
                    "    @Cacheable(\"catalog\")\n"
                    "    String lookup(String sku) {\n"
                    "        return \"value:\" + sku;\n"
                    "    }\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautCacheFixture.java",
                content=(
                    "package fixtures.cache;\n\n"
                    "import io.micronaut.context.annotation.Requires;\n"
                    "import io.micronaut.cache.annotation.Cacheable;\n"
                    "import jakarta.inject.Singleton;\n"
                    "import javax.cache.CacheManager;\n\n"
                    "@Singleton\n"
                    "@Requires(beans = CacheManager.class)\n"
                    "class MicronautCacheFixture {\n"
                    "    @Cacheable(\"catalog\")\n"
                    "    String lookup(String sku) {\n"
                    "        return \"value:\" + sku;\n"
                    "    }\n"
                    "}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="cache_starter_runtime",
        title="Cache Starter Runtime",
        priority="high",
        status="seeded",
        covered_pattern_ids=["catalog.dependency.org_springframework_boot_spring_boot_starter_cache"],
        goals=[
            "verify cache starter migration keeps an explicit provider-backed runtime requirement",
            "verify migrated cache-backed methods retain visible caching intent after dependency migration",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringCacheStarterFixture.java",
                content=(
                    "package fixtures.cache.dependency;\n\n"
                    "import org.springframework.cache.annotation.Cacheable;\n"
                    "import org.springframework.context.annotation.Configuration;\n\n"
                    "@Configuration\n"
                    "class SpringCacheStarterFixture {\n"
                    "    @Cacheable(\"catalog\")\n"
                    "    String lookup(String sku) {\n"
                    "        return \"value:\" + sku;\n"
                    "    }\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautCacheStarterFixture.java",
                content=(
                    "package fixtures.cache.dependency;\n\n"
                    "import io.micronaut.cache.annotation.Cacheable;\n"
                    "import io.micronaut.context.annotation.Requires;\n"
                    "import jakarta.inject.Singleton;\n"
                    "import javax.cache.CacheManager;\n\n"
                    "@Singleton\n"
                    "@Requires(beans = CacheManager.class)\n"
                    "class MicronautCacheStarterFixture {\n"
                    "    @Cacheable(\"catalog\")\n"
                    "    String lookup(String sku) {\n"
                    "        return \"value:\" + sku;\n"
                    "    }\n"
                    "}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="ehcache_provider_runtime",
        title="Ehcache Provider Runtime",
        priority="high",
        status="seeded",
        covered_pattern_ids=["catalog.dependency.org_ehcache_ehcache"],
        goals=[
            "verify direct Ehcache migration keeps provider-specific runtime wiring explicit",
            "verify the migrated target still communicates dependence on external Ehcache configuration",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringEhcacheProviderFixture.java",
                content=(
                    "package fixtures.cache.ehcache;\n\n"
                    "import org.springframework.context.annotation.Configuration;\n\n"
                    "@Configuration\n"
                    "class SpringEhcacheProviderFixture {\n"
                    "    String configLocation() {\n"
                    "        return \"classpath:ehcache.xml\";\n"
                    "    }\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautEhcacheProviderFixture.java",
                content=(
                    "package fixtures.cache.ehcache;\n\n"
                    "import io.micronaut.context.annotation.Factory;\n"
                    "import io.micronaut.context.annotation.Requires;\n"
                    "import jakarta.inject.Singleton;\n"
                    "import javax.cache.CacheManager;\n\n"
                    "@Factory\n"
                    "class MicronautEhcacheProviderFixture {\n"
                    "    @Singleton\n"
                    "    @Requires(beans = CacheManager.class)\n"
                    "    String configLocation() {\n"
                    "        return \"classpath:ehcache.xml\";\n"
                    "    }\n"
                    "}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="scheduled_task_boot",
        title="Scheduled Task Boot",
        priority="high",
        status="seeded",
        covered_pattern_ids=["legacy_promoted.annotation.enablescheduling"],
        goals=[
            "verify scheduled jobs remain registered after migration",
            "verify cron/fixed-delay semantics remain visible in the migrated form",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringSchedulingFixture.java",
                content=(
                    "package fixtures.scheduling;\n\n"
                    "import org.springframework.scheduling.annotation.EnableScheduling;\n"
                    "import org.springframework.scheduling.annotation.Scheduled;\n"
                    "import org.springframework.stereotype.Component;\n\n"
                    "@Component\n"
                    "@EnableScheduling\n"
                    "class SpringSchedulingFixture {\n"
                    "    @Scheduled(fixedDelay = 5000)\n"
                    "    void refresh() {\n"
                    "    }\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautSchedulingFixture.java",
                content=(
                    "package fixtures.scheduling;\n\n"
                    "import io.micronaut.context.annotation.Requires;\n"
                    "import io.micronaut.scheduling.TaskScheduler;\n"
                    "import io.micronaut.scheduling.annotation.Scheduled;\n"
                    "import jakarta.inject.Singleton;\n\n"
                    "@Singleton\n"
                    "@Requires(beans = TaskScheduler.class)\n"
                    "class MicronautSchedulingFixture {\n"
                    "    @Scheduled(fixedDelay = \"5s\")\n"
                    "    void refresh() {\n"
                    "    }\n"
                    "}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="async_execution",
        title="Async Execution",
        priority="high",
        status="seeded",
        covered_pattern_ids=["legacy_promoted.annotation.enableasync"],
        goals=[
            "verify async methods remain asynchronous after migration",
            "verify executor-backed execution is represented explicitly",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringAsyncFixture.java",
                content=(
                    "package fixtures.asyncwork;\n\n"
                    "import java.util.concurrent.CompletableFuture;\n"
                    "import org.springframework.scheduling.annotation.Async;\n"
                    "import org.springframework.scheduling.annotation.EnableAsync;\n"
                    "import org.springframework.stereotype.Service;\n\n"
                    "@Service\n"
                    "@EnableAsync\n"
                    "class SpringAsyncFixture {\n"
                    "    @Async\n"
                    "    CompletableFuture<String> process() {\n"
                    "        return CompletableFuture.completedFuture(\"done\");\n"
                    "    }\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautAsyncFixture.java",
                content=(
                    "package fixtures.asyncwork;\n\n"
                    "import io.micronaut.context.annotation.Requires;\n"
                    "import io.micronaut.scheduling.annotation.Async;\n"
                    "import jakarta.inject.Singleton;\n"
                    "import java.util.concurrent.CompletableFuture;\n"
                    "import java.util.concurrent.ExecutorService;\n\n"
                    "@Singleton\n"
                    "@Requires(beans = ExecutorService.class)\n"
                    "class MicronautAsyncFixture {\n"
                    "    @Async\n"
                    "    CompletableFuture<String> process() {\n"
                    "        return CompletableFuture.completedFuture(\"done\");\n"
                    "    }\n"
                    "}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="jpa_repository_bootstrap",
        title="JPA Repository Bootstrap",
        priority="high",
        status="seeded",
        covered_pattern_ids=["legacy_promoted.annotation.enablejparepositories"],
        goals=[
            "verify repository beans are created after migration",
            "verify a representative repository interaction remains possible",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringJpaRepositoryFixture.java",
                content=(
                    "package fixtures.jpa;\n\n"
                    "import org.springframework.data.jpa.repository.JpaRepository;\n"
                    "import org.springframework.data.jpa.repository.config.EnableJpaRepositories;\n"
                    "import org.springframework.stereotype.Repository;\n\n"
                    "@EnableJpaRepositories\n"
                    "class SpringJpaRepositoryFixture {\n"
                    "}\n\n"
                    "@Repository\n"
                    "interface BookRepository extends JpaRepository<BookEntity, Long> {\n"
                    "}\n\n"
                    "class BookEntity {\n"
                    "    Long id;\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautJpaRepositoryFixture.java",
                content=(
                    "package fixtures.jpa;\n\n"
                    "import io.micronaut.context.annotation.Requires;\n"
                    "import io.micronaut.data.annotation.Repository;\n"
                    "import io.micronaut.data.jpa.repository.JpaRepository;\n"
                    "import jakarta.inject.Singleton;\n"
                    "import jakarta.persistence.EntityManagerFactory;\n\n"
                    "@Singleton\n"
                    "@Requires(beans = EntityManagerFactory.class)\n"
                    "class MicronautJpaRepositoryFixture {\n"
                    "}\n\n"
                    "@Repository\n"
                    "interface BookRepository extends JpaRepository<BookEntity, Long> {\n"
                    "}\n\n"
                    "class BookEntity {\n"
                    "    Long id;\n"
                    "}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="jpa_auditing",
        title="JPA Auditing",
        priority="high",
        status="seeded",
        covered_pattern_ids=["legacy_promoted.annotation.enablejpaauditing"],
        goals=[
            "verify audit fields remain populated after persistence operations",
            "verify auditing bootstrap is represented explicitly in the migrated form",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringJpaAuditingFixture.java",
                content=(
                    "package fixtures.jpaaudit;\n\n"
                    "import java.time.Instant;\n"
                    "import org.springframework.data.annotation.CreatedDate;\n"
                    "import org.springframework.data.jpa.repository.config.EnableJpaAuditing;\n\n"
                    "@EnableJpaAuditing\n"
                    "class SpringJpaAuditingFixture {\n"
                    "}\n\n"
                    "class AuditEntity {\n"
                    "    @CreatedDate\n"
                    "    Instant createdAt;\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautJpaAuditingFixture.java",
                content=(
                    "package fixtures.jpaaudit;\n\n"
                    "import io.micronaut.context.annotation.Requires;\n"
                    "import java.time.Instant;\n"
                    "import jakarta.persistence.EntityManagerFactory;\n"
                    "import jakarta.persistence.PrePersist;\n"
                    "import jakarta.inject.Singleton;\n\n"
                    "@Singleton\n"
                    "@Requires(beans = EntityManagerFactory.class)\n"
                    "class MicronautJpaAuditingFixture {\n"
                    "}\n\n"
                    "class AuditEntity {\n"
                    "    Instant createdAt;\n\n"
                    "    @PrePersist\n"
                    "    void markCreated() {\n"
                    "        createdAt = Instant.now();\n"
                    "    }\n"
                    "}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="framework_toggle",
        title="Framework Toggle",
        priority="medium",
        status="seeded",
        covered_pattern_ids=["legacy_promoted.annotation.enablewebmvc"],
        goals=[
            "verify the migrated app boots without Spring MVC enablement annotations",
            "verify representative routes still resolve through Micronaut controllers",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringEnableWebMvcFixture.java",
                content=(
                    "package fixtures.frameworktoggle;\n\n"
                    "import org.springframework.context.annotation.Configuration;\n"
                    "import org.springframework.web.bind.annotation.GetMapping;\n"
                    "import org.springframework.web.bind.annotation.RestController;\n"
                    "import org.springframework.web.servlet.config.annotation.EnableWebMvc;\n\n"
                    "@Configuration\n"
                    "@EnableWebMvc\n"
                    "class SpringEnableWebMvcFixture {\n"
                    "}\n\n"
                    "@RestController\n"
                    "class SpringToggleController {\n"
                    "    @GetMapping(\"/healthz\")\n"
                    "    String health() {\n"
                    "        return \"ok\";\n"
                    "    }\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautEnableWebMvcFixture.java",
                content=(
                    "package fixtures.frameworktoggle;\n\n"
                    "import io.micronaut.http.annotation.Controller;\n"
                    "import io.micronaut.http.annotation.Get;\n"
                    "import jakarta.inject.Singleton;\n\n"
                    "@Singleton\n"
                    "class MicronautEnableWebMvcFixture {\n"
                    "}\n\n"
                    "@Controller\n"
                    "class MicronautToggleController {\n"
                    "    @Get(\"/healthz\")\n"
                    "    String health() {\n"
                    "        return \"ok\";\n"
                    "    }\n"
                    "}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="http_client_configuration",
        title="HTTP Client Configuration",
        priority="medium",
        status="seeded",
        covered_pattern_ids=["legacy_promoted.configuration.resttemplate_configuration"],
        goals=[
            "verify a migrated HTTP client bean is created with explicit configuration",
            "verify outbound behavior can be represented through a Micronaut client interface",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringRestTemplateConfigurationFixture.java",
                content=(
                    "package fixtures.httpclient;\n\n"
                    "import org.springframework.boot.web.client.RestTemplateBuilder;\n"
                    "import org.springframework.context.annotation.Bean;\n"
                    "import org.springframework.context.annotation.Configuration;\n"
                    "import org.springframework.web.client.RestTemplate;\n\n"
                    "@Configuration\n"
                    "class SpringRestTemplateConfigurationFixture {\n"
                    "    @Bean\n"
                    "    RestTemplate restTemplate(RestTemplateBuilder builder) {\n"
                    "        return builder.rootUri(\"https://inventory.internal\").build();\n"
                    "    }\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautHttpClientConfigurationFixture.java",
                content=(
                    "package fixtures.httpclient;\n\n"
                    "import io.micronaut.context.annotation.Factory;\n"
                    "import io.micronaut.http.annotation.Get;\n"
                    "import io.micronaut.http.client.annotation.Client;\n\n"
                    "@Factory\n"
                    "class MicronautHttpClientConfigurationFixture {\n"
                    "}\n\n"
                    "@Client(\"https://inventory.internal\")\n"
                    "interface InventoryClient {\n"
                    "    @Get(\"/items\")\n"
                    "    String items();\n"
                    "}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="response_wrapper_endpoint",
        title="Response Wrapper Endpoint",
        priority="medium",
        status="seeded",
        covered_pattern_ids=["legacy_promoted.type.optional_responseentity"],
        goals=[
            "verify optional responses still produce the intended HTTP contract",
            "verify empty results preserve explicit not-found handling",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringOptionalResponseEntityFixture.java",
                content=(
                    "package fixtures.responsewrapper;\n\n"
                    "import java.util.Optional;\n"
                    "import org.springframework.http.ResponseEntity;\n"
                    "import org.springframework.web.bind.annotation.GetMapping;\n"
                    "import org.springframework.web.bind.annotation.PathVariable;\n"
                    "import org.springframework.web.bind.annotation.RestController;\n\n"
                    "@RestController\n"
                    "class SpringOptionalResponseEntityFixture {\n"
                    "    @GetMapping(\"/books/{id}\")\n"
                    "    ResponseEntity<String> find(@PathVariable String id) {\n"
                    "        return Optional.ofNullable(\"1\".equals(id) ? \"book-1\" : null)\n"
                    "            .map(ResponseEntity::ok)\n"
                    "            .orElseGet(() -> ResponseEntity.notFound().build());\n"
                    "    }\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautOptionalResponseFixture.java",
                content=(
                    "package fixtures.responsewrapper;\n\n"
                    "import io.micronaut.http.HttpResponse;\n"
                    "import io.micronaut.http.annotation.Controller;\n"
                    "import io.micronaut.http.annotation.Get;\n\n"
                    "@Controller(\"/books\")\n"
                    "class MicronautOptionalResponseFixture {\n"
                    "    @Get(\"/{id}\")\n"
                    "    HttpResponse<String> find(String id) {\n"
                    "        if (\"1\".equals(id)) {\n"
                    "            return HttpResponse.ok(\"book-1\");\n"
                    "        }\n"
                    "        return HttpResponse.notFound();\n"
                    "    }\n"
                    "}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="mvc_customization",
        title="MVC Customization",
        priority="medium",
        status="seeded",
        covered_pattern_ids=["legacy_promoted.type.webmvcconfigurer"],
        goals=[
            "verify custom route or converter configuration is represented explicitly after migration",
            "verify framework customization moves into Micronaut-supported extension points",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringWebMvcConfigurerFixture.java",
                content=(
                    "package fixtures.mvccustomization;\n\n"
                    "import org.springframework.context.annotation.Configuration;\n"
                    "import org.springframework.format.FormatterRegistry;\n"
                    "import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;\n\n"
                    "@Configuration\n"
                    "class SpringWebMvcConfigurerFixture implements WebMvcConfigurer {\n"
                    "    @Override\n"
                    "    public void addFormatters(FormatterRegistry registry) {\n"
                    "        registry.addConverter(String.class, Region.class, Region::new);\n"
                    "    }\n"
                    "}\n\n"
                    "record Region(String value) {}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautTypeConverterFixture.java",
                content=(
                    "package fixtures.mvccustomization;\n\n"
                    "import io.micronaut.core.convert.ConversionContext;\n"
                    "import io.micronaut.core.convert.TypeConverter;\n"
                    "import jakarta.inject.Singleton;\n"
                    "import java.util.Optional;\n\n"
                    "@Singleton\n"
                    "class MicronautTypeConverterFixture implements TypeConverter<String, Region> {\n"
                    "    @Override\n"
                    "    public Optional<Region> convert(String object, Class<Region> targetType, ConversionContext context) {\n"
                    "        return Optional.of(new Region(object));\n"
                    "    }\n"
                    "}\n\n"
                    "record Region(String value) {}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="application_startup",
        title="Application Startup",
        priority="medium",
        status="seeded",
        covered_pattern_ids=["legacy_promoted.code_pattern.commandlinerunner"],
        goals=[
            "verify startup hooks still run once during application boot",
            "verify migration makes lifecycle intent explicit in Micronaut form",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringCommandLineRunnerFixture.java",
                content=(
                    "package fixtures.startup;\n\n"
                    "import org.springframework.boot.CommandLineRunner;\n"
                    "import org.springframework.stereotype.Component;\n\n"
                    "@Component\n"
                    "class SpringCommandLineRunnerFixture implements CommandLineRunner {\n"
                    "    @Override\n"
                    "    public void run(String... args) {\n"
                    "        System.out.println(\"startup\");\n"
                    "    }\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautStartupEventFixture.java",
                content=(
                    "package fixtures.startup;\n\n"
                    "import io.micronaut.context.event.ApplicationEventListener;\n"
                    "import io.micronaut.runtime.server.event.ServerStartupEvent;\n"
                    "import jakarta.inject.Singleton;\n\n"
                    "@Singleton\n"
                    "class MicronautStartupEventFixture implements ApplicationEventListener<ServerStartupEvent> {\n"
                    "    @Override\n"
                    "    public void onApplicationEvent(ServerStartupEvent event) {\n"
                    "        System.out.println(\"startup\");\n"
                    "    }\n"
                    "}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="application_event_listener",
        title="Application Event Listener",
        priority="medium",
        status="seeded",
        covered_pattern_ids=["legacy_promoted.code_pattern.applicationlistener"],
        goals=[
            "verify event listener migration preserves the target event subscription",
            "verify listener behavior remains explicit in Micronaut event form",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringApplicationListenerFixture.java",
                content=(
                    "package fixtures.events;\n\n"
                    "import org.springframework.context.ApplicationListener;\n"
                    "import org.springframework.context.event.ContextRefreshedEvent;\n"
                    "import org.springframework.stereotype.Component;\n\n"
                    "@Component\n"
                    "class SpringApplicationListenerFixture implements ApplicationListener<ContextRefreshedEvent> {\n"
                    "    @Override\n"
                    "    public void onApplicationEvent(ContextRefreshedEvent event) {\n"
                    "        System.out.println(event.getApplicationContext().getId());\n"
                    "    }\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautApplicationListenerFixture.java",
                content=(
                    "package fixtures.events;\n\n"
                    "import io.micronaut.context.event.ApplicationEventListener;\n"
                    "import io.micronaut.runtime.event.ApplicationStartupEvent;\n"
                    "import jakarta.inject.Singleton;\n\n"
                    "@Singleton\n"
                    "class MicronautApplicationListenerFixture implements ApplicationEventListener<ApplicationStartupEvent> {\n"
                    "    @Override\n"
                    "    public void onApplicationEvent(ApplicationStartupEvent event) {\n"
                    "        System.out.println(event.getSource().getClass().getSimpleName());\n"
                    "    }\n"
                    "}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="security_authorization_flow",
        title="Security Authorization Flow",
        priority="high",
        status="seeded",
        covered_pattern_ids=["catalog.dependency.org_springframework_boot_spring_boot_starter_security"],
        goals=[
            "verify secured endpoints keep explicit authorization intent after migration",
            "verify the Micronaut target shape makes security rules visible and reviewable",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringSecurityAuthorizationFixture.java",
                content=(
                    "package fixtures.security;\n\n"
                    "import org.springframework.security.access.annotation.Secured;\n"
                    "import org.springframework.web.bind.annotation.GetMapping;\n"
                    "import org.springframework.web.bind.annotation.RestController;\n\n"
                    "@RestController\n"
                    "class SpringSecurityAuthorizationFixture {\n"
                    "    @Secured(\"ROLE_ADMIN\")\n"
                    "    @GetMapping(\"/admin\")\n"
                    "    String admin() {\n"
                    "        return \"admin\";\n"
                    "    }\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautSecurityAuthorizationFixture.java",
                content=(
                    "package fixtures.security;\n\n"
                    "import io.micronaut.http.annotation.Controller;\n"
                    "import io.micronaut.http.annotation.Get;\n"
                    "import io.micronaut.security.annotation.Secured;\n"
                    "import io.micronaut.security.rules.SecurityRule;\n\n"
                    "@Controller(\"/admin\")\n"
                    "class MicronautSecurityAuthorizationFixture {\n"
                    "    @Secured(SecurityRule.IS_AUTHENTICATED)\n"
                    "    @Get\n"
                    "    String admin() {\n"
                    "        return \"admin\";\n"
                    "    }\n"
                    "}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="validation_contract",
        title="Validation Contract",
        priority="high",
        status="seeded",
        covered_pattern_ids=["catalog.dependency.org_springframework_boot_spring_boot_starter_validation"],
        goals=[
            "verify request validation remains explicit after migration",
            "verify invalid payload handling still has a concrete contract to test",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringValidationContractFixture.java",
                content=(
                    "package fixtures.validation;\n\n"
                    "import jakarta.validation.Valid;\n"
                    "import jakarta.validation.constraints.NotBlank;\n"
                    "import org.springframework.validation.annotation.Validated;\n"
                    "import org.springframework.web.bind.annotation.PostMapping;\n"
                    "import org.springframework.web.bind.annotation.RestController;\n\n"
                    "@RestController\n"
                    "@Validated\n"
                    "class SpringValidationContractFixture {\n"
                    "    @PostMapping(\"/customers\")\n"
                    "    String create(@Valid CustomerCommand command) {\n"
                    "        return command.name();\n"
                    "    }\n"
                    "}\n\n"
                    "record CustomerCommand(@NotBlank String name) {}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautValidationContractFixture.java",
                content=(
                    "package fixtures.validation;\n\n"
                    "import io.micronaut.http.annotation.Body;\n"
                    "import io.micronaut.http.annotation.Controller;\n"
                    "import io.micronaut.http.annotation.Post;\n"
                    "import io.micronaut.validation.Validated;\n"
                    "import jakarta.validation.Valid;\n"
                    "import jakarta.validation.constraints.NotBlank;\n\n"
                    "@Controller(\"/customers\")\n"
                    "@Validated\n"
                    "class MicronautValidationContractFixture {\n"
                    "    @Post\n"
                    "    String create(@Body @Valid CustomerCommand command) {\n"
                    "        return command.name();\n"
                    "    }\n"
                    "}\n\n"
                    "record CustomerCommand(@NotBlank String name) {}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="observability_health",
        title="Observability Health",
        priority="high",
        status="seeded",
        covered_pattern_ids=["catalog.dependency.org_springframework_boot_spring_boot_starter_actuator"],
        goals=[
            "verify custom health indicator intent remains explicit after migration",
            "verify observability code still exposes the same operational signal",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringObservabilityHealthFixture.java",
                content=(
                    "package fixtures.observability;\n\n"
                    "import org.springframework.boot.actuate.health.Health;\n"
                    "import org.springframework.boot.actuate.health.HealthIndicator;\n"
                    "import org.springframework.stereotype.Component;\n\n"
                    "@Component\n"
                    "class SpringObservabilityHealthFixture implements HealthIndicator {\n"
                    "    @Override\n"
                    "    public Health health() {\n"
                    "        return Health.up().withDetail(\"catalog\", \"ready\").build();\n"
                    "    }\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautObservabilityHealthFixture.java",
                content=(
                    "package fixtures.observability;\n\n"
                    "import io.micronaut.management.health.indicator.HealthIndicator;\n"
                    "import io.micronaut.management.health.indicator.HealthResult;\n"
                    "import jakarta.inject.Singleton;\n"
                    "import org.reactivestreams.Publisher;\n"
                    "import reactor.core.publisher.Mono;\n"
                    "import java.util.Map;\n\n"
                    "@Singleton\n"
                    "class MicronautObservabilityHealthFixture implements HealthIndicator {\n"
                    "    @Override\n"
                    "    public Publisher<HealthResult> getResult() {\n"
                    "        return Mono.just(HealthResult.builder(\"catalog\")\n"
                    "            .status(\"UP\")\n"
                    "            .details(Map.of(\"catalog\", \"ready\"))\n"
                    "            .build());\n"
                    "    }\n"
                    "}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="redis_data_access",
        title="Redis Data Access",
        priority="high",
        status="seeded",
        covered_pattern_ids=[
            "catalog.dependency.org_springframework_boot_spring_boot_starter_data_redis",
            "catalog.dependency.redis_clients_jedis",
        ],
        goals=[
            "verify redis access code is moved to Micronaut-supported client APIs",
            "verify both template-style and direct-client intent remain reviewable after migration",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringRedisTemplateFixture.java",
                content=(
                    "package fixtures.redis;\n\n"
                    "import org.springframework.data.redis.core.StringRedisTemplate;\n"
                    "import org.springframework.stereotype.Service;\n\n"
                    "@Service\n"
                    "class SpringRedisTemplateFixture {\n"
                    "    private final StringRedisTemplate redisTemplate;\n\n"
                    "    SpringRedisTemplateFixture(StringRedisTemplate redisTemplate) {\n"
                    "        this.redisTemplate = redisTemplate;\n"
                    "    }\n\n"
                    "    String lookup(String key) {\n"
                    "        return redisTemplate.opsForValue().get(key);\n"
                    "    }\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="SpringJedisFixture.java",
                content=(
                    "package fixtures.redis;\n\n"
                    "import redis.clients.jedis.Jedis;\n\n"
                    "class SpringJedisFixture {\n"
                    "    String lookup(String key) {\n"
                    "        try (Jedis jedis = new Jedis(\"localhost\")) {\n"
                    "            return jedis.get(key);\n"
                    "        }\n"
                    "    }\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautRedisLettuceFixture.java",
                content=(
                    "package fixtures.redis;\n\n"
                    "import io.micronaut.redis.lettuce.StatefulRedisConnection;\n"
                    "import jakarta.inject.Singleton;\n\n"
                    "@Singleton\n"
                    "class MicronautRedisLettuceFixture {\n"
                    "    private final StatefulRedisConnection<String, String> connection;\n\n"
                    "    MicronautRedisLettuceFixture(StatefulRedisConnection<String, String> connection) {\n"
                    "        this.connection = connection;\n"
                    "    }\n\n"
                    "    String lookup(String key) {\n"
                    "        return connection.sync().get(key);\n"
                    "    }\n"
                    "}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="kafka_messaging",
        title="Kafka Messaging",
        priority="high",
        status="seeded",
        covered_pattern_ids=["catalog.dependency.org_springframework_kafka_spring_kafka"],
        goals=[
            "verify listener-based messaging intent remains explicit after migration",
            "verify topic annotations are visible in the migrated Micronaut form",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringKafkaMessagingFixture.java",
                content=(
                    "package fixtures.messaging.kafka;\n\n"
                    "import org.springframework.kafka.annotation.KafkaListener;\n"
                    "import org.springframework.stereotype.Component;\n\n"
                    "@Component\n"
                    "class SpringKafkaMessagingFixture {\n"
                    "    @KafkaListener(topics = \"orders\")\n"
                    "    void consume(String payload) {\n"
                    "    }\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautKafkaMessagingFixture.java",
                content=(
                    "package fixtures.messaging.kafka;\n\n"
                    "import io.micronaut.configuration.kafka.annotation.KafkaListener;\n"
                    "import io.micronaut.configuration.kafka.annotation.Topic;\n\n"
                    "@KafkaListener\n"
                    "class MicronautKafkaMessagingFixture {\n"
                    "    void receive(@Topic(\"orders\") String payload) {\n"
                    "    }\n"
                    "}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="rabbitmq_messaging",
        title="RabbitMQ Messaging",
        priority="high",
        status="seeded",
        covered_pattern_ids=["catalog.dependency.org_springframework_amqp_spring_rabbit"],
        goals=[
            "verify queue listener intent remains explicit after migration",
            "verify queue bindings are easy to inspect in the Micronaut target form",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringRabbitMessagingFixture.java",
                content=(
                    "package fixtures.messaging.rabbit;\n\n"
                    "import org.springframework.amqp.rabbit.annotation.RabbitListener;\n"
                    "import org.springframework.stereotype.Component;\n\n"
                    "@Component\n"
                    "class SpringRabbitMessagingFixture {\n"
                    "    @RabbitListener(queues = \"billing\")\n"
                    "    void consume(String payload) {\n"
                    "    }\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautRabbitMessagingFixture.java",
                content=(
                    "package fixtures.messaging.rabbit;\n\n"
                    "import io.micronaut.rabbitmq.annotation.Queue;\n"
                    "import io.micronaut.rabbitmq.annotation.RabbitListener;\n\n"
                    "@RabbitListener\n"
                    "class MicronautRabbitMessagingFixture {\n"
                    "    void receive(@Queue(\"billing\") String payload) {\n"
                    "    }\n"
                    "}\n"
                ),
            ),
        ],
    ),
    FixturePack(
        pack_id="declarative_http_client",
        title="Declarative HTTP Client",
        priority="high",
        status="seeded",
        covered_pattern_ids=["catalog.dependency.org_springframework_cloud_spring_cloud_starter_openfeign"],
        goals=[
            "verify declarative client interfaces remain explicit after migration",
            "verify outbound endpoint mappings are preserved in the target form",
        ],
        source_files=[
            FixtureSourceFile(
                filename="SpringOpenFeignFixture.java",
                content=(
                    "package fixtures.httpclient.declarative;\n\n"
                    "import org.springframework.cloud.openfeign.FeignClient;\n"
                    "import org.springframework.web.bind.annotation.GetMapping;\n\n"
                    "@FeignClient(name = \"inventory\", url = \"https://inventory.internal\")\n"
                    "interface SpringOpenFeignFixture {\n"
                    "    @GetMapping(\"/items\")\n"
                    "    String items();\n"
                    "}\n"
                ),
            ),
            FixtureSourceFile(
                filename="MicronautDeclarativeClientFixture.java",
                content=(
                    "package fixtures.httpclient.declarative;\n\n"
                    "import io.micronaut.http.annotation.Get;\n"
                    "import io.micronaut.http.client.annotation.Client;\n\n"
                    "@Client(\"https://inventory.internal\")\n"
                    "interface MicronautDeclarativeClientFixture {\n"
                    "    @Get(\"/items\")\n"
                    "    String items();\n"
                    "}\n"
                ),
            ),
        ],
    ),
]


def build_fixture_pack_index(registry_payload: Dict[str, object]) -> Dict[str, object]:
    requirements = [item for item in registry_payload.get("requirements", []) if isinstance(item, dict)]
    registry_ids = {item["pattern_id"] for item in requirements if item.get("pattern_id")}
    packs = [pack for pack in DEFAULT_FIXTURE_PACKS if set(pack.covered_pattern_ids).issubset(registry_ids)]
    covered_pattern_ids = sorted({pattern_id for pack in packs for pattern_id in pack.covered_pattern_ids})
    uncovered_pattern_ids = sorted(
        item["pattern_id"] for item in requirements if item.get("pattern_id") not in covered_pattern_ids
    )
    uncovered_priority_high = sorted(
        item["pattern_id"]
        for item in requirements
        if item.get("priority") == "high" and item.get("pattern_id") not in covered_pattern_ids
    )
    uncovered_priority_medium_low = sorted(
        item["pattern_id"]
        for item in requirements
        if item.get("priority") in {"medium", "low"} and item.get("pattern_id") not in covered_pattern_ids
    )

    return {
        "schema_version": 1,
        "pack_type": "generic_fixture_pack_index",
        "pack_count": len(packs),
        "covered_pattern_count": len(covered_pattern_ids),
        "covered_pattern_ids": covered_pattern_ids,
        "uncovered_pattern_ids": uncovered_pattern_ids,
        "uncovered_high_priority_pattern_ids": uncovered_priority_high,
        "uncovered_medium_low_priority_pattern_ids": uncovered_priority_medium_low,
        "packs": [pack.to_dict() for pack in packs],
    }


def write_fixture_packs(corpus_root: str = "corpus") -> Dict[str, object]:
    repository = PatternCorpusRepository(root=corpus_root)
    repository.initialize_layout()
    registry_result = write_fixture_registry(corpus_root=corpus_root)

    registry_path = Path(registry_result["registry_path"])
    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    pack_index = build_fixture_pack_index(registry_payload)

    target_root = Path(corpus_root) / "validated_patterns" / "release" / "legacy_reviewed" / "fixture_packs"
    target_root.mkdir(parents=True, exist_ok=True)

    index_path = target_root / "index.json"
    index_path.write_text(json.dumps(pack_index, indent=2), encoding="utf-8")

    pack_dirs: List[str] = []
    for pack in pack_index["packs"]:
        pack_root = target_root / pack["pack_id"]
        pack_root.mkdir(parents=True, exist_ok=True)
        metadata_path = pack_root / "pack.json"
        metadata_path.write_text(json.dumps(pack, indent=2), encoding="utf-8")
        for source_file in pack["source_files"]:
            (pack_root / source_file["filename"]).write_text(source_file["content"], encoding="utf-8")
        pack_dirs.append(str(pack_root))

    report = {
        "ok": not pack_index["uncovered_high_priority_pattern_ids"],
        "pack_count": pack_index["pack_count"],
        "covered_pattern_count": pack_index["covered_pattern_count"],
        "uncovered_pattern_ids": pack_index["uncovered_pattern_ids"],
        "uncovered_high_priority_pattern_ids": pack_index["uncovered_high_priority_pattern_ids"],
        "uncovered_medium_low_priority_pattern_ids": pack_index["uncovered_medium_low_priority_pattern_ids"],
    }
    report_path = target_root / "pack_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "index_path": str(index_path),
        "report_path": str(report_path),
        "pack_count": pack_index["pack_count"],
        "covered_pattern_count": pack_index["covered_pattern_count"],
        "pack_directories": pack_dirs,
        "uncovered_pattern_ids": pack_index["uncovered_pattern_ids"],
        "uncovered_high_priority_pattern_ids": pack_index["uncovered_high_priority_pattern_ids"],
        "uncovered_medium_low_priority_pattern_ids": pack_index["uncovered_medium_low_priority_pattern_ids"],
    }


def main():
    parser = argparse.ArgumentParser(description="Seed generic fixture packs for the reviewed backlog")
    parser.add_argument("--corpus-root", default="corpus", help="Corpus root directory")
    parser.add_argument("--write", action="store_true", help="Write fixture pack outputs")
    args = parser.parse_args()

    if args.write:
        print(json.dumps(write_fixture_packs(corpus_root=args.corpus_root), indent=2, sort_keys=True))
        return

    print(json.dumps({"message": "Use --write to materialize fixture packs."}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
