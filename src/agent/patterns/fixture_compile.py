import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from src.agent.patterns.fixture_packs import write_fixture_packs
from src.agent.patterns.repository import PatternCorpusRepository


IMPORT_PATTERN = re.compile(r"^\s*import\s+([\w\.]+);", re.MULTILINE)


STUB_SOURCES: Dict[str, str] = {
    "io.micronaut.cache.annotation.Cacheable": """
package io.micronaut.cache.annotation;
public @interface Cacheable { String value() default ""; }
""",
    "io.micronaut.context.annotation.Factory": """
package io.micronaut.context.annotation;
public @interface Factory {}
""",
    "io.micronaut.configuration.kafka.annotation.KafkaListener": """
package io.micronaut.configuration.kafka.annotation;
public @interface KafkaListener {}
""",
    "io.micronaut.configuration.kafka.annotation.Topic": """
package io.micronaut.configuration.kafka.annotation;
public @interface Topic { String value() default ""; }
""",
    "io.micronaut.context.annotation.Requires": """
package io.micronaut.context.annotation;
public @interface Requires {
    Class<?>[] beans() default {};
    String property() default "";
    String value() default "";
    String missingProperty() default "";
    String notEquals() default "";
}
""",
    "io.micronaut.context.event.ApplicationEventListener": """
package io.micronaut.context.event;
public interface ApplicationEventListener<T> { void onApplicationEvent(T event); }
""",
    "io.micronaut.core.convert.ConversionContext": """
package io.micronaut.core.convert;
public interface ConversionContext {}
""",
    "io.micronaut.core.convert.TypeConverter": """
package io.micronaut.core.convert;
import java.util.Optional;
public interface TypeConverter<S, T> {
    Optional<T> convert(S object, Class<T> targetType, ConversionContext context);
}
""",
    "io.micronaut.data.annotation.Repository": """
package io.micronaut.data.annotation;
public @interface Repository {}
""",
    "io.micronaut.data.jpa.repository.JpaRepository": """
package io.micronaut.data.jpa.repository;
public interface JpaRepository<T, ID> {}
""",
    "io.micronaut.http.HttpRequest": """
package io.micronaut.http;
public interface HttpRequest<T> {}
""",
    "io.micronaut.http.HttpResponse": """
package io.micronaut.http;
public class HttpResponse<T> {
    public static <T> HttpResponse<T> ok(T body) { return new HttpResponse<>(); }
    public static <T> HttpResponse<T> badRequest(T body) { return new HttpResponse<>(); }
    public static <T> HttpResponse<T> notFound() { return new HttpResponse<>(); }
}
""",
    "io.micronaut.http.MutableHttpResponse": """
package io.micronaut.http;
public class MutableHttpResponse<T> extends HttpResponse<T> {
    private final Headers headers = new Headers();
    public Headers getHeaders() { return headers; }
    public static class Headers {
        public void add(String name, String value) {}
    }
}
""",
    "io.micronaut.http.annotation.Body": """
package io.micronaut.http.annotation;
public @interface Body {}
""",
    "io.micronaut.http.annotation.Controller": """
package io.micronaut.http.annotation;
public @interface Controller { String value() default ""; }
""",
    "io.micronaut.http.annotation.Error": """
package io.micronaut.http.annotation;
public @interface Error {
    boolean global() default false;
    Class<? extends Throwable> exception() default Throwable.class;
}
""",
    "io.micronaut.http.annotation.Filter": """
package io.micronaut.http.annotation;
public @interface Filter { String value() default ""; }
""",
    "io.micronaut.http.annotation.Get": """
package io.micronaut.http.annotation;
public @interface Get { String value() default ""; }
""",
    "io.micronaut.http.annotation.Post": """
package io.micronaut.http.annotation;
public @interface Post { String value() default ""; }
""",
    "io.micronaut.http.client.annotation.Client": """
package io.micronaut.http.client.annotation;
public @interface Client { String value() default ""; }
""",
    "io.micronaut.management.health.indicator.HealthIndicator": """
package io.micronaut.management.health.indicator;
import org.reactivestreams.Publisher;
public interface HealthIndicator {
    Publisher<HealthResult> getResult();
}
""",
    "io.micronaut.management.health.indicator.HealthResult": """
package io.micronaut.management.health.indicator;
import java.util.Map;
public class HealthResult {
    public static Builder builder(String name) { return new Builder(); }
    public static class Builder {
        public Builder status(String value) { return this; }
        public Builder details(Map<String, ?> details) { return this; }
        public HealthResult build() { return new HealthResult(); }
    }
}
""",
    "io.micronaut.rabbitmq.annotation.Queue": """
package io.micronaut.rabbitmq.annotation;
public @interface Queue { String value() default ""; }
""",
    "io.micronaut.rabbitmq.annotation.RabbitListener": """
package io.micronaut.rabbitmq.annotation;
public @interface RabbitListener {}
""",
    "io.micronaut.redis.lettuce.StatefulRedisConnection": """
package io.micronaut.redis.lettuce;
public interface StatefulRedisConnection<K, V> {
    SyncCommands<K, V> sync();
    interface SyncCommands<K, V> {
        V get(K key);
    }
}
""",
    "io.micronaut.security.annotation.Secured": """
package io.micronaut.security.annotation;
public @interface Secured { String[] value() default {}; }
""",
    "io.micronaut.security.rules.SecurityRule": """
package io.micronaut.security.rules;
public final class SecurityRule {
    private SecurityRule() {}
    public static final String IS_AUTHENTICATED = "isAuthenticated()";
}
""",
    "io.micronaut.validation.Validated": """
package io.micronaut.validation;
public @interface Validated {}
""",
    "io.micronaut.http.filter.FilterChain": """
package io.micronaut.http.filter;
import io.micronaut.http.HttpRequest;
import io.micronaut.http.MutableHttpResponse;
import org.reactivestreams.Publisher;
public interface FilterChain {
    Publisher<MutableHttpResponse<?>> proceed(HttpRequest<?> request);
}
""",
    "io.micronaut.http.filter.HttpServerFilter": """
package io.micronaut.http.filter;
import io.micronaut.http.HttpRequest;
import io.micronaut.http.MutableHttpResponse;
import org.reactivestreams.Publisher;
public interface HttpServerFilter {
    Publisher<MutableHttpResponse<?>> doFilter(HttpRequest<?> request, FilterChain chain);
}
""",
    "io.micronaut.runtime.event.ApplicationStartupEvent": """
package io.micronaut.runtime.event;
public class ApplicationStartupEvent {
    public Object getSource() { return this; }
}
""",
    "io.micronaut.runtime.server.event.ServerStartupEvent": """
package io.micronaut.runtime.server.event;
public class ServerStartupEvent {}
""",
    "io.micronaut.scheduling.TaskScheduler": """
package io.micronaut.scheduling;
public interface TaskScheduler {}
""",
    "io.micronaut.scheduling.annotation.Async": """
package io.micronaut.scheduling.annotation;
public @interface Async {}
""",
    "io.micronaut.scheduling.annotation.Scheduled": """
package io.micronaut.scheduling.annotation;
public @interface Scheduled { String fixedDelay() default ""; }
""",
    "jakarta.inject.Singleton": """
package jakarta.inject;
public @interface Singleton {}
""",
    "jakarta.validation.Valid": """
package jakarta.validation;
public @interface Valid {}
""",
    "jakarta.validation.constraints.NotBlank": """
package jakarta.validation.constraints;
public @interface NotBlank {}
""",
    "jakarta.servlet.FilterChain": """
package jakarta.servlet;
import java.io.IOException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
public interface FilterChain {
    void doFilter(HttpServletRequest request, HttpServletResponse response) throws IOException, ServletException;
}
""",
    "jakarta.servlet.ServletException": """
package jakarta.servlet;
public class ServletException extends Exception {
    public ServletException() {}
    public ServletException(String message) { super(message); }
}
""",
    "jakarta.servlet.http.HttpServletRequest": """
package jakarta.servlet.http;
public interface HttpServletRequest {}
""",
    "jakarta.servlet.http.HttpServletResponse": """
package jakarta.servlet.http;
public interface HttpServletResponse {
    void addHeader(String name, String value);
}
""",
    "javax.cache.CacheManager": """
package javax.cache;
public interface CacheManager {}
""",
    "javax.persistence.EntityManagerFactory": """
package javax.persistence;
public interface EntityManagerFactory {}
""",
    "jakarta.persistence.EntityManagerFactory": """
package jakarta.persistence;
public interface EntityManagerFactory {}
""",
    "javax.persistence.PrePersist": """
package javax.persistence;
public @interface PrePersist {}
""",
    "jakarta.persistence.PrePersist": """
package jakarta.persistence;
public @interface PrePersist {}
""",
    "org.reactivestreams.Publisher": """
package org.reactivestreams;
public interface Publisher<T> {}
""",
    "org.springframework.boot.CommandLineRunner": """
package org.springframework.boot;
public interface CommandLineRunner { void run(String... args) throws Exception; }
""",
    "org.springframework.boot.actuate.health.Health": """
package org.springframework.boot.actuate.health;
public class Health {
    public static Builder up() { return new Builder(); }
    public static class Builder {
        public Builder withDetail(String key, Object value) { return this; }
        public Health build() { return new Health(); }
    }
}
""",
    "org.springframework.boot.actuate.health.HealthIndicator": """
package org.springframework.boot.actuate.health;
public interface HealthIndicator { Health health(); }
""",
    "org.springframework.boot.web.client.RestTemplateBuilder": """
package org.springframework.boot.web.client;
import org.springframework.web.client.RestTemplate;
public class RestTemplateBuilder {
    public RestTemplateBuilder rootUri(String uri) { return this; }
    public RestTemplate build() { return new RestTemplate(); }
}
""",
    "org.springframework.cache.annotation.Cacheable": """
package org.springframework.cache.annotation;
public @interface Cacheable { String value() default ""; }
""",
    "org.springframework.cache.annotation.EnableCaching": """
package org.springframework.cache.annotation;
public @interface EnableCaching {}
""",
    "org.springframework.cloud.openfeign.FeignClient": """
package org.springframework.cloud.openfeign;
public @interface FeignClient {
    String name() default "";
    String url() default "";
}
""",
    "org.springframework.context.ApplicationListener": """
package org.springframework.context;
public interface ApplicationListener<T> { void onApplicationEvent(T event); }
""",
    "org.springframework.context.annotation.Bean": """
package org.springframework.context.annotation;
public @interface Bean {}
""",
    "org.springframework.context.annotation.Configuration": """
package org.springframework.context.annotation;
public @interface Configuration {}
""",
    "org.springframework.context.event.ContextRefreshedEvent": """
package org.springframework.context.event;
public class ContextRefreshedEvent {
    public ApplicationContext getApplicationContext() { return new ApplicationContext(); }
    public static class ApplicationContext {
        public String getId() { return "ctx"; }
    }
}
""",
    "org.springframework.data.annotation.CreatedDate": """
package org.springframework.data.annotation;
public @interface CreatedDate {}
""",
    "org.springframework.data.jpa.repository.JpaRepository": """
package org.springframework.data.jpa.repository;
public interface JpaRepository<T, ID> {}
""",
    "org.springframework.data.jpa.repository.config.EnableJpaAuditing": """
package org.springframework.data.jpa.repository.config;
public @interface EnableJpaAuditing {}
""",
    "org.springframework.data.jpa.repository.config.EnableJpaRepositories": """
package org.springframework.data.jpa.repository.config;
public @interface EnableJpaRepositories {}
""",
    "org.springframework.data.redis.core.StringRedisTemplate": """
package org.springframework.data.redis.core;
public class StringRedisTemplate {
    public ValueOperations opsForValue() { return new ValueOperations(); }
    public static class ValueOperations {
        public String get(String key) { return null; }
    }
}
""",
    "org.springframework.format.FormatterRegistry": """
package org.springframework.format;
import java.util.function.Function;
public interface FormatterRegistry {
    <S, T> void addConverter(Class<S> sourceType, Class<T> targetType, Function<? super S, ? extends T> converter);
}
""",
    "org.springframework.http.HttpStatus": """
package org.springframework.http;
public enum HttpStatus { BAD_REQUEST }
""",
    "org.springframework.http.ResponseEntity": """
package org.springframework.http;
public class ResponseEntity<T> {
    public static <T> ResponseEntity<T> ok(T body) { return new ResponseEntity<>(); }
    public static BodyBuilder notFound() { return new BodyBuilder(); }
    public static class BodyBuilder {
        public <T> ResponseEntity<T> build() { return new ResponseEntity<>(); }
    }
}
""",
    "org.springframework.scheduling.annotation.Async": """
package org.springframework.scheduling.annotation;
public @interface Async {}
""",
    "org.springframework.scheduling.annotation.EnableAsync": """
package org.springframework.scheduling.annotation;
public @interface EnableAsync {}
""",
    "org.springframework.scheduling.annotation.EnableScheduling": """
package org.springframework.scheduling.annotation;
public @interface EnableScheduling {}
""",
    "org.springframework.security.access.annotation.Secured": """
package org.springframework.security.access.annotation;
public @interface Secured { String[] value() default {}; }
""",
    "org.springframework.scheduling.annotation.Scheduled": """
package org.springframework.scheduling.annotation;
public @interface Scheduled { long fixedDelay() default -1L; }
""",
    "org.springframework.kafka.annotation.KafkaListener": """
package org.springframework.kafka.annotation;
public @interface KafkaListener { String[] topics() default {}; }
""",
    "org.springframework.amqp.rabbit.annotation.RabbitListener": """
package org.springframework.amqp.rabbit.annotation;
public @interface RabbitListener { String[] queues() default {}; }
""",
    "org.springframework.stereotype.Component": """
package org.springframework.stereotype;
public @interface Component {}
""",
    "org.springframework.stereotype.Repository": """
package org.springframework.stereotype;
public @interface Repository {}
""",
    "org.springframework.stereotype.Service": """
package org.springframework.stereotype;
public @interface Service {}
""",
    "org.springframework.web.bind.annotation.ControllerAdvice": """
package org.springframework.web.bind.annotation;
public @interface ControllerAdvice {}
""",
    "org.springframework.web.bind.annotation.ExceptionHandler": """
package org.springframework.web.bind.annotation;
public @interface ExceptionHandler { Class<? extends Throwable>[] value() default {}; }
""",
    "org.springframework.web.bind.annotation.GetMapping": """
package org.springframework.web.bind.annotation;
public @interface GetMapping { String value() default ""; }
""",
    "org.springframework.web.bind.annotation.ModelAttribute": """
package org.springframework.web.bind.annotation;
public @interface ModelAttribute {}
""",
    "org.springframework.web.bind.annotation.PathVariable": """
package org.springframework.web.bind.annotation;
public @interface PathVariable { String value() default ""; }
""",
    "org.springframework.web.bind.annotation.PostMapping": """
package org.springframework.web.bind.annotation;
public @interface PostMapping { String value() default ""; }
""",
    "org.springframework.web.bind.annotation.ResponseBody": """
package org.springframework.web.bind.annotation;
public @interface ResponseBody {}
""",
    "org.springframework.web.bind.annotation.ResponseStatus": """
package org.springframework.web.bind.annotation;
import org.springframework.http.HttpStatus;
public @interface ResponseStatus { HttpStatus value(); }
""",
    "org.springframework.web.bind.annotation.RestController": """
package org.springframework.web.bind.annotation;
public @interface RestController {}
""",
    "org.springframework.web.client.RestTemplate": """
package org.springframework.web.client;
public class RestTemplate {}
""",
    "org.springframework.validation.annotation.Validated": """
package org.springframework.validation.annotation;
public @interface Validated {}
""",
    "org.springframework.web.filter.OncePerRequestFilter": """
package org.springframework.web.filter;
import java.io.IOException;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
public abstract class OncePerRequestFilter {
    protected abstract void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
            throws ServletException, IOException;
}
""",
    "org.springframework.web.servlet.config.annotation.EnableWebMvc": """
package org.springframework.web.servlet.config.annotation;
public @interface EnableWebMvc {}
""",
    "org.springframework.web.servlet.config.annotation.WebMvcConfigurer": """
package org.springframework.web.servlet.config.annotation;
import org.springframework.format.FormatterRegistry;
public interface WebMvcConfigurer {
    default void addFormatters(FormatterRegistry registry) {}
}
""",
    "reactor.core.publisher.Flux": """
package reactor.core.publisher;
import java.util.function.Consumer;
import org.reactivestreams.Publisher;
public class Flux<T> implements Publisher<T> {
    public static <T> Flux<T> just(T... items) { return new Flux<>(); }
    public static <T> Flux<T> from(Publisher<T> publisher) { return new Flux<>(); }
    public Flux<T> doOnNext(Consumer<? super T> consumer) { return this; }
}
""",
    "reactor.core.publisher.Mono": """
package reactor.core.publisher;
import org.reactivestreams.Publisher;
public class Mono<T> implements Publisher<T> {
    public static <T> Mono<T> just(T item) { return new Mono<>(); }
}
""",
    "redis.clients.jedis.Jedis": """
package redis.clients.jedis;
public class Jedis implements AutoCloseable {
    public Jedis(String host) {}
    public String get(String key) { return null; }
    @Override
    public void close() {}
}
""",
}

STUB_DEPENDENCIES: Dict[str, Set[str]] = {
    "io.micronaut.http.MutableHttpResponse": {"io.micronaut.http.HttpResponse"},
    "reactor.core.publisher.Flux": {"org.reactivestreams.Publisher"},
    "reactor.core.publisher.Mono": {"org.reactivestreams.Publisher"},
}


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_imports(source_text: str) -> Set[str]:
    return set(IMPORT_PATTERN.findall(source_text))


def _materialize_java_source(root: Path, fqcn: str, source: str) -> Path:
    relative_path = Path(*fqcn.split(".")).with_suffix(".java")
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.strip() + "\n", encoding="utf-8")
    return target


def _required_stub_imports(source_texts: Iterable[str]) -> Tuple[Set[str], List[str]]:
    imports: Set[str] = set()
    missing: List[str] = []
    for source_text in source_texts:
        for fqcn in _extract_imports(source_text):
            if fqcn.startswith("java."):
                continue
            if fqcn in STUB_SOURCES:
                imports.add(fqcn)
            else:
                missing.append(fqcn)
    expanded_imports = set(imports)
    pending = list(imports)
    while pending:
        current = pending.pop()
        for dependency in STUB_DEPENDENCIES.get(current, set()):
            if dependency not in STUB_SOURCES:
                missing.append(dependency)
                continue
            if dependency not in expanded_imports:
                expanded_imports.add(dependency)
                pending.append(dependency)

    return expanded_imports, sorted(set(missing))


def _compile_pack(pack: Dict[str, object], target_root: Path, javac_cmd: str) -> Dict[str, object]:
    source_files = list(pack.get("source_files", []))
    source_texts = [str(source_file.get("content", "")) for source_file in source_files]
    required_stubs, missing_stubs = _required_stub_imports(source_texts)
    if missing_stubs:
        return {
            "pack_id": pack["pack_id"],
            "ok": False,
            "status": "missing_stubs",
            "missing_stub_imports": missing_stubs,
            "source_file_count": len(source_files),
            "stub_count": len(required_stubs),
            "stderr": "Missing stub sources for imported framework types.",
        }

    errors: List[str] = []
    compiled_source_count = 0
    for source_file in source_files:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            src_root = workspace / "src"
            out_root = workspace / "out"
            src_root.mkdir(parents=True, exist_ok=True)
            out_root.mkdir(parents=True, exist_ok=True)

            java_files: List[Path] = []
            java_path = src_root / str(source_file["filename"])
            java_path.write_text(str(source_file["content"]), encoding="utf-8")
            java_files.append(java_path)

            file_stubs, file_missing_stubs = _required_stub_imports([str(source_file.get("content", ""))])
            if file_missing_stubs:
                errors.append(f"{source_file['filename']}: missing stubs for {', '.join(file_missing_stubs)}")
                continue

            for fqcn in file_stubs:
                java_files.append(_materialize_java_source(src_root, fqcn, STUB_SOURCES[fqcn]))

            process = subprocess.run(
                [javac_cmd, "-d", str(out_root), *[str(path) for path in java_files]],
                capture_output=True,
                text=True,
                cwd=target_root,
            )
            if process.returncode == 0:
                compiled_source_count += 1
            else:
                errors.append(f"{source_file['filename']}: {process.stderr.strip()}")

    return {
        "pack_id": pack["pack_id"],
        "ok": not errors,
        "status": "compiled" if not errors else "compile_failed",
        "missing_stub_imports": [],
        "source_file_count": len(source_files),
        "compiled_source_count": compiled_source_count,
        "stub_count": len(required_stubs),
        "stderr": "\n\n".join(errors),
    }


def evaluate_fixture_compile(corpus_root: str = "corpus") -> Dict[str, object]:
    javac_cmd = shutil.which("javac")
    if not javac_cmd:
        return {
            "ok": False,
            "validation_mode": "offline_stubbed_javac",
            "issues": ["javac not found on PATH"],
            "pack_count": 0,
            "compiled_pack_count": 0,
            "failed_pack_ids": [],
            "items": [],
        }

    target_root = Path(corpus_root) / "validated_patterns" / "release" / "legacy_reviewed"
    pack_index = _load_json(target_root / "fixture_packs" / "index.json")
    items = [_compile_pack(pack, target_root=target_root, javac_cmd=javac_cmd) for pack in pack_index.get("packs", [])]
    failed_pack_ids = sorted(item["pack_id"] for item in items if not item["ok"])
    issues = [f"{item['pack_id']}: {item['status']}" for item in items if not item["ok"]]

    return {
        "ok": not issues,
        "validation_mode": "offline_stubbed_javac",
        "issues": issues,
        "pack_count": len(items),
        "compiled_pack_count": sum(1 for item in items if item["ok"]),
        "failed_pack_ids": failed_pack_ids,
        "items": items,
    }


def write_fixture_compile_report(corpus_root: str = "corpus") -> Dict[str, object]:
    repository = PatternCorpusRepository(root=corpus_root)
    repository.initialize_layout()
    write_fixture_packs(corpus_root=corpus_root)

    target_root = Path(corpus_root) / "validated_patterns" / "release" / "legacy_reviewed"
    report = evaluate_fixture_compile(corpus_root=corpus_root)
    report_path = target_root / "fixture_compile_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"report_path": str(report_path), **report}


def main():
    parser = argparse.ArgumentParser(description="Compile seeded fixture packs offline using local javac and framework stubs")
    parser.add_argument("--corpus-root", default="corpus", help="Corpus root directory")
    parser.add_argument("--write", action="store_true", help="Write fixture compile validation report")
    args = parser.parse_args()

    if args.write:
        print(json.dumps(write_fixture_compile_report(corpus_root=args.corpus_root), indent=2, sort_keys=True))
        return

    print(json.dumps({"message": "Use --write to materialize fixture compile report."}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
