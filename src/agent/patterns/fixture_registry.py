import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from src.agent.patterns.catalog_normalizer import curated_catalog_patterns
from src.agent.patterns.legacy_review import write_legacy_review_outputs
from src.agent.patterns.repository import PatternCorpusRepository


VALID_STATUSES = {"planned", "in_progress", "validated"}
VALID_PRIORITIES = {"high", "medium", "low"}


@dataclass(frozen=True)
class FixtureRequirement:
    pattern_id: str
    fixture_kind: str
    risk_area: str
    priority: str
    status: str
    expected_assertions: List[str]
    notes: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


DEFAULT_FIXTURE_REQUIREMENTS: Dict[str, Dict[str, object]] = {
    "legacy_promoted.annotation.modelattribute": {
        "fixture_kind": "controller_binding",
        "risk_area": "http_binding",
        "priority": "high",
        "expected_assertions": [
            "request payload binds to controller argument as expected",
            "validation errors are surfaced consistently",
        ],
    },
    "legacy_promoted.annotation.exceptionhandler": {
        "fixture_kind": "controller_error_flow",
        "risk_area": "error_handling",
        "priority": "high",
        "expected_assertions": [
            "exception maps to expected status code",
            "error body shape remains stable",
        ],
    },
    "legacy_promoted.annotation.controlleradvice": {
        "fixture_kind": "global_error_advice",
        "risk_area": "error_handling",
        "priority": "high",
        "expected_assertions": [
            "global advice applies across controllers",
            "fallback handlers do not regress",
        ],
    },
    "legacy_promoted.annotation.enablewebmvc": {
        "fixture_kind": "framework_toggle",
        "risk_area": "framework_bootstrap",
        "priority": "medium",
        "expected_assertions": [
            "application boots without the Spring MVC toggle",
            "routing still works for representative endpoints",
        ],
    },
    "legacy_promoted.annotation.enablecaching": {
        "fixture_kind": "runtime_cache_behavior",
        "risk_area": "caching",
        "priority": "high",
        "expected_assertions": [
            "cache-backed method returns memoized response",
            "required cache bean is present",
        ],
    },
    "legacy_promoted.annotation.enablescheduling": {
        "fixture_kind": "scheduled_task_boot",
        "risk_area": "scheduling",
        "priority": "high",
        "expected_assertions": [
            "scheduled job bean is registered",
            "scheduled task executes with expected trigger behavior",
        ],
    },
    "legacy_promoted.annotation.enableasync": {
        "fixture_kind": "async_execution",
        "risk_area": "async_runtime",
        "priority": "high",
        "expected_assertions": [
            "async method runs on expected executor",
            "result propagation matches expected behavior",
        ],
    },
    "legacy_promoted.annotation.enablejparepositories": {
        "fixture_kind": "persistence_bootstrap",
        "risk_area": "jpa",
        "priority": "high",
        "expected_assertions": [
            "repository beans are created",
            "basic repository operation succeeds",
        ],
    },
    "legacy_promoted.annotation.enablejpaauditing": {
        "fixture_kind": "persistence_auditing",
        "risk_area": "jpa",
        "priority": "high",
        "expected_assertions": [
            "audit fields are populated on save",
            "auditing listener lifecycle is active",
        ],
    },
    "legacy_promoted.configuration.resttemplate_configuration": {
        "fixture_kind": "http_client_configuration",
        "risk_area": "http_client",
        "priority": "medium",
        "expected_assertions": [
            "client bean is created with expected configuration",
            "outbound request succeeds against stub server",
        ],
    },
    "legacy_promoted.type.optional_responseentity": {
        "fixture_kind": "response_wrapper_endpoint",
        "risk_area": "http_response",
        "priority": "medium",
        "expected_assertions": [
            "optional response maps to expected HTTP response",
            "empty optional path preserves expected status handling",
        ],
    },
    "legacy_promoted.type.mono_flux_webflux": {
        "fixture_kind": "reactive_endpoint",
        "risk_area": "reactive_runtime",
        "priority": "high",
        "expected_assertions": [
            "publisher endpoint emits expected items",
            "backpressure or completion behavior is acceptable",
        ],
    },
    "legacy_promoted.type.filterchain": {
        "fixture_kind": "server_filter_chain",
        "risk_area": "http_filtering",
        "priority": "high",
        "expected_assertions": [
            "filter order is preserved for representative requests",
            "request/response mutation behavior remains correct",
        ],
    },
    "legacy_promoted.type.webmvcconfigurer": {
        "fixture_kind": "mvc_customization",
        "risk_area": "framework_customization",
        "priority": "medium",
        "expected_assertions": [
            "custom configuration is represented in Micronaut form",
            "affected routes or converters still behave correctly",
        ],
    },
    "legacy_promoted.code_pattern.commandlinerunner": {
        "fixture_kind": "application_startup",
        "risk_area": "lifecycle",
        "priority": "medium",
        "expected_assertions": [
            "startup hook fires once on boot",
            "startup side effects are preserved",
        ],
    },
    "legacy_promoted.code_pattern.applicationlistener": {
        "fixture_kind": "application_event_listener",
        "risk_area": "lifecycle",
        "priority": "medium",
        "expected_assertions": [
            "listener subscribes to the expected event",
            "event ordering does not regress",
        ],
    },
}


DEFAULT_CATALOG_FIXTURE_REQUIREMENTS: Dict[str, Dict[str, object]] = {
    "catalog.dependency.org_springframework_boot_spring_boot_starter_security": {
        "fixture_kind": "security_authorization_flow",
        "risk_area": "security",
        "priority": "high",
        "expected_assertions": [
            "secured endpoints keep equivalent authorization intent",
            "role or scope annotations remain explicit after migration",
        ],
        "notes": "Security starter migration needs fixture-backed validation for authn/authz semantics.",
    },
    "catalog.dependency.org_springframework_boot_spring_boot_starter_validation": {
        "fixture_kind": "validation_contract",
        "risk_area": "validation",
        "priority": "high",
        "expected_assertions": [
            "bean validation still rejects invalid requests",
            "validated service or controller methods keep the intended contract",
        ],
        "notes": "Validation starter migration needs fixture-backed validation for request and bean validation behavior.",
    },
    "catalog.dependency.org_springframework_boot_spring_boot_starter_actuator": {
        "fixture_kind": "observability_health",
        "risk_area": "observability",
        "priority": "high",
        "expected_assertions": [
            "custom health indicators remain visible after migration",
            "management endpoint intent stays explicit in Micronaut form",
        ],
        "notes": "Actuator migration needs fixture-backed validation for health and management exposure behavior.",
    },
    "catalog.dependency.org_springframework_boot_spring_boot_starter_cache": {
        "fixture_kind": "runtime_cache_behavior",
        "risk_area": "caching",
        "priority": "high",
        "expected_assertions": [
            "cache-backed method returns memoized response after migration",
            "the migrated target keeps an explicit cache-provider requirement",
        ],
        "notes": "Cache starter migration needs fixture-backed validation for provider selection and runtime cache behavior.",
    },
    "catalog.dependency.org_ehcache_ehcache": {
        "fixture_kind": "ehcache_provider_runtime",
        "risk_area": "caching",
        "priority": "high",
        "expected_assertions": [
            "provider-specific cache wiring remains explicit after migration",
            "the migrated target still expresses the external Ehcache configuration dependency clearly",
        ],
        "notes": "Direct Ehcache migration needs fixture-backed validation for provider-specific runtime wiring.",
    },
    "catalog.dependency.org_springframework_boot_spring_boot_starter_data_redis": {
        "fixture_kind": "redis_data_access",
        "risk_area": "redis",
        "priority": "high",
        "expected_assertions": [
            "redis-backed reads and writes remain explicit after migration",
            "redis client usage is migrated onto Micronaut-supported APIs",
        ],
        "notes": "Spring Data Redis migration needs fixture-backed validation for runtime client behavior.",
    },
    "catalog.dependency.redis_clients_jedis": {
        "fixture_kind": "redis_client_swap",
        "risk_area": "redis",
        "priority": "high",
        "expected_assertions": [
            "Jedis-style operations migrate to the Micronaut Redis client shape",
            "connection intent remains explicit after migration",
        ],
        "notes": "Jedis-to-Lettuce migration needs fixture-backed validation for direct client usage.",
    },
    "catalog.dependency.org_springframework_kafka_spring_kafka": {
        "fixture_kind": "kafka_messaging",
        "risk_area": "messaging",
        "priority": "high",
        "expected_assertions": [
            "message listener intent remains explicit after migration",
            "topic binding semantics are preserved in the migrated code shape",
        ],
        "notes": "Kafka migration needs fixture-backed validation for listener and topic annotations.",
    },
    "catalog.dependency.org_springframework_amqp_spring_rabbit": {
        "fixture_kind": "rabbitmq_messaging",
        "risk_area": "messaging",
        "priority": "high",
        "expected_assertions": [
            "queue listener intent remains explicit after migration",
            "queue binding annotations are represented in Micronaut form",
        ],
        "notes": "RabbitMQ migration needs fixture-backed validation for queue listener behavior.",
    },
    "catalog.dependency.org_springframework_cloud_spring_cloud_starter_openfeign": {
        "fixture_kind": "declarative_http_client",
        "risk_area": "http_client",
        "priority": "high",
        "expected_assertions": [
            "declarative client interfaces stay explicit after migration",
            "path and method mappings remain equivalent in the migrated shape",
        ],
        "notes": "OpenFeign migration needs fixture-backed validation for declarative client interfaces.",
    },
}


def build_fixture_registry(review_report: Dict[str, object]) -> Dict[str, object]:
    needs_fixture_ids = list(review_report.get("needs_fixture_ids", []))
    reasons_by_pattern_id = dict(review_report.get("reasons_by_pattern_id", {}))
    requirements: List[FixtureRequirement] = []
    seen_pattern_ids = set()

    for pattern_id in needs_fixture_ids:
        template = DEFAULT_FIXTURE_REQUIREMENTS.get(
            pattern_id,
            {
                "fixture_kind": "integration_fixture",
                "risk_area": "general_behavior",
                "priority": "medium",
                "expected_assertions": ["behavior matches expected migrated outcome"],
            },
        )
        requirements.append(
            FixtureRequirement(
                pattern_id=pattern_id,
                fixture_kind=str(template["fixture_kind"]),
                risk_area=str(template["risk_area"]),
                priority=str(template["priority"]),
                status="planned",
                expected_assertions=list(template["expected_assertions"]),
                notes=str(reasons_by_pattern_id.get(pattern_id, "Fixture validation required before GA.")),
            )
        )
        seen_pattern_ids.add(pattern_id)

    catalog_patterns = {pattern.pattern_id: pattern for pattern in curated_catalog_patterns()}
    for pattern_id, template in DEFAULT_CATALOG_FIXTURE_REQUIREMENTS.items():
        if pattern_id in seen_pattern_ids or pattern_id not in catalog_patterns:
            continue
        requirements.append(
            FixtureRequirement(
                pattern_id=pattern_id,
                fixture_kind=str(template["fixture_kind"]),
                risk_area=str(template["risk_area"]),
                priority=str(template["priority"]),
                status="planned",
                expected_assertions=list(template["expected_assertions"]),
                notes=str(template.get("notes") or catalog_patterns[pattern_id].description),
            )
        )
        seen_pattern_ids.add(pattern_id)

    return {
        "schema_version": 1,
        "registry_type": "fixture_validation_registry",
        "requirement_count": len(requirements),
        "requirements": [requirement.to_dict() for requirement in requirements],
    }


def validate_fixture_registry(review_report: Dict[str, object], registry_payload: Dict[str, object]) -> List[str]:
    issues: List[str] = []
    required_ids = set(review_report.get("needs_fixture_ids", []))
    required_ids.update(DEFAULT_CATALOG_FIXTURE_REQUIREMENTS)
    registry_requirements = list(registry_payload.get("requirements", []))
    registry_ids = {item.get("pattern_id") for item in registry_requirements}

    missing = sorted(required_ids - registry_ids)
    extra = sorted(registry_ids - required_ids)

    if missing:
        issues.append(f"Missing fixture requirements for: {', '.join(missing)}")
    if extra:
        issues.append(f"Unexpected fixture requirements for: {', '.join(extra)}")

    for item in registry_requirements:
        status = str(item.get("status", "")).strip()
        priority = str(item.get("priority", "")).strip()
        if status not in VALID_STATUSES:
            issues.append(f"Invalid fixture status for {item.get('pattern_id')}: {status}")
        if priority not in VALID_PRIORITIES:
            issues.append(f"Invalid fixture priority for {item.get('pattern_id')}: {priority}")
        assertions = item.get("expected_assertions")
        if not isinstance(assertions, list) or not assertions:
            issues.append(f"Fixture requirement {item.get('pattern_id')} must define expected assertions.")

    return issues


def write_fixture_registry(corpus_root: str = "corpus") -> Dict[str, object]:
    repository = PatternCorpusRepository(root=corpus_root)
    repository.initialize_layout()
    review_result = write_legacy_review_outputs(corpus_root=corpus_root)

    review_report = review_result["report"]
    registry_payload = build_fixture_registry(review_report)
    issues = validate_fixture_registry(review_report, registry_payload)

    target_root = Path(corpus_root) / "validated_patterns" / "release" / "legacy_reviewed"
    registry_path = target_root / "fixture_registry.json"
    registry_path.write_text(json.dumps(registry_payload, indent=2), encoding="utf-8")

    validation_report = {
        "ok": not issues,
        "issues": issues,
        "requirement_count": registry_payload["requirement_count"],
    }
    validation_path = target_root / "fixture_registry_report.json"
    validation_path.write_text(json.dumps(validation_report, indent=2), encoding="utf-8")

    return {
        "registry_path": str(registry_path),
        "validation_report_path": str(validation_path),
        "ok": not issues,
        "requirement_count": registry_payload["requirement_count"],
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate and validate the fixture registry for non-GA reviewed legacy mappings")
    parser.add_argument("--corpus-root", default="corpus", help="Corpus root directory")
    parser.add_argument("--write", action="store_true", help="Write fixture registry outputs")
    args = parser.parse_args()

    if args.write:
        print(json.dumps(write_fixture_registry(corpus_root=args.corpus_root), indent=2, sort_keys=True))
        return

    print(json.dumps({"message": "Use --write to materialize fixture registry outputs."}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
