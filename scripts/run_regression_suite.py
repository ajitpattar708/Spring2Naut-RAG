#!/usr/bin/env python3
import argparse
import os
import sys
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


FAST_MODULES = [
    "tests.test_ga_release_gate",
    "tests.test_main_cli",
    "tests.test_regression_contracts",
    "tests.test_code_transform_agent",
    "tests.test_config_agent",
    "tests.test_dependency_agent",
    "tests.test_validation_agent",
    "tests.test_verification_agent",
    "tests.test_orchestrator_analysis",
    "tests.test_knowledge_base",
    "tests.test_kb_version_filtering",
]

CORPUS_MODULES = [
    "tests.test_dependency_audit",
    "tests.test_orchestrator_integration",
    "tests.test_release_export",
    "tests.test_kb_release_smoke",
    "tests.test_chroma_audit",
    "tests.test_catalog_release",
    "tests.test_dataset_cleaner",
    "tests.test_fixture_compile",
    "tests.test_fixture_execution",
    "tests.test_fixture_packs",
    "tests.test_fixture_registry",
    "tests.test_github_candidates",
    "tests.test_github_normalizer",
    "tests.test_kb_validator",
    "tests.test_legacy_bootstrap",
    "tests.test_legacy_promotion",
    "tests.test_legacy_review",
    "tests.test_official_normalizer",
    "tests.test_official_seeds",
    "tests.test_pattern_repository",
    "tests.test_pattern_schema",
    "tests.test_promotion",
    "tests.test_runtime_fallbacks",
    "tests.test_vector_audit",
    "tests.test_version_compatibility",
    "tests.test_versioning",
    "tests.test_example_hygiene",
]


def _suite_for_tier(tier: str) -> unittest.TestSuite:
    loader = unittest.defaultTestLoader
    if tier == "full":
        return loader.discover("tests")

    module_names = FAST_MODULES if tier == "fast" else CORPUS_MODULES
    return loader.loadTestsFromNames(module_names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Spring2Naut regression suites.")
    parser.add_argument(
        "--tier",
        choices=["fast", "corpus", "full"],
        default="fast",
        help="fast = core migration guardrails, corpus = KB/corpus pipeline checks, full = discover all tests",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=2,
        help="unittest verbosity level",
    )
    args = parser.parse_args(argv)

    suite = _suite_for_tier(args.tier)
    result = unittest.TextTestRunner(verbosity=args.verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
