#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT_DIR"

STAGE="${1:-}"
if [[ "$STAGE" == "--stage" ]]; then
  STAGE="${2:-pre-push}"
elif [[ -z "$STAGE" ]]; then
  STAGE="pre-push"
fi

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "[guardrails] Python is required but was not found." >&2
  exit 1
fi

echo "[guardrails] Stage: $STAGE"
echo "[guardrails] Running fast regression suite..."
"$PYTHON_BIN" scripts/run_regression_suite.py --tier fast

if [[ "$STAGE" == "pre-commit" ]]; then
  echo "[guardrails] Commit-time guardrails passed."
  exit 0
fi

echo "[guardrails] Running corpus regression suite..."
"$PYTHON_BIN" scripts/run_regression_suite.py --tier corpus

if [[ "$STAGE" == "release" ]]; then
  echo "[guardrails] Running GA release gate..."
  "$PYTHON_BIN" scripts/run_ga_release_gate.py
  echo "[guardrails] Release-stage guardrails passed."
  exit 0
fi

echo "[guardrails] Auditing clean Maven dependency fixture..."
"$PYTHON_BIN" -m src.agent.agents.dependency_audit \
  --build-file tests/fixtures/dependency_audit/maven_clean/pom.xml \
  --dependency-tree tests/fixtures/dependency_audit/maven_clean/dependency_tree.txt \
  --spring-version 3.4.5 \
  --micronaut-version 4.10.8 \
  --fail-on blocking >/dev/null

echo "[guardrails] Auditing clean Gradle dependency fixture..."
"$PYTHON_BIN" -m src.agent.agents.dependency_audit \
  --build-file tests/fixtures/dependency_audit/gradle_clean/build.gradle.kts \
  --dependency-tree tests/fixtures/dependency_audit/gradle_clean/dependencies.txt \
  --spring-version 3.4.5 \
  --micronaut-version 4.10.8 \
  --fail-on blocking >/dev/null

echo "[guardrails] All local GA guardrails passed."
