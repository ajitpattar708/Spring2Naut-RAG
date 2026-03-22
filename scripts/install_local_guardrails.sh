#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT_DIR"

git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
chmod +x .githooks/pre-push
chmod +x scripts/run_local_guardrails.sh

echo "Local guardrails installed."
echo "pre-commit hook: .githooks/pre-commit"
echo "pre-push hook: .githooks/pre-push"
echo "manual run: ./scripts/run_local_guardrails.sh"
echo "pre-commit gate: fast regression suite"
echo "pre-push gate: fast + corpus regression suites + clean Maven/Gradle dependency audit fixtures"
