# Spring2Naut-RAG

Spring2Naut-RAG helps migrate Spring Boot `3.x` projects toward Micronaut `4.x` with deterministic transforms first, RAG-backed pattern lookup, optional LLM refinement, and build validation.

## What It Does

- transforms Java, config, and build files
- supports Maven and Gradle projects
- uses a local Chroma knowledge base built from governed migration patterns
- writes migration and verification reports into the output project

## Prerequisites

- Python 3.8+
- Java 17+
- Maven or Gradle on `PATH`
- Optional but recommended: Ollama running locally

## Install

```bash
pip install -r requirements_file.txt
```

Optional package install:

```bash
pip install .
spring2naut --help
```

## Quick Start

1. Initialize the knowledge base for the exact version pair:

```bash
python main.py init --mode trusted --spring-version 3.0.0 --micronaut-version 4.10.8
```

2. Run migration:

```bash
python main.py migrate examples/spring examples/micronaut \
  --spring-version 3.0.0 \
  --micronaut-version 4.10.8
```

If the source project contains both `pom.xml` and `build.gradle`/`build.gradle.kts`, pass `--build-tool maven` or `--build-tool gradle`. Otherwise the agent auto-detects the build tool.

Example with explicit Maven:

```bash
python main.py migrate examples/spring examples/micronaut \
  --spring-version 3.0.0 \
  --micronaut-version 4.10.8 \
  --build-tool maven
```

## Init Modes

- `trusted`: default and recommended; governed rules only
- `legacy`: old legacy dataset only
- `hybrid`: trusted plus legacy/raw dataset
- `candidate`: trusted plus your staged custom patterns
- `max`: widest experimental local dataset
- `all`: trusted plus oldest legacy encrypted datasets and max materialization

Aliases:
- `extended` = `legacy`
- `both` = `hybrid`
- `full` = `all`

## What Success Looks Like

Healthy `init` run:

```text
Spring2Naut RAG Initialization
Targeting: Spring 3.0.0 -> Micronaut 4.10.8
Mode: trusted
...
[OK] Intelligence Engine indexed with 356 patterns
...
KB Smoke OK: True
```

Healthy `migrate` run:

```text
Agentic Migration Initialized
Targeting: Spring 3.0.0 -> Micronaut 4.10.8
...
[VDB] engine=chromadb status=ready
Starting migration from examples/spring to examples/micronaut
```

## Custom Patterns

Add your own patterns in `corpus/staged_patterns/candidates/index.json`, then rebuild with:

```bash
python main.py init --mode candidate --spring-version 3.0.0 --micronaut-version 4.10.8
```

## Output

- migrated project: your target output directory
- migration report: `<output>/reports/migration_report.json`
- verification report: `<output>/reports/verification_report.json`
- trusted KB manifest: `migration_db/kb_manifest.json`

See `USER_GUIDE.md` for the step-by-step flow.
