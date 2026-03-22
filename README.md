# Spring2Naut-RAG: Spring to Micronaut Migration Agent

Transform Spring Boot 3.x applications toward Micronaut 4.x using a deterministic-first migration agent with governed retrieval, optional local LLM refinement, and validation/reporting loops.

## Key Features

- **🚀 Automated Self-Healing (Try-Compile-Fix)**: The agent doesn't just migrate; it validates. It automatically attempts to build your project and uses AI to fix any compilation errors it finds.
- **🎯 High-Fidelity Mapping**: Utilizes a governed trusted migration knowledge base plus larger optional experimental corpora so annotations, dependencies, and code patterns can be transformed with clear trust boundaries.
- **🛡️ Local-First Security**: Designed to run locally with Ollama so proprietary source code can stay inside your environment.
- **📦 Full Project Transformation**: Handles everything from build configurations (Maven/Gradle) and source code transformation to configuration files (`application.yml`) and dependency injection.
- **🔎 Dependency Compatibility Audit**: Extracts direct and transitive Maven or Gradle dependencies, flags Spring carry-over, Micronaut version drift, `javax`-era libraries, and multi-version conflicts before you treat a migration as production-ready.
- **🧱 Deterministic Core Rewrites**: Common Spring imports, annotations, `ResponseEntity` usage, controller-advice/error-handler patterns, constructor-injection upgrades, paging/model/view-controller patterns, Spring test-slice rewrites, and narrow `@ConditionalOnProperty` to `@Requires` conversion are now rewritten locally before any LLM fallback is used.
- **🛑 Safe LLM Guardrails**: LLM refinement and self-fix output is now validated before it can overwrite code, blocking prose responses, package drift, cross-file class swaps, and file/type mismatches.
- **🧩 Multi-LLM Support**: Works seamlessly with your choice of AI: Ollama (Local/Free), OpenAI, Claude, or Groq.

## What End Users Can Use This For

Use Spring2Naut-RAG when you want to:
- migrate a Spring Boot `3.x.x` service toward Micronaut `4.x.x`
- keep migration work local with Ollama instead of sending source code to a cloud LLM
- initialize a governed trusted migration knowledge base before running a real migration
- audit direct and transitive dependencies before treating a migration as production-ready
- compare trusted governed retrieval with larger experimental local corpora
- add your own local candidate patterns without changing core Python logic

Use it carefully for:
- pilot migrations with engineering review
- local enterprise experimentation
- repeatable migration validation through reports, smoke checks, and regression gates

Do not treat it today as:
- a guaranteed one-click GA migration platform for every Spring application
- a replacement for engineering review on complex data, messaging, security, or third-party library ecosystems

## Start To Finish Workflow

The intended end-user workflow is:

1. install Python dependencies and your preferred LLM runtime
2. choose exact source Spring and target Micronaut versions
3. run `init` once for that version pair
4. run `migrate` against your source project
5. review generated reports and build/test the migrated output
6. optionally add custom candidate patterns and rebuild a candidate DB
7. use regression and release gates before sharing results more broadly

## Getting Started

### 1. Prerequisites
- Python 3.8+
- Maven or Gradle (installed and on PATH)
- Your choice of LLM (e.g., [Ollama](https://ollama.com) installed for local use)

Recommended for enterprise-style local runs:
- Java 17+ available for migrated project validation
- Ollama running locally if you want local-only LLM refinement
- enough disk space for local vector DB materialization and migrated output copies

### 2. Installation

From source checkout:
```bash
# Clone the repository
git clone https://github.com/ajitpattar708/Spring2Naut-RAG.git
cd Spring2Naut-RAG

# Install dependencies
pip install -r requirements_file.txt
```

Package-oriented install for release artifacts:
```bash
pip install .
spring2naut --help
```

Release-artifact integrity contract:
- CI emits `dist/release_manifest.json` with artifact names, sizes, SHA-256 hashes, and the GA gate summary used for that build
- CI emits `dist/SHA256SUMS` so users can verify release assets before mirroring or installing them
- both files are uploaded as workflow artifacts and attached to tag-based GitHub releases

### 2A. Choose Your LLM Runtime

Recommended local-only setup:

```bash
ollama pull codellama:7b
export LLM_PROVIDER=ollama
export LLM_MODEL=codellama:7b
export OLLAMA_BASE_URL=http://localhost:11434
```

If you want deterministic-first behavior with less dependence on LLM fallback, still keep the LLM configured. The agent will prefer deterministic rewrites first and only use the LLM when needed.

### 2B. Know The Two Main Commands

You mainly use two commands:

```bash
python main.py init ...
python main.py migrate ...
```

- `init` prepares the governed dataset, trusted vector DB, KB smoke report, and Chroma audit for a specific version pair
- `migrate` actually transforms a Spring project into Micronaut-oriented output

Recommended pattern:

```bash
python main.py init --mode trusted --spring-version 3.0.0 --micronaut-version 4.10.8
python main.py migrate /path/to/spring-project /path/to/output \
  --spring-version 3.0.0 \
  --micronaut-version 4.10.8
```

Recommended enterprise sequence:
- Step 1: identify the source Spring Boot version from `pom.xml` or `build.gradle`
- Step 2: choose the exact Micronaut target version you want
- Step 3: run `init --mode trusted` with that exact version pair
- Step 4: run `migrate` with the same exact version pair
- Step 5: review `output/reports/` and then build/test the migrated project

### 2C. What Healthy Startup Looks Like

When users are first testing the tool, one of the most common questions is whether the runtime booted correctly.
Below are representative successful startup examples for both `init` and `migrate`.

Successful `migrate` startup example:

```text
$ python main.py migrate examples/spring examples/micronaut --spring-version 3.0.0 --micronaut-version 4.10.8
--------------------------------------------------
Agentic Migration Initialized
Targeting: Spring 3.0.0 -> Micronaut 4.10.8
--------------------------------------------------
No sentence-transformers model found with name microsoft/codebert-base. Creating a new one with mean pooling.
[INFO] Intelligence Engine loaded with 356 cached patterns.
[Runtime]
  [LLM] provider=OllamaProvider configured=ollama model=codellama:7b status=reachable
  [LLM] endpoint=http://localhost:11434
  [VDB] engine=chromadb status=ready path=./migration_db
  [VDB] embedding_model=microsoft/codebert-base dimension=768 trusted_rules=356
  [VDB] initialized_for=Spring 3.0.0 -> Micronaut 4.10.8 mode=trusted compatible_rules=130
  [VDB] target_profile=line-aware spring_line=3.0 micronaut_line=4.10 pair_line_specific=15
  [VDB] using init snapshot for Micronaut 4.10.8
  [VDB] managed deps loaded=3549
  [VDB] snapshot channel=local_maven_repo
  [VDB] target_platform_snapshot=3549 managed path=corpus/validated_patterns/release/target_platforms/spring_3_0_0__micronaut_4_10_8.json
  [VDB] collections: annotations=39, dependencies=23, configurations=2, code_patterns=287, types=5
Starting migration from examples/spring to examples/micronaut
```

Successful `init` startup example:

```text
python main.py init --mode trusted --spring-version 3.0.0 --micronaut-version 4.10.8
--------------------------------------------------
Spring2Naut RAG Initialization
Corpus Root: corpus
Targeting: Spring 3.0.0 -> Micronaut 4.10.8
Mode: trusted
--------------------------------------------------
[INFO] Indexing 356 patterns into Vector DB using microsoft/codebert-base...
[INFO] This is a one-time operation. Subsequent runs will be near-instant.
  > Progress: 356/356 (100.0%) | Est. Remaining: 0m 0s
[OK] Intelligence Engine indexed with 356 patterns in 0m 0s.

==================================================
INITIALIZATION SUMMARY
==================================================
Raw Dataset Rules: 2849
Governed Release Rules: 356
Official Patterns: 39
Indexed Trusted Rules: 356
Trusted Rules Compatible With Target Pair: 356
Trusted Target Runtime Dataset: corpus/validated_patterns/release/target_runtime_datasets/spring_3_0_0__micronaut_4_10_8.json
Trusted Target Profile: line-aware (spring_line=3.0, micronaut_line=4.10)
Trusted Line-Specific Rules: spring=15, micronaut=15, pair=15
Trusted Compatible Rule Categories: annotations=39, code_patterns=287, configurations=2, dependencies=23, types=5
Trusted Target Profile Report: corpus/validated_patterns/release/target_profiles/spring_3_0_0__micronaut_4_10_8.json
Target Platform Managed Dependencies: 3549
Target Platform Snapshot: corpus/validated_patterns/release/target_platforms/spring_3_0_0__micronaut_4_10_8.json
Trusted DB Path: migration_db
Trusted KB Manifest: migration_db/kb_manifest.json
Governed Release Runtime Dataset: corpus/validated_patterns/release/runtime_dataset.json
KB Smoke OK: True
KB Smoke Report: corpus/validated_patterns/release/kb_smoke_report.json
Chroma Audit Trust: high
Distribution Ready: True
Chroma Audit Report: corpus/validated_patterns/release/chroma_audit_report.json
```

What to look for in a healthy startup:
- `status=reachable` for the configured LLM, if you expect LLM support
- `engine=chromadb status=ready` for the vector DB
- `initialized_for=Spring ... -> Micronaut ...` matching the exact version pair you initialized
- `KB Smoke OK: True` during `init`
- no early fatal error before `Starting migration from ...` during `migrate`

### 3. Usage
Run the migration with a single command:
```bash
python main.py migrate <path-to-spring-project> <path-to-output-directory>
```

Build tool selection:
- by default, the agent auto-detects the project build tool from the source tree
- if the project contains only Maven files, it uses Maven
- if the project contains only Gradle files, it uses Gradle
- if both `pom.xml` and `build.gradle` or `build.gradle.kts` exist, explicitly pass `--build-tool maven` or `--build-tool gradle`

### 3A. Run The Bundled Sample

If you want to test the repository sample end to end, use the Spring Petclinic project under `examples/spring/spring-petclinic` as the source and `examples/micronaut` as the generated output root.

Recommended sequence for the sample:

```bash
python main.py init --mode trusted --spring-version 3.0.0 --micronaut-version 4.10.8
```

Migrate the sample using Maven validation:

```bash
python main.py migrate examples/spring examples/micronaut \
  --spring-version 3.0.0 \
  --micronaut-version 4.10.8 \
  --build-tool maven
```

Migrate the sample using Gradle validation:

```bash
python main.py migrate examples/spring examples/micronaut \
  --spring-version 3.0.0 \
  --micronaut-version 4.10.8 \
  --build-tool gradle
```

What these flags mean:
- `--mode trusted`: use the governed trusted dataset and trusted DB only
- `--spring-version 3.0.0`: match the Spring Boot version used by the sample source build
- `--micronaut-version 4.10.8`: target this exact Micronaut runtime/BOM line
- `--build-tool maven|gradle`: force the validation/build path when you want to test one build system explicitly

Sample output location after migration:
- generated project: `examples/micronaut/spring-petclinic`
- reports: `examples/micronaut/reports`

For enterprise migrations, pass explicit source and target versions so the agent can keep dependency selection aligned to the requested Micronaut line:
```bash
python main.py migrate /path/to/spring-project /path/to/output \
  --spring-version 3.4.5 \
  --micronaut-version 4.10.8
```

If a source tree contains both `pom.xml` and `build.gradle` or `build.gradle.kts`, force the intended build path explicitly:
```bash
python main.py migrate /path/to/spring-project /path/to/output \
  --spring-version 3.0.0 \
  --micronaut-version 4.10.8 \
  --build-tool maven
```

Use exact three-part versions such as `3.4.5`, `3.3.3`, `4.10.8`, or `4.5.7`. The migration flow is now guarded against vague targets like `4.x` when running enterprise migration logic.

How to choose the versions:
- `--spring-version` should match the source project's real Spring Boot version from `pom.xml` or `build.gradle`.
- For Maven projects, this is typically the `spring-boot-starter-parent` version.
- `--micronaut-version` should be the exact Micronaut target version you want to migrate to.
- Use the same version pair for both `init` and `migrate` so the governed KB and migration execution stay aligned.

Example for a project using:
```xml
<artifactId>spring-boot-starter-parent</artifactId>
<version>3.0.0</version>
```

Use:
```bash
python main.py init --mode trusted --spring-version 3.0.0 --micronaut-version 4.10.8
python main.py migrate /path/to/spring-project /path/to/output \
  --spring-version 3.0.0 \
  --micronaut-version 4.10.8
```

Every migration now prints:
- a startup runtime banner with the active LLM provider, configured model, endpoint, LLM reachability status, vector DB path, embedding model, and trusted indexed rule counts
- the KB manifest target pair when available, plus a warning if the current migration target differs from the last trusted `init` target pair
- the forced build tool when `--build-tool` is supplied
- a warning when the target output directory already exists and is being cleared before migration
- a per-file summary line showing `deterministic_hits`, `vdb_hits`, `vdb_misses`, `llm_used`, and `llm_reason`
- elapsed time for the full run
- a colored verification summary for the migrated source tree
- build validation status and retry count

The migration engine now also hardens several enterprise-safe deterministic rewrites that used to be fragile in real projects:
- Spring runtime-hints classes are converted into valid Micronaut-side placeholders instead of broken code
- Spring local utility patterns such as `Assert.notNull`, `ToStringCreator`, `Formatter`, `Validator`, JAXB annotations, and Spring property sorting helpers are normalized into compilable local equivalents or safe placeholders
- project-local imports under `org.springframework...` are preserved when they refer to application code, while framework imports are still purged

Every migration now writes:
- `output_dir/reports/migration_report.json`
- `output_dir/reports/verification_report.json`
- `output_dir/reports/source_dependency_audit_report.json`
- `output_dir/reports/migrated_dependency_audit_report.json`
- `output_dir/reports/source_resolved_dependency_inventory.json`
- `output_dir/reports/migrated_resolved_dependency_inventory.json`

The resolved dependency inventory now preserves:
- `evidence_quality`
- `target_platform_summary`
- `repository_intelligence_summary`

This makes it easier to see whether exact Micronaut target-platform proof came from the local Maven cache or from weaker partial evidence.

The startup runtime banner is designed to answer common operator questions immediately:
- which LLM provider is active
- which model the run is configured to use
- whether the local LLM endpoint is actually reachable from the migration process
- which vector DB path and embedding model are active
- how many trusted rules are currently loaded for retrieval

## Documentation
For detailed configuration and advanced usage, see the [User Guide](USER_GUIDE.md).

For the current implementation-based architecture, design, and GA-readiness assessment, see the [Reference Architecture and GA Readiness](docs/REFERENCE_ARCHITECTURE.md).

For the new version-aware corpus model that will back future official-doc and curated GitHub ingestion, see the [Pattern Schema](docs/PATTERN_SCHEMA.md).

For the end-to-end corpus workflow, commands, trust model, and promotion path, see the [Corpus Pipeline](docs/CORPUS_PIPELINE.md).

For the recommended GA packaging and world-wide distribution approach, see the [GA Distribution Strategy](docs/GA_DISTRIBUTION.md).

## Quality Checks
Run the regression suite:
```bash
python scripts/run_regression_suite.py --tier fast
python scripts/run_regression_suite.py --tier full
```

Regression tiers:
- `fast`: stable local gate for CLI contracts, deterministic migration logic, KB lookup behavior, validation parsing, verification, and orchestrator status/report paths
- `corpus`: broader dependency-audit, orchestrator integration, corpus/governance/promotion, and fixture pipeline checks
- `full`: full test discovery across the repository, recommended for CI

Recent guardrail additions:
- fast-tier regression contracts now protect the target-platform evidence summary shown to operators, including the `CONFIGURED LOCALLY` path for Micronaut target lines that are proven in the local build config but not yet fully resolved into a managed-module inventory in that runtime
- corpus-tier dependency-audit coverage now also protects the audit-note wording for `exact_resolved`, `configured_target_line`, and `none` platform-evidence outcomes
- fast-tier regression contracts now also protect the operator-facing target-platform resolution channel, so `channel=local_maven_repo` remains visible when the exact Micronaut BOM was proven from the local Maven cache

Install the shared local guardrails with one command:
```bash
./scripts/install_local_guardrails.sh
```

This sets `core.hooksPath`, makes the hook executable, and gives you a single manual gate:
```bash
./scripts/run_local_guardrails.sh
```

Installed git hooks:
- `pre-commit`: runs the fast regression suite on every commit
- `pre-push`: runs the fast regression suite, then the corpus regression suite, then the clean Maven/Gradle dependency-audit gates before every push
- `release`: available through the shared guardrail runner; executes the explicit GA release gate

GitHub enforcement:
- `push` to `main` or `codex/**`: runs the GitHub CI workflow
- `pull_request`: runs the GitHub CI workflow
- GitHub CI runs `python scripts/run_regression_suite.py --tier full` before the broader corpus, KB, release, and packaging gates

If you prefer to enable the hook manually:
```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
chmod +x .githooks/pre-push
```

The shared local guardrails now support stages:
- `./scripts/run_local_guardrails.sh --stage pre-commit`: fast regression suite only
- `./scripts/run_local_guardrails.sh --stage pre-push`: fast regression suite, corpus regression suite, then clean Maven/Gradle dependency-audit gates
- `./scripts/run_local_guardrails.sh --stage release`: explicit GA release gate with a machine-readable release report

Run the explicit GA release gate directly:
```bash
python3 scripts/run_ga_release_gate.py
```

This writes `reports/ga_release_gate_report.json` and returns non-zero unless:
- the fast regression suite passes
- the corpus regression suite passes
- trusted `init` plus KB smoke plus Chroma trust/distribution audit pass for the requested target pair
- `docs/STRICT_GA_CHECKLIST.md` currently says `GA READY`

Important:
- this gate is intentionally honest
- even if the technical commands pass, it still refuses a GA decision when the strict checklist verdict remains `NOT GA READY`

Run the explicit release-candidate gate when we need to ship a technically clean pilot/prerelease artifact without claiming full GA:
```bash
python3 scripts/run_ga_release_gate.py --release-tier candidate --report reports/release_candidate_gate_report.json
```

Release-candidate behavior:
- requires the same technical gates as the GA path
- still records the strict checklist verdict
- returns success only when the technical gates pass and the strict checklist still allows `safe for pilot migrations with engineering review: yes`
- produces `ship_release_candidate` instead of `ship`

GitHub release automation now has two lanes:
- `.github/workflows/release.yml`: strict GA lane, only honest when the checklist says `GA READY`
- `.github/workflows/release-candidate.yml`: prerelease lane for technically clean `v*-rc*` tags

For release builds, the GitHub workflow now also emits:
- `dist/release_manifest.json`
- `dist/SHA256SUMS`

Those files are the current machine-readable integrity contract for the `2.0.0` release line.

Validate the encrypted rule datasets and Chroma indexing path:
```bash
python -m src.agent.rag.kb_validator
```

Regenerate the cleaned enhanced dataset artifact:
```bash
python -m src.agent.rag.dataset_cleaner \
  --input migration_dataset_enhanced.json.dat \
  --output migration_dataset_enhanced_cleaned.json.dat \
  --format dat \
  --report docs/dataset_cleaning_report.json
```

The runtime and release pipeline both prefer the cleaned enhanced dataset artifact when it exists, so the existing enhanced JSON/DAT corpus remains the base knowledge source rather than being bypassed.

Runtime retrieval is now version-aware as well: active Spring and Micronaut target versions are passed through the agents into the knowledge base so incompatible mappings can be filtered out before transformation.

Initialize or validate the versioned corpus workspace:
```bash
python -m src.agent.patterns.repository --init
python -m src.agent.patterns.repository --validate
```

Initialize the governed dataset, vector DB, smoke checks, and trust audit with one command:
```bash
python main.py init --spring-version 3.4.5 --micronaut-version 4.10.8
```

Beginner rule:
- run `init` first for the same version pair you plan to use in `migrate`
- then run `migrate`
- then review `output_dir/reports/`

This single command now performs:
- official pattern normalization
- validated release dataset generation
- target-pair filtering of the governed release dataset into a persisted trusted runtime subset
- trusted Chroma/vector DB initialization using that target-compatible trusted runtime dataset
- target-profile generation for the requested Spring/Micronaut pair, including line-specific compatible-rule counts and per-category breakdowns
- trusted KB manifest generation at `migration_db/kb_manifest.json`
- persisted Chroma trust/distribution audit

Important version-awareness detail:
- `init --spring-version ... --micronaut-version ...` now writes a target-compatible trusted runtime dataset under `corpus/validated_patterns/release/target_runtime_datasets/`
- the trusted DB is initialized from that filtered dataset, not from the unfiltered governed release export
- `init` still does not create one permanent DB namespace per patch pair; it reuses the trusted DB path but materializes it from the exact compatible subset requested for that run
- it also writes a target profile report under `corpus/validated_patterns/release/target_profiles/`, and records that pair plus the target-profile summary in the trusted KB manifest
- `migrate` then uses the live runtime target versions to filter applicable rules during retrieval and transformation
- during `migrate`, the dependency auditor also now tries to resolve the exact Micronaut target BOM recursively, follows imported BOMs, resolves basic Maven property placeholders, and logs whether that exact target platform resolution succeeded or degraded
- the dependency auditor now checks the local Maven cache first for exact target BOM POMs before falling back to Maven Central, so offline or network-restricted environments can still reach `exact_resolved` when the requested Micronaut artifacts already exist under the local Maven repository
- the local Maven cache path is discovered in this order: `MAVEN_LOCAL_REPOSITORY`, then Maven `settings.xml` `localRepository`, then the default `~/.m2/repository`
- if your enterprise image uses a non-default Maven settings file, point the runtime at it with `MAVEN_SETTINGS_FILE=/path/to/settings.xml`
- when the source audit proves that transitive Micronaut modules drift off the requested target line and the exact target platform map yields a trusted managed version, the build migration can now inject Maven `dependencyManagement` overrides or Gradle constraints automatically
- when the source audit proves that a surviving third-party direct dependency is reintroducing allowlisted legacy `javax.*` APIs such as `javax.validation` or `javax.servlet`, the build migration can now inject Maven exclusions or Gradle `exclude` clauses automatically
- the Java transformer now also applies target-version compatibility markers for known target-sensitive APIs, so version-line-specific review items are surfaced directly in migrated source instead of being hidden only in reports
- if you re-run `init` for `Spring 3.0.0 -> Micronaut 4.10.8` versus `Spring 3.0.0 -> Micronaut 4.5.7`, the target-profile report and manifest can legitimately differ because line-specific governed rules may apply to one Micronaut line and not the other
- use the same version pair in `init` and `migrate` for the cleanest enterprise operator experience

Initialization modes:
- `trusted` (default): builds and indexes only the governed enterprise-ready release dataset
- `legacy`: builds and indexes the old encrypted/raw dataset path explicitly
- `hybrid`: builds the trusted DB and a separate legacy/raw-plus-trusted experimental DB
- `candidate`: builds a trusted-plus-staged-candidates dataset for controlled evaluation
- `max`: builds the widest local experimental dataset from all currently materialized local sources
- `extended`: compatibility alias for `legacy`
- `both`: compatibility alias for `hybrid`

Recommended end-user command:

```bash
python main.py init --mode trusted --spring-version 3.0.0 --micronaut-version 4.10.8
```

Mode selection guidance:
- use `trusted` for normal enterprise migrations
- use `legacy` only when you intentionally want the older encrypted/raw dataset materialized for exploration
- use `hybrid` when you want governed trusted retrieval plus a separate broader experimental DB that includes legacy/raw patterns
- use `candidate` only when you want to test your own staged patterns
- use `max` only for exploration, research, or widest local experiments
- treat `extended` and `both` as backward-compatible aliases, not the preferred names

What each mode is for:
- `trusted`: use this when you want the safest default enterprise path with the governed reviewed dataset only
- `legacy`: use this when you want the old encrypted/raw dataset path loaded into its own DB explicitly
- `hybrid`: use this when you want both the trusted governed DB and a broader separate experimental DB that includes legacy/raw patterns
- `candidate`: use this when you or your team added custom staged patterns and want to test them without changing the default trusted DB
- `max`: use this when you want the widest locally materialized experimental dataset for research, discovery, or recall-heavy exploration
- `extended`: compatibility alias for `legacy`
- `both`: compatibility alias for `hybrid`

Important trust boundary:
- `trusted` and `candidate` do not load the old encrypted/raw dataset path into their DBs
- `legacy`, `hybrid`, and `max` are the only modes that intentionally load legacy encrypted/raw patterns

Normal `python main.py migrate ...` runs use the trusted DB by default.
The legacy, hybrid, candidate, and max DBs are experimental surfaces and should not replace the trusted DB for enterprise migration runs.

Examples:
```bash
python main.py init --mode trusted --spring-version 3.0.0 --micronaut-version 4.10.8
python main.py init --mode legacy --spring-version 3.0.0 --micronaut-version 4.10.8
python main.py init --mode hybrid --spring-version 3.0.0 --micronaut-version 4.10.8
python main.py init --mode candidate --spring-version 3.0.0 --micronaut-version 4.10.8
python main.py init --mode max --spring-version 3.0.0 --micronaut-version 4.10.8
python main.py init --mode extended --spring-version 3.0.0 --micronaut-version 4.10.8
python main.py init --mode both --spring-version 3.0.0 --micronaut-version 4.10.8
python main.py init --mode max --spring-version 3.0.0 --micronaut-version 4.10.8
python main.py init --mode both --spring-version 3.0.0 --micronaut-version 4.10.8
```

## What Happens During Migration

When you run:

```bash
python main.py migrate /path/to/spring-project /path/to/output \
  --spring-version 3.0.0 \
  --micronaut-version 4.10.8
```

the agent does this:

1. analyzes the source project structure
2. detects or uses the forced build tool
3. audits source dependencies and target-platform compatibility
4. copies the source into a fresh output directory
5. rewrites build files, configs, and Java sources
6. verifies source-to-target file coverage
7. runs build validation and limited self-refinement
8. writes machine-readable reports under `output_dir/reports/`

The migration path is:
- deterministic rewrite first
- trusted vector retrieval second
- LLM refinement only when needed
- build validation and report generation at the end

## What To Review After Migration

Always review these outputs:
- `output_dir/reports/migration_report.json`
- `output_dir/reports/verification_report.json`
- `output_dir/reports/source_dependency_audit_report.json`
- `output_dir/reports/migrated_dependency_audit_report.json`
- `output_dir/reports/source_resolved_dependency_inventory.json`
- `output_dir/reports/migrated_resolved_dependency_inventory.json`

Recommended operator review order:
- check `verification_report.json` for leftover Spring markers
- check migrated dependency audit findings
- build and test the migrated project
- review files where `llm_used=true` or `llm_reason` is non-empty

## Common End-User Commands

Trusted local initialization:

```bash
python main.py init --mode trusted --spring-version 3.0.0 --micronaut-version 4.10.8
```

Maven migration:

```bash
python main.py migrate /path/to/spring-project /path/to/output \
  --spring-version 3.0.0 \
  --micronaut-version 4.10.8 \
  --build-tool maven
```

Gradle migration:

```bash
python main.py migrate /path/to/spring-project /path/to/output \
  --spring-version 3.0.0 \
  --micronaut-version 4.10.8 \
  --build-tool gradle
```

Candidate-pattern experimentation:

```bash
python main.py init --mode candidate --spring-version 3.0.0 --micronaut-version 4.10.8
export VECTOR_DB_PATH=./migration_db_candidate
python main.py migrate /path/to/spring-project /path/to/output \
  --spring-version 3.0.0 \
  --micronaut-version 4.10.8
```

Run local guardrails:

```bash
./scripts/run_local_guardrails.sh --stage pre-push
```

Clear end-user sequence:
1. run `python main.py init --mode trusted --spring-version ... --micronaut-version ...`
2. run `python main.py migrate <source> <target> --spring-version ... --micronaut-version ... --build-tool ...`
3. inspect `target/reports/` for warnings and review items
4. build and test the migrated application itself

Run the explicit release-candidacy gate:

```bash
python3 scripts/run_ga_release_gate.py
```

## Add Your Own Patterns

End users can extend the local dataset without changing core Python logic.

Recommended path for local custom patterns:

1. add your custom pattern to:
   - `corpus/staged_patterns/candidates/index.json`
2. optionally add a matching standalone JSON file under:
   - `corpus/staged_patterns/candidates/`
3. rebuild the Chroma DB with candidate mode
4. point `migrate` to the candidate DB for that run

Why this path is recommended:
- it lets you test your own local patterns without changing the governed trusted enterprise catalog
- it keeps your additions separate from the default trusted release DB

### Minimal pattern shape

Add a new entry under the `patterns` array in `corpus/staged_patterns/candidates/index.json`:

```json
{
  "pattern_id": "custom.annotation.request_header",
  "pattern_type": "annotation",
  "spring_pattern": "@RequestHeader",
  "micronaut_pattern": "@Header",
  "description": "Custom local mapping for request header injection.",
  "spring_versions": {
    "minimum": "3.0.0",
    "maximum": "3.6.99"
  },
  "micronaut_versions": {
    "minimum": "4.0.0",
    "maximum": "4.10.99"
  },
  "status": "candidate",
  "confidence": 0.9,
  "complexity": "low",
  "category": "annotations",
  "source_kind": "manual",
  "evidence": [],
  "examples": [],
  "metadata": {
    "automated_migration_supported": true,
    "user_added": true
  }
}
```

### Supported `pattern_type` values

Common values:
- `annotation`
- `configuration`
- `dependency`
- `dependency_injection`
- `type`
- `application`
- `code_pattern`
- `import`

### Load your custom patterns into Chroma

Rebuild the candidate DB:

```bash
python main.py init --mode candidate --spring-version 3.0.0 --micronaut-version 4.10.8
```

This writes the candidate runtime dataset and candidate Chroma DB, typically at:
- `corpus/validated_patterns/candidate/runtime_dataset.json`
- `migration_db_candidate`

### Use your custom candidate DB during migration

By default, `migrate` uses the trusted DB at `./migration_db`.

If you want the migration run to use your custom candidate patterns, point the runtime to the candidate DB:

```bash
export VECTOR_DB_PATH=./migration_db_candidate
python main.py migrate /path/to/spring-project /path/to/output \
  --spring-version 3.0.0 \
  --micronaut-version 4.10.8
```

When you want to go back to the default trusted DB:

```bash
unset VECTOR_DB_PATH
```

### If you want the widest local experimental pattern set

Instead of candidate mode, you can build the `max` DB:

```bash
python main.py init --mode max --spring-version 3.0.0 --micronaut-version 4.10.8
export VECTOR_DB_PATH=./migration_db_max
python main.py migrate /path/to/spring-project /path/to/output \
  --spring-version 3.0.0 \
  --micronaut-version 4.10.8
```

### Important guidance

- update `corpus/staged_patterns/candidates/index.json`, because that index is what the candidate release build reads
- keep `spring_pattern` and `micronaut_pattern` specific and non-generic
- keep version windows honest so incompatible rules are filtered out
- candidate and max DBs are for local experimentation and custom extensions, not the default enterprise-governed path
- if Chroma is unavailable in your environment, the dataset files can still be written, but the live vector DB may not be indexed

Console summary fields:
- `Raw Dataset Rules`: cleaned larger raw dataset size
- `Governed Release Rules`: trusted reviewed release size
- `Indexed Trusted Rules`: rules indexed into the trusted DB
- `Trusted Rules Compatible With Target Pair`: governed rules whose version windows match the `init` target pair
- `Trusted Target Runtime Dataset`: the persisted filtered trusted dataset that was actually indexed for the requested pair
- `Trusted Target Profile`: whether the requested pair is covered only by broad `3.x/4.x` rules or also by line-specific governed rules
- `Trusted Line-Specific Rules`: how many compatible governed rules are narrowed to the requested Spring line, Micronaut line, or both
- `Trusted Compatible Rule Categories`: per-category count of compatible governed rules for the requested pair
- `Trusted Target Profile Report`: persisted JSON report under `corpus/validated_patterns/release/target_profiles/`
- `Trusted KB Manifest`: persisted init metadata that records the trusted DB target pair, target lines, compatible governed rule count, and target-profile summary
- `Indexed Extended Rules`: rules indexed into the extended DB when `--mode extended` or `--mode both` is used
- `Candidate Runtime Rules`: total candidate dataset rows before retrieval deduplication
- `Indexed Candidate Rules`: trusted release plus staged-candidate rules
- `Max Runtime Rules`: widest local known-pattern dataset rows before retrieval deduplication
- `Indexed Max Rules`: widest local experimental rule set currently available on disk

Current local corpus scale after the latest hardening pass:
- `Governed Release Rules`: 130
- `Candidate Runtime Rules`: 75

## Limits And Honest Expectations

Current honest usage guidance:
- good for controlled internal testing: yes
- good for pilot migrations with engineering review: yes
- ready for unqualified enterprise GA release: no

Why:
- some migration surfaces still need broader deterministic coverage
- some third-party dependency ecosystems still require review
- build/self-fix paths are improved but not universally deterministic
- the release gate can still block GA even when many technical checks pass

Use these two artifacts to judge readiness:
- `docs/STRICT_GA_CHECKLIST.md`
- `reports/ga_release_gate_report.json` after running `python3 scripts/run_ga_release_gate.py`
- `Indexed Candidate Rules`: 75
- `Raw Dataset Rules`: 2849
- `Max Runtime Rules`: 6911
- `Indexed Max Rules`: 2924

Why this is not 10k or 20k yet:
- the repo does not currently contain a hidden 10k-plus reviewed Spring-to-Micronaut corpus
- the `2849` raw dataset is the real cleaned legacy/raw baseline currently present on disk
- larger counts will require new ingestion and governance work, not just a CLI flag
- the `6911` max dataset is intentionally deduplicated down to `2924` indexed retrieval records so semantically repeated mappings do not pollute vector search quality

Materialize the official documentation seed catalog:
```bash
python -m src.agent.patterns.official_seeds --write
```

The official seed catalog now includes Spring Boot support-policy coverage plus version-specific Spring Boot `3.0` through `3.5` release-note sources so corpus promotion can stay minor-line aware.

Normalize official documentation seeds into versioned patterns:
```bash
python -m src.agent.patterns.official_normalizer --write
```

Materialize curated GitHub candidate sources:
```bash
python -m src.agent.patterns.github_candidates --write
```

Normalize curated GitHub candidate sources into candidate patterns:
```bash
python -m src.agent.patterns.github_normalizer --write
```

Promote safe GitHub candidates into staged review:
```bash
python -m src.agent.patterns.promotion --write
```

Build a validated GA release dataset by merging validated corpus patterns on top of the existing cleaned enhanced dataset:
```bash
python -m src.agent.patterns.release --write
```

Bootstrap the existing cleaned runtime dataset into the versioned corpus archive:
```bash
python -m src.agent.patterns.legacy_bootstrap --write
```

Promote the safest consolidated legacy mappings into the validated release set:
```bash
python -m src.agent.patterns.legacy_promotion --write
```

Review promoted legacy mappings into `GA-ready` vs `needs fixture validation` buckets:
```bash
python -m src.agent.patterns.legacy_review --write
```

Generate the fixture-validation registry for the held-back backlog:
```bash
python -m src.agent.patterns.fixture_registry --write
```

Seed generic fixture packs for the reviewed backlog:
```bash
python -m src.agent.patterns.fixture_packs --write
```

Validate fixture execution readiness for the reviewed backlog:
```bash
python -m src.agent.patterns.fixture_execution --write
```

Compile seeded fixture packs offline with local `javac` and framework stubs:
```bash
python -m src.agent.patterns.fixture_compile --write
```

Build the validated release dataset, load it into ChromaDB, and run sample retrieval smoke tests:
```bash
python -m src.agent.rag.kb_release_smoke --write
```

Audit persisted Spring-to-Micronaut Chroma metadata and derive a retrieval trust level:
```bash
python -m src.agent.rag.chroma_audit --write
```

Audit direct and transitive Maven or Gradle dependencies for migration risks:
```bash
python -m src.agent.agents.dependency_audit \
  --build-file /path/to/pom.xml \
  --project-path /path/to/project \
  --spring-version 3.4.5 \
  --micronaut-version 4.10.8 \
  --report reports/dependency_audit.json \
  --fail-on blocking
```

Current fixture execution state:

- 26 reviewed fixture requirements tracked
- 26 reviewed requirements seeded and execution-ready
- 0 blocking fixture execution issues in the current report
- 0 remaining fixture backlog items in the current seeded corpus
- 25 seeded packs cover those 26 requirements because one Redis pack validates both Spring Data Redis and Jedis migration shapes
- 25 of 25 seeded packs also pass offline compile validation
- the fixture corpus now covers both legacy framework-risk mappings and reviewed dependency-domain migration shapes for security, validation, observability, Redis, Kafka, RabbitMQ, and declarative HTTP clients
- validated runtime release dataset now contains 130 canonical retrieval-safe rules after collapsing 1,140 duplicate mapping pairs
- the official validated release layer now contributes 39 version-aware normalized patterns, including endpoint, binding, bean-factory, stereotype, dependency-injection, exception/status, property, transaction, cache, scheduling, async, configuration-properties, Spring MVC model/view, and Spring Data paging guidance
- overlapping annotation and code-pattern mappings are now governed by the official validated layer first, reducing the GA-ready legacy contribution in the release export to 3 higher-value residual patterns while increasing the audited runtime surface to 130 governed rules
- the maintained Spring Petclinic regression path now verifies at `0 blocking / 0 review`, with Spring-only framework gaps surfaced in the console as governed manual-review guidance instead of blind vector misses
- the maintained Maven Petclinic end-to-end migration path now completes with `Build Validation: PASSED (1 attempt)` in deterministic-only mode against `examples/micronaut`
- safe project-local imports under `org.springframework.samples...` such as `NamedEntity` and `Person` are now restored after framework-import cleanup so domain model inheritance does not break the migrated build
- persisted Chroma audit now reports trust level `high`
- persisted release Chroma audit now also reports `distribution_ready = true` for the current validated release dataset
- the validated release dependency collection now contains 23 reviewed enterprise dependency mappings instead of being empty, and release smoke tests now require dependency retrieval to succeed for `spring-boot-starter-web`, `spring-boot-starter-jdbc`, `spring-boot-starter-cache`, `org.ehcache:ehcache`, and `org.springdoc:springdoc-openapi-ui`
- dependency compatibility auditing now covers direct and transitive Maven and Gradle dependencies locally and flags blocking/review issues before enterprise sign-off
- the dependency audit now understands Micronaut parent/BOM-managed direct dependency versions and separates them from risky missing-version cases
- Maven migration cleanup now strips common Spring-era build blockers such as `spring-boot-maven-plugin`, `git-commit-id-plugin`, Spring plugin repositories, and Spring formatting/checkstyle/jacoco plugins when generating the Micronaut POM
- the dependency audit now also uses a curated compatibility catalog for high-risk and common enterprise libraries such as Springfox, springdoc WebMVC starters, Spring Cloud Gateway, OpenFeign, Jedis, Spring Security, Actuator, Kafka, RabbitMQ, and Redis starters
- the Maven dependency audit now also captures effective-POM-resolved direct dependencies and separate compile/runtime resolved inventories when Maven evidence is available
- the Gradle dependency audit now also captures separate `compileClasspath` and `runtimeClasspath` resolved inventories and persists those evidence artifacts in the same report structure used by Maven
- the Gradle dependency audit now also infers unique resolved direct dependencies from those scope graphs, so Micronaut replacement findings can prefer the resolved Gradle version over Maven Central latest when local build evidence proves it
- the dependency audit now separates Maven Central "latest available" evidence from BOM-compatible recommended versions so Micronaut platform-managed upgrades are not mistaken for blind latest-version upgrades
- when Maven effective-POM dependency management is available, curated Micronaut replacement findings now prefer the exact BOM-managed version over the repository latest
- the Maven dependency audit now also inspects published Maven Central POM descriptors for curated replacement jars and records declared compile/runtime dependency counts plus risky Spring or `javax` carry-over signals
- the Maven dependency audit now also walks one additional level into resolvable child dependencies from those published POM descriptors so immediate downstream Spring or `javax` carry-over can be flagged before auto-upgrading
- curated compatibility-catalog repository intelligence now enriches both Maven and Gradle audit paths, so risky replacement candidates surface the same explicit repository finding codes in either build tool flow
- repository descriptor risks now surface as explicit audit findings and are also summarized under `repository_intelligence_summary` in the JSON audit report, making the agent easier to test and gate in CI
- a golden repository-risk Maven fixture now regression-checks the JSON audit summary and threshold-failure behavior for a known unsafe migration case
- the orchestrator now also has golden Maven and Gradle migration-report subset fixtures for both minimal and risky migrations so end-to-end JSON output stays regression-checked without depending on volatile timestamps or temp paths
- the shared regression suite now also includes a more realistic Maven REST-service fixture covering `@Value`, constructor injection, `@RequestBody` to `@Body`, `@RequestMapping(method=...)`, and `ResponseEntity` to `HttpResponse`
- the shared regression suite now also covers deterministic narrow `@ConditionalOnProperty(prefix/name/havingValue)` and direct-name `@ConditionalOnProperty(name/havingValue)` migration to Micronaut `@Requires(property=..., value=...)`
- dependency audit reports can now be persisted as JSON and CI fails on blocking findings for clean generic Maven and Gradle fixtures
- orchestrated migrations now also write source and post-migration dependency audit reports under `output_dir/reports/`
- orchestrated migrations now also write a top-level `output_dir/reports/migration_report.json` with summary counts, validation outcome, warning state, versions, and audit report paths
- orchestrated migrations now also write `output_dir/reports/verification_report.json` with file-by-file migrated-source findings such as leftover Spring test APIs, Markdown fences, and unresolved web/data patterns
- orchestrated migrations now also surface post-migration dependency audit summaries in the top-level warning flow
- Gradle dependency audit reports now also materialize compile/runtime evidence files under `output_dir/reports/dependency_evidence/` when those scope graphs are available
- the shared test suite now includes an end-to-end orchestrator regression that migrates a Maven-shaped sample project and compiles the migrated Java offline with Micronaut stubs
- a repo-managed `.githooks/pre-push` can now run the fast local GA guardrail set before pushes
- deterministic Java rewrites now also cover explicit `@RequestMapping(method = RequestMethod.X)` controller methods, common `ResponseEntity` factory idioms, qualifier-aware constructor injection, `@Value("${...}")` to `@Property(...)`, `@ControllerAdvice` plus `@ExceptionHandler` flows, and `@RequestBody` to `@Body` migration before any LLM refinement is attempted
- the Java transformer now uses an AST-aware analysis pass via `javalang` before fallback regex rewrites so annotation discovery is structure-aware and avoids rewriting string literals or comments as if they were Java code

## Support
Professional support and custom enterprise datasets are available for large-scale migrations.

---
© 2024 Spring2Naut-RAG | Licensed under MIT
