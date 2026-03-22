# User Guide - Spring2Naut-RAG

A practical guide to transforming Spring Boot projects toward Micronaut with the current Spring2Naut-RAG CLI.

## What This Tool Is For

Spring2Naut-RAG is for end users who want to:
- migrate Spring Boot `3.x.x` codebases toward Micronaut `4.x.x`
- run the migration locally with Ollama when privacy matters
- initialize a governed trusted vector knowledge base before migration
- inspect dependency compatibility and migration reports after each run
- extend the local dataset with candidate patterns for their own organization

Current honest usage guidance:
- good for controlled internal testing: yes
- good for pilot migrations with engineering review: yes
- ready for unqualified enterprise GA release: no

## End-To-End Flow

Use this order:

1. install dependencies
2. configure your LLM runtime
3. choose exact Spring and Micronaut versions
4. run `init`
5. run `migrate`
6. review reports and build/test the migrated output
7. optionally add custom patterns and rebuild a candidate DB
8. run guardrails or the release gate before sharing results broadly

## Prerequisites

- Python 3.8 or higher
- Maven or Gradle (installed and on PATH)
- LLM Provider (choose one):
  - **Ollama** (recommended for privacy, free, local)
  - OpenAI API (Higher accuracy for complex logic)
  - Anthropic Claude / Groq

## Quick Start

### Step 0: Clone And Enter The Repo

```bash
git clone https://github.com/ajitpattar708/Spring2Naut-RAG.git
cd Spring2Naut-RAG
```

### Step 1: Install Dependencies

```bash
pip install -r requirements_file.txt
```

Optional package-style local install:

```bash
pip install .
spring2naut --help
```

Release artifact verification:
- the release workflow now emits `dist/release_manifest.json` and `dist/SHA256SUMS`
- use those files to verify which wheel/sdist artifacts were produced and what GA gate decision accompanied them

### Step 2: Set Up LLM (Choose One)

#### Option A: Ollama (Free, Local)

Install Ollama from [ollama.com](https://ollama.com).

```bash
# Pull the recommended model
ollama pull codellama:7b
```

#### Option B: OpenAI (Recommended for Accuracy)

```bash
export OPENAI_API_KEY=your-api-key
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4-turbo
```

Local-first recommendation:
- prefer Ollama for enterprise/private source code
- keep the Ollama server running before you start `migrate`
- even when an LLM is configured, the agent still prefers deterministic rewrites first

### Step 2A: Pick Exact Versions

Use explicit three-part versions only:
- Spring examples: `3.0.0`, `3.3.3`, `3.4.5`
- Micronaut examples: `4.5.7`, `4.9.2`, `4.10.8`

How to pick them:
- `--spring-version`: match the real source project version from Maven parent, Gradle plugin, or effective build
- `--micronaut-version`: the exact target Micronaut version you want
- use the same pair for `init` and `migrate`

### Step 3: Initialize The Governed Vector DB

Before running a real RAG-based migration, initialize the governed dataset and local Chroma vector DB:

```bash
python main.py init --mode trusted --spring-version 3.4.5 --micronaut-version 4.10.8
```

This single command:
- builds the official governed patterns
- builds the validated release dataset
- writes a target-compatible trusted runtime dataset for the requested Spring and Micronaut pair
- loads that filtered trusted runtime dataset into the local vector DB
- writes a target-profile report for that exact pair, including compatible-rule category counts and line-specific rule counts
- writes a trusted KB manifest to `migration_db/kb_manifest.json`
- runs KB smoke checks
- writes the persisted Chroma trust/distribution audit

What `init` versions really do:
- `--spring-version` and `--micronaut-version` now materialize a filtered trusted runtime dataset under `corpus/validated_patterns/release/target_runtime_datasets/`
- the trusted DB is initialized from that filtered dataset for the requested pair
- they also generate a persisted target-profile report under `corpus/validated_patterns/release/target_profiles/`, and stamp the trusted KB manifest with the intended target pair and target lines
- later, `migrate` still uses its own runtime versions to filter applicable rules from the trusted KB
- during `migrate`, the dependency auditor now also tries to resolve the exact Micronaut target BOM recursively, follows imported BOMs, resolves basic Maven property placeholders, and prints platform-intelligence status in the migration logs
- the dependency auditor now checks the local Maven cache first for requested Micronaut BOM POMs before falling back to Maven Central, so exact target-platform evidence can still be proven offline when those artifacts already exist locally
- the local Maven cache path is discovered in this order: `MAVEN_LOCAL_REPOSITORY`, then Maven `settings.xml` `localRepository`, then the default `~/.m2/repository`
- if your workstation or CI image uses a non-default Maven settings file, point the runtime at it with `MAVEN_SETTINGS_FILE=/path/to/settings.xml`
- when the source audit proves that transitive Micronaut modules drift off the requested target line and the exact target platform map yields a trusted managed version, `migrate` can now inject Maven `dependencyManagement` overrides or Gradle constraints automatically
- when the source audit proves that a surviving third-party direct dependency is reintroducing allowlisted legacy `javax.*` APIs such as `javax.validation` or `javax.servlet`, `migrate` can now inject Maven exclusions or Gradle `exclude` clauses automatically
- the Java transformer now also emits target-version compatibility review markers for selected target-sensitive APIs so version-line-specific migration risk shows up in migrated source files too
- this means `init --spring-version 3.0.0 --micronaut-version 4.10.8` and `init --spring-version 3.0.0 --micronaut-version 4.5.7` can produce different filtered trusted datasets and different target-profile evidence even though they reuse the same trusted DB path
- for enterprise usage, keep the same version pair across `init` and `migrate` unless you intentionally want to reuse the trusted DB for a nearby target line

Initialization modes:
- `trusted` (default): recommended enterprise mode; indexes only the governed trusted release dataset
- `legacy`: explicitly indexes the older encrypted/raw dataset path into its own experimental DB
- `hybrid`: builds the trusted governed DB and a separate experimental DB that combines trusted plus legacy/raw patterns
- `candidate`: indexes the trusted release dataset plus staged candidate rules for controlled evaluation
- `max`: indexes the widest local experimental dataset built from all materialized local sources
- `extended`: compatibility alias for `legacy`
- `both`: compatibility alias for `hybrid`

Recommended user choice:
- use `trusted` for normal enterprise migration work
- use `legacy` only when you intentionally want to explore the older encrypted/raw patterns
- use `hybrid` when you want both the trusted DB and a broader experimental DB that also includes legacy/raw patterns
- use `candidate` only when testing your own added patterns
- use `max` only for broader local experimentation
- treat `extended` and `both` as backward-compatible aliases

What each mode is for:
- `trusted`: safest default for real migrations; only the governed reviewed release dataset is indexed
- `legacy`: explicit old encrypted/raw dataset path for exploration and fallback investigation
- `hybrid`: trusted governed dataset plus a separate broader experimental DB that includes legacy/raw patterns
- `candidate`: trusted dataset plus your staged candidate patterns for controlled testing
- `max`: widest locally materialized experimental dataset for research or wider recall
- `extended`: compatibility alias for `legacy`
- `both`: compatibility alias for `hybrid`

Important trust boundary:
- `trusted` and `candidate` do not load the old encrypted/raw dataset path into their DBs
- `legacy`, `hybrid`, and `max` are the modes that intentionally pull those older encrypted/raw patterns into the indexed dataset path

Normal `python main.py migrate ...` runs use the trusted DB by default.
Treat the legacy, hybrid, candidate, and max DBs as exploratory surfaces, not the default enterprise-governed retrieval path.

Examples:

```bash
python main.py init --mode trusted --spring-version 3.0.0 --micronaut-version 4.10.8
python main.py init --mode legacy --spring-version 3.0.0 --micronaut-version 4.10.8
python main.py init --mode hybrid --spring-version 3.0.0 --micronaut-version 4.10.8
python main.py init --mode candidate --spring-version 3.0.0 --micronaut-version 4.10.8
python main.py init --mode max --spring-version 3.0.0 --micronaut-version 4.10.8
python main.py init --mode extended --spring-version 3.0.0 --micronaut-version 4.10.8
python main.py init --mode both --spring-version 3.0.0 --micronaut-version 4.10.8
```

### Step 3A: Add Your Own Patterns To The Dataset

If you want to teach the tool a new mapping locally, the simplest supported end-user path is to add it as a candidate pattern and rebuild the candidate Chroma DB.

Recommended user flow:
1. edit `corpus/staged_patterns/candidates/index.json`
2. add your new pattern under the `patterns` array
3. run `init --mode candidate`
4. point migration to `./migration_db_candidate`

Why this is the safest user path:
- it does not require changing internal Python authoring code
- it keeps your local customizations separate from the default trusted DB
- you can test your pattern without affecting the governed enterprise-ready release set

Example pattern entry:

```json
{
  "pattern_id": "custom.code.rest_template_exchange",
  "pattern_type": "code_pattern",
  "spring_pattern": "RestTemplate.exchange",
  "micronaut_pattern": "HttpClient.exchange",
  "description": "Local custom mapping for RestTemplate exchange usage.",
  "spring_versions": {
    "minimum": "3.0.0",
    "maximum": "3.6.99"
  },
  "micronaut_versions": {
    "minimum": "4.0.0",
    "maximum": "4.10.99"
  },
  "status": "candidate",
  "confidence": 0.85,
  "complexity": "medium",
  "category": "code_patterns",
  "source_kind": "manual",
  "evidence": [],
  "examples": [],
  "metadata": {
    "automated_migration_supported": true,
    "user_added": true
  }
}
```

Useful `pattern_type` values:
- `annotation`
- `configuration`
- `dependency`
- `dependency_injection`
- `type`
- `application`
- `code_pattern`
- `import`

Then rebuild the candidate DB:

```bash
python main.py init --mode candidate --spring-version 3.0.0 --micronaut-version 4.10.8
```

This produces:
- `corpus/validated_patterns/candidate/runtime_dataset.json`
- `migration_db_candidate`

By default, `python main.py migrate ...` uses `./migration_db`.
To make the migration run use your custom candidate patterns, point the runtime to the candidate DB:

```bash
export VECTOR_DB_PATH=./migration_db_candidate
python main.py migrate <path-to-spring-project> <path-to-output-directory> \
    --spring-version 3.0.0 \
    --micronaut-version 4.10.8
```

When you want to switch back to the default trusted governed DB:

```bash
unset VECTOR_DB_PATH
```

If you want the widest local experimental DB instead of only your candidate additions:

```bash
python main.py init --mode max --spring-version 3.0.0 --micronaut-version 4.10.8
export VECTOR_DB_PATH=./migration_db_max
python main.py migrate <path-to-spring-project> <path-to-output-directory> \
    --spring-version 3.0.0 \
    --micronaut-version 4.10.8
```

Important notes:
- update `corpus/staged_patterns/candidates/index.json`, because that index is what candidate dataset materialization reads
- keep patterns specific; vague mappings reduce retrieval trust
- keep version windows accurate so the KB can filter incompatible rules
- candidate and max DBs are user-extension and experimentation paths, not the default enterprise-governed trusted path
- if Chroma is unavailable locally, the dataset JSON can still be generated even though the live DB may not be indexed

What the summary means:
- `Raw Dataset Rules`: cleaned larger raw dataset size
- `Governed Release Rules`: trusted reviewed release size
- `Indexed Trusted Rules`: rules indexed into the trusted DB
- `Trusted Rules Compatible With Target Pair`: governed rules currently compatible with the exact `init` pair you requested
- `Trusted Target Runtime Dataset`: persisted filtered trusted dataset that was actually indexed for that exact pair
- `Trusted KB Manifest`: the persisted metadata file that records the last trusted init target pair and compatible governed rule count
- `Trusted Target Profile`: whether the requested pair is covered only by broad family rules or also by line-specific governed rules
- `Trusted Line-Specific Rules`: how many compatible governed rules are narrowed to the requested Spring line, Micronaut line, or both
- `Trusted Compatible Rule Categories`: per-category count of compatible governed rules for the requested pair
- `Trusted Target Profile Report`: persisted JSON report for the last trusted init target pair
- `Indexed Extended Rules`: rules indexed into the extended DB when enabled
- `Candidate Runtime Rules`: total candidate dataset rows before retrieval deduplication
- `Indexed Candidate Rules`: trusted release plus staged candidate rules
- `Max Runtime Rules`: widest local known-pattern dataset rows before retrieval deduplication
- `Indexed Max Rules`: widest local experimental rule set currently available

How to choose the versions:
- `--spring-version` should match the source project's actual Spring Boot version.
- For Maven projects, use the version from `spring-boot-starter-parent` in `pom.xml`.
- For Gradle projects, use the Spring Boot plugin version or the effective Spring Boot version used by the build.
- `--micronaut-version` should be the exact Micronaut target version you want.
- Use the same version pair in both `init` and `migrate`.

Example:

If your Maven project contains:

```xml
<artifactId>spring-boot-starter-parent</artifactId>
<version>3.0.0</version>
```

Then run:

```bash
python main.py init --mode trusted --spring-version 3.0.0 --micronaut-version 4.10.8
python main.py migrate <path-to-spring-project> <path-to-output-directory> \
    --spring-version 3.0.0 \
    --micronaut-version 4.10.8
```

### Step 4A: Run The Repository Sample

If you want a known local sample before trying your own application, use the bundled Spring sample in [examples/spring/spring-petclinic](/Users/ajpattar/rag-agent/Spring2Naut-RAG/examples/spring/spring-petclinic).

Initialize the trusted dataset for the sample version pair:

```bash
python main.py init --mode trusted --spring-version 3.0.0 --micronaut-version 4.10.8
```

Run the sample migration with Maven:

```bash
python main.py migrate examples/spring examples/micronaut \
    --spring-version 3.0.0 \
    --micronaut-version 4.10.8 \
    --build-tool maven
```

Run the sample migration with Gradle:

```bash
python main.py migrate examples/spring examples/micronaut \
    --spring-version 3.0.0 \
    --micronaut-version 4.10.8 \
    --build-tool gradle
```

How to use these commands:
- use the same exact version pair for `init` and `migrate`
- use `--build-tool maven` when you want Maven validation/build output
- use `--build-tool gradle` when you want Gradle validation/build output
- the generated sample project is written under [examples/micronaut/spring-petclinic](/Users/ajpattar/rag-agent/Spring2Naut-RAG/examples/micronaut/spring-petclinic)
- migration reports are written under [examples/micronaut/reports](/Users/ajpattar/rag-agent/Spring2Naut-RAG/examples/micronaut/reports)

Recommended end-user sequence:
1. find the real Spring Boot version in the source build file
2. choose the exact Micronaut target version
3. run `python main.py init --mode trusted --spring-version <source-version> --micronaut-version <target-version>`
4. run `python main.py migrate <source-project> <output-dir> --spring-version <source-version> --micronaut-version <target-version> --build-tool <maven|gradle>`
5. inspect `output-dir/reports/`
6. build and test the migrated project itself

Current local corpus scale:
- `Governed Release Rules`: 130
- `Candidate Runtime Rules`: 75
- `Indexed Candidate Rules`: 75
- `Raw Dataset Rules`: 2849
- `Max Runtime Rules`: 6911
- `Indexed Max Rules`: 2924

Important:
- there is no hidden local 10k-plus trusted corpus in this repository today
- if you want 10k to 20k patterns, we must ingest and govern more source material first
- the max dataset is larger because it expands known local patterns across concrete version anchors, but the retrieval index still deduplicates that down to canonical records for better search quality

### Step 4: Run Migration

Use the modular migration engine to transform your project. The tool will automatically detect your build system (Maven/Gradle), migrate dependencies, transform code, and perform a self-refinement loop.

```bash
python main.py migrate <path-to-spring-project> <path-to-output-directory> \
    --spring-version 3.4.5 \
    --micronaut-version 4.10.8
```

If the input tree contains both Maven and Gradle build files, force the intended path:

```bash
python main.py migrate <path-to-spring-project> <path-to-output-directory> \
    --spring-version 3.0.0 \
    --micronaut-version 4.10.8 \
    --build-tool maven
```

This is especially useful for samples such as Petclinic where both `pom.xml` and `build.gradle` may exist in the same project root.
If the target output directory already exists, the migration run now clears it first and prints a warning before writing fresh generated output.

At the start of each migration, the tool now prints a runtime banner that shows:
- the configured LLM provider and model
- the LLM endpoint and whether it is reachable from the running process
- the vector DB engine and path
- the embedding model and dimension
- the trusted indexed rule count and available collection counts
- the trusted KB manifest target pair when present
- a warning if the current migration target differs from the last trusted `init` target pair
- the forced build tool when `--build-tool` is supplied

During per-file Java migration, the tool now also prints a compact summary line:
- `deterministic_hits`: how many symbols or structural patterns were handled locally without retrieval
- `vdb_hits`: how many mappings were taken from the vector DB
- `vdb_misses`: how many looked-up symbols had no trustworthy KB match
- `llm_used`: whether LLM refinement was actually invoked for that file
- `llm_reason`: the explicit reason the file crossed from deterministic/RAG handling into LLM refinement

This makes the execution order easier to follow:
1. deterministic rewrite first
2. vector lookup for uncovered symbols
3. LLM refinement only when complex Spring leftovers still remain

After each run, check these generated artifacts under `<path-to-output-directory>/reports/`:
- `migration_report.json`: top-level summary, versions, warnings, elapsed time, and build-validation outcome
- `verification_report.json`: file-by-file migrated-source audit for leftover Spring APIs and risky unresolved patterns
- `source_dependency_audit_report.json`: dependency risks in the original project
- `migrated_dependency_audit_report.json`: dependency risks still present after migration
- `source_resolved_dependency_inventory.json` and `migrated_resolved_dependency_inventory.json`: persisted dependency-evidence snapshots, including `evidence_quality`, `target_platform_summary`, and `repository_intelligence_summary`

### Step 5: Validate The Migrated Output

Recommended end-user review order:

1. open `<output>/reports/verification_report.json`
2. open `<output>/reports/migrated_dependency_audit_report.json`
3. build the migrated project directly with Maven or Gradle
4. run tests for the migrated project
5. inspect files where `llm_used` was true or where manual-review markers were inserted

Typical build commands:

Maven:

```bash
cd <path-to-output-directory>
mvn test
```

Gradle:

```bash
cd <path-to-output-directory>
./gradlew test
```

### Common End-User Command Patterns

Trusted initialization:

```bash
python main.py init --mode trusted --spring-version 3.0.0 --micronaut-version 4.10.8
```

Maven migration:

```bash
python main.py migrate <path-to-spring-project> <path-to-output-directory> \
    --spring-version 3.0.0 \
    --micronaut-version 4.10.8 \
    --build-tool maven
```

Gradle migration:

```bash
python main.py migrate <path-to-spring-project> <path-to-output-directory> \
    --spring-version 3.0.0 \
    --micronaut-version 4.10.8 \
    --build-tool gradle
```

Candidate-pattern testing:

```bash
python main.py init --mode candidate --spring-version 3.0.0 --micronaut-version 4.10.8
export VECTOR_DB_PATH=./migration_db_candidate
python main.py migrate <path-to-spring-project> <path-to-output-directory> \
    --spring-version 3.0.0 \
    --micronaut-version 4.10.8
```

Switch back to trusted DB:

```bash
unset VECTOR_DB_PATH
```

Default sequence for most users:

```bash
python main.py init --mode trusted --spring-version 3.0.0 --micronaut-version 4.10.8
python main.py migrate <path-to-spring-project> <path-to-output-directory> \
    --spring-version 3.0.0 \
    --micronaut-version 4.10.8 \
    --build-tool maven
```

### Regression Guardrails

Use the named regression suite instead of remembering long unittest commands:

```bash
python scripts/run_regression_suite.py --tier fast
python scripts/run_regression_suite.py --tier corpus
python scripts/run_regression_suite.py --tier full
```

What the tiers mean:
- `fast`: best local safety net for CLI contracts, deterministic migration logic, KB lookup behavior, validation parsing, verification, and orchestrator status/report flow
- `corpus`: broader dependency-audit, orchestrator integration, corpus, release, fixture, and governance pipeline checks
- `full`: all tests discovered in `tests/`, intended for CI and release validation

Recommended enforcement path:
- local commit: install the shared git hooks and let `pre-commit` run the `fast` suite
- local push: let `pre-push` run `fast`, then `corpus`, then the clean Maven/Gradle dependency-audit fixtures
- GitHub push or pull request: CI runs `python scripts/run_regression_suite.py --tier full` and then the broader KB/release/package checks

Install the local hooks once per clone:

```bash
./scripts/install_local_guardrails.sh
```

Recent safety-net additions:
- `fast` now includes a regression contract for the operator-facing `Platform Evidence` summary, so the audited Micronaut target-platform state does not silently regress from `CONFIGURED LOCALLY` back into a misleading generic unresolved signal
- broader dependency-audit coverage now also protects the persisted audit-note path for `exact_resolved`, `configured_target_line`, and `none`
- `fast` now also protects the operator-facing target-platform resolution channel, so `channel=local_maven_repo` stays visible when exact Micronaut BOM proof came from the local Maven cache

Run the explicit GA release gate:

```bash
python3 scripts/run_ga_release_gate.py
```

What it does:
- runs the `fast` regression suite
- runs the `corpus` regression suite
- runs trusted `init` for the requested target pair in a temporary corpus root
- requires KB smoke and Chroma trust/distribution audit to pass
- reads `docs/STRICT_GA_CHECKLIST.md`
- writes `reports/ga_release_gate_report.json`
- in CI release builds, the workflow also generates `dist/release_manifest.json` and `dist/SHA256SUMS`

Important:
- this gate can still return `do_not_ship_as_ga` even when the technical commands pass
- that is expected if the strict checklist verdict still says `NOT GA READY`

Run the explicit release-candidate gate when you need a technically verified prerelease without claiming full enterprise GA:

```bash
python3 scripts/run_ga_release_gate.py --release-tier candidate --report reports/release_candidate_gate_report.json
```

What the candidate lane does:
- uses the same technical gates as the GA lane
- keeps the strict checklist verdict visible
- succeeds only when the technical gates pass and the strict checklist still permits pilot use with engineering review
- returns `ship_release_candidate` instead of `ship`

Release workflows:
- `.github/workflows/release.yml`: strict GA publishing lane
- `.github/workflows/release-candidate.yml`: prerelease publishing lane for `v*-rc*` tags

## Advanced Features

### Automated Self-Refinement (Try-Compile-Fix)

The tool includes a sophisticated "validation loop". After the initial transformation, the **ValidationAgent** attempts to build the migrated project. If compilation errors are detected:
1. The error log is parsed to identify the exact file and cause.
2. The code and errors are sent back to the LLM for a targeted "self-fix".
3. The process repeats (up to 3 times) until the build succeeds.

The migration flow now also includes a structural verification pass before the build step. This compares source and migrated Java files one by one and flags issues such as:
- Spring test slices like `@WebMvcTest` or `@DataJpaTest`
- `MockMvc`, `@LocalServerPort`, or unresolved `WebEnvironment`
- leftover Spring MVC model/paging APIs in the generated Micronaut code
- Spring dependencies that still remain in the migrated build file

Current deterministic safety behavior for unsupported Spring test slices:
- simple `@WebMvcTest` + `MockMvc` HTTP interactions are now converted into real `@MicronautTest` + `HttpClient` tests for supported request/assertion shapes
- complex `@WebMvcTest` cases that depend on Spring MVC model/view matchers still fall back to explicit `@MicronautTest` + `@Disabled(...)` manual-review placeholders instead of pretending to be fully migrated
- `@DataJpaTest` is normalized into active `@MicronautTest` when the repository-test shape is structurally safe enough for deterministic conversion
- `MockMvc` request chains are replaced with explicit unsupported helper calls so the migrated source no longer silently pretends those tests were fully converted
- `SpringBootTest(webEnvironment = RANDOM_PORT)`, `@LocalServerPort`, and `RestTemplateBuilder`-style integration-test scaffolding are normalized toward Micronaut client usage when the shape is simple enough to rewrite safely
- Spring MVC model/view flows now get deterministic rewrites for `Model`, `ModelMap`, `ModelAndView`, `Page`, `Pageable`, and `PageRequest` when the pattern is safe enough to convert locally
- governed KB coverage now includes reviewed enterprise guidance for `ModelAttribute`, `InitBinder`, `BindingResult`, `WebDataBinder`, and `JCacheManagerCustomizer`, so the console shows a governed/manual-review signal instead of a blind vector miss
- deterministic controller validation handling now rewrites common `BindingResult` flows into validator-backed Micronaut helper methods and auto-synthesizes `Validator` injection when needed
- recent dependency cleanup parity now applies to both Maven and Gradle migration paths for common Spring build leftovers such as Thymeleaf starter replacement and Devtools removal
- when the configured LLM is not reachable, the agent now skips actual LLM refinement work instead of spending long periods pretending to refine files
- fenced code blocks and trailing explanatory notes from model output are now stripped before generated Java files are written
- invalid LLM/self-fix responses are now rejected if they change the package, replace the file's main type with a different class, drop the package declaration, or include prose instead of raw Java
- deterministic safety coverage now also includes Spring runtime hints, local Spring utility helpers (`Assert.notNull`, `ToStringCreator`), `Formatter`/`Validator` placeholders, JAXB annotation cleanup, Spring property-sort helper replacement, Spring repository/query signatures, Spring validation-support helpers, and Micronaut page API normalization

### Source Code Privacy & Security

For security-conscious environments, the system ensures your code stays private:
- **Local Execution**: When using Ollama, no source code or transformation logic ever leaves your local machine.
- **No Data Retention**: The agent processes code in memory and does not store or "train" on your proprietary source.

## What Gets Migrated

### Automatically Migrated
- **Annotations**: Full mapping of Spring Web, DI, and Data annotations.
- **Build Config**: Full conversion of `pom.xml` and `build.gradle` scripts.
- **Source Code**: Field injection to constructor injection, package replacements.
- **Configurations**: `application.properties`/`yml` to Micronaut metadata.

### What End Users Should Still Review

- complex Spring Security behavior
- advanced data and repository semantics
- custom starters and organization-specific shared libraries
- messaging integrations with non-trivial producer/consumer configuration
- AOP-heavy or reflection-heavy Spring code
- third-party libraries that bring deep transitive dependency trees

## Troubleshooting

### Build Failures After Migration
If the self-refinement loop reaches its retry limit, check the terminal output for the remaining errors. Common causes include:
- Missing custom dependencies in the mapping dataset.
- Extremely complex Spring AOP patterns.
- Local toolchain/runtime problems such as Gradle native library issues can still fail build validation even when the migration logic itself is behaving correctly.

### Current Petclinic Status

The maintained Maven Petclinic path is now green on the local deterministic enterprise flow:
- structural verification is now `0 blocking / 0 review`
- Maven migration cleanup removes common Spring-era build blockers such as `spring-boot-maven-plugin`, `git-commit-id-plugin`, Spring plugin repositories, and Spring formatting/checkstyle/jacoco plugins from the generated Micronaut POM
- the deterministic transformer now keeps project-local `org.springframework...` imports when they refer to application code, while still purging real framework imports
- safe import restoration now preserves project-local model inheritance imports such as `NamedEntity` and `Person`, which fixed the Petclinic domain-model build break
- copied Thymeleaf templates are now normalized for Micronaut-safe rendering, including Spring preprocessed URL forms such as `@{__${owner.id}__/edit}`, menu links like `@{__${link}__}`, and safe-navigation expressions like `visit?.description`
- copied Spring MVC Thymeleaf templates are now relocated into Micronaut’s `src/main/resources/views` layout during migration instead of being left in Spring’s `src/main/resources/templates` location
- the migration report now raises template-normalization review warnings if leftover Spring Thymeleaf preprocess markers such as `__${...}__` or `?.` still exist in generated HTML resources
- the validated command `python main.py migrate examples/spring examples/micronaut --spring-version 3.0.0 --micronaut-version 4.10.8 --build-tool maven` completed with `Build Validation: PASSED (1 attempt)`
- Spring-only framework gaps that are not safe for blind auto-rewrite still surface as governed review guidance instead of raw vector misses or invalid Java placeholders
- the local Gradle build validation in this environment can still fail for host-native Gradle issues (`libnative-platform.dylib` on macOS aarch64), so Maven is currently the cleaner regression reference fixture here

### Intelligence Engine Setup

The agent utilizes a high-performance, encrypted intelligence engine to map complex code patterns.

**Recommended setup:**
- Run `python main.py init --spring-version 3.4.5 --micronaut-version 4.10.8` once before enterprise migration testing.
- This prepares the governed release dataset, initializes the local vector DB, and writes the trust audit report.
- After that, normal `python main.py migrate ...` migration commands can use the initialized RAG path directly.
- Before a real release decision, run `python3 scripts/run_ga_release_gate.py` and use the generated report as the ship/no-ship artifact.



---
**Need Help?** Professional support is available for enterprise migrations.
