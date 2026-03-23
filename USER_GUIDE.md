# User Guide

## Recommended Flow

1. Install dependencies.
2. Pick the exact Spring and Micronaut versions.
3. Run `init` once for that pair.
4. Run `migrate` on your project.
5. Review reports and build the migrated output.

## 1. Install

```bash
pip install -r requirements_file.txt
```

Optional:

```bash
pip install .
spring2naut --help
```

## 2. Choose Versions

Use exact versions and keep the same pair in both commands.

Example:

```bash
--spring-version 3.0.0 --micronaut-version 4.10.8
```

## 3. Initialize

Recommended mode:

```bash
python main.py init --mode trusted --spring-version 3.0.0 --micronaut-version 4.10.8
```

When to use each mode:

- `trusted`: normal migration work
- `candidate`: test your own added patterns
- `legacy`: inspect the old legacy dataset
- `hybrid`: trusted plus legacy
- `max`: widest experimental dataset
- `all`: everything local, including oldest legacy encrypted datasets

Aliases:

- `extended` = `legacy`
- `both` = `hybrid`
- `full` = `all`

Healthy `init` output should show:

- the exact version pair
- pattern indexing progress
- `KB Smoke OK: True`

## 4. Migrate

Basic command:

```bash
python main.py migrate <spring-project> <output-dir> \
  --spring-version 3.0.0 \
  --micronaut-version 4.10.8
```

Example:

```bash
python main.py migrate examples/spring examples/micronaut \
  --spring-version 3.0.0 \
  --micronaut-version 4.10.8
```

Build tool behavior:

- if only Maven files exist, Maven is used
- if only Gradle files exist, Gradle is used
- if both exist, pass `--build-tool maven` or `--build-tool gradle`

Examples:

```bash
python main.py migrate examples/spring examples/micronaut \
  --spring-version 3.0.0 \
  --micronaut-version 4.10.8 \
  --build-tool maven
```

```bash
python main.py migrate examples/spring examples/micronaut \
  --spring-version 3.0.0 \
  --micronaut-version 4.10.8 \
  --build-tool gradle
```

Healthy startup should reach lines like:

```text
Agentic Migration Initialized
[VDB] engine=chromadb status=ready
Starting migration from ...
```

## 5. Add Your Own Patterns

If you want to add your own mappings:

1. Edit `corpus/staged_patterns/candidates/index.json`
2. Add your pattern entry
3. Rebuild with `candidate` mode

```bash
python main.py init --mode candidate --spring-version 3.0.0 --micronaut-version 4.10.8
```

Use `candidate` when you want your local patterns available without changing the default trusted dataset.

## 6. Where To Look After Migration

- migrated project: your output folder
- migration report: `<output>/reports/migration_report.json`
- verification report: `<output>/reports/verification_report.json`
- trusted KB manifest: `migration_db/kb_manifest.json`

## 7. Mode Guidance

- use `trusted` for normal work
- use `candidate` for your own patterns
- use `legacy`, `hybrid`, `max`, or `all` only for exploratory runs
