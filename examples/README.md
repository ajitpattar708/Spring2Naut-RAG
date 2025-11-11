# Examples Directory

This directory is for your Spring Boot projects to migrate.

## Structure

```
examples/
├── spring/          # Place your Spring Boot project here
└── micronaut/       # Migrated Micronaut project will be created here
```

## Usage

1. **Place your Spring Boot project** in the `spring/` directory
2. **Run migration**:
   ```bash
   python migration_agent_main.py migrate \
       examples/spring \
       examples/micronaut \
       --spring-version 3.4.5 \
       --micronaut-version 4.10.8
   ```
3. **Migrated project** will be created in `examples/micronaut/`

## Note

- The `spring/` and `micronaut/` directories are excluded from git
- You can create your own Spring Boot projects here for testing
- This directory structure is provided as a convenience

