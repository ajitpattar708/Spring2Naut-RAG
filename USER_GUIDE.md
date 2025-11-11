# User Guide - Spring Boot to Micronaut Migration Agent

A simple guide to get started with the migration agent.

## Prerequisites

- Python 3.8 or higher
- Maven (for building migrated projects)
- LLM Provider (choose one):
  - **Ollama** (recommended, free, local)
  - OpenAI API key
  - Anthropic Claude API key
  - Groq API key

## Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements_file.txt
```

### Step 2: Set Up LLM (Choose One)

#### Option A: Ollama (Free, Local)

```bash
# Install Ollama
# Windows: Download from https://ollama.com/download
# Linux/Mac: curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server
ollama serve

# In another terminal, pull model
ollama pull codellama:7b
```

Set environment variables:
```bash
export LLM_PROVIDER=ollama
export LLM_MODEL=codellama:7b
```

#### Option B: OpenAI

```bash
export OPENAI_API_KEY=your-api-key
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4-turbo
```

#### Option C: Anthropic Claude

```bash
export ANTHROPIC_API_KEY=your-api-key
export LLM_PROVIDER=claude
export LLM_MODEL=claude-3-opus-20240229
```

#### Option D: Groq

```bash
export GROQ_API_KEY=your-api-key
export LLM_PROVIDER=groq
export LLM_MODEL=llama3-70b-8192
```

### Step 3: Initialize Knowledge Base

```bash
python migration_agent_main.py init
```

This loads migration patterns into the vector database (takes 1-2 minutes).

### Step 4: Migrate Your Project

```bash
python migration_agent_main.py migrate \
    <path-to-spring-project> \
    <path-to-output-directory> \
    --spring-version <version> \
    --micronaut-version <version>
```

**Example:**
```bash
python migration_agent_main.py migrate \
    /path/to/my-spring-app \
    /path/to/my-micronaut-app \
    --spring-version 3.4.5 \
    --micronaut-version 4.10.8
```

## Available Commands

| Command | Description |
|---------|-------------|
| `init` | Initialize knowledge base (first time setup) |
| `migrate <source> <output> [options]` | Migrate a Spring Boot project to Micronaut |
| `test` | Test with sample code |

### Command Options

- `--spring-version <version>` - Spring Boot version (e.g., 3.4.5)
- `--micronaut-version <version>` - Micronaut version (e.g., 4.10.8)

**Note:** Advanced commands like `export` and `merge` are available for dataset management but not needed for basic usage.

## What Gets Migrated

### Automatically Migrated

- **Annotations**: `@RestController` → `@Controller`, `@GetMapping` → `@Get`, etc.
- **Dependencies**: Spring Boot → Micronaut equivalents in `pom.xml`
- **Configuration**: `application.yml`/`.properties` conversion
- **Code Patterns**: Controllers, Services, Repositories, Configs
- **Imports**: Spring packages → Micronaut packages

### Manual Steps May Be Required

- Complex security configurations
- Custom Spring AOP
- Spring Cloud components
- WebSocket/STOMP configurations
- Multi-module projects

## After Migration

1. **Review the migration report** - Check `migration-report.json` in output directory
2. **Build the project**:
   ```bash
   cd <output-directory>
   mvn clean compile
   ```
3. **Run tests**:
   ```bash
   mvn test
   ```
4. **Review changes** - Check the diff of migrated files
5. **Test manually** - Test all API endpoints

## Troubleshooting

### Knowledge Base Not Found
```bash
python migration_agent_main.py init
```

### Ollama Not Available
The agent works without Ollama but with reduced capabilities. Start Ollama:
```bash
ollama serve
```

### Import Errors After Migration
Add missing Micronaut dependencies to `pom.xml`:
```xml
<dependency>
    <groupId>io.micronaut</groupId>
    <artifactId>micronaut-inject</artifactId>
</dependency>
```

### Out of Memory
- Use a smaller embedding model
- Process files in batches
- Close other applications

## More Information

- **Full Documentation**: See [README.md](README.md)
- **Configuration**: See Configuration section in README.md

## Tips

1. **Start Small**: Test with a small module first
2. **Version Control**: Commit your code before migration
3. **Backup**: Keep original project safe
4. **Review**: Always review migrated code before deploying
5. **Test Thoroughly**: Run all tests after migration

---

**Need Help?** Open an issue on GitHub or check the documentation files.

