# 🚀 Spring Boot to Micronaut Migration Agent

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-beta-yellow.svg)](https://github.com/yourusername/Spring2Naut-RAG)

> ⚠️ **Beta Release** - This project is in active development. While functional, it's recommended for testing and feedback. See [GA_READINESS_CHECK.md](GA_READINESS_CHECK.md) for current status.

An intelligent AI-powered RAG (Retrieval-Augmented Generation) agent that automatically migrates Spring Boot 3.x.x projects to Micronaut 4.x.x projects with version-specific compatibility handling.

## ✨ Features

- 🤖 **AI-Powered Migration** - Uses RAG (Retrieval-Augmented Generation) with LLM fallback for complex transformations
- 📚 **Knowledge Base** - Vector database (ChromaDB) with 115,000+ migration patterns
- 🔍 **Semantic Search** - CodeBERT embeddings for accurate pattern matching
- 🧠 **Multi-LLM Support** - Ollama, OpenAI, Claude, or Groq
- 🔄 **Version-Aware** - Handles Spring Boot 3.x.x → Micronaut 4.x.x with patch version compatibility
- 📦 **Dependency Resolution** - Intelligent dependency version mapping
- ⚙️ **Configuration Migration** - Automatic `application.yml`/`.properties` conversion
- 🏗️ **Build Validation** - Automatically compiles migrated projects
- 🎯 **Multi-Agent System** - Specialized agents for dependencies, code, config, and validation

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [User Guide](#user-guide) ⭐ **Start Here**
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Maven (for building migrated projects)
- Ollama (optional, for local LLM) or API keys for OpenAI/Claude/Groq

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Spring2Naut-RAG.git
cd Spring2Naut-RAG

# Install dependencies
pip install -r requirements_file.txt

# Initialize knowledge base
python migration_agent_main.py init
```

### Basic Usage

```bash
# Migrate a Spring Boot project
python migration_agent_main.py migrate \
    <path-to-spring-project> \
    <path-to-output-directory> \
    --spring-version 3.4.5 \
    --micronaut-version 4.10.8
```

## 📖 User Guide

**For detailed step-by-step instructions, see [USER_GUIDE.md](USER_GUIDE.md)**

The user guide includes:
- Complete setup instructions
- LLM provider configuration (Ollama, OpenAI, Claude, Groq)
- Command reference
- Troubleshooting tips
- What gets migrated automatically
- Post-migration checklist

## 📖 Installation

### Step 1: Install Python Dependencies

```bash
pip install -r requirements_file.txt
```

### Step 2: Set Up LLM Provider (Choose One)

#### Option A: Ollama (Local, Free)

```bash
# Install Ollama
# Windows: Download from https://ollama.com/download
# Linux/Mac: curl -fsSL https://ollama.com/install.sh | sh

# Pull CodeLlama model
ollama pull codellama:7b
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

This loads the migration dataset into ChromaDB vector database.

## 💻 Usage

### Migrate a Project

```bash
python migration_agent_main.py migrate \
    <source-spring-project> \
    <output-micronaut-project> \
    --spring-version <version> \
    --micronaut-version <version>
```

**Example:**
```bash
python migration_agent_main.py migrate \
    examples/spring \
    examples/micronaut \
    --spring-version 3.4.5 \
    --micronaut-version 4.10.8
```

### Available Commands

```bash
# Initialize knowledge base
python migration_agent_main.py init

# Migrate a project
python migration_agent_main.py migrate <source> <output> [options]

# Test with sample code
python migration_agent_main.py test

# Export knowledge base to dataset
python migration_agent_main.py export

# Merge dataset into knowledge base
python migration_agent_main.py merge <dataset.json> [--mode add|replace]
```

### Command Options

```bash
--spring-version <version>     Spring Boot version (e.g., 3.4.5)
--micronaut-version <version>  Micronaut version (e.g., 4.10.8)
```

## ⚙️ Configuration

The agent can be configured via environment variables:

### LLM Configuration

```bash
# LLM Provider (ollama, openai, claude, groq)
export LLM_PROVIDER=ollama

# LLM Model
export LLM_MODEL=codellama:7b  # or gpt-4-turbo, claude-3-opus-20240229, etc.

# API Keys (if using cloud providers)
export OPENAI_API_KEY=your-key
export ANTHROPIC_API_KEY=your-key
export GROQ_API_KEY=your-key
```

### Embedding Model

```bash
# Embedding model for RAG search
export EMBEDDING_MODEL=microsoft/codebert-base  # or all-MiniLM-L6-v2
```

### ChromaDB Configuration

```bash
# ChromaDB persistence directory
export CHROMADB_PATH=./migration_db
```

### Ollama Configuration

```bash
# Ollama server URL
export OLLAMA_BASE_URL=http://localhost:11434
```

See [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) for complete configuration options.

## 🏗️ Architecture

The migration agent uses a multi-agent system:

```
MigrationOrchestrator
    ├── DependencyAgent      (Maven/Gradle dependency migration)
    ├── CodeTransformAgent       (Java code transformation)
    ├── ConfigAgent              (application.yml migration)
    └── ValidationAgent          (Migration validation)
```

### Knowledge Base

- **Vector Database:** ChromaDB with semantic search
- **Embedding Model:** CodeBERT (768 dimensions) with MiniLM fallback
- **Dataset:** 115,000+ migration patterns
- **LLM Fallback:** For complex transformations not in dataset

See [ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md) for detailed architecture.

## 📚 Examples

### Example Project

The repository includes a complete example:

- **Source:** `examples/spring/` - Spring Boot 3.4.5 project
- **Migrated:** `examples/micronaut/` - Micronaut 4.10.8 project

**Features demonstrated:**
- REST Controllers
- Services and Repositories
- Configuration classes (Redis, Coherence, Cache, DataSource)
- Filters (Authentication, Logging)
- Gateway MVC configuration
- OCI integration (Vault, Redis, Auth)

### Run Example Migration

```bash
python migration_agent_main.py migrate \
    examples/spring \
    examples/micronaut \
    --spring-version 3.4.5 \
    --micronaut-version 4.10.8
```

## 🧪 Testing

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for comprehensive testing instructions.

### Quick Test

```bash
# Test with sample code
python migration_agent_main.py test

# Test with example project
python migration_agent_main.py migrate \
    examples/spring \
    examples/micronaut \
    --spring-version 3.4.5 \
    --micronaut-version 4.10.8
```

### Verify Migration

```bash
# Build migrated project
cd examples/micronaut
mvn clean compile
```

## 🔧 What Gets Migrated

### ✅ Supported Migrations

- **Annotations:** `@RestController` → `@Controller`, `@GetMapping` → `@Get`, etc.
- **Dependencies:** Spring Boot → Micronaut equivalents
- **Configuration:** `application.yml`/`.properties` conversion
- **Code Patterns:** Controllers, Services, Repositories, Configs
- **Filters:** Jakarta Servlet → Micronaut HttpServerFilter
- **Cache:** Spring Cache → Micronaut Cache
- **Redis:** Spring Data Redis → Micronaut Redis
- **Data Source:** Spring DataSource → Micronaut DataSource

### ⚠️ Limitations

- Currently supports **Maven** only (Gradle support coming soon)
- **Java** only (Kotlin support coming soon)
- Single-module projects (multi-module support coming soon)
- Some Spring Cloud features require manual migration

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Adding Migration Patterns

To add new migration patterns, edit `migration_dataset.json`:

```json
{
  "spring_pattern": "@YourAnnotation",
  "micronaut_pattern": "@MicronautEquivalent",
  "category": "annotation",
  "description": "Migration description",
  "complexity": "low|medium|high"
}
```

Then run:
```bash
python migration_agent_main.py init
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Micronaut Framework** - https://micronaut.io
- **Spring Boot** - https://spring.io/projects/spring-boot
- **ChromaDB** - Vector database for embeddings
- **CodeBERT** - Code embedding model
- **Ollama** - Local LLM server

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/Spring2Naut-RAG/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/Spring2Naut-RAG/discussions)
- **Documentation:** See `docs/` directory

## 🗺️ Roadmap

- [ ] Gradle build tool support
- [ ] Kotlin code migration
- [ ] Multi-module project support
- [ ] Spring Security → Micronaut Security
- [ ] Spring Cloud → Micronaut Cloud
- [ ] Reactive (WebFlux) → Micronaut Reactive
- [ ] Interactive migration wizard
- [ ] Migration preview mode

---

**Made with ❤️ for the Micronaut community**

