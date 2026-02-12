# User Guide - Spring2Naut-RAG (GA v1.0.0)

A comprehensive guide to transforming your Spring Boot projects to Micronaut using the professional-grade Agentic Migration Tool.

## Prerequisites

- Python 3.8 or higher
- Maven or Gradle (installed and on PATH)
- LLM Provider (choose one):
  - **Ollama** (recommended for privacy, free, local)
  - OpenAI API (Higher accuracy for complex logic)
  - Anthropic Claude / Groq

## Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements_file.txt
```

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

### Step 3: Run Migration

Use the modular migration engine to transform your project. The tool will automatically detect your build system (Maven/Gradle), migrate dependencies, transform code, and perform a self-refinement loop.

```bash
python main.py <path-to-spring-project> <path-to-output-directory> \
    --spring-version 3.4.5 \
    --micronaut-version 4.10.8
```

## Advanced Features

### Automated Self-Refinement (Try-Compile-Fix)

The tool includes a sophisticated "validation loop". After the initial transformation, the **ValidationAgent** attempts to build the migrated project. If compilation errors are detected:
1. The error log is parsed to identify the exact file and cause.
2. The code and errors are sent back to the LLM for a targeted "self-fix".
3. The process repeats (up to 3 times) until the build succeeds.

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

## Troubleshooting

### Build Failures After Migration
If the self-refinement loop reaches its retry limit, check the terminal output for the remaining errors. Common causes include:
- Missing custom dependencies in the mapping dataset.
- Extremely complex Spring AOP patterns.

### Intelligence Engine Setup

The agent utilizes a high-performance, encrypted intelligence engine to map complex code patterns.

**What you need to do:**
- **Nothing.** The engine activates automatically at runtime.
- You will see `[INFO] Initializing intelligence engine...` in your terminal—this confirms the system is ready.



---
**Need Help?** Professional support is available for enterprise migrations.

