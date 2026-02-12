# Spring2Naut-RAG: Professional Spring to Micronaut Migration Agent (GA v1.0.0)

Automatically transform your Spring Boot 3.x applications into modern Micronaut 4.x services with AI-driven precision and automated self-healing.

## Key Features

- **🚀 Automated Self-Healing (Try-Compile-Fix)**: The agent doesn't just migrate; it validates. It automatically attempts to build your project and uses AI to fix any compilation errors it finds.
- **🎯 High-Fidelity Mapping**: Utilizes a deep knowledge base of over 10,000+ migration patterns to ensure your annotations, dependencies, and code patterns are correctly transformed.
- **🛡️ Enterprise Grade Security**: Designed with security in mind. The agent can run entirely offline with local LLMs (like Ollama) to ensure your proprietary source code never leaves your infrastructure.
- **📦 Full Project Transformation**: Handles everything from build configurations (Maven/Gradle) and source code transformation to configuration files (`application.yml`) and dependency injection.
- **🧩 Multi-LLM Support**: Works seamlessly with your choice of AI: Ollama (Local/Free), OpenAI, Claude, or Groq.

## Getting Started

### 1. Prerequisites
- Python 3.8+
- Maven or Gradle (installed and on PATH)
- Your choice of LLM (e.g., [Ollama](https://ollama.com) installed for local use)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/ajitpattar708/Spring2Naut-RAG.git
cd Spring2Naut-RAG

# Install dependencies
pip install -r requirements_file.txt
```

### 3. Usage
Run the migration with a single command:
```bash
python main.py <path-to-spring-project> <path-to-output-directory>
```

## Documentation
For detailed configuration and advanced usage, see the [User Guide](USER_GUIDE.md).

## Support
Professional support and custom enterprise datasets are available for large-scale migrations.

---
© 2024 Spring2Naut-RAG | Licensed under MIT


