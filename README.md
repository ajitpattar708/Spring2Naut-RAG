# Spring2Naut-RAG: Spring Boot to Micronaut Migration Agent (GA v1.0.0)

An intelligent, production-ready AI-powered RAG (Retrieval-Augmented Generation) agent that automatically migrates Spring Boot 3.x projects to Micronaut 4.x.

## Key Features

- **Automated Self-Refinement (Try-Compile-Fix)**: Automatically runs builds and uses LLM-driven logic to fix compilation errors.
- **Advanced RAG Knowledge Base**: Uses CodeBERT embeddings and a protected, encrypted dataset for high-fidelity pattern matching.
- **Architectural Modularity**: Clean separation of concerns between core logic, specialized agents, and knowledge retrieval.
- **IP Protection**: Support for encrypted datasets and remote knowledge services to protect proprietary migration logic.
- **Version-Aware Compatibility**: Intelligent handling of API changes between specific Spring and Micronaut versions.
- **Multi-LLM Support**: Compatibility with Ollama (Local), OpenAI, Claude, and Groq.

## Architecture

The system utilizes a multi-agent orchestration pattern to ensure robust transformations:

```text
MigrationOrchestrator
├── DependencyAgent      (Build file migration: Maven/Gradle)
├── CodeTransformAgent   (Source code transformation using RAG)
└── ValidationAgent      (Build verification and error extraction)
```

## Getting Started

### Prerequisites

- Python 3.8+
- Maven or Gradle (installed and on PATH)
- Ollama (optional for local inference) or API keys for cloud providers

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Spring2Naut-RAG.git
   cd Spring2Naut-RAG
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements_file.txt
   ```

### Usage

Run the migration using the main CLI:

```bash
python main.py <source-project-path> <output-directory> \
    --spring-version 3.4.5 \
    --micronaut-version 4.10.8
```

## Security and IP Protection

The agent is designed to protect proprietary migration logic. The `LocalMigrationKnowledgeBase` supports encrypted `.dat` files. For enterprise deployments, the knowledge base can be configured to use a remote service, ensuring that the core transformation logic never leaves your secure environment.

## License

This project is licensed under the MIT License.


