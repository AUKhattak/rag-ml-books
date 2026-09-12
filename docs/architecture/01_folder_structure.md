# Folder Structure

This document outlines the high-level folder structure of the `rag-ml-books` project and briefly explains the purpose of each directory.

## Root Directories

- **`artifacts/`**: Stores output files, generated models, or other reproducible build artifacts.
- **`data/`**: Contains datasets used in the project.
  - `raw/`: Unprocessed, original data files.
  - `processed/`: Cleaned and transformed data ready for use.
  - `evaluation/`: Datasets specifically used for model and pipeline evaluation.
- **`docs/`**: Project documentation.
  - `api/`: API documentation.
  - `architecture/`: High-level architecture, design decisions, and folder structure.
  - `decisions/`: Architecture Decision Records (ADRs).
  - `diagrams/`: Visual diagrams explaining system flows.
  - `guides/`: User or developer guides.
- **`notebooks/`**: Jupyter notebooks for exploration, prototyping, and data analysis.
- **`scripts/`**: Utility scripts for deployment, automation, or routine tasks.
- **`src/`**: The main application source code.
  - `api/`: API endpoints and routing logic.
  - `config/`: Configuration management and settings.
  - `embedding/`: Logic for generating vector embeddings from text.
  - `evaluation/`: Code for evaluating RAG pipeline performance and metrics.
  - `generation/`: LLM generation components (prompts, model integrations).
  - `ingestion/`: Data loading, parsing, and chunking pipelines.
  - `monitoring/`: Logging, tracing, and telemetry.
  - `retrieval/`: Vector database integrations and search algorithms.
  - `storage/`: Interfaces and logic for saving/loading data and metadata.
  - `utils/`: Shared helper functions and generic utilities.
- **`tests/`**: Automated tests (unit, integration, and end-to-end tests).
