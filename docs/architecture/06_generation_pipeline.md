# ML Books Generation Pipeline

## Overview

The generation pipeline represents the final stage of the RAG system. It takes the user's natural language question, executes the retrieval pipeline to fetch relevant contextual chunks, constructs a specialized prompt, and passes it to a Large Language Model (LLM) to generate a well-reasoned, highly accurate answer accompanied by exact citations.

## 📁 Directory Structure

```text
rag-ml-books/
├── src/
│   ├── generation/
│   │   ├── generator.py                     # Main RAG orchestration logic
│   │   └── prompts.py                       # System and user prompt templates
│   ├── llm/
│   │   ├── retry.py                         # API retry and error handling
│   │   └── errors.py                        # Custom LLM exception classes
│   └── catalog/
│       └── llm_models.py                    # LLM configuration registry
└── scripts/
    └── answer.py                            # End-to-end RAG CLI
```

## Components

### 1. Generator (`src/generation/generator.py`)
Orchestrates the complete "Retrieve and Generate" loop.
- **Dependency Injection**: Takes an instantiated `Retriever` and the LLM model configuration.
- **Provider Abstraction**: Currently defaults to Google's Gemini models (`gemini-flash` via `genai.Client`), utilizing a dedicated wrapper.
- **Data Encapsulation**: Returns a structured `GeneratedAnswer` object containing the raw query, the LLM's text answer, a parsed list of `Citation` objects, and the raw retrieved chunks for debugging.
- **Resilience**: Delegates API calls to the `retry` infrastructure to automatically handle transient network errors or rate limits.

### 2. Prompt Templates (`src/generation/prompts.py`)
Constructs the actual payload sent to the LLM.
- **System Prompt**: Enforces boundaries on the LLM (e.g., "only use the provided context", "be objective").
- **User Prompt**: Dynamically formats the user's query alongside the retrieved text chunks, explicitly injecting metadata (like chunk index, book, page, section) so the model can cite its sources natively.

### 3. LLM Utilities (`src/llm/`)
Provides robust infrastructure for third-party LLM APIs.
- **Retry Logic**: Exponential backoff implementations tailored for API limits.
- **Error Types**: Standardized exceptions (`LLMError`, `RetryExhausted`) ensuring the `Generator` doesn't crash entirely on API hiccups.

### 4. Orchestration (`scripts/answer.py`)
The main end-to-end user interface script.
- Takes the query, initializes the `Retriever` (using `bge-small`) and the `Generator` (using `gemini-flash`).
- Formats the resulting text response nicely for the terminal.
- Optionally displays the raw chunks alongside the answer if the `--show-chunks` flag is provided.

**Generation Output Schema:**
```json
{
  "query": "What is backpropagation?",
  "answer": "Backpropagation is an algorithm used to calculate the gradient of a loss function...",
  "citations": [
    {
      "index": 1,
      "book": "Deep Learning",
      "page": 204,
      "section": "6.5 Back-Propagation"
    }
  ],
  "chunks": [ ...raw chunk dictionaries... ]
}
```

## Usage

### Run End-to-End RAG
```bash
python scripts/answer.py "What is the difference between L1 and L2 regularization?" --k 5
```
This script will embed the question, retrieve the top 5 chunks from the books, construct the context, ping the Gemini API, and return a comprehensive answer with appended citations to the exact pages in the texts.
