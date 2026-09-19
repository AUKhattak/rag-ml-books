# Evaluation

## Overview

The evaluation framework measures the quality of the retrieval stage using an automated, reproducible **LLM-as-Judge** approach. Instead of requiring manual annotation for every run, a lightweight LLM is prompted to act as a strict factual judge, deciding whether a retrieved chunk contains the specific facts that would answer a query.

## Directory Structure

```text
rag-ml-books/
├── data/
│   └── evaluation/
│       ├── golden_dataset/
│       │   └── qa_pairs_with_context.jsonl    # Human-authored ground-truth queries
│       ├── judge_cache/                       # Cached LLM judge responses (SHA-256 keyed)
│       └── results/
│           └── retrieval_vN.json             # Saved evaluation run outputs
├── src/
│   └── evaluation/
│       ├── retrieval_eval.py                 # Metrics logic and RetrievalEvaluator class
│       └── judge.py                          # LLM-as-judge prompt, adapter, and cache
└── scripts/
    └── evaluate_retrieval.py                 # Main evaluation orchestration script
```

---

## Golden Dataset

The evaluation is grounded in a **golden dataset** — a manually curated set of question-answer pairs drawn from the source books, stored in JSONL format at:

```
data/evaluation/golden_dataset/qa_pairs_with_context.jsonl
```

**Golden Entry Schema:**
```json
{
  "id": "Unique identifier for this query",
  "book": "Deep Learning",
  "query": "What is the vanishing gradient problem?",
  "difficulty": "easy | medium | hard",
  "expected_answer_summary": "A brief reference answer containing the key facts that must appear in a retrieved chunk.",
  "notes": "Optional additional context or annotation notes."
}
```

Queries are labelled with a **difficulty** (`easy`, `medium`, `hard`) to enable per-difficulty metric breakdowns.

---

## LLM-as-Judge (`src/evaluation/judge.py`)

Rather than simple keyword or string matching, each retrieved chunk is evaluated by an LLM to determine whether it genuinely contains the required facts.

### How it Works

1. A structured prompt is built from the golden query, the expected answer summary, and the retrieved chunk content.
2. The judge LLM is asked to answer with a single word: `YES` or `NO`.
3. The judge is strict — being on the **same topic** is not enough. The specific key facts from the reference answer must be explicitly present in the chunk.

### Judge Prompt Template

```
You are a strict retrieval-quality judge for a RAG system on machine learning textbooks.

You will see a QUERY, a REFERENCE ANSWER, and a RETRIEVED CHUNK.

Your job: decide whether the retrieved chunk contains the specific facts stated in the reference answer.

Rules:
- YES only if the chunk explicitly states or clearly contains the key facts in the reference answer.
- NO if the chunk is merely on the same topic or discusses the topic at a different level of detail.
- NO if the chunk contradicts the reference answer.
- NO if the chunk is too vague or generic to confirm the reference facts.
- Being on the same topic is NOT enough. The facts must be present.

Answer with a single word: YES or NO.
```

### Judge Configuration

| Setting         | Value                              |
|-----------------|------------------------------------|
| **Model**       | `gemini-flash-lite` (Gemini API)   |
| **Temperature** | `0.0` (deterministic responses)    |
| **Max Retries** | 5 (with exponential backoff)       |
| **Rate Limit**  | ~4.5s between calls (free tier)    |

### Response Caching

To avoid redundant and costly LLM calls across runs, the judge responses are cached on disk. Each cache entry is keyed by a **SHA-256 hash** of the prompt, stored as individual JSON files under `data/evaluation/judge_cache/`. Re-running evaluation after a partial failure resumes seamlessly from the cache.

---

## Metrics (`src/evaluation/retrieval_eval.py`)

The `RetrievalEvaluator` class runs each golden query through the `Retriever`, evaluates each returned chunk with the judge, and aggregates results into the following standard IR metrics:

| Metric           | Description |
|------------------|-------------|
| **Recall@1**     | Fraction of queries where the very first retrieved chunk was judged relevant. |
| **Recall@k**     | Fraction of queries where at least one of the top-k chunks was judged relevant. |
| **MRR** (Mean Reciprocal Rank) | Average of `1/rank` where `rank` is the position of the first relevant chunk. Rewards finding the right chunk earlier. |
| **Precision@k**  | Fraction of all judged chunks (across all queries) that were relevant. |

All metrics are reported at three levels:
- **Overall** — aggregated across all queries
- **Per Book** — broken down by source book
- **Per Difficulty** — broken down by `easy`, `medium`, `hard`

---

## Usage

### Run Evaluation
```bash
python scripts/evaluate_retrieval.py
python scripts/evaluate_retrieval.py --k 5
python scripts/evaluate_retrieval.py --golden path/to/golden.jsonl --output run_name.json
```

### Output

The script prints a detailed evaluation report to the terminal and saves a structured JSON result file to `data/evaluation/results/`.

**Results JSON Schema:**
```json
{
  "config": {
    "k": 5,
    "model_key": "bge-small",
    "judge_model_key": "gemini-flash-lite",
    "index": "data/processed/indexes/books_v2.faiss",
    "golden_file": "data/evaluation/golden_dataset/qa_pairs_with_context.jsonl"
  },
  "overall": {
    "n": 50,
    "k": 5,
    "recall_at_1": 0.72,
    "recall_at_k": 0.88,
    "mrr": 0.7961,
    "precision_at_k": 0.45
  },
  "per_book": { ... },
  "per_difficulty": { ... },
  "per_query": [ ... ]
}
```

---

## Future Evaluation Plans

The current evaluation focuses exclusively on **retrieval quality** — whether the right chunks were surfaced. The next phase will extend coverage to the full RAG pipeline, evaluating the quality of the generated answer itself using the [DeepEval](https://docs.confident-ai.com/) framework.

DeepEval is an open-source LLM evaluation library that provides out-of-the-box metrics for RAG systems, each backed by an LLM judge to produce a score between 0 and 1.

---

### Faithfulness

Measures whether the generated answer is factually grounded in the retrieved context. A faithful answer only makes claims that are directly supported by the chunks passed to the LLM — it does not hallucinate or invent information.

```python
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

metric = FaithfulnessMetric(threshold=0.8)
test_case = LLMTestCase(
    input="What is backpropagation?",
    actual_output=generated_answer,
    retrieval_context=retrieved_chunks,
)
metric.measure(test_case)
print(metric.score, metric.reason)
```

**What it catches:** Answers that go beyond the source material, introduce incorrect facts, or contradict the retrieved context.

---

### Answer Relevancy

Measures whether the generated answer directly addresses the user's question. A high score means the answer is on-topic and complete; a low score indicates the model went off-track or gave a generic response.

```python
from deepeval.metrics import AnswerRelevancyMetric

metric = AnswerRelevancyMetric(threshold=0.8)
```

---

### Contextual Precision & Recall

Evaluate the quality of the retrieved context itself from the generation perspective:

- **Contextual Precision**: Are the retrieved chunks actually useful for answering the question (no noise)?
- **Contextual Recall**: Does the retrieved context contain all the information needed for a complete answer?

```python
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric

precision_metric = ContextualPrecisionMetric(threshold=0.7)
recall_metric = ContextualRecallMetric(threshold=0.7)
```

---

### Toxicity

Measures whether the generated answer contains harmful, offensive, or inappropriate content. This is particularly relevant when the model is given ambiguous or adversarial queries that could elicit undesired outputs.

```python
from deepeval.metrics import ToxicityMetric

metric = ToxicityMetric(threshold=0.1)  # Low threshold — any toxicity is a failure
test_case = LLMTestCase(
    input="How do I manipulate people using psychology?",
    actual_output=generated_answer,
)
metric.measure(test_case)
```

**What it catches:** Harmful instructions, hate speech, or manipulative content that the RAG pipeline should never produce, even if triggered by a crafted query.

---

### Application-Level Evaluation

Beyond individual metric scores, DeepEval supports **end-to-end test suites** that can be integrated into CI/CD pipelines to automatically gate deployments.

```python
import deepeval
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

def test_rag_answer():
    test_case = LLMTestCase(
        input="What is dropout regularization?",
        actual_output=generator.answer("What is dropout regularization?").answer,
        retrieval_context=[c["content"] for c in retrieved_chunks],
    )
    assert_test(test_case, [
        FaithfulnessMetric(threshold=0.8),
        AnswerRelevancyMetric(threshold=0.8),
        ToxicityMetric(threshold=0.1),
    ])
```

Run the full suite with:

```bash
deepeval test run tests/test_rag.py
```

This produces a structured report that can be tracked over time to detect regressions as the models, prompts, or chunking strategy evolve.

---

### Planned Metric Summary

| Metric                  | Library   | Evaluates              | Target Threshold |
|-------------------------|-----------|------------------------|------------------|
| Recall@k / MRR          | Custom    | Retrieval quality      | Currently active |
| Faithfulness            | DeepEval  | Answer grounding       | ≥ 0.80           |
| Answer Relevancy        | DeepEval  | Answer on-topic score  | ≥ 0.80           |
| Contextual Precision    | DeepEval  | Chunk noise level      | ≥ 0.70           |
| Contextual Recall       | DeepEval  | Context completeness   | ≥ 0.70           |
| Toxicity                | DeepEval  | Safety / harm          | ≤ 0.10           |
