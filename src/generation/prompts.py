"""
Prompt templates for answer generation.

Kept as plain module-level strings so they can be versioned with the code
and unit-tested without loading an LLM.
"""

from __future__ import annotations


SYSTEM_PROMPT = """You are a precise assistant answering questions about machine learning textbooks.

Rules:
- Answer using ONLY the provided context chunks.
- If the context does not contain the answer, reply exactly: "I don't have enough information in the provided sources to answer this."
- Do not invent facts, page numbers, or citations.
- Prefer concrete, specific answers over general summaries.
- Write in clear, concise prose. Aim for 3-6 sentences unless the question demands more.
- Do not repeat the question. Do not include a preamble like "Based on the context...".
"""


def build_context_block(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block.

    Each chunk is annotated with its book and page so the LLM can cite it.
    """
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        book = chunk.get("book", "Unknown")
        page = chunk.get("page", "?")
        section = chunk.get("section", "") or ""
        section_str = f", {section}" if section else ""
        header = f"[{i}] {book}, page {page}{section_str}"
        content = chunk.get("content", "").strip()
        lines.append(f"{header}\n{content}")
    return "\n\n---\n\n".join(lines)


def build_user_prompt(query: str, chunks: list[dict]) -> str:
    """Compose the final user prompt from the query and retrieved chunks."""
    context = build_context_block(chunks)
    return f"""Context chunks:

{context}

---

Question: {query}

Answer using only the chunks above. If the answer is not present, say so."""