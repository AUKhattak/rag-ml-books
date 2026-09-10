import json
from pathlib import Path
from collections import Counter

chunk_file = "data/processed/chunks/chunks_v1.jsonl"

print("📊 Analyzing chunks...")
chunks = []
with open(chunk_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

print(f"\n✅ Total chunks: {len(chunks)}")

# Check by book
book_counts = Counter(c['book_title'] for c in chunks)
print(f"\n📚 Chunks per book:")
for book, count in book_counts.most_common():
    print(f"   {book}: {count} chunks")

# Check chunk sizes
sizes = [len(c['content']) for c in chunks]
print(f"\n📏 Chunk sizes:")
print(f"   Min: {min(sizes)} chars")
print(f"   Max: {max(sizes)} chars")
print(f"   Avg: {sum(sizes)/len(sizes):.0f} chars")

# Show a few sample chunks
print(f"\n📝 Sample chunks (showing content preview):")
for i, chunk in enumerate(chunks[:5]):
    content_preview = chunk['content'][:150].replace('\n', ' ')
    print(f"\nChunk {i+1}:")
    print(f"   Book: {chunk['book_title']}")
    print(f"   Page: {chunk['page_number']}")
    print(f"   Chunk: {chunk['chunk_index']+1}/{chunk['total_chunks']}")
    print(f"   Size: {len(chunk['content'])} chars")
    print(f"   Preview: {content_preview}...")