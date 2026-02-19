# Retrieval Strategies

## Pipeline

```
Query → Hybrid Retrieval (Vector + Keyword Fallback) → Deduplication → Reranking → Context Builder → Traceable Output
```

---

## Strategies

**1. Hybrid Retrieval with Keyword Fallback** — `retriever/hybrid_retriever.py`
Dense vector search with metadata filters (`year`, `type`). Falls back to keyword search if results are insufficient, preventing empty result sets on narrow filters.

**2. Reranking** — `retriever/reranker.py`
CrossEncoder (`ms-marco-MiniLM-L-6-v2`) scores each `(query, chunk)` pair jointly for deeper semantic alignment. Adds a precision layer on top of fast-but-imprecise vector search.

**3. Deduplication** — `retriever/hybrid_retriever.py`
Removes duplicate chunks produced by overlapping vector and keyword paths before reranking, avoiding biased or bloated context.

**4. Token-Limited Context Building** — `pipelines/context_builder.py`
Chunks are added greedily in reranked order up to a `max_tokens` budget (default: 2000), keeping context tight and noise-free.

**5. Traceable Sources** — `pipelines/context_builder.py`
Each chunk is tagged inline with its provenance and a `sources` metadata list is returned for citation and auditability:
```
[Source: sampple.pdf | Page: 12 | Year: 2023 | Type: proxy]
```

---

**output**
![results_output](screenshots/day2/results_output.png)