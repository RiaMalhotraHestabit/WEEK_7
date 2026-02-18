import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")


def chunk_text(document, chunk_size=600, overlap=100):
    text = document["text"]
    metadata = document["metadata"]

    tokens = encoding.encode(text)

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)

        chunks.append({
            "text": chunk_text,
            "metadata": {
                **metadata,
                "chunk_id": chunk_id
            }
        })

        start += chunk_size - overlap
        chunk_id += 1

    return chunks
