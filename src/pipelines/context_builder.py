class ContextBuilder:
    def __init__(self, max_tokens=2000):
        self.max_tokens = max_tokens

    def build(self, chunks):
        context = ""
        total_length = 0
        sources = []

        for chunk in chunks:
            text = chunk["text"]
            metadata = chunk["metadata"]

            if total_length + len(text) > self.max_tokens:
                break

            context += (
                f"\n\n[Source: {metadata['source']} | "
                f"Page: {metadata['page']} | "
                f"Year: {metadata['year']} | "
                f"Type: {metadata['type']}]\n"
            )
            context += text

            sources.append(metadata)
            total_length += len(text)

        return context.strip(), sources
