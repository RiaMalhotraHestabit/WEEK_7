import collections

class MemoryStore:
    """Stores last N messages per session."""
    def __init__(self, max_len=5):
        self.max_len = max_len
        self.store = collections.deque(maxlen=self.max_len)

    def add_message(self, role, content):
        """role: 'user' or 'assistant'"""
        self.store.append({'role': role, 'content': content})

    def get_history(self):
        return list(self.store)