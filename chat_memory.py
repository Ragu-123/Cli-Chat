# chat_memory.py
class ConversationBuffer:
    """
    Sliding window memory buffer to store recent user/assistant messages.
    """
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.history = []

    def add_message(self, role, text):
        # Role must be 'user' or 'assistant'
        self.history.append((role, text))
        if len(self.history) > self.window_size * 2:
            self.history = self.history[-self.window_size * 2:]

    def get_history(self):
        return self.history
