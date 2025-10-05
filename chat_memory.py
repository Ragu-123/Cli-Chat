class ConversationBuffer:
    """
    Sliding-window conversation memory storing recent user and assistant messages.
    """
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.history = []

    def add_message(self, role, text):
        """
        Add a message tuple (role, text) to the history and trim to the configured window size.
        """
        self.history.append((role, text))
        if len(self.history) > self.window_size * 2:
            self.history = self.history[-self.window_size * 2:]

    def get_history(self):
        """
        Return the current conversation history as a list of (role, text) tuples.
        """
        return self.history
