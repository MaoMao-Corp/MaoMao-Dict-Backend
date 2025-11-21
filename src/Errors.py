class TargetPositionError(Exception):
    """
    Exception raised when the position of
    the target words exceeds the length of
    the tokenized sentence
    Args:
        message -- explanation of the error
    """
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
