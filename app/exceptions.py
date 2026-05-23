class AppError(Exception):
    """Base application error."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class EmptyTextError(AppError):
    def __init__(self, message: str = "Text must not be empty") -> None:
        super().__init__(message, status_code=422)


class ModelUnavailableError(AppError):
    def __init__(self, message: str = "Embedding model is unavailable") -> None:
        super().__init__(message, status_code=503)


class UpstreamTimeoutError(AppError):
    def __init__(self, message: str = "Upstream embedding service timed out") -> None:
        super().__init__(message, status_code=504)


class InvalidUpstreamResponseError(AppError):
    def __init__(self, message: str = "Invalid response from embedding model") -> None:
        super().__init__(message, status_code=502)
