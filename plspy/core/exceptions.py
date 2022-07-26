# define exceptions for error handling


class Error(Exception):
    """Base class for the following exceptions."""

    pass


class InputMatrixDimensionMismatchError(Error):
    """Exception raised when the input dimensions of M and W don't match
    the expected dimensions of M and W.
    """

    pass


class ImproperShapeError(Error):
    """Exception raised when a matrix has the incorrect shape."""

    pass


class ConditionMatrixMalformedError(Error):
    """Raised when the Condition matrix is not of shape (n,)."""

    pass


class NotImplementedError(Error):
    """Raised when a function has yet to be implemented."""

    pass


class MissingParameterError(Error):
    """Raised when a required parameter is not passed in."""

    pass


class OutOfRangeError(Error):
    """Raised when an out-of-range index is referenced."""

    pass
