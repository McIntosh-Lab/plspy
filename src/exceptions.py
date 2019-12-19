# define exceptions for error handling


class Error(Exception):
    """Base class for the following excpetions.
    """

    pass


class InputMatrixDimensionMismatchError(Error):
    """Exception raised when the dimensions of M and W don't match
    the dimensions of M and W.
    """

    pass


class ImproperShapeError(Error):
    """Exception raised when a matrix has the incorrect shape.
    """

    pass


class ConditionMatrixMalformedError(Error):
    """Raised when the Condition matrix is not of shape (n,).
    """

    pass


class NotImplementedError(Error):
    """Raised when a function has yet to be implemented.
    """

    pass
