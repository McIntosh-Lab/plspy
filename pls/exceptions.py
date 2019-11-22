#define exceptions for error handling

class Error(Exception):
	"""Base class for the following excpetions
	"""
	pass


class InputMatrixDimensionMismatchError(Error):
	"""Exception raised when the dimensions of M and W don't match
	the dimensions of M and W
	"""
	pass
