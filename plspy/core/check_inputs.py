from . import exceptions


def check_input_cond_order_match(X, cond_order, groups_sizes):
    """ """

    # check dimensions match
    if len(cond_order.shape) != len(X.shape):
        raise exceptions.InputMatrixDimensionMismatchError(
            "Dimension of condition orders does not match "
            "dimension of input matrix X."
        )
    elif X.shape[0] != cond_order.shape[0]:
        pass
    elif X.shape[2] != cond_order.shape[2]:
        # properties and subject info mismatch
        # raise exceptions.InputMatrixDimensionMismatchError(
        #     "")
        pass
