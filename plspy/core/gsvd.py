import numpy as np
from scipy.linalg import fractional_matrix_power, lapack

from . import exceptions


def gsvd(A, M=None, W=None, exp=0.5, full_matrices=False, compute_uv=True):
    """Performs Generalized Singular Value Decomposition given an
    input matrix `A`, row-wise constraint matrix `M`, column-wise
    constraint matrix `W`, and an exponent value `exp`.

    Parameters
    ----------
    A : array_like
        Input matrix of dimension `m` x `n`
    M : array_like
        Row-wise constraint matrix of size `m`x`m`
    W : array_like
        Column-wise constraint matrix of size `n`x`n`
    exp : float, optional
          Exponent value with with to raise `M` and `W` while transforming `A`
          and computing `Uhat`, `Vhat`. Defaults to `0.5`.
    full_matrices : boolean
        Flag to specify whether full or partial U,V matrices should
        be returned.

    Returns
    -------
    Uhat: np_array
          Eigenvectors of matrix `A`*`A`^T; left singular vectors
    S: np_array
       vector containing diagonal of the singular values
    Vhat: np_array
          Eigenvectors of matrix `A`^T*`A`; right singular vectors
    """

    # user_spec_m = True
    # user_spec_w = True

    A = np.array(A)
    # if no M/W matrix specified, use identity matrix
    if M is None or M == []:
        M = np.identity(A.shape[0])
        # user_spec_m = False
    # cast to numpy array
    else:
        M = np.array(M)
    if W is None or W == []:
        W = np.identity(A.shape[1])
        # user_spec_w = False
    else:
        W = np.array(W)

    # handle dimension mismatches of input matrices
    M_wrong_dim = M.shape[0] != A.shape[0]
    W_wrong_dim = W.shape[0] != A.shape[1]
    if M_wrong_dim:
        raise exceptions.InputMatrixDimensionMismatchError(
            "Dimension of M {} doesn't match"
            "number of rows of A ({})".format(M.shape, A.shape[0])
        )
    if W_wrong_dim:
        raise exceptions.InputMatrixDimensionMismatchError(
            "Dimension of W {} doesn't match"
            "number of columns of A ({})".format(W.shape, A.shape[1])
        )

    # transpose A and swap M and W if more columns than rows
    flipped = False
    if A.shape[0] < A.shape[1]:
        A = np.transpose(A)
        # swap M and W
        M, W = W, M
        flipped = True

    # shortcut to scipy function to raise matrix to non-integer power
    # save some typing
    matpow = fractional_matrix_power

    # raise matrices to exponent provided
    Mexp = matpow(M, exp)
    Wexp = matpow(W, exp)
    print(Wexp)

    # create A-hat according to formula:
    Ahat = np.matmul(np.matmul(Mexp, A), Wexp)

    # use LAPACK call to save computation overhead
    # U,S,Vt = np.linalg.svd(Ahat)
    U, S, Vt, i = lapack.dgesdd(
        Ahat, full_matrices=full_matrices, compute_uv=compute_uv
    )

    if not compute_uv:
        return S
    # obtain matrices of generalized eigenvectors
    Uhat = np.matmul(matpow(M, -exp), U)
    Vhat = np.matmul(matpow(W, -exp), np.transpose(Vt))

    # correct sign according to first entry in left eigenvector
    sign = np.sign(Uhat[0, 0])
    Uhat *= sign
    Vhat *= sign

    if flipped:
        # swap back again
        Uhat, Vhat = Vhat, Uhat

    return (Uhat, S, Vhat)
