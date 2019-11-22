import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.linalg import lapack


def gsvd(A, M=[], W=[], exp=0.5):
	"""performs Generalized Singular Value Decomposition given an
	input matrix `A`, row-wise constraint matrix `M`, column-wise
	constraint matrix `W`, and an exponent value `exp`.

	Parameters
	----------
	A : numpy_array
		Input matrix of dimension `m`x`n`
	M : numpy_array, optional
		Row-wise constraint matrix of size `m`x`m`
	W : numpy_array, optional
		Column-wise constraint matrix of size `n`x`n`
	exp : float, optional
		Exponent value with with to raise `M` and `W` while transforming `A`
		and computing Uhat, Vhat. Defaults to `0.5`.

	Returns
	-------
	Uhat: np_array
		Eigenvectors of matrix `A`*`A`^T; left singular vectors
	S: np_array
		vector containing diagonal of the singular values
	Vhat: np_array
		Eigenvectors of matrix `A`^T*`A`; right singular vectors
	"""

	#if no M/W matrix specified, use identity matrix	
	if M == []:
		M = np.identity(A.shape[0])
	if W == []:
		W = np.identity(A.shape[1])

	if (M.shape[0] != A.shape[0]) or (W.shape[0] != A.shape[1]):
		

	#transpose A and swap M and W if more columns than rows
	flipped = False
	if A.shape[0] < A.shape[1]:
		A = np.transpose(A)
		#swap M and W
		M, W = W, M
		flipped = True

	#shortcut to scipy function to raise matrix to non-integer power
	#save some typing
	matpow = fractional_matrix_power

	#raise matrices to exponent provided
	Mexp = matpow(M,exp)
	Wexp = matpow(W,exp)
	
	#create A-hat according to formula:
	Ahat = np.matmul(np.matmul(Mexp,A),Wexp)
	
	#use LAPACK call to save computation overhead
	#U,S,Vt = np.linalg.svd(Ahat)
	U,S,Vt,i = lapack.dgesdd(Ahat)
	
	#obtain matrices of generalized eigenvectors	
	Uhat = np.matmul(matpow(M,-exp),U)
	Vhat = np.matmul(matpow(W,-exp),np.transpose(Vt))

	#correct sign according to first entry in left eigenvector
	sign = np.sign(Uhat[0,0])
	Uhat *= sign
	Vhat *= sign

	if flipped:
		#swap back again
		Uhat, Vhat = Vhat, Uhat

	return (Uhat,S,Vhat)


#def eigen(X):
#
#	#machine epsilon used as tolerance
#	eps = np.finfo(float).eps
#
#	U,D = np.eig(X)
#	
#	D = np.diag(D)
	
		
	
		


