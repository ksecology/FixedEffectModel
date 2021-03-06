import numpy as np
import scipy.linalg as la
import warnings


def waldtest(coef, V):
    R = np.eye(len(coef))
    beta = np.dot(R, coef)
    RVR = np.dot(np.dot(R, V), R.T)
    RVR_inv = pinvx(RVR)
    w = np.dot(np.dot(beta.T, RVR_inv), beta)
    return w


def pinvx(x):
    chp, badvars = cholx(x)
    c = np.linalg.inv(chp)
    inv = np.dot(c.T, c)
    if badvars is None:
        return inv
    else:
        for i in sorted(badvars):
            inv = np.insert(inv, i-1, 0, 1)
            inv = np.insert(inv, i-1, 0, 0)
        return inv


def cholx(x, eps=1e-6):
    n = x.shape[0]
    dpstrf = la.get_lapack_funcs('pstrf', [x])
    res = dpstrf(x, tol=eps)
    rank = res[2]
    if rank == n:
        chp = np.linalg.cholesky(x)
        badvars = None
    else:
        warnings.warn('The variance matrix is either rank-deficient or indefinite.')
        pivot = res[1]
        badvars = pivot[rank:n]
        new_var = list(range(1, n+1))
        final_var = [item-1 for item in new_var if item not in list(badvars)]
        twod_final_var = np.array(final_var).reshape(len(final_var), 1)
        chp = np.linalg.cholesky(x[twod_final_var, final_var])

    return chp, badvars
