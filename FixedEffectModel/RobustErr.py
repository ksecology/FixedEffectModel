import numpy as np


def robust_err(demean, consist_col, n, k, rank):
    """

    This function is used to calculate robust variance matrix based on equation
    (x'px)^-1 * sum(u_i*X_i)'(u_i*X_i)*(x'px)^-1

    :param demean: demeaned dataframe with relevant data
    :param consist_col: List of continuous_variables
    :param n: data size
    :param k: number of continuous variables
    :param rank: degree of freedom of category variables
    :return: robust variance matrix
    """
    xpx = np.dot(demean[consist_col].values.T, demean[consist_col].values)
    xpx_inv = np.linalg.inv(xpx)
    a2 = np.zeros((k, k))
    epsilon_xi = demean[consist_col].values * demean[['resid']].values.reshape(n, 1)
    a2 += np.dot(epsilon_xi.T, epsilon_xi)
    m = np.dot(xpx_inv, a2)
    beta = np.dot(m, xpx_inv)
    return beta * n / (n - k - rank)
