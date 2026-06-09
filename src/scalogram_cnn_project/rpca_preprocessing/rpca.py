import numpy as np

## Notes
# This is a Python implementation of the Robust Principal
# Component Analysis (RPCA) algorithm.

# The algorithm is based on the paper "Robust Principal
# Component Analysis?" by Emmanuel J. Candès, Xiaodong Li,
# Yi Ma, and John Wright.

# RPCA returns the low-rank matrix L and the sparse matrix S
# that best approximate the input matrix X. L captures the
# underlying structure in X, while S captures the outliers.


def RPCA(X, lamb, mu, tolerance, max_iteration):

    [m, n] = X.shape
    unobserved = np.isnan(X)
    X[unobserved] = 0
    normX = np.linalg.norm(X, "fro")

    # Default values for lamb, mu, tol and max_iteration
    if lamb == None:
        lamb = 1 / np.sqrt(m)

    if mu == None:
        mu = 10 * lamb

    if tolerance == None:
        tolerance = 1e-9

    if max_iteration == None:
        max_iteration = 1000

    # Initial solution
    L = np.zeros((m, n))
    S = np.zeros((m, n))
    Y = np.zeros((m, n))

    for i in range(max_iteration):
        L = Do(X - S + (1 / mu) * Y, 1 / mu)
        S = So(X - L + (1 / mu) * Y, lamb / mu)

        Z = X - L - S
        Z[unobserved] = 0

        Y = Y + mu * Z

        err = np.linalg.norm(Z, "fro") / normX

        if err < tolerance:
            break

    return L, S


def So(X, tau):
    """
    Shrinkage operator.
    Parameters:
    tau : float or numpy array
        The threshold value.
    X : numpy array
        The input array.

    Returns:
    numpy array
        The result of applying the shrinkage operator to X.
    """
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)


def Do(X, tau):
    """
    Singular value thresholding operator.
    Parameters:
    tau : float
        The threshold value.
    X : numpy array
        The input array.

    Returns:
    numpy array
        The result of applying the singular value thresholding operator to X.
    """
    U, S, V = np.linalg.svd(X, full_matrices=False)
    return np.dot(U, np.dot(np.diag(S - tau), V))

