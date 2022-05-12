import numpy
import numpy as np
import math
import time
from scipy.linalg import lu

import matrixAndVectorOperation


def jacobi(A, b, tolerance=1e-9, max_iterations=10000):
    x = np.zeros_like(b, dtype=np.double)

    T = A - np.diag(np.diagonal(A))

    for k in range(max_iterations):

        x_old = x.copy()

        x[:] = (b - np.dot(T, x)) / np.diagonal(A)

        if np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < tolerance:
            break
    return x


def gauss_seidel(A, b, tolerance=1e-10, max_iterations=10000):
    x = np.zeros_like(b, dtype=np.double)

    for k in range(max_iterations):
        x_old = x.copy()
        for i in range(A.shape[0]):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, (i + 1):], x_old[(i + 1):])) / A[i, i]

        if np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < tolerance:
            break

    return x

def LU(X, b):
    A = np.array(X)
    A = A.astype('float64')
    n = A.shape[0]

    U = A.copy()
    L = np.eye(n, dtype=np.double)

    for i in range(n):
        factor = U[i + 1:, i] / U[i, i]
        L[i + 1:, i] = factor
        U[i + 1:] -= factor[:, np.newaxis] * U[i]

    y = np.zeros_like(b, dtype=np.double);

    y[0] = b[0] / L[0, 0]

    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    n = U.shape[0]

    x = np.zeros_like(y, dtype=np.double);

    x[-1] = y[-1] / U[-1, -1]

    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i:], x[i:])) / U[i, i]

    return x

def compare(A, C, b, N, e, f, a1, a2, a3):

    A_copy = np.array(A)
    print("\n\nporownanie czasow własnych procedur i tych z pakietow numpy/scipy \n")
    tstart = time.time()
    x = jacobi(A_copy, b, N)
    tend = time.time()
    time_difference = tend - tstart
    print()
    print("Jacobi:")
    print("-czas obliczenia: " + str(time_difference))

    tstart = time.time()
    x = gauss_seidel(A_copy, b)
    tend = time.time()
    time_difference = tend - tstart
    print()
    print("GaussSeidel:")
    print("-czas obliczenia: " + str(time_difference))

    tstart = time.time()
    x = LU(A_copy, b)
    tend = time.time()
    time_difference = tend - tstart
    print()
    print("LU:")
    print("-czas obliczenia: " + str(time_difference))

    #ZadanieE
    time_jacobi_numpy = []
    time_gauss_seidel_numpy = []
    time_LU_numpy = []
    N = [100, 500, 1000, 2000, 3000]
    a1 = 5 + e
    for i in range(len(N)):
        A = np.array(matrixAndVectorOperation.create_A_matrix(N[i], a1, a2, a3))
        b = matrixAndVectorOperation.create_b_vecotr(f, N[i])
        tstart = time.time()
        jacobi(A, b)
        tend = time.time()
        time_difference = tend - tstart
        time_jacobi_numpy.append(time_difference)
        tstart = time.time()
        gauss_seidel(A, b)
        tend = time.time()
        time_difference = tend - tstart
        time_gauss_seidel_numpy.append(time_difference)
        tstart = time.time()
        LU(A, b)
        tend = time.time()
        time_difference = tend - tstart
        time_LU_numpy.append( time_difference)

    return time_jacobi_numpy, time_gauss_seidel_numpy, time_LU_numpy