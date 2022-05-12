import math

import matrixAndVectorOperation

def LU(A):
    N = len(A[0])
    L = matrixAndVectorOperation.create_ones_diagonal_matrix(1, N)
    U = matrixAndVectorOperation.create_zero_square_matrix(N)

    for i in range(N):
        for j in range(i + 1):
            temp = 0
            for k in range(j):
                temp += L[j][k] * U[k][i]
            U[j][i] = A[j][i] - temp

        for j in range(i + 1, N):
            for k in range(i):
                L[j][i] -= L[j][k] * U[k][i] / U[i][i]
            L[j][i] += A[j][i] / U[i][i]

    return L, U

def forward_sub(L, b):
    N = len(L)
    y = matrixAndVectorOperation.create_zero_vector(N)
    for i in range(N):
        temp = b[i]
        for j in range(i):
            temp -= L[i][j] * y[j]
        y[i] = temp / L[i][i]

    return y

def back_sub(U, y):
    N = len(U)
    x = matrixAndVectorOperation.create_zero_vector(N)
    for i in reversed(range(N)):
        temp = y[i]
        for j in range(i+1, N):
            temp -= U[i][j] * x[j]
        x[i] = temp / U[i][i]
    return x

def LU_solve(A, b):
    L, U = LU(A)
    y = forward_sub(L, b)
    x = back_sub(U, y)

    res = matrixAndVectorOperation.matrix_vector_multiplication(A, x)
    res = matrixAndVectorOperation.sub_vectors(res, b)
    norm_res = matrixAndVectorOperation.norm(res)
    return x, norm_res

