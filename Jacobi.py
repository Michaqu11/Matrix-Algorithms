import math

import matrixAndVectorOperation

norm_residium = 10 ** -9


def jacobi(A, b, N):

    x = [0 for i in range(N)]
    iter = 0
    norm_res = 1
    while  10e6 > norm_res > norm_residium:
        iter += 1
        x_copy = x.copy()
        for i in range(N):
            sumToDeduct = 0
            for j in range(N):
                if i != j:
                    sumToDeduct += A[i][j] * x_copy[j]
            x[i] = (b[i] - sumToDeduct) / A[i][i]

        # res = M*r - b;
        res = matrixAndVectorOperation.matrix_vector_multiplication(A, x)
        res = matrixAndVectorOperation.sub_vectors(res, b)
        norm_res = matrixAndVectorOperation.norm(res)

    return norm_res, iter



