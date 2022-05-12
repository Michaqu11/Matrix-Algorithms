import math


def norm(res, N):
    norm_res = 0
    for i in range(N):
        norm_res += res[i][0] ** 2
    return math.sqrt(norm_res)

def create_A_matrix(N, a1, a2, a3):
    A = []
    row = N*[0]
    row[0] = a1
    row[1] = a2
    row[2] = a3
    A.append(row.copy())

    for i in range(1, N):
        row.pop()
        if i == 1:
            row.insert(0, a2)
        elif i == 2:
            row.insert(0, a3)
        else:
            row.insert(0, 0)

        A.append(row.copy())

    return A
def create_b_vecotr(f, N):
    b = []
    for i in range(N):
        b.append(math.sin(i * (f+1)))
    return b
def matrix_vector_multiplication(matrix, vector):
    N = len(vector)
    result = [0 for i in range(len(vector))]
    for i in range(N):
        for j in range(N):
            result[i] += matrix[i][j] * vector[j]
    return result


def sub_vectors(vector1, vector2):
    N = len(vector1)
    result = [0 for i in range(N)]
    for i in range(N):
        result[i] = vector1[i] - vector2[i]
    return result

def create_zero_vector(N):

    return [ 0 for i in range(N)]


def create_zero_square_matrix(N):
    result = []
    for i in range(N):
        row = N * [0]
        result.append(row)

    return result


def create_ones_diagonal_matrix(a, N):
    result = create_zero_square_matrix(N)
    for i in range(N):
        result[i][i] = a

    return result


def norm(res):
    norm_res = 0
    for i in range(len(res)):
        norm_res += res[i] ** 2
    return math.sqrt(norm_res)


