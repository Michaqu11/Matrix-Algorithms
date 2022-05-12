import math
import time

import compareWithNumpyAndScipy
from matplotlib import pyplot

import GaussSeidel
import Jacobi
from LU import LU_solve
import matrixAndVectorOperation



def jacobi(A, b, N):
    tstart = time.time()
    result, iter = Jacobi.jacobi(A, b, N)
    tend = time.time()
    time_difference = tend - tstart
    print()
    print("Jacobi:")
    print("-norma z wektora residuum: " + str(result))
    print("-liczba iteracji: " + str(iter))
    print("-czas obliczenia: " + str(time_difference))
    return time_difference

def gauss_seidel(A, b, N):
    tstart = time.time()
    result, iter = GaussSeidel.gauss_seidel(A, b, N)
    tend = time.time()
    time_difference = tend - tstart
    print()
    print("GaussSeidel:")
    print("-norma z wektora residuum: " + str(result))
    print("-liczba iteracji: " + str(iter))
    print("-czas obliczenia: " + str(time_difference))
    return time_difference

def LU(C, b):
    tstart = time.time()
    result, norm_res = LU_solve(C, b)
    tend = time.time()
    time_difference = tend - tstart
    print()
    print("LU:")
    print("-norma z wektora residuum: " + str(norm_res))
    print("-czas obliczenia: " + str(time_difference))
    return time_difference


if __name__ == '__main__':

    #184841
    c = 1
    d = 4
    e = 8
    f = 4
    N = 900+10*d+c
    a1 = 5 + e
    a2 = a3 = -1

    #Zadanie A
    b = matrixAndVectorOperation.create_b_vecotr(f, N)
    print("wektor b:", b)

    #Zadanie B
    A = matrixAndVectorOperation.create_A_matrix(N, a1, a2, a3)
    jacobi(A, b, N)
    gauss_seidel(A, b, N)


    # Zadanie C
    a1 = 3
    a2 = a3 = -1
    N = 900+10*d+c
    C = matrixAndVectorOperation.create_A_matrix(N, a1, a2, a3)
    jacobi(C, b, N)
    gauss_seidel(C, b, N)

    #Zadanie D
    LU(C, b)

    #ZadanieE
    time_jacobi = []
    time_gauss_seidel = []
    time_LU = []

    N = [100, 500, 1000, 2000, 3000]
    a1 = 5 + e
    for i in range(len(N)):
        print()
        print(N[i], ": ")
        A = matrixAndVectorOperation.create_A_matrix(N[i], a1, a2, a3)
        b = matrixAndVectorOperation.create_b_vecotr(f, N[i])
        time_jacobi.append(jacobi(A, b, N[i]))
        time_gauss_seidel.append(gauss_seidel(A, b, N[i]))
        time_LU.append(LU(A, b))


    pyplot.plot(N, time_jacobi, label="Jacobi", color="blue")
    pyplot.plot(N, time_gauss_seidel, label="Gauss-Seidel", color="red")
    pyplot.plot(N, time_LU, label="LU_decomposition", color="green")
    pyplot.legend()
    pyplot.grid(True)
    pyplot.ylabel('Czas[s]')
    pyplot.xlabel('liczba niewiadomych')
    pyplot.title('Zależność czasu od liczby niewiadomych')
    pyplot.show()
    N = 900 + 10 * d + c
    a1 = 5 + e
    A = matrixAndVectorOperation.create_A_matrix(N, a1, a2, a3)
    a1 = 3
    C = matrixAndVectorOperation.create_A_matrix(N, a1, a2, a3)
    b = matrixAndVectorOperation.create_b_vecotr(f, N)
    time_jacobi_numpy, time_gauss_seidel_numpy, time_LU_numpy = compareWithNumpyAndScipy.compare(A, C, b, N, e, f, a1, a2, a3)

    N = [100, 500, 1000, 2000, 3000]
    print("\n\n\n PORWNANIE CZASOWE")
    for i in range(len(N)):
        print("Liczba niewiadomych: " + str(N[i]))
        print("-czas własnych procedur: \njacobi: " + str(time_jacobi[i]) + " GS:" + str(time_gauss_seidel[i]) + " LU:" + str(time_LU[i]))
        print("-czas operacji z pakietow: jacobi: " + str(time_jacobi_numpy[i]) + " GS:" + str(time_gauss_seidel_numpy[i]) + " LU:" + str(time_LU_numpy[i]))
        print()

    pyplot.plot(N, time_jacobi, label="Jacobi", color="blue")
    pyplot.plot(N, time_gauss_seidel, label="Gauss-Seidel", color="red")
    pyplot.plot(N, time_LU, label="LU_decomposition", color="green")
    pyplot.plot(N, time_jacobi_numpy, label="Jacobi_numpy", color="yellow")
    pyplot.plot(N, time_gauss_seidel_numpy, label="Gauss-Seidel_numpy", color="orange")
    pyplot.plot(N, time_LU_numpy, label="LU_decomposition_numpy", color="purple")
    pyplot.legend()
    pyplot.grid(True)
    pyplot.ylabel('Czas[s]')
    pyplot.xlabel('liczba niewiadomych')
    pyplot.title('Zależność czasu od liczby niewiadomych')
    pyplot.show()
