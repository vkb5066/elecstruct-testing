from random import random, seed
from numpy import array, transpose, allclose, conj, dot, sqrt
from copy import deepcopy

def SysMake(size, min, max):
    #Make coefficient matrix A
    mat = [[None for i in range(0, size)] for j in range(0, size)]

    for i in range(0, size):
        mat[i][i] = min + random()*(max-min)
        for j in range(i + 1, size):
            re = min + random()*(max-min)
            im = min + random()*(max-min)
            mat[i][j] = re + 1j*im
            mat[j][i] = re - 1j*im

    ##and check if it is really hermitian
    if(not allclose(array(mat), conj(transpose(mat)))):
        print("I suck")
        exit(1)

    #Make resultant vector b
    res = [None for i in range(0, size)]
    for i in range(0, size):
        re = min + random() * (max - min)
        im = min + random() * (max - min)
        res[i] = re + 1j*im

    return mat, res

def CAV(z): ##absolute value of a complex number
    return sqrt(z.real*z.real + z.imag*z.imag)

#Returns a symmetrix, real, tridiagonal matrix in two arrays: the diagonals and off-diagonals
def Jacobi(A, size, nSweeps):
    diag = sum([A[i][i]*A[i][i] for i in range(0, size)])/size
    for n in range(0, nSweeps):
        #print("n =", n, "...")
        for i in range(0, size - 1):
            for j in range(i + 1, size):
                if(CAV(A[i][j]) < 1.0E-6):
                    continue

                ##Setup
                sgn = +1.0 if A[i][i].real - A[j][j].real > 0.0 else -1.0
                t = 2.0*CAV(A[i][j])*sgn / \
                    (
                            CAV(A[i][i] - A[j][j]) + \
                            sqrt(
                                    CAV(A[i][i] - A[j][j])*CAV(A[i][i] - A[j][j]) + \
                                    4.0*CAV(A[i][j])*CAV(A[i][j])
                                )
                    )
                c = 1.0 / sqrt(1 + t*t)
                s = t*c ##t/sqrt(1+t^2)
                sp = s*A[i][j]/CAV(A[i][j])
                sm = s*CAV(A[i][j])/A[i][j]

                ##Do rotations (w/o functions - for "speed")
                A[i][i] = A[i][i] + t*CAV(A[i][j])
                A[j][j] = A[j][j] - t*CAV(A[i][j])
                A[i][j] = 0.0 + 0.0j
                #A[j][i] = 0.0 + 0.0j ##not necessary (?) we only access the upper triangle

                for r in range(0, i):
                    x = c*A[r][i] + sm*A[r][j]
                    A[r][j] = -sp*A[r][i] + c*A[r][j]
                    A[r][i] = x
                for r in range(i + 1, j):
                    x = c*A[i][r] + sp*conj(A[r][j])
                    A[r][j] = -sp*conj(A[i][r]) + c*A[r][j]
                    A[i][r] = x
                for r in range(j + 1, size):
                    x = c*A[i][r] + sp*A[j][r]
                    A[j][r] = -sm*A[i][r] + c*A[j][r]
                    A[i][r] = x

        diagN = sum([A[i][i]*A[i][i] for i in range(0, size)])/size
        print("dE:", diagN - diag)
        if(abs(diagN - diag) < 0.0001):
            break
        diag = diagN


    #just for my sanity: above only updates upper diagonal, but I want to see a full matrix update
    #for i in range(0, size - 1):
    #    for j in range(i, size):
    #        A[j][i] = conj(A[i][j])
    return A






"""
**************************************************************************************************************
*****  TESTING ***********************************************************************************************
**************************************************************************************************************
"""
SIZE = 150
MIN, MAX = -10, +10
seed(69101)
A, b = SysMake(SIZE, MIN, MAX)
A, b = array(A), array(b)

from scipy.linalg import eigvals
e = array(sorted(eigvals(A).real))
#print(e, "\n\n")
A = Jacobi(A, SIZE, 10)
f = array(sorted([A[i][i].real for i in range(0, SIZE)]))
for a in A:
    from numpy import round
    print(round(a, 3))
print("\n\n")
print((e - f).real)
