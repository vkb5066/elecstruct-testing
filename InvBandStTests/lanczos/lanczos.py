from random import random
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



#Returns a symmetrix, real, tridiagonal matrix in two arrays: the diagonals and off-diagonals
def Lanczos(A, v, size, k):
    k = min(k, size)
    #                       alphas                       betas
    retDiag, retOff = array([None for i in range(0, k)]), array([None for i in range(0, k)])

    ##init
    w = dot(A, v)
    alpha = dot(conj(w), v)
    w = w - alpha*v

    retDiag[0] = alpha
    retOff[0] = 0.0 ##no beta-1
    #loops
    for j in range(1, k):
        beta = sqrt(dot(conj(w), w))
        if(beta < 1.0E-9):
            print("ruh roh")
            exit(1)

        vLast = deepcopy(v)
        v = w/beta
        w = dot(A, v)
        alpha = dot(conj(w), v)
        w = w - alpha*v - beta*vLast

        retDiag[j] = alpha
        retOff[j] = beta

    return retDiag, retOff


"""
**************************************************************************************************************
*****  TESTING ***********************************************************************************************
**************************************************************************************************************
"""
SIZE = 1000
MIN, MAX = -1, +1
A, b = SysMake(SIZE, MIN, MAX)
A, b = array(A), array(b)

K = 1000
v0 = array([  MIN + random()*(MAX-MIN) + 1j*(MIN + random()*(MAX-MIN)) for i in range(0, SIZE)])
norm = sqrt(dot(conj(v0), v0))
v0 /= norm

TDiag, TOff = Lanczos(A=A, v=array([1/sqrt(SIZE) for i in range(0, SIZE)]), size=SIZE, k=K)
A_ = array([[0.0 + 0.0j for i in range(0, K)] for j in range(0, K)])
for i in range(0, K):
    A_[i][i] = TDiag[i]
for i in range(0, K - 1):
    A_[i][i + 1] = TOff[i + 1]
    A_[i + 1][i] = TOff[i + 1]

from scipy.linalg import eigvals

print(array(sorted(eigvals(A).real)[:K//2]) - array(sorted(eigvals(A_).real)[:K//2]))
print("\n\n")
