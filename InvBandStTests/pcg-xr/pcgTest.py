from random import random, seed
from numpy import array, transpose, allclose, conj, dot, sqrt, vdot, zeros, round, eye
from numpy.linalg import inv, eig
from copy import deepcopy




"""
bro
"""

def IsHerm(A):
    return allclose(conj(transpose(A)), A, 1e-6, 1e-6)

def SysMake(size, min, max, mino, maxo):
    #Make coefficient matrix A
    mat = [[None for i in range(0, size)] for j in range(0, size)]

    for i in range(0, size):
        mat[i][i] = min + random()*(max-min) + 0.j
        for j in range(i + 1, size):
            re = mino + random()*(maxo-mino)
            im = mino + random()*(maxo-mino)
            mat[i][j] = re + 1j*im
            mat[j][i] = re - 1j*im

    ##and check if it is really hermitian
    if(not allclose(array(mat), conj(transpose(mat)))):
        print("I suck")
        exit(1)

    return mat






def PCG(n, A, P, neigs, nitr, blocksize):
    X = array([[0. for i in range(0, nEigs)] for j in range(0, n)], dtype=complex)
    for i in range(0, neigs):
        X[i][i] = 1.0 + 0.0j

    for i in range(0, nitr):
        for m in range(0, blocksize):





"""
**************************************************************************************************************
*****  TESTING ***********************************************************************************************
**************************************************************************************************************
"""


N_EIGS = 5
SIZE = 50
MIN, MAX = 10, 250
MINO, MAXO = -3, 2
REFE = 120
seed(111)


A = SysMake(size=SIZE, min=MIN, max=MAX, mino=MINO, maxo=MAXO)
print("orig herm: ", IsHerm(A))
A = array(A)
A2 = dot(A - REFE*eye(SIZE), A - REFE*eye(SIZE))




#Base, original eigenvalues
va, ve = eig(A2)
sind = va.argsort()[:N_EIGS + SIZE - N_EIGS]
va = va[sind]
ve = ve[:,sind]
