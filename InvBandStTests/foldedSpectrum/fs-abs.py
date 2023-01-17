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







"""
**************************************************************************************************************
*****  TESTING ***********************************************************************************************
**************************************************************************************************************
"""


N_EIGS = 3
SIZE = 3
MIN, MAX = +3, +250
MINO, MAXO = -3, 2
SPARSITY = 0.001
seed(69101)

SQ = True
CB = not SQ

A = SysMake(size=SIZE, min=MIN, max=MAX, mino=MINO, maxo=MAXO)
print("orig herm: ", IsHerm(A))
A = array(A)

#Base, original eigenvalues
va, ve = eig(A)
sind = va.argsort()[:N_EIGS]
va = va[sind]
ve = ve[:,sind]

div = 1

#Folded Spectrum ...
eref = 50
I = eye(SIZE)
AF = abs(A - eref*I)

print(round(A, 3))
print(round(AF, 0))

print("folded herm:", IsHerm(A))
#... unshifted eigenvalues
vafu, vefu = eig(AF)
sind = vafu.argsort()[:N_EIGS]
vafu = vafu[sind]
vefu = vefu[:,sind]
#... shifted eigenvalues
vaf, vef = deepcopy(vafu), deepcopy(vefu)
for i in range(0, N_EIGS):
    sign = +1.0
    if(vaf[i] < 0):
        sign = -1.0
    vaf[i] = sign*(sign*vaf[i])**(1/div) + eref


print("orig eigs:")
for e in range(0, N_EIGS):
    print(round(va[e], 5))

print("folded eigs (u), (folded eigs)^(1/div) (u):")
for e in range(0, N_EIGS):
    print(round(vafu[e], 5), round((vafu[e])**(1/div), 5))

print("folded eigs (s):")
for e in range(0, N_EIGS):
    print(round(vaf[e], 5))
