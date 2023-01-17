from numpy import allclose, conj, transpose
def IsHerm(A):
    return allclose(conj(transpose(A)), A, 1e-6, 1e-6)

from random import random
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









###############
#   TESTING   #
###############
from random import seed
from numpy import array
TARGET = 100
N_EIGS = 8
SIZE = 1000
MIN, MAX = +3, +250
MINO, MAXO = -3, 2
seed(69101)


A = SysMake(size=SIZE, min=MIN, max=MAX, mino=MINO, maxo=MAXO)
print(IsHerm(A))
A = array(A)
P = SysMake(size=SIZE, min=MIN, max=MAX, mino=0, maxo=0)
for i in range(0, SIZE):
    P[i][i] = 1/A[i][i]
from scipy.sparse import coo_matrix
A = coo_matrix(A)
P = coo_matrix(P)

from pydavidson import JDh
from scipy.sparse.linalg import inv
e,v=JDh(A,k=N_EIGS,v0=None,M=None,    #calculate k eigenvalues for mat, with no initial guess.
        tol=1e-6,maxiter=1000,    #set the tolerence and maximum iteration as a stop criteria.
        which='SL',sigma=TARGET,    #calculate selected(SL) region near sigma.
        #set up the solver for linear equation A*x=b(here, Jacobi-Davison correction function),
        #we use bicgstab(faster but less accurate than '(l)gmres') here.
        #preconditioning is used during solving this Jacobi-Davison correction function.
        linear_solver_maxiter=100,linear_solver='bicgstab',linear_solver_precon=True,
        iprint=1)
