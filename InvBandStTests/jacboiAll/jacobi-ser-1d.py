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
    ii, ij, jj = None, None, None
    ri, rj, ir = None, None, None

    diag = sum([A[TransformInds(i, i, size)]*A[TransformInds(i, i, size)] for i in range(0, size)])/size
    for n in range(0, nSweeps):
        #print("n =", n, "...")
        for i in range(0, size - 1):
            for j in range(i + 1, size):
                ii = TransformInds(i, i, size)
                ij = TransformInds(i, j, size)
                jj = TransformInds(j, j, size)

                if(CAV(A[ij]) < 1.0E-6):
                    continue

                ##Setup
                sgn = +1.0 if A[ii].real - A[jj].real > 0.0 else -1.0
                t = 2.0*CAV(A[ij])*sgn / \
                    (
                            CAV(A[ii] - A[jj]) + \
                            sqrt(
                                    CAV(A[ii] - A[jj])*CAV(A[ii] - A[jj]) + \
                                    4.0*CAV(A[ij])*CAV(A[ij])
                                )
                    )

                c = 1.0 / sqrt(1 + t*t)
                s = t*c ##t/sqrt(1+t^2)
                sp = s*A[ij]/CAV(A[ij])
                sm = s*CAV(A[ij])/A[ij]

                ##Do rotations (w/o functions - for "speed")
                A[ii] = A[ii] + t*CAV(A[ij])
                A[jj] = A[jj] - t*CAV(A[ij])
                A[ij] = 0.0 + 0.0j
                #A[j][i] = 0.0 + 0.0j

                for r in range(0, i):
                    ri = TransformInds(r, i, size)
                    rj = TransformInds(r, j, size)
                    x =     c*A[ri] + sm*A[rj]
                    A[rj] = c*A[rj] - sp*A[ri]
                    A[ri] = x
                for r in range(i + 1, j):
                    ir = TransformInds(i, r, size)
                    rj = TransformInds(r, j, size)
                    x =     c*A[ir] + sp*conj(A[rj])
                    A[rj] = c*A[rj] - sp*conj(A[ir])
                    A[ir] = x
                for r in range(j + 1, size):
                    ir = TransformInds(i, r, size)
                    jr = TransformInds(j, r, size)
                    x =     c*A[ir] + sp*A[jr]
                    A[jr] = c*A[jr] - sm*A[ir]
                    A[ir] = x

        diagN = sum([A[TransformInds(i, i, size)]*A[TransformInds(i, i, size)] for i in range(0, size)])/size
        print("dE:", diagN - diag)
        if(abs(diagN - diag) < 0.0001):
            break
        diag = diagN


    #just for my sanity: above only updates upper diagonal, but I want to see a full matrix update
    #for i in range(0, size - 1):
    #    for j in range(i, size):
    #        A[j][i] = conj(A[i][j])
    return A


def TransformMatrix(A, size):
    sizeN = size*(size+1) // 2
    ret = [None for i in range(0, sizeN)]
    counter = 0
    for i in range(0, size):
        for j in range(i, size):
            ret[counter] = A[i][j]
            counter += 1
    return ret

def TransformInds(i, j, size):
    if(i <= j):
        return (2*size - i - 1)*i//2 + j
    return (2*size - j - 1)*j//2 + i


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
A = Jacobi(TransformMatrix(A, SIZE), SIZE, 10)
f = array(sorted([A[TransformInds(i, i, SIZE)].real for i in range(0, SIZE)]))
print((e - f).real)
