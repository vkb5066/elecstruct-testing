#MINRES is an algorithm that iterativly solves Ax=b for a hermitian matrix A
#ref: https://en.wikipedia.org/wiki/MINRES


from random import random
from numpy import array, transpose, allclose, conj
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




"""
**************************************************************************************************************
*****  TESTING ***********************************************************************************************
**************************************************************************************************************
"""

A, b = SysMake(2, -1, 1)














"""
**************************************************************************************************************
*****  GARBAGE ***********************************************************************************************
**************************************************************************************************************

#Converts a standard array pair of indices i, j into a lower-packed-mode index according to
##https://www.ibm.com/docs/en/essl/6.2?topic=representation-lower-packed-storage-mode
def PairToLPInd(i, j, size):

    return (size-j)*i + j

#Convert a NxN matrix into an array in lower-packed mode as defined here
#https://www.ibm.com/docs/en/essl/6.2?topic=representation-lower-packed-storage-mode
#and here
#https://www.ibm.com/docs/en/essl/6.2?topic=matrix-complex-hermitian-storage-representation
def ConvToLP(A, size):
    ret = [None for i in range(0, size*(size+1)//2)]

    itr = 0
    for i in range(0, size):
        for j in range(i, size):
            ret[itr] = A[i][j]
            itr += 1
    return ret

A = [[1, 2, 3, 4, 5],
     [2, 6, 7, 8, 9],
     [3, 7, 10, 11, 12],
     [4, 8, 11, 13, 14],
     [5, 9, 12, 14, 15]]
print(ConvToLP(A, 5))
I, J = 2, 1
print(PairToLPInd(I, J, 5),'     ', ConvToLP(A, 5)[PairToLPInd(I, J, 5)], A[I][J])
"""
