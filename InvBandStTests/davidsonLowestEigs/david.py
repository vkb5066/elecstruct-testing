#!/bin/python
from __future__ import division
from __future__ import print_function
import math
import numpy as np
import time




from random import random
from numpy import allclose, array, conj, transpose
def SysMake(size, min, max, mino, maxo):
    #Make coefficient matrix A
    mat = [[None for i in range(0, size)] for j in range(0, size)]

    for i in range(0, size):
        mat[i][i] = min + random()*(max-min) + 0.j
        for j in range(i + 1, size):
            re = (mino + random()*(maxo-mino))
            im = (mino + random()*(maxo-mino))
            mat[i][j] = re + 1j*im
            mat[j][i] = re - 1j*im

    ##and check if it is really hermitian
    if(not allclose(array(mat), conj(transpose(mat)))):
        print("I suck")
        exit(1)

    return array(mat)




''' Block Davidson, Joshua Goings (2013)

    Block Davidson method for finding the first few
	lowest eigenvalues of a large, diagonally dominant,
    sparse Hermitian matrix (e.g. Hamiltonian)
'''

n = 120					# Dimension of matrix
tol = 1e-8				# Convergence tolerance
mmax = n//2				# Maximum number of iterations

''' Create sparse, diagonally dominant matrix A with 
	diagonal containing 1,2,3,...n. The eigenvalues
    should be very close to these values. You can 
    change the sparsity. A smaller number for sparsity
    increases the diagonal dominance. Larger values
    (e.g. sparsity = 1) create a dense matrix
'''

def IsHerm(A):
    return np.allclose(np.conj(np.transpose(A)), A, 1e-6, 1e-6)
def IsReal(A):
    return np.allclose(A.real, 0.0, 1e-6, 1e-6)
def IsOkay1(A, l, x): ##Checks Ax = lambda x
    return np.allclose(np.dot(A, x), l*x, 1e-6, 1e-6)
def IsOkay(A, ls, xs):
    ok = True
    for i in range(0, len(ls)):
        ok = IsOkay1(A=A, l=ls[i], x=xs[:,i])
        if(not ok):
            break
    return ok



sparsity = 0.001
A = np.zeros((n,n))
A = A + sparsity*np.random.randn(n,n)
A = A + sparsity*np.random.randn(n,n)*-1j
A = A + sparsity*np.random.randn(n,n)*+1j
for i in range(0,n):
    A[i,i] = i + 1
    for j in range(0, n):
        A[j,i] = np.conj(A[i,j])
SIZE = 1000
MIN, MAX = +3, +250
MINO, MAXO = -3, 2
SPARSITY = 0.001
A = SysMake(size=n, min=MIN, max=MAX, mino=MINO, maxo=MAXO)
print(IsHerm(A))

k = 12					# number of initial guess vectors
eig = 12				# number of eignvalues to solve
t = np.eye(n,k)			# set of k unit vectors as guess
V = np.zeros((n,n), dtype=complex)		# array of zeros to hold guess vec
I = np.eye(n)			# identity matrix same dimen as A

# Begin block Davidson routine

start_davidson = time.time()

for m in range(k,mmax,k):
    if m <= k:
        for j in range(0,k):
            V[:,j] = t[:,j]/np.linalg.norm(t[:,j])
        theta_old = 1
    elif m > k:
        theta_old = theta[:eig]
    V[:,:m],R = np.linalg.qr(V[:,:m])
    T = np.dot(np.conj(V[:,:m]).T,np.dot(A,V[:,:m]))
    #print("T: Hermitian?", IsHerm(T))
    THETA,S = np.linalg.eig(T) ##note that T is ALWAYS hermitian here
    idx = THETA.argsort()
    theta = THETA[idx]
    s = S[:,idx]
    for j in range(0,k):
        w = np.dot((A - theta[j]*I),np.dot(V[:,:m],s[:,j]))
        q = w/(theta[j]-A[j,j])
        V[:,(m+j)] = q
    norm = np.linalg.norm(theta[:eig] - theta_old)
    if norm < tol:
        break

end_davidson = time.time()

# End of block Davidson. Print results.
print("\n")
#print(len(S), len(theta), IsOkay(A=A, ls=THETA, xs=S))
#print(A.shape, THETA.shape, theta.shape, S.shape, s.shape)

print("davidson = ", np.round(theta[:eig], 3),";",
    end_davidson - start_davidson, "seconds")

# Begin Numpy diagonalization of A

start_numpy = time.time()

E,Vec = np.linalg.eig(A)
E = np.sort(E)

end_numpy = time.time()

# End of Numpy diagonalization. Print results.

print("numpy = ", np.round(E[:eig], 3),";",
     end_numpy - start_numpy, "seconds")
