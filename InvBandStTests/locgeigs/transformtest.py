from random import random as random
from random import seed
from numpy import empty, conj, transpose, allclose, seterr
from numpy.linalg import eigh as numpyeigh
seterr(all='raise')

def sysmake(n):
    A, B = None, None

    hpd = 0
    while(not hpd):
        A = empty(shape=(n, n), dtype=complex)
        for i in range(0, n):
            for j in range(i+1, n):
                A[i][j] = random() + 1j*random()
                A[j][i] = A[i][j].real - 1j*A[i][j].imag
            A[i][i] = 10*random() + 0j

        #check if A is hermitian
        if(allclose(A, conj(transpose(A)))):
            print("(A) hermitian: ok")
            hpd = 1

    hpd = 0
    while(not hpd):
        B = empty(shape=(n, n), dtype=complex)
        for i in range(0, n):
            for j in range(i+1, n):
                B[i][j] = random() + 1j*random()
                B[j][i] = B[i][j].real - 1j*B[i][j].imag
            B[i][i] = 10*random() + 0j

        #check if B is hermitian positive definite
        if(allclose(B, conj(transpose(B)))):
            print("(B) hermitian: ok")
            e, v = numpyeigh(B)
            hpd_ = 1
            for i in range(0, n):
                if(e[i] < 0):
                    hpd_ = 0
                    break
            if(hpd_):
                print("(B) positive definite: ok (", e, ")")
            hpd = hpd_

    return A, B



from numpy import sqrt
def chol(n, A):
    i, j = 0, 0
    su = 0.0 + 0.0j

    for i in range(0, n):
        for j in range(0, i):
            su = 0.0 + 0.0j ##this sum is complex
            for k in range(0, j):
                su += A[i][k] * conj(A[j][k])
            A[i][j] = (1.0/A[j][j].real) * (A[i][j] - su)

        su = 0.0 + 0.j ##this sum has zero complex part
        for j in range(0, i):
            su += A[i][j].real*A[i][j].real + A[i][j].imag*A[i][j].imag + 0j
        A[i][i] = sqrt(A[i][i].real - su.real)

    ##just to match numpy - if you only access the lower triangle, this is unnecessary
    #for i in range(0, n):
    #    for j in range(i+1, n):
    #        A[i][j] = 0.0 + 0.0j

    return A

#solve LX = A for X
def getx_full(n, L, A):
    X = empty(shape=(n, n), dtype=complex)
    for i in range(0, n):
        for j in range(0, n):
            X[i][j] = A[i][j]
            for k in range(0, i):
                X[i][j] -= L[i][k]*X[k][j]
            X[i][j] /= L[i][i]
    return X
#solve CL^*^T = X for C
def getc_full(n, L, X):
    C = empty(shape=(n, n), dtype=complex)
    C.fill(None)
    for i in range(0, n):
        for j in range(0, n):
            C[i][j] = X[i][j]
            for k in range(0, j):
                    C[i][j] -= C[i][k]*conj(L[j][k])
            C[i][j] /= conj(L[j][j]) ##pure real?
    return C

#solve LX = A for X, but only computes the upper triangle of X (replaces A)
def getx_half(n, L, A):
    for i in range(0, n):
        for j in range(i, n):
            for k in range(0, i):
                A[i][j] -= L[i][k]*A[k][j]
            A[i][j] /= L[i][i].real
    return A
#solve CL^*^T = X for C, but only accesses the upper triangle of X (all of C is computed, replaces X)
def getc_half(n, L, X):
    for i in range(0, n):
        ##diagonals
        for j in range(0, i):
            X[i][i] -= X[i][j]*conj(L[i][j])
        X[i][i] /= conj(L[i][i].real)
        ##off-diagonals
        for j in range(i+1, n):
            for k in range(0, j):
                X[i][j] -= X[i][k]*conj(L[j][k])
            X[i][j] /= conj(L[j][j].real)  ##pure real?
            X[j][i] = conj(X[i][j])
    return X

#solve L^*^T R = V for R (the eigenvectors of the un-transformed problem)
#note that the memory access pattern of this function is disgustingly bad
def retvecscol(n, L, V):
    R = empty(shape=(n, n), dtype=complex)
    for i in range(n-1, -1, -1): ##go from n-1 to 0 (both inclusive), decrementing by -1
        for j in range(n-1, -1, -1): ##ditto ^
            R[i][j] = V[i][j]
            for k in range(i+1, n):
                R[i][j] -= conj(L[k][i])*R[k][j]
            R[i][j] /= conj(L[i][i]) ##pure real?
    return R
#solve L^*^T R = V for R (the eigenvectors of the un-transformed problem).  Overwrites V
def retvecsrow(n, L, V):
    for i in range(n-1, -1, -1): ##go from n-1 to 0 (both inclusive), decrementing by -1
        for j in range(n-1, -1, -1): ##ditto ^
            for k in range(j+1, n):
                V[i][j] -= conj(L[k][j])*V[i][k]
            V[i][j] /= conj(L[j][j]) ##pure real?
    return V

#seed(0x0B00B135)
N = 14
A, B = sysmake(N)

from numpy import round
from scipy.linalg import eigh
print("\n\n")
scva, scve = eigh(a=A, b=B)
#print("real eigvals:\n", scva)
#print("real eigvecs:\n", round(scve, 3))



from numpy import max
from numpy.linalg import cholesky, inv
#------method one: the super dumb way
#L = cholesky(B)
#C = inv(L) @ A @ inv(conj(transpose(L)))
#myva, myve = numpyeigh(C)
#myve = inv(conj(transpose(L))) @ myve
#------method two: the less dumb way
#from numpy.linalg import solve
#L_ = cholesky(B)
#X_ = solve(L_, A)
#C_ = solve(inv(X_), inv(conj(transpose(L_)))) ##numpy can only solve Ax=B, not XA=B, so we have to do this trash
#myva, myve = numpyeigh(C)
#myve = solve(conj(transpose(L)), myve)
#------method three: my less dumb way
#L = chol(N, B)
#X = getx_full(N, L, A)
#C = getc_full(N, L, X)
#myva, myve = numpyeigh(C)
#myve = retvecscol(N, L, myve)
#------method four: my smart way
L = chol(N, B)
X = getx_half(N, L, A)
C = getc_half(N, L, X)
myva, myve = numpyeigh(C)
if(0):
    myve = retvecscol(N, L, myve)
else:
    myve = transpose(myve) ##the case for my qrh()
    myve = retvecsrow(N, L, myve)
    myve = transpose(myve) ##to match numpy
#print("my eigenvals:\n", myva)
#print("my eigvecs:\n", round(myve, 3))
print("\ncorrect?")
print(allclose(myva, scva))
print(allclose(myve, scve))
