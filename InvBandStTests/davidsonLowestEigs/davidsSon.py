from random import random, seed
from numpy import array, transpose, allclose, conj, dot, sqrt, vdot
from numpy.linalg import inv
from copy import deepcopy

"""
AN IMPLEMENTATION OF THE GENERALIZED DAVIDSON METHOD WITH RESTARTS

BASED OFF OF:
TITLE: THE DAVIDSON METHOD
AUTHORS: M. CROUZEIX, B. PHILIPPE, AND M. SADKANE
JOURNAL: SOCIETY FOR INDUSTRIAL AND APPLIED MATHEMATICS (SIAM)
DATE: JAN 1994

EDITED TO WORK WITH HERMITIAN MATRICES (AS OPPOSED TO REAL SYMMETRIC) 
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


#Modified (safe) grahm schmidt on a list of row vectors V
def MGS(V, nVecs, vecDim):
    ##Make an nVecs (rows) x vecDim (cols) matrix
    #so the ith row of the return vector will be the ith orthogonal vector of V
    Q = array([[0. for i in range(0, vecDim)] for j in range(0, nVecs)], dtype=complex)

    for j in range(0, nVecs):
        Q[j] = V[j] / sqrt(dot(conj(V[j]), V[j]))
        for k in range(j + 1, nVecs):
            V[k] = V[k] - dot(  dot(transpose(conj(Q[j])), V[k]),   Q[j]  )

    return Q



from numpy.linalg import eig, pinv
from numpy import eye
#Davidson diagonalization for the lowest nEigs eigenpairs of a DIAGONALLY DOMINANT
#matrix A
def David(n, A, nEigs, maxBasisSize=None, V0exists=False, V0=None):
    if(maxBasisSize == None):
        maxBasisSize = max(min(int(2*nEigs + 1), n), nEigs)
    maxBasisSize = int(6.25*nEigs)#50#n#2*nEigs

    assert n >= nEigs
    assert maxBasisSize >= 2*nEigs
    assert n > maxBasisSize

    #Just the diagonals of A, for preconditioning
    C = array([[0. for i in range(0, n)] for j in range(0, n)], dtype=complex)
    for i in range(0, n):
        C[i][i] = A[i][i]

    #The tridiagonal portions of A ... TODO: Get rid of this since it seems useless
    TRI = deepcopy(C)
    TRI[0][1] = A[0][1]
    for i in range(1, n-1):
        TRI[i+1][i] = A[i+1][i]
        TRI[i-1][i] = A[i-1][i]
    TRI[n-2][n-1] = A[n-1][n-1]

    V = array([[0. for i in range(0, maxBasisSize)] for j in range(0, n)], dtype=complex)
    for i in range(0, maxBasisSize):
        V[i][i] = 1.0
    if(V0exists):
        for i in range(0, V0.shape[1]):
            V[:,i] = V0[:,i]

    ##columns of ritz vectors
    X = array([[0. for i in range(0, nEigs)] for j in range(0, n)], dtype=complex)
    ##columns of residuals
    R = array([[0. for i in range(0, nEigs)] for j in range(0, n)], dtype=complex)
    ##columns of new search directions
    T = array([[0. for i in range(0, nEigs)] for j in range(0, n)], dtype=complex)


    currE, lastE = 100, 0
    currBasisSize = nEigs
    counter = 0
    while(True):
        Vk = V[:,:currBasisSize]
        W = A @ Vk
        H = transpose(conj(Vk)) @ W

        va, ve = eig(H)
        sind = va.argsort()[:nEigs]
        va = va[sind]
        ve = ve[:,sind]

        #Get ritz vectors
        #These approximate the eigenvectors of A and are in order with the eigenvalues
        for i in range(0, nEigs):
            X[:,i] = Vk @ ve[:,i]


        #Converg check
        ##the eigenvalue furthest from the bottom is the least well converged so only check that one.
        ##normally, you'd check both values/vectors, but there is some proof (I forget the name...) that the
        ##wavefunctions are an order of magnitude less important than their corresponding observables
        currE = va[-1]
        if(counter):
            dev = abs(currE - lastE)
            print(counter, dev)
            if(dev < 0.001):
                return va, X
        else:
            print(counter, "---")
        lastE = currE


        #Grab residuals
        #Normally, this'd be above the convergence check, but I only check eigenvalues, so this isn't needed
        #unless we aren't converged yet
        for i in range(0, nEigs):
            R[:, i] = va[i]*X[:, i] - W@ve[:, i]

        for i in range(0, nEigs):
            ##Classic preconditioner (cheap):
            for j in range(0, n):
                C[j][j] = 1. / (va[i] - A[j][j])

            #Safe preconditioner (a bit more expensive than ^^^, but guaranteed to converge for restarts):
            # find the minimum of 1 / |lambda(i) - A(j, j)| for 0 <= j < n ...
            # ... which corresponds to 1 / (the maximum value of |lambda - A(j, j)| for 0 <= j < n)
            #mi = 1./max([abs(va[i] - A[k][k]) for k in range(0, n)])
            #for j in range(0, n):
            #    C[j][j] = mi
            #Expt. preconditioner (you'd think that this would be good, but it sucks):
            #C = (inv(va[i]*eye(n) - TRI))

            T[:,i] = C @ R[:,i]

        #Increase the basis size or restart
        if(currBasisSize + nEigs < maxBasisSize): ##increase of basis
            for i in range(0, nEigs):
                V[:,currBasisSize + i] = T[:,i]
            TMP = MGS(V=transpose(V), nVecs=currBasisSize+nEigs, vecDim=n)
            V[:,:currBasisSize+nEigs] = transpose(TMP)
        else: ##restart procedure
            currBasisSize = nEigs
            for j in range(0, nEigs):
                V[:,j] = X[:,j]
                V[:,j+nEigs] = T[:,j]
            TMP = MGS(V=transpose(V), nVecs=2*nEigs, vecDim=n)
            V[:,:2*nEigs] = transpose(TMP)

        currBasisSize += nEigs
        counter += 1






"""
**************************************************************************************************************
*****  TESTING ***********************************************************************************************
**************************************************************************************************************
"""

from numpy import zeros, round, dot, array, allclose
from numpy.random import randn
import time

N_EIGS = 8
SIZE = 100
MIN, MAX = +3, +250
MINO, MAXO = -3, 2
SPARSITY = 0.001
seed(69101)

A = SysMake(size=SIZE, min=MIN, max=MAX, mino=MINO, maxo=MAXO)
print(IsHerm(A))
A = array(A)

from scipy.sparse.linalg import eigs
sstart = time.time()
sva, svi = eigs(A=A, k=N_EIGS)
send = time.time()

estart = time.time()
va, ve = eig(A)
eend = time.time()
sind = va.argsort()[:N_EIGS]
va = va[sind]
ve = ve[:,sind]

dstart = time.time()
eigs, vecs = David(SIZE, A, N_EIGS)
dend = time.time()

#Check that eigenvalues are the same, to 3 decimal places at least
print("EIGENVALUES:", allclose(eigs, va, 1e-3, 1e-3))
#Check that the eigenvectors obey the eigenvalue equations ...
okay = True
for i in range(0, N_EIGS):
    okay = allclose(A@vecs[:,i] - eigs[i]*vecs[:,i], 0.0, 1e-6, 1e-6)
    if(not okay):
        print(i, max(A@vecs[:,i] - eigs[i]*vecs[:,i]))

print("EIGENVECTORS(1):", okay)
#... and that they are normalized
okay = True
for i in range(0, N_EIGS):
    okay = allclose(vecs[:,i], vecs[:,i]/sqrt(dot(conj(vecs[:,i]), vecs[:,i])), 0.0 + 0.0j, 1e-6, 1e-6)
    if(not okay):
        break
print("EIGENVECTORS(2):", okay)

#Print timings
print("TIME: david:", round(dend-dstart, 2),
      " ... numpy:", round(eend-estart, 2),
      " ... scipy:", round(send - sstart, 2))


#Example of a warm-start: re-using the vectors from the previous run (and, presumably, the vectors
#from a run with a very similar hamiltonian to the previous run) speeds up convergence
##randomize A a little bit
for i in range(0, SIZE):
    A[i][i] += (-1.0 + 2.0*random()) ##between -1 and +1
    for j in range(i+1, SIZE):
        A[i][j] += ((-1.0 + 2.0*random()) + (-1.0 + 2.0*random())*1j)*0.01
        A[j][i] = conj(A[i][j])
dstart = time.time()
eigs, vecs = David(SIZE, A, N_EIGS, V0exists=True, V0=vecs)
dend = time.time()
print("TIME: david2:", round(dend-dstart, 2))
