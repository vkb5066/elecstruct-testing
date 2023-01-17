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

EDITED TO WORK WITH HERMITIAN MATRICES (AS OPPOSED TO REAL SYMMETRIC) AND TO PRIORITIZE 
ROW ACCESSES (INSTEAD OF COLUMN ACCESSES) AS MUCH AS POSSIBLE
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
#This gets called many times, so ideally, we'd allocate a maximum amount of memory once and pass its address
#to this functions instead of constantly re-making Q
def MGS(V, nVecs, vecDim):
    ##Make an nVecs (rows) x vecDim (cols) matrix
    #so the ith row of the return vector will be the ith orthogonal vector of V
    Q = array([[0. for i in range(0, vecDim)] for j in range(0, nVecs)], dtype=complex)

    for i in range(0, nVecs):
        Q[i] = V[i] / sqrt(dot(conj(V[i]), V[i]))
        for j in range(i + 1, nVecs):
            V[j] = V[j] - dot(  dot(transpose(conj(Q[i])), V[j]),   Q[i]  )

    return Q


from numpy.linalg import eig, pinv, eigh, solve
from numpy import eye
#Davidson diagonalization for the lowest nEigs eigenpairs of a DIAGONALLY DOMINANT
#hermitian matrix A
#TODO: A -> flat format, V0 -> matrix format
def David(n, A, nEigs, maxBasisSize=None, V0exists=False, V0=None, eTol=1e-3,
          fs=False, eref=None):
    if(maxBasisSize == None):
        maxBasisSize = min(int(6.25*nEigs), n-1) ##emperically determined "good enough" max basis size

    assert n >= nEigs
    assert maxBasisSize >= 2*nEigs
    assert n > maxBasisSize

    #Get important portions of the preconditioner before fucking up A
#    tavg, vavg = 0., 0.
#    for i in range(0, n):
#        tavg += A[i][i].real
#        for j in range(i+1, n):
#            vavg += A[i][j] + conj(A[j][i])
#    tavg = tavg / n
#    vavg = (  vavg / (n*n - n)  ).real
#    Dpre = array([0. for i in range(0, n)], dtype=float)
#    for i in range(0, n):
#        Dpre[i] = tavg*tavg / (  (1/2*A[i][i].real*A[i][i].real + vavg - eref)**2 + tavg*tavg  )

    ##main diagonals of A, these are used often enough to warrent their own storage
    #D0 = array([0. for i in range(0, n)], dtype=float)
    #for i in range(0, n):
    #    D0[i] = A[i][i].real


    #P = array([0. for i in range(0, n)], dtype=complex) ##yes, this is really complex, not real
    #A_ = A - eref * eye(n)
    #A2 = dot(A_, A_)
    #if(fs):
    #    for i in range(0, n):
    #        P[i] = 0.0 + 0.0j
    #        for j in range(0, n):
    #            P[i] += A2[j][i] / A_[j][i]
    #        P[i] = P[i] / n
    #A = deepcopy(A2)


    #The initial orthonormal vector set - unit vectors are decent choices w/o any prior info since
    #our hamiltonian is (expected to be) diagonally dominant ...
    V = array([[0. for i in range(0, n)] for j in range(0, maxBasisSize)], dtype=complex) ##(m.b.s. x n)
    for i in range(0, maxBasisSize):
        V[i][i] = 1.0 + 0.0j
    if(V0exists): ## ... but a better choice would be eigenvectors of a k-point w/ similar momentum
        ##Fill up as much of V as we reasonably can
        ##Strictly speaking, we should guarentee that V is now orthogonal.  Emperically, as long as
        ##rows(V0) ~ nEigs and cols(V0) ~ n, even if we append 0s or strip away portions of V0 to fit in V,
        ##everything still works out - the re-orthogonalization would take more time than would be gained
        ##by fixing the errors in V's orthogonality (assuming that V0 is orthogonal to begin with!)
        for i in range(0, min(maxBasisSize, V0.shape[0])):
            for j in range(0, min(n, V0.shape[1])):
                V[i][j] = V0[i][j]

        #V = MGS(V=V, nVecs=maxBasisSize, vecDim=n)

    ##main diagonals of A, these are used often enough to warrent their own storage
    D = array([0. for i in range(0, n)], dtype=float)
    for i in range(0, n):
        D[i] = A[i][i].real


    ##rows of ritz vectors (nEigs x n)
    X = array([[0. for i in range(0, n)] for j in range(0, nEigs)], dtype=complex)
    ##rows of new search directions (nEigs x n)
    T = array([[0. for i in range(0, n)] for j in range(0, nEigs)], dtype=complex)

    #W holds A dot V^T, size (n x maxBasisSize)
    W = array([[0. for i in range(0, maxBasisSize)] for j in range(0, n)], dtype=complex)
    #H is the hermitian matrix V^+ dot A dot V, size maxBasisSize x maxBasisSize
    H = array([[0. for i in range(0, maxBasisSize)] for j in range(0, maxBasisSize)], dtype=complex)

    currE, lastE = 2.*eTol, 0.
    currBasisSize = nEigs
    counter = 0
    while(True):
        #W = A @ V^T
        for i in range(0, n):
            for j in range(0, currBasisSize):
                W[i][j] = 0.0 + 0.0j
                for k in range(0, n):
                    W[i][j] += A[i][k] * V[j][k]
        #H = V* @ W, w/ H hermitian (hence this being a bit more complex than the normal dot prod formula)
        for i in range(0, currBasisSize):
            H[i][i] = 0.0 + 0.0j
            for k in range(0, n):
                H[i][i] += conj(V[i][k]) * W[k][i]
            for j in range(0, currBasisSize):
                if(j <= i):
                    continue
                H[i][j] = 0.0 + 0.0j
                for k in range(0, n):
                    H[i][j] += conj(V[i][k]) * W[k][j]
                H[j][i] = conj(H[i][j])

        #Eigenvalues of smaller projected subspace (again, H is hermitian!)
        va, ve = eig(H[0:currBasisSize,0:currBasisSize])
        sind = va.argsort()[:nEigs]
        va = va[sind]
        ve = ve[:,sind]
        ve = transpose(ve) ##this will be the case for my C implementation of eigh
        #va, ve = eigh(H[0:currBasisSize,0:currBasisSize])
        #sind = va.argsort()[:nEigs] ##already sorted from eigh
        #va = va[:nEigs]
        #ve = ve[:,:nEigs]
        #ve = transpose(ve) ##this will be the case for my C implementation of eigh


        #Get ritz vectors
        #These approximate the eigenvectors of A and are in order with the eigenvalues
        for i in range(0, nEigs):
            for j in range(0, n):
                X[i][j] = 0.0 + 0.0j
                for k in range(0, currBasisSize):
                    X[i][j] += ve[i][k] * V[k][j]

        #Converg check
        ##the eigenvalue furthest from the bottom is the least well converged so only check that one.
        ##normally, you'd check both values/vectors, but there is some proof (I forget the name...) that the
        ##wavefunctions are an order of magnitude less important than their corresponding observables
        currE = va[nEigs-1]
        if(counter):
            dev = abs(currE - lastE)
            print(counter, dev)
            if(dev < eTol):
                return va, X
        else:
            print(counter, "---")
        lastE = currE

        #Set new search directions with Wang and Zunger's modified preconditioner
        #In the absense of any other use for the residuals, I'm putting their computation mixed into
        #the preconditioner: the residuals are expensive, we already dynamically allocate a ton of memory
        #in this algo, and doing things one piece at a time is for cowards (and people who want to be able
        #to read their code later) - I am neither of these
        #If the residual matrix R holds residuals as the rows, then R is:
        #R = va*X - W@ve
        #And the (ith, jth) component of R is
        #R(i,j) = va(i)X(i,j)  -  [ W @ ve(i) ](ij)    for 0 <= i < nEigs, 0 <= j < n
        #Which is necessary for finding the search directions with rows of T holding the search vectors:
        #T(i, j) = R(i, j) / (va(i) - D(j)) i.e. solve (va*I - K)t = r for t, with K ~ A (but K != A)
        for i in range(0, nEigs):
            for j in range(0, n):
                T[i][j] = va[i]*X[i][j]
                for k in range(0, currBasisSize): ##the ith, jth component of W @ ve
                    T[i][j] -= ve[i][k] * W[j][k]
                T[i][j] /= (va[i] - D[j])
                #T[i][j] *= tavg*tavg / (  (1/2*(D[j])**2 + vavg - eref)**2 + tavg*tavg  )
                #T[i][j] /= (va[i] - Dpre[j])
                #T[i][j] /= (va[i] - P[j])

        #Increase the basis size or restart
        if(currBasisSize + nEigs < maxBasisSize): ##increase of basis
            for i in range(0, nEigs):
                for j in range(0, n):
                    V[currBasisSize+i][j] = T[i][j]
            V[0:currBasisSize+nEigs,0:n] = MGS(V=V, nVecs=currBasisSize+nEigs, vecDim=n)
        else: ##restart procedure - basis size set to 2*nEigs
            currBasisSize = nEigs ##end of dav step adds to this again to bring back to 2*nEigs
            for i in range(0, nEigs):
                for j in range(0, n):
                    V[i][j] = X[i][j]
                    V[i+nEigs][j] = T[i][j]
            V[0:2*nEigs,0:n] = MGS(V=V, nVecs=2*nEigs, vecDim=n)

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
seed(69101)

A = SysMake(size=SIZE, min=MIN, max=MAX, mino=MINO, maxo=MAXO)
print(IsHerm(A))
A = array(A)
A2 = dot(A - 150*eye(SIZE), A - 150*eye(SIZE))
print(IsHerm(A2))

from scipy.sparse.linalg import eigs
sstart = time.time()
sva, svi = eigs(A=A2, k=N_EIGS)
send = time.time()

estart = time.time()
va, ve = eig(A2)
eend = time.time()
sind = va.argsort()[:N_EIGS]
va = va[sind]
ve = ve[:,sind]

dstart = time.time()
eigs, vecs = David(SIZE, A2, N_EIGS)#, fs=True, eref=150)
dend = time.time()
vecs = transpose(vecs) ##to test w/ numpy formatting

#Check that eigenvalues are the same, to 3 decimal places at least
print("EIGENVALUES:", allclose(eigs, va, 1e-3, 1e-3))
#Check that the eigenvectors obey the eigenvalue equations ...
okay = True
for i in range(0, N_EIGS):
    okay = allclose(A2@vecs[:,i] - eigs[i]*vecs[:,i], 0.0, 1e-6, 1e-6)
    if(not okay):
        print(i, max(A2@vecs[:,i] - eigs[i]*vecs[:,i]))

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
      " ... scipy:", round(send-sstart, 2))


#Example of a warm-start: re-using the vectors from the previous run (and, presumably, the vectors
#from a run with a very similar hamiltonian to the previous run) speeds up convergence
##randomize A a little bit
ADDSIZE = SIZE + 6
B = SysMake(size=ADDSIZE, min=MIN, max=MAX, mino=MINO, maxo=MAXO)
for i in range(0, SIZE):
    B[i][i] = A[i][i] + (-1.0 + 2.0*random()) ##between -1 and +1
    for j in range(i+1, SIZE):
        B[i][j] = A[i][j] + ((-1.0 + 2.0*random()) + (-1.0 + 2.0*random())*1j)*0.01
        B[j][i] = conj(B[i][j])
vecs = transpose(vecs) ##bring back to row order - prev transposed to comply with numpy's wishes
dstart = time.time()
eigs, vecs = David(ADDSIZE, B, N_EIGS, V0exists=True, V0=vecs)#, fs=True, eref=150)
dend = time.time()
print("TIME: david2:", round(dend-dstart, 2))
