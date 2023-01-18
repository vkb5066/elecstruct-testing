






import headerrunparams as hrps
import headerfft as hfft
from numpy import zeros, empty, array, conj, transpose, sqrt, dot
from scipy.fft import fftn, ifftn
from numpy.linalg import eig, norm


#If an mxn matrix is stored as a single array, this accesses it's ith, jth element
#only need access to the number of columns, rows don't matter
def acc(n, i, j):
    return n*i + j

def _acc3(i, j, k, lx, ly, lz):
    #return i*(ly*lz) + j*(lz) + k
    return (i*ly + j)*lz + k


#Computes action <psi| H
def action(vg, psig, psiv, ##V(r) on grid (1d arr), Psi(G)'s coeffs on grid (1d arr), Psi(G)'s coefs as vector
           gridsizes, npw, mills, ##[lenx, leny, lenz], len(mills) = len(psiv), mills
           res, ##the action <psi| H, in-line with psiv (but not overwriting it! we need psiv later)
           diags, buff):
    pows = [hfft.NLO2(s) for s in gridsizes]

    psig.fill(0.0 + 0.0j)  ##memset -> 0
    ##Get coeffs of psi from V onto the real space grid
    psig = hfft.GetPsig(npw=npw, mills=mills, coeffs=psiv, sizes=gridsizes, psigrid=psig)
    psig = hfft._fftconv3basedif(arr=psig, buff=buff, pows=pows, sizes=gridsizes) *\
                                 1# gridsizes[0] * gridsizes[1] * gridsizes[2] ##(don't do bit rev!)
    ##Compute the action in real space
    ##TODO: V(r) should (i think) be pure real ... look into this
    for i in range(0, gridsizes[0]*gridsizes[1]*gridsizes[2]):
        psig[i] = psig[i] * vg[i]
    ##Transform the grid back into W - we need V later, so don't overwrite
    psig = hfft._fftconv3basedit(arr=psig, buff=buff, pows=pows, sizes=gridsizes) / \
                                 (gridsizes[0]*gridsizes[1]*gridsizes[2]) ##(don't do bit rev!)
    res = hfft.GetPsiv(npw=npw, mills=mills, coeffs=res, psigrid=psig, sizes=gridsizes)
    # Finish up with the kinetic energy part
    for j in range(0, npw):
        res[j] += psiv[j]*diags[j]

    return res


#In the C code, initialize space for both Q and V in FFTDav
#Have V be the input, and Q the output (Q is the buffer on input)
#Each call to MGS edits and updates Q, then switches V to point to Q, and Q to V (this way, the main
#code can always work with V instead of switching between the two or memsetting to 0 each time
def MGS(V, Q, nVecs, vecDim):
    ##Make an nVecs (rows) x vecDim (cols) matrix
    #so the ith row of the return vector will be the ith orthogonal vector of V
    #Q = empty(shape=(nVecs, vecDim), dtype=complex)

    for i in range(0, nVecs):
        #Normalize V into Q
        norm_ = 0. ##strictly real
        for j in range(0, vecDim):
            ij = acc(vecDim, i, j)
            norm_ += conj(V[ij]) * V[ij]
        norm_ = 1./sqrt(norm_)
        for j in range(0, vecDim):
            ij = acc(vecDim, i, j)
            Q[ij] = V[ij]*norm_

        #Orthogonalize all j > i to i in 0, ... j
        for j in range(i + 1, nVecs):
            qipvj = 0.0 + 0.0j
            for k in range(0, vecDim):
                qipvj += conj(Q[acc(vecDim, i, k)]) * V[acc(vecDim, j, k)]
            for k in range(0, vecDim):
                V[acc(vecDim, j, k)] -= qipvj*Q[acc(vecDim, i, k)]

    return Q





from numpy import reshape
#Note for comments in this function:
# A^T = transpose of A
# A* = complex conjugate of A
# A^+ = hermitian conjugate of A ( = (A*)^T = (A^T)* )
# A @ B = matrix product between A and B

#va, X should be sent into the function as allocated (not initialized) memory ... a single real arr
#and a 2d (pointer->pointer) complex array
#This function does not create any memory that it does not free
#If v0m, v0n = 0, then random initialization for the entire set V is done (send V0 as NULL).
#otherwise, V0 is assumed to be a v0m x v0n array (pointer-to-pointer to make resending old eigenvecs easier)
def DavidFFT(n, mills, gs, kpt, vrgrid, gridsizes, nEigs, maxBasisSize=None, v0m=0, v0n=0,
             V0=None, eTol=1e-3):
    if(maxBasisSize == None):
        maxBasisSize = min(int(6.25*nEigs), n-1) ##emperically determined "good enough" max basis size
                                                 ##prob. will not be the same for c code since this isn't
                                                 ##vectorized
    assert n >= nEigs
    assert maxBasisSize >= 2*nEigs
    assert n > maxBasisSize

    #The initial orthonormal vector set - unit vectors are decent choices w/o any prior info since
    #our hamiltonian is (expected to be) diagonally dominant ...
    # ... at least, you'd think so.  This actually sucks vs random initialization
    #V = array([[0. for i in range(0, n)] for j in range(0, maxBasisSize)], dtype=complex) ##(m.b.s. x n)
    #for i in range(0, maxBasisSize):
    #    V[i][i] = 1.0 + 0.0j

    #Set V
    V = empty(shape=(maxBasisSize*n), dtype=complex)  ##(m.b.s. x n)
    ##first, fill up as much as we can with initial guesses (prob. eigenvecs of a k-point w/ similar p)
    if(v0m and v0n):
        for i in range(0, min(maxBasisSize, v0m)):
            for j in range(0, min(n, v0n)):
                V[acc(n, i, j)] = V0[i][j]

    ##fill in the rest of V (if necessary) randomly, re-normalize it
    from random import random, seed
    seed(69420) ##DON'T RE-SEED IN C CODE
    for i in range(min(maxBasisSize, v0m), maxBasisSize):
        norm_ = 0. ##strictly real
        for j in range(min(n, v0n), n):
            ij = acc(n, i, j)
            V[ij] = random() + random()*1j
            norm_ += conj(V[ij])*V[ij]
        ##Normalize
        norm_ = 1./sqrt(norm_)
        for j in range(0, n):
            V[acc(n, i, j)] *= norm_

    ##strictly speaking, we should guarentee that V is now orthogonal.
    ##realistically, any reasonable initial guess will satisfy this requirement.  Additionally, the random
    ##initialization is _very_ likley to satisfy this as well ... we'd _probably_ lose more than we gained
    ##by calling MGS here
    #V = MGS(V=V, Q=Q, nVecs=maxBasisSize, vecDim=n)


    #Buffer for MGS(V) size mbs x n
    Q = empty(shape=(maxBasisSize*n), dtype=complex)

    ##Main diagonals of the hamiltonian, these are used often enough to warrent their own storage
    D = empty(shape=(n), dtype=float)
    for i in range(0, n):
        D[i] = hrps.LocKin(k=kpt, Gi=gs[i], Gj=gs[i])


    ##columns of ritz vectors (stored as rows, pointer-to-pointer since we'll be returning these as eigvecs)
    X = empty(shape=(nEigs, n), dtype=complex)
    ##columns of new search directions
    T = empty(shape=(nEigs*n), dtype=complex)

    #W holds V dot A*, size maxBasisSize x n
    W = empty(shape=(maxBasisSize*n), dtype=complex)
    #H is the hermitian matrix V^+ dot A dot V, size maxBasisSize x maxBasisSize
    H = empty(shape=(maxBasisSize*maxBasisSize), dtype=complex)

    #Holds the wavefunction fft grid, would really need one of these for each thread that is working
    #on an individual band
    psigridmain = empty(shape=(gridsizes[0]*gridsizes[1]*gridsizes[2]), dtype=complex)

    #Buffer for fft that I've initialized like 20 times already lol
    buff = empty(shape=max([gridsizes[0], gridsizes[1], gridsizes[2]]), dtype=complex)

    #!!!!!
    #A = hrps.BuildHamil(k=hrps.KPT, npw=n, gs=gs, mils=mills)
    #!!!!!


    currE, lastE = 2.*eTol, 0.
    currBasisSize = nEigs
    counter = 0
    from copy import deepcopy
    from numpy import allclose, round
    while(True):
        #W = V @ A*
        #A is the hamiltonian, so V @ A = V(G)[i][i] * A[i][i]  +  V(r) * FFT(A*)
        #(note that the conjugate of A's diagonals isn't important ... they're pure real)
        #(also note that you'd have to build V(G) as V*(G) before transforming it to V(r) ... or if thats
        #too confusing, have two grids - one for V(G) -> V(r) and one for V*(G) -> V(r)
        for i in range(0, currBasisSize): ##loop over bands
            W[i*n:(i+1)*n] = action(vg=vrgrid, psig=psigridmain, psiv=V[i*n:(i+1)*n],
                                    gridsizes=gridsizes, npw=n, mills=mills,
                                    res=W[i*n:(i+1)*n],
                                    diags=D, buff=buff)


        #H = V* @ W^T, w/ H hermitian (hence this being a bit more complex than the normal dot prod formula)
        #TODO: can this be accomplished more efficiently with FFTs?  I don't know of a formula for fft(V*)
        #given FFT(V), so afaik, we'd have to do another whole ass FFT on V* ...
        #H = conj(V) @ A @ transpose(V)
        for i in range(0, currBasisSize):
            ii = acc(currBasisSize, i, i)
            H[ii] = 0.0 + 0.0j
            for j in range(0, n):
                ij = acc(n, i, j)
                H[ii] += conj(V[ij]) * W[ij]
            for j in range(i+1, currBasisSize):
                ij = acc(currBasisSize, i, j)
                H[ij] = 0.0 + 0.0j
                for k in range(0, n):
                    H[ij] += conj(V[acc(n, i, k)]) * W[acc(n,j,k)]
                H[acc(currBasisSize, j, i)] = conj(H[ij])



        #Eigenvalues of smaller projected subspace (again, H is hermitian!)
        #Note: H seems to be mostly diagonal? In this case, the jacobi-davidson routines might be better
        #than QR
        #Note2: H is only mostly diagonal when the algo is close to converging.  Maybe start with
        #QR then transition to JD
        #Note3: The JD algo is very easily/efficiently run in parallel compared to QR
        #print(round(H[0:currBasisSize,0:currBasisSize], 0))
        H2 = reshape(a=H[0:currBasisSize*currBasisSize],
                     newshape=(-1, currBasisSize)) ##only to make this work with eig()
        va, ve = eig(H2)
        sind = va.argsort()[:nEigs]
        va = va[sind]
        ve = ve[:,sind]
        ve = transpose(ve) ##this will be the case for my C implementation of eigh


        #Get ritz vectors
        #These approximate the eigenvectors of A and are in order with the eigenvalues
        for i in range(0, nEigs):
            for j in range(0, n):
                X[i][j] = 0.0 + 0.0j
                for k in range(0, currBasisSize):
                    X[i][j] += ve[i][k] * V[acc(n, k, j)]


        #Converg check
        ##the eigenvalue furthest from the bottom is the least well converged so only check that one.
        ##normally, you'd check both values/vectors, but there is some proof (I forget the name...) that the
        ##wavefunctions are an order of magnitude less important than their corresponding observables
        currE = sum(va)/len(va)
        if(counter):
            dev = abs(currE - lastE)
            print(counter, dev)
            if(dev < eTol):
                return va, X
                #return va, conj(X) ##do this if V(G)-> is sent as V*(G) -> V(r)
        else:
            print(counter, "---")
        lastE = currE

        #Set new search directions with Davidson's original preconditioner
        #In the absense of any other use for the residuals, I'm putting their computation mixed into
        #the preconditioner: the residuals are expensive, we already dynamically allocate a ton of memory
        #in this algo, and doing things one piece at a time is for cowards (and people who want to be able
        #to read their code later) - I am neither of these
        #If the residual matrix R holds residuals as the rows, then R is:
        #R = va*X - W^T@ve
        #And the (ith, jth) component of R is
        #R(i,j) = va(i)X(i,j)  -  [ W^T @ ve(i) ](ij)    for 0 <= i < nEigs, 0 <= j < n
        #Which is necessary for finding the search directions T(i,j) = R(i,j) / (va(i)-D(j))
        for i in range(0, nEigs):
            for j in range(0, n):
                ij = acc(n=n, i=i, j=j)
                T[ij] = va[i]*X[i][j]
                for k in range(0, currBasisSize): ##the ith, jth component of W^T @ ve
                    T[ij] -= ve[i][k] * W[acc(n,k,j)]
                T[ij] /= (va[i] - D[j])


        #Increase the basis size or restart
        if(currBasisSize + nEigs < maxBasisSize): ##increase of basis
            for i in range(0, nEigs):
                for j in range(0, n):
                    V[acc(n, currBasisSize+i, j)] = T[acc(n, i, j)]
            V[0:(currBasisSize+nEigs)*n] = MGS(V=V[0:(currBasisSize+nEigs)*n],
                                               Q=Q[0:(currBasisSize+nEigs)*n],
                                               nVecs=currBasisSize+nEigs, vecDim=n)
        else: ##restart procedure - basis size set to 2*nEigs
            currBasisSize = nEigs ##end of dav step adds to this again to bring back to 2*nEigs
            for i in range(0, nEigs):
                for j in range(0, n):
                    ij = acc(n, i, j)
                    V[ij] = X[i][j]
                    V[acc(n, i+nEigs, j)] = T[ij]
            V[0:(2*nEigs)*n] = MGS(V=V[0:(2*nEigs)*n],
                                   Q=Q[0:(2*nEigs)*n],
                                   nVecs=2*nEigs, vecDim=n)

        currBasisSize += nEigs
        counter += 1




"""
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
"""
