from numpy import array, empty, zeros, reshape, cos, sin, arctan, sqrt, conj
from numpy import transpose, eye
from numpy.linalg import eig
from random import seed, random
from copy import deepcopy

IMAG = 1
"""
todo do the block form (new file) then figure out how to lock ...
... locking eigenvalues is easy ... just keep an array of indices [0 0 0 ... neigs] w/ 0 being unlocked and 1 
being locked, then at each j iteration (faster than i) skip the locked indices.
but the eigenvectors need to be orthogonalized to the locked ones, and we dont want to change the locked
eigenvectors.  the locked ones will be mutually orthogonal, though, so we SHOULD BE ABLE to orthogonalize 
unconverged vectors against them w/o changing the locked ones.  
lok into limiting cases of ord gram schmidt to see if i can figure this out.  or maybe the plane wave src code
"""
"""
https://netlib.org/utk/people/JackDongarra/etemplates/node418.html#fig:prec_cgmBA
https://netlib.org/utk/people/JackDongarra/etemplates/node419.html
"""


#ordinary grahm schmidt
import numpy
def ogs(neig, vdim, a, b, c, locked):
    #print(locked)
    V = empty(shape=(3*neig, vdim), dtype=complex)
    lockedext = empty(shape=3*neig)
    for i in range(0, neig):
        V[0*neig+i,:] = a[i,:]
        V[1*neig+i,:] = b[i,:]
        V[2*neig+i,:] = c[i,:]
        lockedext[i+0*neig] = locked[i]
        lockedext[i+1*neig] = locked[i]
        lockedext[i+2*neig] = locked[i]

    B = numpy.zeros_like(V)

    locks = empty(shape=(3*neig, vdim), dtype=complex)
    c = 0
    for i in range(0, neig):
        if(locked[i]):
            locks[c+0,:] = deepcopy(V[i+0*neig,:])
            locks[c+1,:] = deepcopy(V[i+1*neig,:])
            locks[c+2,:] = deepcopy(V[i+2*neig,:])
            c += 3
    for j in range(0, 3*neig):
        if(lockedext[j]):
            B[j,:] = deepcopy(V[j,:])
            continue
        temp = V[j]
        for k in range(0, c):
            temp = temp - conj(transpose(locks[k])) @ V[j] * locks[k]
        for k in range(0, j):
            if(lockedext[k]):
                continue
            temp = temp - conj(transpose(B[k])) @ V[j] * B[k]
        B[j] = temp / (sqrt(conj(transpose(temp)) @ temp))

    V = deepcopy(B)

    return V[0*neig:1*neig,:], V[1*neig:2*neig,:], V[2*neig:3*neig,:]





def matmake(n):
    R = zeros(shape=(n, n), dtype=complex)

    for i in range(0, n):
        R[i][i] = 1.*random() + 0.0j
        for j in range(i+1, n):
            R[i][j] = 1*((random() - 0.5) + IMAG*1j*(random() - 0.5))
            R[j][i] = R[i][j].real - 1j*R[i][j].imag

    return R


def ritz(neig, vdim, A, a, b, c, nlocks, locks, eigs, vecs):
    V = empty(shape=(3*(neig-nlocks), vdim), dtype=complex)

    co = 0
    for i in range(0, neig):
        if(not locks[i]):
            V[co+0,:] = a[i,:]
            V[co+1,:] = b[i,:]
            V[co+2,:] = c[i,:]
            co += 3

    H = conj(V)@A@transpose(V)
    e_, u = eig(H)
    sind = e_.argsort()[:neig-nlocks]
    e_ = e_[sind]
    u = u[:,sind]
    v_ = transpose(V)@u
    v_ = transpose(v_)

    co = 0
    for i in range(0, neig):
        if(not locks[i]):
            eigs[i] = e_[co].real
            vecs[i,:] = v_[co,:]
            co += 1

    return eigs, vecs

#updates convergence and locks
def updateconv(neigs, nlocked, locks, curre, olde, diffe, tol):
    for j in range(0, neigs):
        if(not locks[j]):
            diffe[j] = abs((curre[j] - olde[j]))
            if(diffe[j] < tol):
                nlocked += 1
                locks[j] = 1

    for j in range(0, neigs):
        olde[j] = curre[j]

    return nlocked, locks, curre, olde, diffe

#computes the action H|psi> by applying the complex conjugate of H leftwards to the row vector |psi>
#(this is so psi can be stored as a row vector - letting H act to the right would require a column
#representation)
#normally, we'd do this with H = T + V ... (T + V)|psi> = T|psi> + ifft(fft(V)fft(|psi>))
def action(psi, H):
    return psi@conj(H)


numpy.seterr(all='raise')
def locg1(npw, A, neigs, tol):

    # --- decs ---
    #subspace basis vectors (neigs rows of dim npw)
    Wi = empty(shape=(neigs, npw), dtype=complex) ##preconditioned search directions
    Xm = empty(shape=(neigs, npw), dtype=complex) ##trial wavefunctions, i (M)inus 1th iteration
    Xi = empty(shape=(neigs, npw), dtype=complex) ##trial wavefunctions, ith iteration

    #action of wavefunctions on hamiltonian
    WiH = empty(shape=(neigs, npw), dtype=complex)
    XmH = empty(shape=(neigs, npw), dtype=complex)
    XiH = empty(shape=(neigs, npw), dtype=complex)

    #diagonal preconditioner
    T = empty(shape=npw, dtype=float)

    #ritz eigenpair stuff
    R = empty(shape=(3*neigs, 3*neigs), dtype=complex) ##
    er = empty(shape=3*neigs, dtype=float)
    vr = empty(shape=(3*neigs, npw), dtype=complex)

    #converged eigenpairs
    e = empty(shape=neigs, dtype=float)
    v = empty(shape=(neigs, npw), dtype=complex)

    #convergence stuff
    nlocks = 0
    locked = empty(shape=neigs, dtype=int)   ## \
    eolds = empty(shape=neigs, dtype=float)  ##  |>  all inline w/ one another
    ediffs = empty(shape=neigs, dtype=float) ## /


    # --- init ---
    #random trial wavefunctions (in the absense of any initial guesses)
    for i in range(0, neigs):
        for j in range(0, npw):
            Xm[i][j] = random() + 1j*random()
            Xi[i][j] = random() + 1j*random()
    for j in range(0, neigs):
        Xi[j,:] /= sqrt(conj(transpose(Xi[j,:]))@Xi[j,:])

    #preconditioner setup
    for i in range(0, npw):
        T[i] = 1 + 0.1*random() ##garbage preconditioner, just using for consistency checks

    #convergence stuff setup
    nlocks = 0
    locked.fill(0)
    ediffs.fill(0.0)


    #############
    # Main Loop #
    #############
    for count in range(0, 1000):

        for i in range(0, neigs):
            if(not locked[i]):
                XiH[i,:] = action(psi=Xi[i,:], H=A)
                lam = 0 ## = <psi|H|psi>
                for j in range(0, npw):
                    lam += XiH[i][j] * conj(Xi[i][j])
                for j in range(0, npw):
                    Wi[i][j] = T[j] * (Xi[i][j] - XiH[i][j]/lam)

        Wi, Xi, Xm = ogs(neig=neigs, vdim=npw, a=Wi, b=Xi, c=Xm, locked=locked)
        e, v = ritz(neig=neigs, vdim=npw, A=A, a=Wi, b=Xi, c=Xm, nlocks=nlocks, locks=locked,
                    eigs=e, vecs=v)

        nlocks, locked, e, eolds, ediffs = updateconv(neigs=neigs, nlocked=nlocks, locks=locked,
                                                      curre=e, olde=eolds, diffe=ediffs,
                                                      tol=tol)
        print(count,"\t", [f'{d:.1e}' for d in ediffs])
        if(nlocks >= neigs):
            break


        ##X(i) -> X(i+1)
        for i in range(0, neigs):
            if(not locked[i]):
                Xm[i,:] = deepcopy(Xi[i,:])
                Xi[i,:] = deepcopy(v[i,:])




    print("finished in", count, "steps")
    return e, v


seed(267)
neig = 5
size = 200
A = matmake(size)
#print(A)
print("begin")

ecg, vcg = locg1(npw=size, A=A, neigs=neig, tol=1e-2)
print("locg1 ...")
ecg = sorted(e.real for e in ecg)
print(ecg)

print("actual ...")
ereal, vreal = eig(A)
ereal = sorted(e.real for e in ereal)[:neig]
print(ereal)

print("diffs ...")
diffs = [(ecg[i] - ereal[i]) / ereal[i] for i in range(0, neig)]
print([f'{abs(d)*100:.2f}%' for d in diffs])
