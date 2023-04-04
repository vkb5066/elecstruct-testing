from numpy import array, empty, zeros, reshape, cos, sin, arctan, sqrt, conj
from numpy import transpose, eye
from numpy.linalg import eig
from scipy.linalg import eigh
from random import seed, random
from copy import deepcopy

IMAG = 1
"""
https://netlib.org/utk/people/JackDongarra/etemplates/node418.html#fig:prec_cgmBA
https://netlib.org/utk/people/JackDongarra/etemplates/node419.html
"""


def matmake(n):
    R = zeros(shape=(n, n), dtype=complex)

    for i in range(0, n):
        R[i][i] = 1.*random() + 0.0j
        for j in range(i+1, n):
            R[i][j] = 1*((random() - 0.5) + IMAG*1j*(random() - 0.5))
            R[j][i] = R[i][j].real - 1j*R[i][j].imag

    return R

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


def ritz(neig, vdim, A, a, b, c, nlocks, locks, eigs, vecs):
    V = empty(shape=(3*(neig-nlocks), vdim), dtype=complex)

    co = 0
    for i in range(0, neig):
        if(not locks[i]):
            V[co+0,:] = a[i,:]
            V[co+1,:] = b[i,:]
            V[co+2,:] = c[i,:]
            co += 3
    """
    co = 0
    for i in range(0, neig):
        if(not locks[i]):
            V[co+0,:] = a[i,:]
            co += 1
    for i in range(0, neig):
        if(not locks[i]):
            V[co+0,:] = b[i,:]
            co += 1
    for i in range(0, neig):
        if(not locks[i]):
            V[co+0,:] = c[i,:]
            co += 1
    """

    e_, u = None, None
    H = (V @ conj(A)) @ conj(transpose(V))
    #TM = (V @ conj(A))
    #for i in range(0, 3*(neig-nlocks)):
    #    TM[i,:] = action(psi=V[i,:], H=A)
    #H = TM @ conj(transpose(V))
    S = conj(V)@transpose(V)

    e_, u = eigh(conj(H), S)
    sind = e_.argsort()[:neig - nlocks]
    e_ = e_[sind]
    u = u[:, sind]
    v_ = transpose(V) @ u
    v_ = transpose(v_)


    co = 0
    for i in range(0, neig):
        if(not locks[i]):
            eigs[i] = e_[co].real
            vecs[i,:] = v_[co,:]
            co += 1

    return eigs, vecs

def ritz1p5(neig, vdim, A, a, b, c, nlocks, locks, eigs, vecs):
    V = empty(shape=(3*neig, vdim), dtype=complex)

    co = 0
    for i in range(0, neig):
        V[co+0,:] = a[i,:]
        V[co+1,:] = b[i,:]
        V[co+2,:] = c[i,:]
        co += 3


    e_, u = None, None
    H = (V @ conj(A)) @ conj(transpose(V))
    #TM = (V @ conj(A))
    #for i in range(0, 3*(neig-nlocks)):
    #    TM[i,:] = action(psi=V[i,:], H=A)
    #H = TM @ conj(transpose(V))
    S = conj(V)@transpose(V)

    e_, u = eigh(conj(H), S)
    sind = e_.argsort()[:neig]
    e_ = e_[sind]
    u = u[:, sind]
    v_ = transpose(V) @ u
    v_ = transpose(v_)


    co = 0
    for i in range(0, neig):
        eigs[i] = e_[co].real
        vecs[i,:] = v_[co,:]
        co += 1

    return eigs, vecs


__GLOBAL_COUNTER = 0
#updates convergence and locks
def updateconv(neigs, nlocked, locks, curre, olde, diffe, stab, tol):
    global __GLOBAL_COUNTER

    for j in range(0, neigs):
        if(not locks[j]):
            diffe[j] = abs(curre[j] - olde[j])
            if(diffe[j] < tol):
                nlocked += 1
                locks[j] = 1

    for j in range(0, neigs):
        olde[j] = curre[j]

    endflag = (neigs == nlocked)

    if(stab == 2):
        nlocked = 0
        locks.fill(0)

    print(__GLOBAL_COUNTER, [f'{d:.1e}' for d in diffe])
    __GLOBAL_COUNTER += 1
    return nlocked, locks, curre, olde, diffe, endflag


#computes the action H|psi> by applying the complex conjugate of H leftwards to the row vector |psi>
#(this is so psi can be stored as a row vector - letting H act to the right would require a column
#representation)
#normally, we'd do this with H = T + V ... (T + V)|psi> = T|psi> + ifft(fft(V)fft(|psi>))
def action(psi, H):
    return psi@conj(H)

numpy.seterr(all='raise')
#stability: 0 = no orthogonalization, includes locking (mostly for good initial guesses, tol >= 1e-3 or so)
#           1 = orthogonalization, includes locking (maybe introduces ~0.05% error near end of spectrum)
#           2 = orthogonalization, does not include locking (works well for any precision up to machine eps)
def locg(npw, A, neigs, stab, tol):

    # --- decs ---
    #subspace basis vectors (neigs rows of dim npw)
    Wi = empty(shape=(neigs, npw), dtype=complex) ##preconditioned search directions
    Xm = empty(shape=(neigs, npw), dtype=complex) ##trial wavefunctions, i (M)inus 1th iteration
    Xi = empty(shape=(neigs, npw), dtype=complex) ##trial wavefunctions, ith iteration

    #action of wavefunctions on hamiltonian
    PH = empty(shape=(3*neigs, npw), dtype=complex)
    WiH = empty(shape=(neigs, npw), dtype=complex)
    XmH = empty(shape=(neigs, npw), dtype=complex)
    XiH = empty(shape=(neigs, npw), dtype=complex)

    #diagonal preconditioner
    T = empty(shape=npw, dtype=float)

    #ritz eigenpair stuff
    HR = empty(shape=(3*neigs, 3*neigs), dtype=complex) ##dense subspace hamiltonian
    SR = empty(shape=(3*neigs, 3*neigs), dtype=complex) ##dense subspace overlap matrix
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
        T[i] = 1 + 1*0.1*random() ##garbage preconditioner, just using for consistency checks

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
                XiH[i, :] = action(psi=Xi[i,:], H=A)
                lam = 0 ## = <psi|H|psi>
                for j in range(0, npw):
                    lam += XiH[i][j] * conj(Xi[i][j])

                for j in range(0, npw):
                    Wi[i][j] = T[j] * (Xi[i][j] - XiH[i][j]/lam)
                #print(action(Wi[i], A)[0])
        if(stab):
            Wi, Xi, Xm = ogs(neig=neigs, vdim=npw, a=Wi, b=Xi, c=Xm, locked=locked)
        e, v = ritz(neig=neigs, vdim=npw, A=A, a=Wi, b=Xi, c=Xm, nlocks=nlocks, locks=locked,
                    eigs=e, vecs=v)
        nlocks, locked, e, eolds, ediffs, endflag = updateconv(neigs=neigs,
                                                               nlocked=nlocks, locks=locked,
                                                               curre=e, olde=eolds, diffe=ediffs,
                                                               stab=stab, tol=tol)
        if(endflag):
            break

        ##X(i) -> X(i+1)
        for i in range(0, neigs):
            if(not locked[i]):
                Xm[i,:] = deepcopy(Xi[i,:])
                Xi[i,:] = deepcopy(v[i,:])




    print("finished in", count, "steps")
    return e, v


"""
--------------------------------------------------------------------------------------------------------------
"""

def ritz2(neig, vdim, V, VH, nlocks, locks, HR, SR, eigs, vecs):
    HR.fill(0)
    SR.fill(0)
    c = 0
    for i in range(0, neig):
        if(not locks[i]):
            d = 0
            for j in range(0, neig):
                if(not locks[j]):
                    HR[c+0][d+0] = VH[i+0*neig,:] @ conj(V[j+0*neig,:])
                    HR[c+1][d+0] = VH[i+1*neig,:] @ conj(V[j+0*neig,:])
                    HR[c+2][d+0] = VH[i+2*neig,:] @ conj(V[j+0*neig,:])
                    HR[c+0][d+1] = VH[i+0*neig,:] @ conj(V[j+1*neig,:])
                    HR[c+1][d+1] = VH[i+1*neig,:] @ conj(V[j+1*neig,:])
                    HR[c+2][d+1] = VH[i+2*neig,:] @ conj(V[j+1*neig,:])
                    HR[c+0][d+2] = VH[i+0*neig,:] @ conj(V[j+2*neig,:])
                    HR[c+1][d+2] = VH[i+1*neig,:] @ conj(V[j+2*neig,:])
                    HR[c+2][d+2] = VH[i+2*neig,:] @ conj(V[j+2*neig,:])

                    SR[c+0][d+0] = conj(V[i+0*neig,:]) @ V[j+0*neig,:]
                    SR[c+1][d+0] = conj(V[i+1*neig,:]) @ V[j+0*neig,:]
                    SR[c+2][d+0] = conj(V[i+2*neig,:]) @ V[j+0*neig,:]
                    SR[c+0][d+1] = conj(V[i+0*neig,:]) @ V[j+1*neig,:]
                    SR[c+1][d+1] = conj(V[i+1*neig,:]) @ V[j+1*neig,:]
                    SR[c+2][d+1] = conj(V[i+2*neig,:]) @ V[j+1*neig,:]
                    SR[c+0][d+2] = conj(V[i+0*neig,:]) @ V[j+2*neig,:]
                    SR[c+1][d+2] = conj(V[i+1*neig,:]) @ V[j+2*neig,:]
                    SR[c+2][d+2] = conj(V[i+2*neig,:]) @ V[j+2*neig,:]
                d += 3
            c += 3
    print(numpy.round(SR))
    #HR = VH @ conj(transpose(V))
    SR = conj(V) @ transpose(V)
    print(numpy.round(SR, 1))

    e_, u = None, None
    e_, u = eigh(conj(HR[0:c,0:c]), SR[0:c,0:c]) ##do we REALLY need eigh instead of eig?  seems like SR is often the identity
    sind = e_.argsort()[:neig - nlocks]
    e_ = e_[sind]
    u = transpose(u[:, sind]) ##this is the output of qrh()

    tm = empty(shape=(3*neig-3*nlocks, vdim), dtype=complex)
    c = 0
    for i in range(0, neig):
        if(not locks[i]):
            tm[c,:] = deepcopy(V[i,:])
            tm[c+1, :] = deepcopy(V[i+neig, :])
            tm[c+2, :] = deepcopy(V[i+2*neig, :])
            c += 3
    #v_ = u @ V ##<- this is NOT the correct V!
    v_ = u @ tm
    #v_ = empty(shape=(neig-nlocks, vdim), dtype=complex)
    #print(v_.shape, u.shape, V.shape)

    co = 0
    for i in range(0, neig):
        if(not locks[i]):
            eigs[i] = e_[co].real
            vecs[i,:] = v_[co,:]
            #vecs[i,:] = transpose(transpose(V)[i,:] @ u[:,co])
            co += 1

    return eigs, vecs


def updateconv2(neigs, nlocked, locks, curre, olde, diffe, stab, tol):
    global __GLOBAL_COUNTER

    for j in range(0, neigs):
        diffe[j] = abs(curre[j] - olde[j])
        if(diffe[j] < tol):
            nlocked += 1
            locks[j] = 1

    for j in range(0, neigs):
        olde[j] = curre[j]

    if(stab == 2):
        nlocked = 0
        locks.fill(0)

    print(__GLOBAL_COUNTER, [f'{d:.1e}' for d in diffe])
    __GLOBAL_COUNTER += 1
    return nlocked, locks, curre, olde, diffe


#shuffles locked vectors s.t. all locks are stored at the end of their arrays and all unconverged
#are stored at the beginning (locks = [1 0 1 1 0] --> [0 0 1 1 1] along with corresponding vectors)
def lockshuffle(neigs, locks, V, VH, curreigs, oldeigs, diffeigs, vecs):
    indbeg, indend = 0, neigs-1
    while(indbeg < indend):
        while(locks[indbeg] == 0 and indbeg < indend):
            indbeg += 1
        while(locks[indend] == 1 and indbeg < indend):
            indend -= 1

        if(indbeg < indend):
            ##shuffle all
            ###locks
            locks[indbeg], locks[indend] = locks[indend], locks[indbeg]

            ###W(i), X(i), X(i-1)
            for i in range(0, 3):
                tmp = deepcopy(V[neigs*i + indbeg,:])
                V[neigs*i + indbeg,:] = deepcopy(V[neigs*i + indend,:])
                V[neigs*i + indend,:] = deepcopy(tmp)

            ###<psi|H^*
            for i in range(0, 3):
                tmp = deepcopy(VH[neigs*i + indbeg,:])
                VH[neigs*i + indbeg,:] = deepcopy(VH[neigs*i + indend,:])
                VH[neigs*i + indend,:] = deepcopy(tmp)

            ###eigvals
            curreigs[indbeg], curreigs[indend] = curreigs[indend], curreigs[indbeg]
            oldeigs[indbeg], oldeigs[indend] = oldeigs[indend], oldeigs[indbeg]
            diffeigs[indbeg], diffeigs[indend] = diffeigs[indend], diffeigs[indbeg]

            ###eigvecs
            tmp = deepcopy(vecs[indbeg,:])
            vecs[indbeg,:] = deepcopy(vecs[indend,:])
            vecs[indend,:] = deepcopy(tmp)

            indbeg += 1
            indend -= 1

    return locks, V, VH, curreigs, oldeigs, diffeigs, vecs


def locg2(npw, A, neigs, stab, tol):

    # --- decs ---
    #subspace basis vectors (neigs rows of dim npw)
    V = empty(shape=(3*neigs, npw), dtype=complex) ##0->1/3 p.s.d.s, 1/3->2/3 psi(i), 2/3->3/3 psi(i-1)

    #action of wavefunctions on hamiltonian
    VH = empty(shape=(3*neigs, npw), dtype=complex) ## <V| H^*

    #diagonal preconditioner
    T = empty(shape=npw, dtype=float)

    #ritz eigenpair stuff
    HR = empty(shape=(3*neigs, 3*neigs), dtype=complex) ##dense subspace hamiltonian
    SR = empty(shape=(3*neigs, 3*neigs), dtype=complex) ##dense subspace overlap matrix
    er = empty(shape=3*neigs, dtype=float)
    vr = empty(shape=(3*neigs, npw), dtype=complex)

    #converged eigenpairs
    e = empty(shape=neigs, dtype=float)
    ve = empty(shape=(neigs, npw), dtype=complex)

    #convergence stuff
    nueigs = neigs ##number of unconverged eigenpairs
    nlocks = 0
    locked = empty(shape=neigs, dtype=int)   ## \
    eolds = empty(shape=neigs, dtype=float)  ##  |>  all inline w/ one another
    ediffs = empty(shape=neigs, dtype=float) ## /


    # --- init ---
    #random trial wavefunctions (in the absense of any initial guesses)
    #this is currently set up in a dumb way to be compatible w/ lobcg1
    for i in range(0, neigs):
        for j in range(0, npw):
            V[2*neigs+i][j] = random() + 1j*random()
            V[1*neigs+i][j] = random() + 1j*random() ##random init for |i-1>, |i>
    for i in range(neigs, 2*neigs):
        V[i,:] /= sqrt(conj(transpose(V[i,:]))@V[i,:]) ##normalize |i>
        VH[neigs+i,:] = action(psi=V[neigs+i,:], H=A) ##maybe get rid of me ....?????.....??????>.....?????...

    #preconditioner setup
    for i in range(0, npw):
        T[i] = 1 + 1*0.1*random() ##garbage preconditioner, just using for consistency checks

    #convergence stuff setup
    nlocks = 0
    locked.fill(0)
    ediffs.fill(0.0)


    #############
    # Main Loop #
    #############
    for count in range(0, 1000):

        for i in range(0, nueigs):
            VH[neigs+i,:] = action(psi=V[neigs+i,:], H=A) ##calc action of |i>
            lam = 0 ## = <psi|H|psi>
            for j in range(0, npw):
                lam += VH[neigs+i][j] * conj(V[neigs+i][j])

            for j in range(0, npw):
                V[i][j] = T[j] * (V[neigs+i][j] - VH[neigs+i][j]/lam) ##set |w>
            VH[i,:] = action(psi=V[i,:], H=A) ##calc action of |w>

        if(stab): ##edit for nueigs...
            V[0*neigs:1*neigs], V[1*neigs:2*neigs], V[2*neigs:3*neigs] = \
                            ogs(neig=neigs, vdim=npw,
                            a=V[0*neigs:1*neigs], b=V[1*neigs:2*neigs], c=V[2*neigs:3*neigs],
                            locked=locked)

        if(1):
            e, ve = ritz1p5(neig=nueigs, vdim=npw, A=A,
                            a=V[0*neigs:1*neigs], b=V[1*neigs:2*neigs], c=V[2*neigs:3*neigs],
                            nlocks=nlocks, locks=locked, eigs=e, vecs=ve)
        else:
            e, ve = ritz2(neig=nueigs, vdim=npw, V=V, VH=VH, nlocks=nlocks, locks=locked, HR=HR, SR=SR,
                         eigs=e, vecs=ve)

        nlocks, locked, e, eolds, ediffs = updateconv2(neigs=nueigs,
                                                       nlocked=nlocks, locks=locked,
                                                       curre=e, olde=eolds, diffe=ediffs,
                                                       stab=stab, tol=tol)
        for i in range(0, neigs):
            if(locked[i]):
                #e[i] = None
                V[i,:].fill(None)
                VH[i,:].fill(None)


        if(nlocks == neigs):
            break
        #print(V[:, 0])
        locked, V, VH, e, eolds, ediffs, ve = lockshuffle(neigs=neigs, locks=locked, V=V, VH=VH,
                                                          curreigs=e, oldeigs=eolds, diffeigs=ediffs,
                                                          vecs=ve)
        nueigs = neigs - nlocks
        #print(V[:,0], "\n")

        if(not stab):
            for i in range(0, nueigs):
                VH[2*neigs+i,:] = deepcopy(VH[1*neigs+i,:])
        ##X(i) -> X(i+1)
        for i in range(0, nueigs):
            V[2*neigs+i,:] = deepcopy(V[neigs+i,:])
            V[neigs+i,:] = deepcopy(ve[i,:])




    print("finished in", count, "steps")
    return e, ve




seed(267)
neig = 5
size = 700
A = 1*matmake(size)
#print(A)
print("begin")

ecg, vcg = locg(npw=size, A=A, neigs=neig, stab=0, tol=1e-4)
print("locg ...")
ecg = sorted(e.real for e in ecg)
print(ecg)

print("actual ...")
ereal, vreal = eig(A)
ereal = sorted(e.real for e in ereal)[:neig]
print(ereal)

print("diffs ...")
diffs = [(ecg[i] - ereal[i]) / ereal[i] for i in range(0, neig)]
print([f'{abs(d)*100:.2f}%' for d in diffs])
diffs = [abs(ecg[i] - ereal[i]) for i in range(0, neig)]
print([f'{d:.2f}' for d in diffs])

exit(3000000)
from scipy.sparse.linalg import lobpcg
X = empty(shape=(size, neig), dtype=complex)
for i in range(0, neig):
    for j in range(0, size):
        X[j][i] = random() + 1j * random()
for j in range(0, neig):
    X[:,j] /= sqrt(conj(transpose(X[:,j]))@X[:,j])
res = lobpcg(A, X, verbosityLevel=0, largest=False)
print(res[0])
