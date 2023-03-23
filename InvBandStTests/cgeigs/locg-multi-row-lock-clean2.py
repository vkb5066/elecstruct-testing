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

def ogs2(neig, vdim, a, b, c, abuf, buf, cbuf, nlocks, locked):
    _ = 0
    """
    my way
    """
    abuf = zeros(shape=(neig, vdim), dtype=complex) ##memset these to 0 in real code
    bbuf = zeros(shape=(neig, vdim), dtype=complex)
    cbuf = zeros(shape=(neig, vdim), dtype=complex)

    #ortho a
    for i in range(0, neig):
        ##skip orthonormalization of locked vectors
        if(locked[i]):
            abuf[i,:] = deepcopy(a[i,:])
            continue
        ##ortho against locked vectors first
        tmp = a[i]
        for j in range(0, neig):
            if(locked[j]):
                tmp = tmp - conj(transpose(a[j])) @ a[i] * a[j]
                tmp = tmp - conj(transpose(b[j])) @ a[i] * b[j]
                tmp = tmp - conj(transpose(c[j])) @ a[i] * c[j]
        ##and now against previous a
        for j in range(0, i):
            if(not locked[j]):
                tmp = tmp - conj(transpose(abuf[j])) @ a[i] * abuf[j]
        abuf[i] = tmp / (sqrt(conj(transpose(tmp)) @ tmp))

    #ortho b
    for i in range(0, neig):
        ##skip orthonormalization of locked vectors
        if(locked[i]):
            bbuf[i,:] = deepcopy(b[i,:])
            continue
        ##ortho against locked vectors first
        tmp = b[i]
        for j in range(0, neig):
            if(locked[j]):
                tmp = tmp - conj(transpose(a[j])) @ b[i] * a[j]
                tmp = tmp - conj(transpose(b[j])) @ b[i] * b[j]
                tmp = tmp - conj(transpose(c[j])) @ b[i] * c[j]
        ##and now against previous a, b
        for j in range(0, i):
            if(not locked[j]):
                tmp = tmp - conj(transpose(abuf[j])) @ b[i] * abuf[j]
                tmp = tmp - conj(transpose(bbuf[j])) @ b[i] * bbuf[j]
        bbuf[i] = tmp / (sqrt(conj(transpose(tmp)) @ tmp))

    #ortho c
    for i in range(0, neig):
        ##skip orthonormalization of locked vectors
        if(locked[i]):
            bbuf[i,:] = deepcopy(b[i,:])
            continue
        ##ortho against locked vectors first
        tmp = b[i]
        for j in range(0, neig):
            if(locked[j]):
                tmp = tmp - conj(transpose(a[j])) @ b[i] * a[j]
                tmp = tmp - conj(transpose(b[j])) @ b[i] * b[j]
                tmp = tmp - conj(transpose(c[j])) @ b[i] * c[j]
        ##and now against previous a, b
        for j in range(0, i):
            if(not locked[j]):
                tmp = tmp - conj(transpose(abuf[j])) @ b[i] * abuf[j]
                tmp = tmp - conj(transpose(bbuf[j])) @ b[i] * bbuf[j]
        bbuf[i] = tmp / (sqrt(conj(transpose(tmp)) @ tmp))



def ritz2(neig, vdim, A, a, b, c, ah, bh, ch, nlocks, locks, eigs, vecs):

    #We can remove this 'V' business after I'm done testing
    V = empty(shape=(3*(neig-nlocks), vdim), dtype=complex)
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

    #Note that this has a pretty awful access pattern, but the alternative is like
    #100+ lines of triple loops infested w/ if/else statements
    #note that H and S are hermitian, but this current routine doesn't take advantage of that !!!!!!!!!!!!!!!!!!!!!!!
    H = empty(shape=(3*(neig-nlocks), 3*(neig-nlocks)), dtype=complex)
    S = empty(shape=(3*(neig-nlocks), 3*(neig-nlocks)), dtype=complex)
    H.fill(None) ##\ for testing only
    S.fill(None) ##/ ditto
    n = neig-nlocks
    ci, cj = 0, 0
    for i in range(0, neig):
        if(not locks[i]):
            cj = 0
            for j in range(0, neig):
                if(not locks[j]):
                    H[ci+0*n][cj+0*n] = ah[i,:] @ conj(a[j,:])
                    H[ci+0*n][cj+1*n] = ah[i,:] @ conj(b[j,:])
                    H[ci+0*n][cj+2*n] = ah[i,:] @ conj(c[j,:])
                    S[ci+0*n][cj+0*n] = conj(a[i,:]) @ a[j,:]
                    S[ci+0*n][cj+1*n] = conj(a[i,:]) @ b[j,:]
                    S[ci+0*n][cj+2*n] = conj(a[i,:]) @ c[j,:]
                    cj += 1
            cj = 0
            for j in range(0, neig):
                if(not locks[j]):
                    H[ci+1*n][cj+0*n] = bh[i,:] @ conj(a[j,:])
                    H[ci+1*n][cj+1*n] = bh[i,:] @ conj(b[j,:])
                    H[ci+1*n][cj+2*n] = bh[i,:] @ conj(c[j,:])
                    S[ci+1*n][cj+0*n] = conj(b[i,:]) @ a[j,:]
                    S[ci+1*n][cj+1*n] = conj(b[i,:]) @ b[j,:]
                    S[ci+1*n][cj+2*n] = conj(b[i,:]) @ c[j,:]
                    cj += 1
            cj = 0
            for j in range(0, neig):
                if(not locks[j]):
                    H[ci+2*n][cj+0*n] = ch[i,:] @ conj(a[j,:])
                    H[ci+2*n][cj+1*n] = ch[i,:] @ conj(b[j,:])
                    H[ci+2*n][cj+2*n] = ch[i,:] @ conj(c[j,:])
                    S[ci+2*n][cj+0*n] = conj(c[i,:]) @ a[j,:]
                    S[ci+2*n][cj+1*n] = conj(c[i,:]) @ b[j,:]
                    S[ci+2*n][cj+2*n] = conj(c[i,:]) @ c[j,:]
                    cj += 1
            ci += 1

    #make 100% sure that the disaster above actually works
    if not (numpy.allclose(H, conj(transpose(H)))):
        exit(0x0B00B135)
    if not (numpy.allclose(S, conj(transpose(S)))):
        exit(0xBEEFBABE)
    if not(numpy.allclose(H, ((V @ conj(A)) @ conj(transpose(V))))):
        exit(80085)
    if not(numpy.allclose(S, conj(V)@transpose(V))):
        exit(58008)

    e_, u = None, None
    #H = (V @ conj(A)) @ conj(transpose(V))
    #S = conj(V)@transpose(V)
    e_, u = eigh(conj(H), S)
    sind = e_.argsort()[:neig - nlocks]
    e_ = e_[sind]
    u = transpose(u[:, sind]) ##the output of qrh()


    v_ = empty(shape=(neig-nlocks, vdim), dtype=complex)
    v_.fill(None) ##for testing ... get rid of me later
    n = neig-nlocks
    #this mess computes u @ V.  the problem is that some ROWS of V represent locked |psi>, but we need
    #to traverse it's COLUMNS for the dot product.  To avoid having an if statement in the innermost (k)
    #loops, I instead implicitly set locked components to zero by negating the locked array and using
    #it to zero-out any locked components, effectivly "skipping" needed entries
    #BEWARE: THIS WILL ONLY WORK IF locked[] IS FILLED WITH 1S OR 0S.
    locks = [int(not locks) for locks in locks]  ##locks -> not locked
    for i in range(0, n):
        for j in range(0, vdim):
            v_[i][j] = 0.0 + 0.0j
            co = -1 ##note this only needs to be as large as neig - probably a ushort is fine
            for k in range(0, neig):
                co += locks[k]
                v_[i][j] += u[i][co] * complex(locks[k])*a[k][j]
            #co = 1*n
            for k in range(0, neig):
                co += locks[k]
                v_[i][j] += u[i][co] * complex(locks[k])*b[k][j]
            #co = 2*n
            for k in range(0, neig):
                co += locks[k]
                v_[i][j] += u[i][co] * complex(locks[k])*c[k][j]
    locks = [int(not locks) for locks in locks] ##back to original

    if(not numpy.allclose(v_, u@V)):
        exit(122333)
    #v_ = u @ V ##the only portion of this entire routine that I couldn't get row-row m.a.p.s in :(

    co = 0
    for i in range(0, neig):
        if(not locks[i]):
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
    Xi = empty(shape=(neigs, npw), dtype=complex) ##trial wavefunctions, ith iteration
    Xm = empty(shape=(neigs, npw), dtype=complex) ##trial wavefunctions, i (M)inus 1th iteration

    #action of wavefunctions on hamiltonian
    PH = empty(shape=(3*neigs, npw), dtype=complex)
    WiH = empty(shape=(neigs, npw), dtype=complex)
    XiH = empty(shape=(neigs, npw), dtype=complex)
    XmH = empty(shape=(neigs, npw), dtype=complex)

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

    if(not stab): ##need to initialize this only once before 1st call to ritz()
        for i in range(0, neigs):
            XmH[i,:] = action(psi=Xm[i, :], H=A)

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
                XiH[i,:] = action(psi=Xi[i,:], H=A)
                lam = 0 ## = <psi|H|psi>
                for j in range(0, npw):
                    lam += XiH[i][j] * conj(Xi[i][j])

                for j in range(0, npw):
                    Wi[i][j] = T[j] * (Xi[i][j] - XiH[i][j]/lam)

            else: ##for testing - to make sure I don't do anything stupid with already locked values
                WiH[i,:].fill(None)
                XiH[i,:].fill(None)
                XmH[i,:].fill(None)
                #... but we cant do the same for |psi> themselves since we need them for orthogonalization

        if(stab): ##stable re-orthogonalization, need to recalc the action on all (unconverged) wavefunctions
            Wi, Xi, Xm = ogs(neig=neigs, vdim=npw, a=Wi, b=Xi, c=Xm, locked=locked)
            for i in range(0, neigs):
                if(not locked[i]):
                    WiH[i,:] = action(psi=Wi[i,:], H=A)
                    XiH[i,:] = action(psi=Xi[i,:], H=A)
                    XmH[i,:] = action(psi=Xm[i,:], H=A)
        else: ##unstable but much faster - only need to calculate action on W
            for i in range(0, neigs):
                if(not locked[i]):
                    WiH[i,:] = action(psi=Wi[i,:], H=A)

        e, v = ritz2(neig=neigs, vdim=npw, A=A,
                     a=Wi, b=Xi, c=Xm, ah=WiH, bh=XiH, ch=XmH,
                     nlocks=nlocks, locks=locked,
                     eigs=e, vecs=v)
        nlocks, locked, e, eolds, ediffs, endflag = updateconv(neigs=neigs,
                                                               nlocked=nlocks, locks=locked,
                                                               curre=e, olde=eolds, diffe=ediffs,
                                                               stab=stab, tol=tol)
        if(endflag):
            break

        ##if no orthogonalization, we can skip an application of <psi|H^*
        if(not stab):
            for i in range(0, neigs):
                XmH[i,:] = deepcopy(XiH[i,:])
        ##X(i) -> X(i+1)
        for i in range(0, neigs):
            if(not locked[i]):
                Xm[i,:] = deepcopy(Xi[i,:])
                Xi[i,:] = deepcopy(v[i,:])




    print("finished in", count, "steps")
    return e, v










seed(267)
neig = 4
size = 700
A = 1*matmake(size)
#print(A)
print("begin")

ecg, vcg = locg(npw=size, A=A, neigs=neig, stab=1, tol=1e-3)
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



"""
--- old function definitions that i totally wont need later :)
"""
def ritz(neig, vdim, A, a, b, c, nlocks, locks, eigs, vecs, ah=None, bh=None, ch=None):
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
