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

def normalize(n, v):
    norm = 0.0
    for i in range(0, n):
        norm += v[i].real*v[i].real + v[i].imag*v[i].imag
    norm = 1./sqrt(norm)
    for i in range(0, n):
        v[i] *= norm
    return v

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
def ogsh(n, orig, against, buff):
    fac = 0.0 + 0.0j
    for i in range(0, n):
        fac += conj(against[i]) * orig[i]
    for i in range(0, n):
        buff[i] -= fac*against[i]
    return buff

def ogs2(neig, vdim, a, b, c, abuf, bbuf, cbuf, locked):
    abuf = deepcopy(a) #\
    bbuf = deepcopy(b) # > memcpy me
    cbuf = deepcopy(c) #/

    #ortho a
    for i in range(0, neig):
        ##skip orthonormalization of locked vectors
        if(locked[i]):
            continue
        ##ortho against locked vectors first
        for j in range(0, neig):
            if(locked[j]):
                abuf[i] = ogsh(n=vdim, orig=a[i], against=a[j], buff=abuf[i])
                abuf[i] = ogsh(n=vdim, orig=a[i], against=b[j], buff=abuf[i])
                abuf[i] = ogsh(n=vdim, orig=a[i], against=c[j], buff=abuf[i])
        ##and now against previous a
        for j in range(0, i):
            if(not locked[j]):
                abuf[i] = ogsh(n=vdim, orig=a[i], against=abuf[j], buff=abuf[i])
        abuf[i] = normalize(n=vdim, v=abuf[i])

    #ortho b
    for i in range(0, neig):
        ##skip orthonormalization of locked vectors
        if(locked[i]):
            continue
        ##ortho against locked vectors first
        for j in range(0, neig):
            if(locked[j]):
                bbuf[i] = ogsh(n=vdim, orig=b[i], against=a[j], buff=bbuf[i])
                bbuf[i] = ogsh(n=vdim, orig=b[i], against=b[j], buff=bbuf[i])
                bbuf[i] = ogsh(n=vdim, orig=b[i], against=c[j], buff=bbuf[i])
        ##and now against previous a, b
        for j in range(0, neig):
            if(not locked[j]):
                bbuf[i] = ogsh(n=vdim, orig=b[i], against=abuf[j], buff=bbuf[i])
        for j in range(0, i):
            if(not locked[j]):
                bbuf[i] = ogsh(n=vdim, orig=b[i], against=bbuf[j], buff=bbuf[i])
        bbuf[i] = normalize(n=vdim, v=bbuf[i])

    #ortho c
    for i in range(0, neig):
        ##skip orthonormalization of locked vectors
        if(locked[i]):
            continue
        ##ortho against locked vectors first
        for j in range(0, neig):
            if(locked[j]):
                cbuf[i] = ogsh(n=vdim, orig=c[i], against=a[j], buff=cbuf[i])
                cbuf[i] = ogsh(n=vdim, orig=c[i], against=b[j], buff=cbuf[i])
                cbuf[i] = ogsh(n=vdim, orig=c[i], against=c[j], buff=cbuf[i])
        ##and now against previous a, b, c
        for j in range(0, neig):
            if(not locked[j]):
                cbuf[i] = ogsh(n=vdim, orig=c[i], against=abuf[j], buff=cbuf[i])
                cbuf[i] = ogsh(n=vdim, orig=c[i], against=bbuf[j], buff=cbuf[i])
        for j in range(0, i):
            if(not locked[j]):
                cbuf[i] = ogsh(n=vdim, orig=c[i], against=cbuf[j], buff=cbuf[i])
        cbuf[i] = normalize(n=vdim, v=cbuf[i])

    return abuf, bbuf, cbuf

#computes conj(a) @ b
def conjhdot(n, conja, b):
    res = 0.0 + 0.0j
    for i in range(0, n):
        res += conj(conja[i]) * b[i]
    return res

def ritz2(neig, vdim, HR, SR, er, vr,
          a, b, c, ah, bh, ch, nlocks, locks, eigs, vecs, stab):
    #Make H^*
    #Note that this has a pretty awful access pattern, but the alternative is like
    #100+ lines of triple loops infested w/ if/else statements
    n = neig-nlocks

    ci, cj = 0, 0
    if(not stab): ##need to compute overlap matrix if no ortho
        for i in range(0, neig):
            if(not locks[i]):
                cj = 0
                for j in range(0, neig):
                    if(not locks[j]):
                        if(cj+0*n >= ci+0*n):
                            HR[ci+0*n][cj+0*n] = conjhdot(n=vdim, conja=ah[i], b=a[j])
                            SR[ci+0*n][cj+0*n] = conjhdot(n=vdim, conja=a[i], b=a[j])
                        if(cj+1*n >= ci+0*n):
                            HR[ci+0*n][cj+1*n] = conjhdot(n=vdim, conja=ah[i], b=b[j])
                            SR[ci+0*n][cj+1*n] = conjhdot(n=vdim, conja=a[i], b=b[j])
                        if(cj+2*n >= ci+0*n):
                            HR[ci+0*n][cj+2*n] = conjhdot(n=vdim, conja=ah[i], b=c[j])
                            SR[ci+0*n][cj+2*n] = conjhdot(n=vdim, conja=a[i], b=c[j])
                        cj += 1
                cj = 0
                for j in range(0, neig):
                    if(not locks[j]):
                        if(cj+0*n >= ci+1*n):
                            HR[ci+1*n][cj+0*n] = conjhdot(n=vdim, conja=bh[i], b=a[j])
                            SR[ci+1*n][cj+0*n] = conjhdot(n=vdim, conja=b[i], b=a[j])
                        if(cj+1*n >= ci+1*n):
                            HR[ci+1*n][cj+1*n] = conjhdot(n=vdim, conja=bh[i], b=b[j])
                            SR[ci+1*n][cj+1*n] = conjhdot(n=vdim, conja=b[i], b=b[j])
                        if(cj+2*n>= ci+1*n):
                            HR[ci+1*n][cj+2*n] = conjhdot(n=vdim, conja=bh[i], b=c[j])
                            SR[ci+1*n][cj+2*n] = conjhdot(n=vdim, conja=b[i], b=c[j])
                        cj += 1
                cj = 0
                for j in range(0, neig):
                    if(not locks[j]):
                        if(cj+0*n >= ci+2*n):
                            HR[ci+2*n][cj+0*n] = conjhdot(n=vdim, conja=ch[i], b=a[j])
                            SR[ci+2*n][cj+0*n] = conjhdot(n=vdim, conja=c[i], b=a[j])
                        if(cj+1*n >= ci+2*n):
                            HR[ci+2*n][cj+1*n] = conjhdot(n=vdim, conja=ch[i], b=b[j])
                            SR[ci+2*n][cj+1*n] = conjhdot(n=vdim, conja=c[i], b=b[j])
                        if(cj+2*n >= ci+2*n):
                            HR[ci+2*n][cj+2*n] = conjhdot(n=vdim, conja=ch[i], b=c[j])
                            SR[ci+2*n][cj+2*n] = conjhdot(n=vdim, conja=c[i], b=c[j])
                        cj += 1
                ci += 1
        for i in range(0, 3*n):
            for j in range(0, i):
                HR[i][j] = conj(HR[j][i])
                SR[i][j] = conj(SR[j][i])
    else: ##ortho makes overlap matrix the identity - no need to calculate S
        for i in range(0, neig):
            if(not locks[i]):
                cj = 0
                for j in range(0, neig):
                    if(not locks[j]):
                        if(cj+0*n >= ci+0*n):
                            HR[ci+0*n][cj+0*n] = conjhdot(n=vdim, conja=ah[i], b=a[j])
                        if(cj+1*n >= ci+0*n):
                            HR[ci+0*n][cj+1*n] = conjhdot(n=vdim, conja=ah[i], b=b[j])
                        if(cj+2*n >= ci+0*n):
                            HR[ci+0*n][cj+2*n] = conjhdot(n=vdim, conja=ah[i], b=c[j])
                        cj += 1
                cj = 0
                for j in range(0, neig):
                    if(not locks[j]):
                        if(cj+0*n >= ci+1*n):
                            HR[ci+1*n][cj+0*n] = conjhdot(n=vdim, conja=bh[i], b=a[j])
                        if(cj+1*n >= ci+1*n):
                            HR[ci+1*n][cj+1*n] = conjhdot(n=vdim, conja=bh[i], b=b[j])
                        if(cj+2*n>= ci+1*n):
                            HR[ci+1*n][cj+2*n] = conjhdot(n=vdim, conja=bh[i], b=c[j])
                        cj += 1
                cj = 0
                for j in range(0, neig):
                    if(not locks[j]):
                        if(cj+0*n >= ci+2*n):
                            HR[ci+2*n][cj+0*n] = conjhdot(n=vdim, conja=ch[i], b=a[j])
                        if(cj+1*n >= ci+2*n):
                            HR[ci+2*n][cj+1*n] = conjhdot(n=vdim, conja=ch[i], b=b[j])
                        if(cj+2*n >= ci+2*n):
                            HR[ci+2*n][cj+2*n] = conjhdot(n=vdim, conja=ch[i], b=c[j])
                        cj += 1
                ci += 1
        for i in range(0, 3*n):
            for j in range(0, i):
                HR[i][j] = conj(HR[j][i])

    if(not stab):
        er, vr = eigh(HR[0:3*n,0:3*n], SR[0:3*n,0:3*n])   ###needed if we skip orthogonalization
    else:
        er, vr = eig(HR[0:3*n,0:3*n])      ###can use this if orthoganlization is used
    sind = er.argsort()[:n]
    er = er[sind]
    vr = transpose(vr[:, sind]) ##the output of qrh()

    #update the eigenvalues to return
    co = 0
    for i in range(0, neig):
        if(not locks[i]):
            eigs[i] = er[co].real
            co += 1

    n = neig-nlocks
    #this mess computes u @ {a,b,c}  the problem is that some ROWS of {} represent locked |psi>, but we need
    #to traverse it's COLUMNS for the dot product.  To avoid having an if statement in the innermost (k)
    #loops, I instead implicitly set locked components to zero by negating the locked array and using
    #it to zero-out any locked components, effectivly "skipping" needed entries
    #BEWARE: THIS WILL ONLY WORK IF locked[] IS FILLED WITH 1S OR 0S.
    locks = [int(not locks) for locks in locks]  ##locks -> not locked
    cq = 0
    #for i in range(0, n):
    for i in range(0, neig):
        if(locks[i]): ##actually 'not locked[i]'
            for j in range(0, vdim):
                vecs[i][j] = 0.0 + 0.0j
                co = -1 ##note this only needs to be as large as neig - probably a ushort is fine
                for k in range(0, neig):
                    co += locks[k]
                    vecs[i][j] += vr[cq][co] * complex(locks[k])*a[k][j]
                #co = 1*n
                for k in range(0, neig):
                    co += locks[k]
                    vecs[i][j] += vr[cq][co] * complex(locks[k])*b[k][j]
                #co = 2*n
                for k in range(0, neig):
                    co += locks[k]
                    vecs[i][j] += vr[cq][co] * complex(locks[k])*c[k][j]
            cq += 1
    locks = [int(not locks) for locks in locks] ##back to original


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
__GLOBAL_ACTION_COUNTER = 0
def action(psi, H):
    global  __GLOBAL_ACTION_COUNTER
    __GLOBAL_ACTION_COUNTER += 1
    return psi@conj(H)

numpy.seterr(all='raise')
from time import time as time
#stability: 0 = no orthogonalization, includes locking (mostly for good initial guesses, tol >= 1e-3 or so)
#           1 = orthogonalization, includes locking (introduces error near end of spectrum)
#           2 = orthogonalization, does not include locking (works for any precision up to machine eps)
def locg(npw, A, neigs, stab, tol):
    global __GLOBAL_ACTION_COUNTER
    timestart = time()

    # --- decs ---
    #subspace basis vectors (neigs rows of dim npw)
    Wi = empty(shape=(neigs, npw), dtype=complex) ##preconditioned search directions
    Xi = empty(shape=(neigs, npw), dtype=complex) ##trial wavefunctions, ith iteration
    Xm = empty(shape=(neigs, npw), dtype=complex) ##trial wavefunctions, i (M)inus 1th iteration

    #action of wavefunctions on hamiltonian
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
    for i in range(0, neigs):
        Xi[i,:] = normalize(n=npw, v=Xi[i,:])

    if(not stab): ##need to initialize this only once before 1st call to ritz()
        for i in range(0, neigs):
            XmH[i,:] = action(psi=Xm[i,:], H=A)

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

        #compute new search directions
        for i in range(0, neigs):
            if(not locked[i]):
                XiH[i,:] = action(psi=Xi[i,:], H=A)
                lam = 0.0 ## = <psi|H|psi>, pure real
                for j in range(0, npw):
                    #lam += XiH[i][j] * conj(Xi[i][j])
                    lam += XiH[i][j].real*Xi[i][j].real + XiH[i][j].imag*Xi[i][j].imag + 0.0j

                lam = 1.0/lam
                for j in range(0, npw):
                    Wi[i][j] = (  T[j] * (Xi[i][j].real - XiH[i][j].real*lam)  ) +\
                          1.0j*(  T[j] * (Xi[i][j].imag - XiH[i][j].imag*lam)  )

        #initialize action on trial wavefunctions (after reorthogonalization if necessary)
        if(stab): ##stable, recalc all (unconverged) psi since ogs() changes them
            Wi, Xi, Xm = ogs2(neig=neigs, vdim=npw,
                              a=Wi, b=Xi, c=Xm, abuf=WiH, bbuf=XiH, cbuf=XmH,
                              locked=locked)
            for i in range(0, neigs):
                if(not locked[i]):
                    WiH[i,:] = action(psi=Wi[i,:], H=A)
                    XiH[i,:] = action(psi=Xi[i,:], H=A)
                    XmH[i,:] = action(psi=Xm[i,:], H=A)
        else: ##unstable, only need to calculate action on |W>
            for i in range(0, neigs):
                if(not locked[i]):
                    WiH[i,:] = action(psi=Wi[i,:], H=A)

        #rayleigh-ritz procedure on trial subspace
        e, v = ritz2(neig=neigs, vdim=npw, HR=HR, SR=SR, er=er, vr=vr,
                     a=Wi, b=Xi, c=Xm, ah=WiH, bh=XiH, ch=XmH,
                     nlocks=nlocks, locks=locked,
                     eigs=e, vecs=v, stab=stab)

        ##if no orthogonalization, we can skip an application of <psi|H^*
        ##this needs to be done before updating the locked vectors, though
        if(not stab):
            for i in range(0, neigs):
                if(not locked[i]):
                    XmH[i,:] = deepcopy(XiH[i,:])

        #update convergence params and locked vectors, if necessary
        nlocks, locked, e, eolds, ediffs, endflag = updateconv(neigs=neigs,
                                                               nlocked=nlocks, locks=locked,
                                                               curre=e, olde=eolds, diffe=ediffs,
                                                               stab=stab, tol=tol)
        if(endflag):
            break

        ##prep for next iteration: X(i) -> X(i+1)
        for i in range(0, neigs):
            if(not locked[i]):
                Xm[i,:] = deepcopy(Xi[i,:])
                Xi[i,:] = deepcopy(v[i,:])


    print("finished in", count, "steps,", time() - timestart, "seconds, with", __GLOBAL_ACTION_COUNTER,
          "applications of H|psi>")
    return e, v










seed(267)
neig = 5
size = 700
A = 10*matmake(size)
#print(A)
print("begin")

ecg, vcg = locg(npw=size, A=A, neigs=neig, stab=1, tol=1e-4)
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
