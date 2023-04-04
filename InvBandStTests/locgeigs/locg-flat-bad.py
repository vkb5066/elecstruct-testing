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

def acc(n, i, j):
    return n*i + j

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
                abuf[i*vdim:(i+1)*vdim] = ogsh(n=vdim, orig=a[i*vdim:(i+1)*vdim], against=a[j*vdim:(j+1)*vdim], buff=abuf[i*vdim:(i+1)*vdim])
                abuf[i*vdim:(i+1)*vdim] = ogsh(n=vdim, orig=a[i*vdim:(i+1)*vdim], against=b[j*vdim:(j+1)*vdim], buff=abuf[i*vdim:(i+1)*vdim])
                abuf[i*vdim:(i+1)*vdim] = ogsh(n=vdim, orig=a[i*vdim:(i+1)*vdim], against=c[j*vdim:(j+1)*vdim], buff=abuf[i*vdim:(i+1)*vdim])
        ##and now against previous a
        for j in range(0, i):
            if(not locked[j]):
                abuf[i*vdim:(i+1)*vdim] = ogsh(n=vdim, orig=a[i*vdim:(i+1)*vdim], against=abuf[j*vdim:(j+1)*vdim], buff=abuf[i*vdim:(i+1)*vdim])
        abuf[i*vdim:(i+1)*vdim] = normalize(n=vdim, v=abuf[i*vdim:(i+1)*vdim])

    #ortho b
    for i in range(0, neig):
        ##skip orthonormalization of locked vectors
        if(locked[i]):
            continue
        ##ortho against locked vectors first
        for j in range(0, neig):
            if(locked[j]):
                bbuf[i*vdim:(i+1)*vdim] = ogsh(n=vdim, orig=b[i*vdim:(i+1)*vdim], against=a[j*vdim:(j+1)*vdim], buff=bbuf[i*vdim:(i+1)*vdim])
                bbuf[i*vdim:(i+1)*vdim] = ogsh(n=vdim, orig=b[i*vdim:(i+1)*vdim], against=b[j*vdim:(j+1)*vdim], buff=bbuf[i*vdim:(i+1)*vdim])
                bbuf[i*vdim:(i+1)*vdim] = ogsh(n=vdim, orig=b[i*vdim:(i+1)*vdim], against=c[j*vdim:(j+1)*vdim], buff=bbuf[i*vdim:(i+1)*vdim])
        ##and now against previous a, b
        for j in range(0, neig):
            if(not locked[j]):
                bbuf[i*vdim:(i+1)*vdim] = ogsh(n=vdim, orig=b[i*vdim:(i+1)*vdim], against=abuf[j*vdim:(j+1)*vdim], buff=bbuf[i*vdim:(i+1)*vdim])
        for j in range(0, i):
            if(not locked[j]):
                bbuf[i*vdim:(i+1)*vdim] = ogsh(n=vdim, orig=b[i*vdim:(i+1)*vdim], against=bbuf[j*vdim:(j+1)*vdim], buff=bbuf[i*vdim:(i+1)*vdim])
        bbuf[i*vdim:(i+1)*vdim] = normalize(n=vdim, v=bbuf[i*vdim:(i+1)*vdim])

    #ortho c
    for i in range(0, neig):
        ##skip orthonormalization of locked vectors
        if(locked[i]):
            continue
        ##ortho against locked vectors first
        for j in range(0, neig):
            if(locked[j]):
                cbuf[i*vdim:(i+1)*vdim] = ogsh(n=vdim, orig=c[i*vdim:(i+1)*vdim], against=a[j*vdim:(j+1)*vdim], buff=cbuf[i*vdim:(i+1)*vdim])
                cbuf[i*vdim:(i+1)*vdim] = ogsh(n=vdim, orig=c[i*vdim:(i+1)*vdim], against=b[j*vdim:(j+1)*vdim], buff=cbuf[i*vdim:(i+1)*vdim])
                cbuf[i*vdim:(i+1)*vdim] = ogsh(n=vdim, orig=c[i*vdim:(i+1)*vdim], against=c[j*vdim:(j+1)*vdim], buff=cbuf[i*vdim:(i+1)*vdim])
        ##and now against previous a, b, c
        for j in range(0, neig):
            if(not locked[j]):
                cbuf[i*vdim:(i+1)*vdim] = ogsh(n=vdim, orig=c[i*vdim:(i+1)*vdim], against=abuf[j*vdim:(j+1)*vdim], buff=cbuf[i*vdim:(i+1)*vdim])
                cbuf[i*vdim:(i+1)*vdim] = ogsh(n=vdim, orig=c[i*vdim:(i+1)*vdim], against=bbuf[j*vdim:(j+1)*vdim], buff=cbuf[i*vdim:(i+1)*vdim])
        for j in range(0, i):
            if(not locked[j]):
                cbuf[i*vdim:(i+1)*vdim] = ogsh(n=vdim, orig=c[i*vdim:(i+1)*vdim], against=cbuf[j*vdim:(j+1)*vdim], buff=cbuf[i*vdim:(i+1)*vdim])
        cbuf[i*vdim:(i+1)*vdim] = normalize(n=vdim, v=cbuf[i*vdim:(i+1)*vdim])

    return abuf, bbuf, cbuf

#computes conj(a) @ b
def conjhdot(n, conja, b):
    res = 0.0 + 0.0j
    for i in range(0, n):
        res += conj(conja[i]) * b[i]
    return res

def denseeigsolver(stab, n, H, S, e, v):
    #if S exists, need to transform general -> easy eigenproblem
    #i.e. make H|psi> = eS|psi>  -->  C|phi> = e|phi>
    if(not stab):
        #S -> LL+ (S is herm. positive def)
        for i in range(0, n):
            for j in range(0, i):
                su = 0.0 + 0.0j  ##this sum is complex
                for k in range(0, j):
                    su += S[acc(n, i, k)] * conj(S[acc(n, j, k)])
                S[acc(n, i, j)] = (1.0/S[acc(n, j, j)].real) * (S[acc(n, i, j)] - su)
            su = 0.0 + 0.j  ##this sum has zero complex part
            for j in range(0, i):
                su += S[acc(n, i, j)].real*S[acc(n, i, j)].real + \
                      S[acc(n, i, j)].imag*S[acc(n, i, j)].imag + 0j
            S[acc(n, i, i)] = sqrt(S[acc(n, i, i)].real - su.real)
        #H -> X: solve L@X  = H (only need 1/2 of X)
        for i in range(0, n):
            for j in range(i, n):
                for k in range(0, i):
                    H[acc(n, i, j)] -= S[acc(n, i, k)] * H[acc(n, k, j)]
                H[acc(n, i, j)] /= S[acc(n, i, i)].real
        #X -> C: solve C@L+ = X (C is hermitian)
        for i in range(0, n):
            for j in range(0, i):
                H[acc(n, i, i)] -= H[acc(n, i, j)]*conj(S[acc(n, i, j)])
            H[acc(n, i, i)] /= conj(S[acc(n, i, i)].real)
            for j in range(i+1, n):
                for k in range(0, j):
                    H[acc(n, i, j)] -= H[acc(n, i, k)]*conj(S[acc(n, j, k)])
                H[acc(n, i, j)] /= S[acc(n, j, i)].real
                H[acc(n, j, i)] = conj(H[acc(n, i, j)])

    h_ = reshape(a=H, newshape=(n, n)) ##to make this work with numpy
    e, v = eig(h_)
    #sind = e.argsort()[:n//3]
    sind = e.argsort()[:n]
    e = e[sind]
    v = transpose(v[:, sind]) ##the output of qrh()

    #again, if S exists, need to get the general eigenvectors back
    #this only works if the vectors are stored as rows
    if(not stab):
        for i in range(n - 1, -1, -1):  ##go from n-1 to 0 (both inclusive), decrementing by -1
            for j in range(n - 1, -1, -1):  ##ditto ^
                for k in range(j + 1, n):
                    v[i][j] -= conj(S[acc(n, k, j)]) * v[i][k]
                v[i][j] /= S[acc(n, j, j)].real  ##pure real?

    return e, v


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
                            HR[acc(3*n, ci+0*n, cj+0*n)] = conjhdot(n=vdim, conja=ah[i*vdim:(i+1)*vdim], b=a[j*vdim:(j+1)*vdim])
                            SR[acc(3*n, ci+0*n, cj+0*n)] = conjhdot(n=vdim, conja=a[i*vdim:(i+1)*vdim], b=a[j*vdim:(j+1)*vdim])
                        if(cj+1*n >= ci+0*n):
                            HR[acc(3*n, ci+0*n, cj+1*n)] = conjhdot(n=vdim, conja=ah[i*vdim:(i+1)*vdim], b=b[j*vdim:(j+1)*vdim])
                            SR[acc(3*n, ci+0*n, cj+1*n)] = conjhdot(n=vdim, conja=a[i*vdim:(i+1)*vdim], b=b[j*vdim:(j+1)*vdim])
                        if(cj+2*n >= ci+0*n):
                            HR[acc(3*n, ci+0*n, cj+2*n)] = conjhdot(n=vdim, conja=ah[i*vdim:(i+1)*vdim], b=c[j*vdim:(j+1)*vdim])
                            SR[acc(3*n, ci+0*n, cj+2*n)] = conjhdot(n=vdim, conja=a[i*vdim:(i+1)*vdim], b=c[j*vdim:(j+1)*vdim])
                        cj += 1
                cj = 0
                for j in range(0, neig):
                    if(not locks[j]):
                        if(cj+0*n >= ci+1*n):
                            HR[acc(3*n, ci+1*n, cj+0*n)] = conjhdot(n=vdim, conja=bh[i*vdim:(i+1)*vdim], b=a[j*vdim:(j+1)*vdim])
                            SR[acc(3*n, ci+1*n, cj+0*n)] = conjhdot(n=vdim, conja=b[i*vdim:(i+1)*vdim], b=a[j*vdim:(j+1)*vdim])
                        if(cj+1*n >= ci+1*n):
                            HR[acc(3*n, ci+1*n, cj+1*n)] = conjhdot(n=vdim, conja=bh[i*vdim:(i+1)*vdim], b=b[j*vdim:(j+1)*vdim])
                            SR[acc(3*n, ci+1*n, cj+1*n)] = conjhdot(n=vdim, conja=b[i*vdim:(i+1)*vdim], b=b[j*vdim:(j+1)*vdim])
                        if(cj+2*n>= ci+1*n):
                            HR[acc(3*n, ci+1*n, cj+2*n)] = conjhdot(n=vdim, conja=bh[i*vdim:(i+1)*vdim], b=c[j*vdim:(j+1)*vdim])
                            SR[acc(3*n, ci+1*n, cj+2*n)] = conjhdot(n=vdim, conja=b[i*vdim:(i+1)*vdim], b=c[j*vdim:(j+1)*vdim])
                        cj += 1
                cj = 0
                for j in range(0, neig):
                    if(not locks[j]):
                        if(cj+0*n >= ci+2*n):
                            HR[acc(3*n, ci+2*n, cj+0*n)] = conjhdot(n=vdim, conja=ch[i*vdim:(i+1)*vdim], b=a[j*vdim:(j+1)*vdim])
                            SR[acc(3*n, ci+2*n, cj+0*n)] = conjhdot(n=vdim, conja=c[i*vdim:(i+1)*vdim], b=a[j*vdim:(j+1)*vdim])
                        if(cj+1*n >= ci+2*n):
                            HR[acc(3*n, ci+2*n, cj+1*n)] = conjhdot(n=vdim, conja=ch[i*vdim:(i+1)*vdim], b=b[j*vdim:(j+1)*vdim])
                            SR[acc(3*n, ci+2*n, cj+1*n)] = conjhdot(n=vdim, conja=c[i*vdim:(i+1)*vdim], b=b[j*vdim:(j+1)*vdim])
                        if(cj+2*n >= ci+2*n):
                            HR[acc(3*n, ci+2*n, cj+2*n)] = conjhdot(n=vdim, conja=ch[i*vdim:(i+1)*vdim], b=c[j*vdim:(j+1)*vdim])
                            SR[acc(3*n, ci+2*n, cj+2*n)] = conjhdot(n=vdim, conja=c[i*vdim:(i+1)*vdim], b=c[j*vdim:(j+1)*vdim])
                        cj += 1
                ci += 1
        for i in range(0, 3*n):
            for j in range(0, i):
                HR[acc(3*n, i, j)] = conj(HR[acc(3*n, j, i)])
                SR[acc(3*n, i, j)] = conj(SR[acc(3*n, j, i)])
    else: ##ortho makes overlap matrix the identity - no need to calculate S
        for i in range(0, neig):
            if(not locks[i]):
                cj = 0
                for j in range(0, neig):
                    if(not locks[j]):
                        if(cj+0*n >= ci+0*n):
                            HR[acc(3*n, ci+0*n, cj+0*n)] = conjhdot(n=vdim, conja=ah[i*vdim:(i+1)*vdim], b=a[j*vdim:(j+1)*vdim])
                        if(cj+1*n >= ci+0*n):
                            HR[acc(3*n, ci+0*n, cj+1*n)] = conjhdot(n=vdim, conja=ah[i*vdim:(i+1)*vdim], b=b[j*vdim:(j+1)*vdim])
                        if(cj+2*n >= ci+0*n):
                            HR[acc(3*n, ci+0*n, cj+2*n)] = conjhdot(n=vdim, conja=ah[i*vdim:(i+1)*vdim], b=c[j*vdim:(j+1)*vdim])
                        cj += 1
                cj = 0
                for j in range(0, neig):
                    if(not locks[j]):
                        if(cj+0*n >= ci+1*n):
                            HR[acc(3*n, ci+1*n, cj+0*n)] = conjhdot(n=vdim, conja=bh[i*vdim:(i+1)*vdim], b=a[j*vdim:(j+1)*vdim])
                        if(cj+1*n >= ci+1*n):
                            HR[acc(3*n, ci+1*n, cj+1*n)] = conjhdot(n=vdim, conja=bh[i*vdim:(i+1)*vdim], b=b[j*vdim:(j+1)*vdim])
                        if(cj+2*n>= ci+1*n):
                            HR[acc(3*n, ci+1*n, cj+2*n)] = conjhdot(n=vdim, conja=bh[i*vdim:(i+1)*vdim], b=c[j*vdim:(j+1)*vdim])
                        cj += 1
                cj = 0
                for j in range(0, neig):
                    if(not locks[j]):
                        if(cj+0*n >= ci+2*n):
                            HR[acc(3*n, ci+2*n, cj+0*n)] = conjhdot(n=vdim, conja=ch[i*vdim:(i+1)*vdim], b=a[j*vdim:(j+1)*vdim])
                        if(cj+1*n >= ci+2*n):
                            HR[acc(3*n, ci+2*n, cj+1*n)] = conjhdot(n=vdim, conja=ch[i*vdim:(i+1)*vdim], b=b[j*vdim:(j+1)*vdim])
                        if(cj+2*n >= ci+2*n):
                            HR[acc(3*n, ci+2*n, cj+2*n)] = conjhdot(n=vdim, conja=ch[i*vdim:(i+1)*vdim], b=c[j*vdim:(j+1)*vdim])
                        cj += 1
                ci += 1
        for i in range(0, 3*n):
            for j in range(0, i):
                HR[acc(3*n, i, j)] = conj(HR[acc(3*n, j, i)])

    er, vr = denseeigsolver(stab=stab, n=3*n, H=HR[0:3*n*3*n], S=SR[0:3*n*3*n], e=er, v=vr)

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
                    vecs[i][j] += vr[cq][co] * complex(locks[k])*a[acc(neig, k, j)]
                #co = 1*n
                for k in range(0, neig):
                    co += locks[k]
                    vecs[i][j] += vr[cq][co] * complex(locks[k])*b[acc(neig, k, j)]
                #co = 2*n
                for k in range(0, neig):
                    co += locks[k]
                    vecs[i][j] += vr[cq][co] * complex(locks[k])*c[acc(neig, k, j)]
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
    Wi = empty(shape=(neigs*npw), dtype=complex) ##preconditioned search directions
    Xi = empty(shape=(neigs*npw), dtype=complex) ##trial wavefunctions, ith iteration
    Xm = empty(shape=(neigs*npw), dtype=complex) ##trial wavefunctions, i (M)inus 1th iteration

    #action of wavefunctions on hamiltonian
    WiH = empty(shape=(neigs*npw), dtype=complex)
    XiH = empty(shape=(neigs*npw), dtype=complex)
    XmH = empty(shape=(neigs*npw), dtype=complex)

    #diagonal preconditioner
    T = empty(shape=npw, dtype=float)

    #ritz eigenpair stuff
    HR = empty(shape=(3*neigs*3*neigs), dtype=complex) ##dense subspace hamiltonian
    SR = empty(shape=(3*neigs*3*neigs), dtype=complex) ##dense subspace overlap matrix
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
            Xm[acc(npw, i, j)] = random() + 1j*random()
            Xi[acc(npw, i, j)] = random() + 1j*random()
    for i in range(0, neigs):
        Xi[i*npw:(i+1)*npw] = normalize(n=npw, v=Xi[i*npw:(i+1)*npw])

    if(not stab): ##need to initialize this only once before 1st call to ritz()
        for i in range(0, neigs):
            XmH[i*npw:(i+1)*npw] = action(psi=Xm[i*npw:(i+1)*npw], H=A)

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
                XiH[i*npw:(i+1)*npw] = action(psi=Xi[i*npw:(i+1)*npw], H=A)
                lam = 0.0 ## = <psi|H|psi>, pure real
                for j in range(0, npw):
                    #lam += XiH[i][j] * conj(Xi[i][j])
                    lam += XiH[acc(npw, i, j)].real*Xi[acc(npw, i, j)].real + \
                           XiH[acc(npw, i, j)].imag*Xi[acc(npw, i, j)].imag + 0.0j

                lam = 1.0/lam
                for j in range(0, npw):
                    Wi[acc(npw, i, j)] = (  T[j] * (Xi[acc(npw, i, j)].real - XiH[acc(npw, i, j)].real*lam)  ) +\
                                    1.0j*(  T[j] * (Xi[acc(npw, i, j)].imag - XiH[acc(npw, i, j)].imag*lam)  )

        #initialize action on trial wavefunctions (after reorthogonalization if necessary)
        if(stab): ##stable, recalc all (unconverged) psi since ogs() changes them
            Wi, Xi, Xm = ogs2(neig=neigs, vdim=npw,
                              a=Wi, b=Xi, c=Xm, abuf=WiH, bbuf=XiH, cbuf=XmH,
                              locked=locked)
            for i in range(0, neigs):
                if(not locked[i]):
                    WiH[i*npw:(i+1)*npw] = action(psi=Wi[i*npw:(i+1)*npw], H=A)
                    XiH[i*npw:(i+1)*npw] = action(psi=Xi[i*npw:(i+1)*npw], H=A)
                    XmH[i*npw:(i+1)*npw] = action(psi=Xm[i*npw:(i+1)*npw], H=A)
        else: ##unstable, only need to calculate action on |W>
            for i in range(0, neigs):
                if(not locked[i]):
                    WiH[i*npw:(i+1)*npw] = action(psi=Wi[i*npw:(i+1)*npw], H=A)

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
                    XmH[i*npw:(i+1)*npw] = deepcopy(XiH[i*npw:(i+1)*npw])

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
                Xm[i*npw:(i+1)*npw] = deepcopy(Xi[i*npw:(i+1)*npw])
                for j in range(0, npw):
                    Xi[acc(npw, i, j)] = v[i][j]


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
