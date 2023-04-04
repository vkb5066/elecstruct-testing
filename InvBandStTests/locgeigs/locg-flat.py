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
    ii = 0
    for i in range(0, neig):
        ##skip orthonormalization of locked vectors
        if(locked[i]):
            ii += vdim
            continue
        ##ortho against locked vectors first
        jj = 0
        for j in range(0, neig):
            if(locked[j]):
                abuf[ii:ii+vdim] = ogsh(n=vdim, orig=a[ii:ii+vdim], against=a[jj:jj+vdim], buff=abuf[ii:ii+vdim])
                abuf[ii:ii+vdim] = ogsh(n=vdim, orig=a[ii:ii+vdim], against=b[jj:jj+vdim], buff=abuf[ii:ii+vdim])
                abuf[ii:ii+vdim] = ogsh(n=vdim, orig=a[ii:ii+vdim], against=c[jj:jj+vdim], buff=abuf[ii:ii+vdim])
            jj += vdim
        ##and now against previous a
        jj = 0
        for j in range(0, i):
            if(not locked[j]):
                abuf[ii:ii+vdim] = ogsh(n=vdim, orig=a[ii:ii+vdim], against=abuf[jj:jj+vdim], buff=abuf[ii:ii+vdim])
            jj += vdim
        abuf[ii:ii+vdim] = normalize(n=vdim, v=abuf[ii:ii+vdim])
        ii += vdim

    #ortho b
    ii = 0
    for i in range(0, neig):
        ##skip orthonormalization of locked vectors
        if(locked[i]):
            ii += vdim
            continue
        ##ortho against locked vectors first
        jj = 0
        for j in range(0, neig):
            if(locked[j]):
                bbuf[ii:ii+vdim] = ogsh(n=vdim, orig=b[ii:ii+vdim], against=a[jj:jj+vdim], buff=bbuf[ii:ii+vdim])
                bbuf[ii:ii+vdim] = ogsh(n=vdim, orig=b[ii:ii+vdim], against=b[jj:jj+vdim], buff=bbuf[ii:ii+vdim])
                bbuf[ii:ii+vdim] = ogsh(n=vdim, orig=b[ii:ii+vdim], against=c[jj:jj+vdim], buff=bbuf[ii:ii+vdim])
            jj += vdim
        ##and now against previous a, b
        jj = 0
        for j in range(0, neig):
            if(not locked[j]):
                bbuf[ii:ii+vdim] = ogsh(n=vdim, orig=b[ii:ii+vdim], against=abuf[jj:jj+vdim], buff=bbuf[ii:ii+vdim])
            jj += vdim
        jj = 0
        for j in range(0, i):
            if(not locked[j]):
                bbuf[ii:ii+vdim] = ogsh(n=vdim, orig=b[ii:ii+vdim], against=bbuf[jj:jj+vdim], buff=bbuf[ii:ii+vdim])
            jj += vdim
        bbuf[ii:ii+vdim] = normalize(n=vdim, v=bbuf[ii:ii+vdim])
        ii += vdim

    #ortho c
    ii = 0
    for i in range(0, neig):
        ##skip orthonormalization of locked vectors
        if(locked[i]):
            ii += vdim
            continue
        ##ortho against locked vectors first
        jj = 0
        for j in range(0, neig):
            if(locked[j]):
                cbuf[ii:ii+vdim] = ogsh(n=vdim, orig=c[ii:ii+vdim], against=a[jj:jj+vdim], buff=cbuf[ii:ii+vdim])
                cbuf[ii:ii+vdim] = ogsh(n=vdim, orig=c[ii:ii+vdim], against=b[jj:jj+vdim], buff=cbuf[ii:ii+vdim])
                cbuf[ii:ii+vdim] = ogsh(n=vdim, orig=c[ii:ii+vdim], against=c[jj:jj+vdim], buff=cbuf[ii:ii+vdim])
            jj += vdim
        ##and now against previous a, b, c
        jj = 0
        for j in range(0, neig):
            if(not locked[j]):
                cbuf[ii:ii+vdim] = ogsh(n=vdim, orig=c[ii:ii+vdim], against=abuf[jj:jj+vdim], buff=cbuf[ii:ii+vdim])
                cbuf[ii:ii+vdim] = ogsh(n=vdim, orig=c[ii:ii+vdim], against=bbuf[jj:jj+vdim], buff=cbuf[ii:ii+vdim])
            jj += vdim
        jj = 0
        for j in range(0, i):
            if(not locked[j]):
                cbuf[ii:ii+vdim] = ogsh(n=vdim, orig=c[ii:ii+vdim], against=cbuf[jj:jj+vdim], buff=cbuf[ii:ii+vdim])
            jj += vdim
        cbuf[ii:ii+vdim] = normalize(n=vdim, v=cbuf[ii:ii+vdim])
        ii += vdim

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
                su += S[acc(n, i, j)].real*S[acc(n, i, j)].real + S[acc(n, i, j)].imag*S[acc(n, i, j)].imag + 0j
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
                H[acc(n, i, j)] /= S[acc(n, j, j)].real
                H[acc(n, j, i)] = conj(H[acc(n, i, j)])

    h_ = H.reshape((n, n)) ##just because numpy can't deal with flattened matrices
    e, v = eig(h_)
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
                v[i][j] /= S[acc(n, j, j)].real

    return e, v


def ritz2(neig, vdim, HR, SR, er, vr,
          a, b, c, ah, bh, ch, nlocks, locks, eigs, vecs, stab):

    #Make H^*
    #Note that this has a pretty awful access pattern, but the alternative is like
    #100+ lines of triple loops infested w/ if/else statements
    n = neig-nlocks
    n0, n1, n2, n3 = 0, 1*n, 2*n, 3*n
    ci, cj = 0, 0
    if(not stab): ##need to compute overlap matrix if no ortho
        ii = 0
        for i in range(0, neig):
            if(not locks[i]):
                cj = 0
                jj = 0
                for j in range(0, neig):
                    if(not locks[j]):
                        if(cj+n0 >= ci+n0):
                            HR[acc(n3, ci+n0, cj+n0)] = conjhdot(n=vdim, conja=ah[ii:ii+vdim], b=a[jj:jj+vdim])
                            SR[acc(n3, ci+n0, cj+n0)] = conjhdot(n=vdim, conja=a[ii:ii+vdim],  b=a[jj:jj+vdim])
                        if(cj+n1 >= ci+n0):
                            HR[acc(n3, ci+n0, cj+n1)] = conjhdot(n=vdim, conja=ah[ii:ii+vdim], b=b[jj:jj+vdim])
                            SR[acc(n3, ci+n0, cj+n1)] = conjhdot(n=vdim, conja=a[ii:ii+vdim],  b=b[jj:jj+vdim])
                        if(cj+n2 >= ci+n0):
                            HR[acc(n3, ci+n0, cj+n2)] = conjhdot(n=vdim, conja=ah[ii:ii+vdim], b=c[jj:jj+vdim])
                            SR[acc(n3, ci+n0, cj+n2)] = conjhdot(n=vdim, conja=a[ii:ii+vdim],  b=c[jj:jj+vdim])
                        cj += 1
                    jj += vdim
                cj = 0
                jj = 0
                for j in range(0, neig):
                    if(not locks[j]):
                        if(cj+n0 >= ci+n1):
                            HR[acc(n3, ci+n1, cj+n0)] = conjhdot(n=vdim, conja=bh[ii:ii+vdim], b=a[jj:jj+vdim])
                            SR[acc(n3, ci+n1, cj+n0)] = conjhdot(n=vdim, conja=b[ii:ii+vdim],  b=a[jj:jj+vdim])
                        if(cj+n1 >= ci+n1):
                            HR[acc(n3, ci+n1, cj+n1)] = conjhdot(n=vdim, conja=bh[ii:ii+vdim], b=b[jj:jj+vdim])
                            SR[acc(n3, ci+n1, cj+n1)] = conjhdot(n=vdim, conja=b[ii:ii+vdim],  b=b[jj:jj+vdim])
                        if(cj+n2>= ci+n1):
                            HR[acc(n3, ci+n1, cj+n2)] = conjhdot(n=vdim, conja=bh[ii:ii+vdim], b=c[jj:jj+vdim])
                            SR[acc(n3, ci+n1, cj+n2)] = conjhdot(n=vdim, conja=b[ii:ii+vdim],  b=c[jj:jj+vdim])
                        cj += 1
                    jj += vdim
                cj = 0
                jj = 0
                for j in range(0, neig):
                    if(not locks[j]):
                        if(cj+n0 >= ci+n2):
                            HR[acc(n3, ci+n2, cj+n0)] = conjhdot(n=vdim, conja=ch[ii:ii+vdim], b=a[jj:jj+vdim])
                            SR[acc(n3, ci+n2, cj+n0)] = conjhdot(n=vdim, conja=c[ii:ii+vdim],  b=a[jj:jj+vdim])
                        if(cj+n1 >= ci+n2):
                            HR[acc(n3, ci+n2, cj+n1)] = conjhdot(n=vdim, conja=ch[ii:ii+vdim], b=b[jj:jj+vdim])
                            SR[acc(n3, ci+n2, cj+n1)] = conjhdot(n=vdim, conja=c[ii:ii+vdim],  b=b[jj:jj+vdim])
                        if(cj+n2 >= ci+n2):
                            HR[acc(n3, ci+n2, cj+n2)] = conjhdot(n=vdim, conja=ch[ii:ii+vdim], b=c[jj:jj+vdim])
                            SR[acc(n3, ci+n2, cj+n2)] = conjhdot(n=vdim, conja=c[ii:ii+vdim],  b=c[jj:jj+vdim])
                        cj += 1
                    jj += vdim
                ci += 1
            ii += vdim
        ii = 0
        for i in range(0, n3):
            for j in range(0, i):
                HR[ii + j] = conj(HR[acc(n3, j, i)])
                SR[ii + j] = conj(SR[acc(n3, j, i)])
            ii += n3
    else: ##ortho makes overlap matrix the identity - no need to calculate S
        ii = 0
        for i in range(0, neig):
            if(not locks[i]):
                cj = 0
                jj = 0
                for j in range(0, neig):
                    if(not locks[j]):
                        if(cj+n0 >= ci+n0):
                            HR[acc(n3, ci+n0, cj+n0)] = conjhdot(n=vdim, conja=ah[ii:ii+vdim], b=a[jj:jj+vdim])
                        if(cj+n1 >= ci+n0):
                            HR[acc(n3, ci+n0, cj+n1)] = conjhdot(n=vdim, conja=ah[ii:ii+vdim], b=b[jj:jj+vdim])
                        if(cj+n2 >= ci+n0):
                            HR[acc(n3, ci+n0, cj+n2)] = conjhdot(n=vdim, conja=ah[ii:ii+vdim], b=c[jj:jj+vdim])
                        cj += 1
                    jj += vdim
                cj = 0
                jj = 0
                for j in range(0, neig):
                    if(not locks[j]):
                        if(cj+n0 >= ci+n1):
                            HR[acc(n3, ci+n1, cj+n0)] = conjhdot(n=vdim, conja=bh[ii:ii+vdim], b=a[jj:jj+vdim])
                        if(cj+n1 >= ci+n1):
                            HR[acc(n3, ci+n1, cj+n1)] = conjhdot(n=vdim, conja=bh[ii:ii+vdim], b=b[jj:jj+vdim])
                        if(cj+n2>= ci+n1):
                            HR[acc(n3, ci+n1, cj+n2)] = conjhdot(n=vdim, conja=bh[ii:ii+vdim], b=c[jj:jj+vdim])
                        cj += 1
                    jj += vdim
                cj = 0
                jj = 0
                for j in range(0, neig):
                    if(not locks[j]):
                        if(cj+n0 >= ci+n2):
                            HR[acc(n3, ci+n2, cj+n0)] = conjhdot(n=vdim, conja=ch[ii:ii+vdim], b=a[jj:jj+vdim])
                        if(cj+n1 >= ci+n2):
                            HR[acc(n3, ci+n2, cj+n1)] = conjhdot(n=vdim, conja=ch[ii:ii+vdim], b=b[jj:jj+vdim])
                        if(cj+n2 >= ci+n2):
                            HR[acc(n3, ci+n2, cj+n2)] = conjhdot(n=vdim, conja=ch[ii:ii+vdim], b=c[jj:jj+vdim])
                        cj += 1
                    jj += vdim
                ci += 1
            ii += vdim
        ii = 0
        for i in range(0, n3):
            for j in range(0, i):
                HR[ii + j] = conj(HR[acc(n3, j, i)])
            ii += n3

    er, vr = denseeigsolver(stab=stab, n=n3, H=HR[0:n3*n3], S=SR[0:n3*n3], e=er, v=vr)

    #update the eigenvalues to return
    co = 0
    for i in range(0, neig):
        if(not locks[i]):
            eigs[i] = er[co].real
            co += 1


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
                    vecs[i][j] += vr[cq][co] * complex(locks[k])*a[acc(vdim, k, j)]
                #co = 1*n
                for k in range(0, neig):
                    co += locks[k]
                    vecs[i][j] += vr[cq][co] * complex(locks[k])*b[acc(vdim, k, j)]
                #co = 2*n
                for k in range(0, neig):
                    co += locks[k]
                    vecs[i][j] += vr[cq][co] * complex(locks[k])*c[acc(vdim, k, j)]
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
    for ij in range(0, neigs*npw):
        Xm[ij] = random() + 1j*random()
        Xi[ij] = random() + 1j*random()
    for ii in range(0, neigs*npw, npw):
        Xi[ii:ii+npw] = normalize(n=npw, v=Xi[ii:ii+npw])

    if(not stab): ##need to initialize this only once before 1st call to ritz()
        for ii in range(0, neigs*npw, npw):
            XmH[ii:ii+npw] = action(psi=Xm[ii:ii+npw], H=A)

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
        ii = 0
        for i in range(0, neigs):
            if(not locked[i]):
                XiH[ii:ii+npw] = action(psi=Xi[ii:ii+npw], H=A)
                lam = 0.0 ## = <psi|H|psi>, pure real
                for ij in range(ii, ii+npw):
                    lam += XiH[ij].real*Xi[ij].real + XiH[ij].imag*Xi[ij].imag + 0.0j

                lam = 1.0/lam
                j = 0
                for ij in range(ii, ii+npw):
                    Wi[ij] = (  T[j] * (Xi[ij].real - XiH[ij].real*lam)  ) +\
                        1.0j*(  T[j] * (Xi[ij].imag - XiH[ij].imag*lam)  )
                    j += 1
            ii += npw

        #initialize action on trial wavefunctions (after reorthogonalization if necessary)
        if(stab): ##stable, recalc all (unconverged) psi since ogs() changes them
            Wi, Xi, Xm = ogs2(neig=neigs, vdim=npw,
                              a=Wi, b=Xi, c=Xm, abuf=WiH, bbuf=XiH, cbuf=XmH,
                              locked=locked)
            ii = 0
            for i in range(0, neigs):
                if(not locked[i]):
                    WiH[ii:ii+npw] = action(psi=Wi[ii:ii+npw], H=A)
                    XiH[ii:ii+npw] = action(psi=Xi[ii:ii+npw], H=A)
                    XmH[ii:ii+npw] = action(psi=Xm[ii:ii+npw], H=A)
                ii += npw
        else: ##unstable, only need to calculate action on |W>
            ii = 0
            for i in range(0, neigs):
                if(not locked[i]):
                    WiH[ii:ii+npw] = action(psi=Wi[ii:ii+npw], H=A)
                ii += npw

        #rayleigh-ritz procedure on trial subspace
        e, v = ritz2(neig=neigs, vdim=npw, HR=HR, SR=SR, er=er, vr=vr,
                     a=Wi, b=Xi, c=Xm, ah=WiH, bh=XiH, ch=XmH,
                     nlocks=nlocks, locks=locked,
                     eigs=e, vecs=v, stab=stab)

        ##if no orthogonalization, we can skip an application of <psi|H^*
        ##this needs to be done before updating the locked vectors, though
        if(not stab):
            ii = 0
            for i in range(0, neigs):
                if(not locked[i]):
                    XmH[ii:ii+npw] = deepcopy(XiH[ii:ii+npw])
                ii += npw

        #update convergence params and locked vectors, if necessary
        nlocks, locked, e, eolds, ediffs, endflag = updateconv(neigs=neigs,
                                                               nlocked=nlocks, locks=locked,
                                                               curre=e, olde=eolds, diffe=ediffs,
                                                               stab=stab, tol=tol)
        if(endflag):
            break

        ##prep for next iteration: X(i) -> X(i+1)
        ii = 0
        for i in range(0, neigs):
            if(not locked[i]):
                Xm[ii:ii+npw] = deepcopy(Xi[ii:ii+npw])
                j = 0
                for ij in range(ii, ii+npw):
                    Xi[ij] = v[i][j]
                    j += 1
            ii += npw


    print("finished in", count, "steps,", time() - timestart, "seconds, with", __GLOBAL_ACTION_COUNTER,
          "applications of H|psi>")
    return e, v










seed(267)
neig = 5
size = 700
A = 10*matmake(size)
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
