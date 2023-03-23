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


#random init of a normalized vector
def ropt(m, n):
    r = empty(shape=(m, n), dtype=complex)
    for i in range(0, n):
        for j in range(0, m):
            r[j][i] = random() + IMAG*1j*random()
    for i in range(0, m):
    #normalize ROWS
        r[i,:] /= sqrt(conj(transpose(r[i,:]))@r[i,:])
    return r

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
        #print(j)
        for k in range(0, c):
            temp = temp - conj(transpose(locks[k])) @ V[j] * locks[k]
        for k in range(0, j):
            if(lockedext[k]):
                continue
            #print("  ", k)
            temp = temp - conj(transpose(B[k])) @ V[j] * B[k]
        B[j] = temp / (sqrt(conj(transpose(temp)) @ temp))
    #print(B)
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


def ritz(neig, vdim, A, a, b, c, locks):
    nlocks = 0
    for i in range(0, neig):
        nlocks += locks[i]
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

    e, v = empty(shape=neig, dtype=float), empty(shape=(neig, vdim), dtype=complex)

    co = 0
    for i in range(0, neig):
        if(not locks[i]):
            e[i] = e_[co].real
            v[i,:] = v_[co,:]
            co += 1

    return e, v

numpy.seterr(all='raise')
def locg1(npw, A, neigs, tol):

    #subspace basis vectors
    w = empty(shape=(neigs, npw), dtype=complex)
    xim1 = empty(shape=(neigs, npw), dtype=complex)
    xi = empty(shape=(neigs, npw), dtype=complex)
    for i in range(0, neigs):
        for j in range(0, npw):
            xim1[i][j] = random() + 1j*random()
            xi[i][j] = random() + 1j*random()
    for j in range(0, neigs):
        xim1[j, :] /= sqrt(conj(transpose(xim1[j, :])) @ xim1[j, :])
        xi[j,:] /= sqrt(conj(transpose(xi[j,:]))@xi[j,:])

    #search direction stuff
    lams = empty(shape=neigs, dtype=complex) ##probably real, not complex ...

    #preconditioner setup goes here
    T = eye(npw, dtype=complex)
    for i in range(0, npw):
        T[i][i] *= 1 + 0.1*random() ##garbage preconditioner, just using for consistency checks

    #convergence stuff
    converged = False
    locked = zeros(shape=neigs, dtype=int)   ## \
    eolds = empty(shape=neigs, dtype=float)  ##  |>  all inline w/ one another
    ediffs = zeros(shape=neigs, dtype=float) ## /

    e, v = empty(shape=neigs, dtype=float), empty(shape=(neigs, npw), dtype=complex)
    matmuls = 0
    for i in range(0, 1000):

        for j in range(0, neigs):
            if(not locked[j]):
                Ax = A@transpose(xi[j,:])
                w[j,:] =  T @ (xi[j,:] - transpose(Ax)/(conj(xi[j,:])@Ax))
                matmuls += 1

        w, xi, xim1 = ogs(neig=neigs, vdim=npw, a=w, b=xi, c=xim1, locked=locked)
        e_, v_ = ritz(neig=neigs, vdim=npw, A=A, a=w, b=xi, c=xim1, locks=locked)

        for j in range(0, neigs):
            if(not locked[j]):
                matmuls += 2 ##we already did A@xi, this is for A@w and A@xim1 (actually we could prob. store A@xim1 to avoid re-calculating it
                e[j] = e_[j].real
                v[j,:] = v_[j,:]
            else:
                e[j] = e[j]
                v[j,:] = v[j,:]
        #print(e)
        if(i):
            converged = True
            for j in range(0, neigs):
                ediffs[j] = abs((e[j] - eolds[j]).real)
                if(ediffs[j] > tol):
                    locked[j] = 0
                    converged = False
                else:
                    locked[j] = 1
            #print(i, locked)
        print(i,"\t", [f'{d:.1e}' for d in ediffs])
        if(converged):
            break
        for j in range(0, neigs):
            eolds[j] = e[j].real


        ##convert xi -> xip1
        for j in range(0, neigs):
            if(not locked[j]):
                xim1[j,:] = deepcopy(xi[j,:])
                xi[j,:] = deepcopy(v[j,:])
            else:
                xim1[j,:] = deepcopy(xim1[j,:])
                xi[j,:] = deepcopy(xi[j,:])

        if(i == 10):
            from time import sleep
            sleep(5)
            #exit(5)



    print("finished in", i, "steps and", matmuls, "matmuls")
    return e, v


seed(267)
neig = 4
size = 100
A = matmake(size)
#print(A)
print("begin")

ecg, vcg = locg1(npw=size, A=A, neigs=neig, tol=1e-4)
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
