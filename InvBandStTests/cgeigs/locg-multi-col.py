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
    for i in range(0, m):
        for j in range(0, n):

            r[i][j] = random() + IMAG*1j*random()
    for i in range(0, n):
    #normalize COLS
        r[:,i] /= sqrt(conj(transpose(r[:,i]))@r[:,i])
    return r

#ordinary grahm schmidt
import numpy
def ogs(neig, vdim, a, b, c, locked):
    #print(locked)
    V = empty(shape=(vdim, 3*neig), dtype=complex)
    lockedext = empty(shape=3*neig)
    for i in range(0, neig):
        V[:,0*neig + i] = a[:,i]
        V[:,1*neig + i] = b[:,i]
        V[:,2*neig + i] = c[:,i]
        lockedext[i + 0*neig] = locked[i]
        lockedext[i + 1*neig] = locked[i]
        lockedext[i + 2*neig] = locked[i]

    V = transpose(V)
    B = numpy.zeros_like(V)

    if(0):
        for j in range(0, 3*neig):
            temp = V[j]
            for k in range(0, j):
                temp = temp - conj(transpose(B[k])) @ V[j] * B[k]
            B[j] = temp / (sqrt(conj(transpose(temp)) @ temp))
    else:
        for j in range(0, 3*neig):
            if(lockedext[j]):
                B[j,:] = deepcopy(V[j,:])
            temp = V[j]
            for k in range(0, j):
                temp = temp - conj(transpose(B[k])) @ V[j] * B[k]
            B[j] = temp / (sqrt(conj(transpose(temp)) @ temp))

    V = deepcopy(transpose(B))

    return V[:,0*neig:neig], V[:,1*neig:2*neig], V[:,2*neig:3*neig]





def matmake(n):
    R = zeros(shape=(n, n), dtype=complex)

    for i in range(0, n):
        R[i][i] = (i%150) + 1 + 100*(2*random() - 1) + 0.0j
        for j in range(i+1, n):
            R[i][j] = 1*((2*random() - 1) + IMAG*1j*(2*random() - 1))
            R[j][i] = R[i][j].real - 1j*R[i][j].imag

    return R


def ritz(neig, vdim, A, a, b, c):
    V = empty(shape=(vdim, 3*neig), dtype=complex)
    for i in range(0, neig):
        V[:,0*neig + i] = a[:,i]
        V[:,1*neig + i] = b[:,i]
        V[:,2*neig + i] = c[:,i]

    H = conj(transpose(V))@A@V
    e, u = eig(H)
    sind = e.argsort()[:neig]
    e = e[sind]
    u = u[:,sind]
    v = V@u
    return e, v

numpy.seterr(all='raise')
def locg1(A, neigs, tol):
    npw = A.shape[0]

    #subspace basis vectors
    w = ropt(m=npw, n=neigs)
    xim1 = ropt(m=npw, n=neigs)
    xi = ropt(m=npw, n=neigs)
    for j in range(0, neigs):
        xi[:,j] /= sqrt(conj(transpose(xi[:,j]))@xi[:,j]) ##only xi needs normalized before use

    #search direction stuff
    lams = empty(shape=neigs, dtype=complex) ##probably real, not complex ...

    #preconditioner setup goes here
    T = eye(npw, dtype=complex)

    #convergence stuff
    converged = False
    locked = zeros(shape=neigs, dtype=int)   ## \
    eolds = empty(shape=neigs, dtype=float)  ##  |>  all inline w/ one another
    ediffs = zeros(shape=neigs, dtype=float) ## /

    e, v = None, None
    for i in range(0, 10000):

        for j in range(0, neigs):
            if(not locked[j]): #                                 tol^2 may be unnecessary in c code
                lams[j] = conj(transpose(xi[:,j]))@(A@xi[:,j]) + 0*tol*tol  #/ (conj(transpose(xi[:,j]))@xi[:,j])
                w[:,j] = T@(xi[:,j] - A@xi[:,j]/lams[j])
                #print(numpy.linalg.norm(w[:,j])) ##may be an indicator of convergence

        w, xi, xim1 = ogs(neig=neigs, vdim=npw, a=w, b=xi, c=xim1, locked=locked)
        e, v = ritz(neig=neigs, vdim=npw, A=A, a=w, b=xi, c=xim1)

        if(i):
            converged = True
            for j in range(0, neigs):
                ediffs[j] = abs((e[j] - eolds[j]).real)
                if(ediffs[j] > tol):
                    locked[j] = 0
                    converged = False
                else:
                    locked[j] = 1
        print(i,"\t", [f'{d:.0e}' for d in ediffs])
        if(converged):
            break
        for j in range(0, neigs):
            eolds[j] = e[j].real

        ##convert xi -> xip1
        for i in range(0, neigs):
            if(not locked[i]):
                xim1[:,i] = deepcopy(xi[:,i])
                xi[:,i] = deepcopy(v[:,i])
            else:
                xim1[:,i] = deepcopy(xim1[:,i])
                xi[:,i] = deepcopy(xi[:,i])

    print("finished in", i, "steps")
    return e, v


seed(267)
neig = 5
A = matmake(300)
#print(A)
print("begin")

ecg, vcg = locg1(A=A, neigs=neig, tol=1e-8)
print("locg1 ...")
print(sorted(e.real for e in ecg))

print("actual ...")
ereal, vreal = eig(A)
print(sorted(e.real for e in ereal)[:neig])





"""
this is cringe
def _ogs(neig, vdim, a, b, c, locked):
    #print(locked)
    V = empty(shape=(vdim, 3*neig), dtype=complex)

    mapa = [] ##mapa[i] = j means that the column V[j] corresponds to a[i] originally

    ##all locked vectors go from 0 to 3*nlocks ...
    nlocks = 0
    li = 0
    for i in range(0, neig):
        if(locked[i]):
            nlocks += 1
            V[:,li+0] = a[:,i]
            V[:,li+1] = b[:,i]
            V[:,li+2] = c[:,i]
            li += 3
    ##... and the rest from 3*nlocks to 3*neigs
    for i in range(0, neig):
        if(not locked[i]):
            #print("", li)
            V[:,li+0] = a[:,i]
            V[:,li+1] = b[:,i]
            V[:,li+2] = c[:,i]
            li += 3

    V = transpose(V)
    B = numpy.zeros_like(V)

    for i in range(0, 3*nlocks):
        B[i, :] = deepcopy(V[i, :])
    for j in range(3*nlocks, 3*neig):
        temp = V[j]
        for k in range(0, j):
            temp = temp - conj(transpose(B[k]))@V[j]*B[k]
        B[j] = temp / (sqrt(conj(transpose(temp))@temp))
    V = deepcopy(transpose(B))

    li = 0
    for i in range(0, neig):
        if(locked[i]):
            a[:,i] = V[:,li+0]
            b[:,i] = V[:,li+1]
            c[:,i] = V[:,li+2]
            li += 3
    ##... and the rest from 3*nlocks to 3*neigs
    for i in range(0, neig):
        if(not locked[i]):
            #print("", li)
            a[:,i] = V[:,li+0]
            b[:,i] = V[:,li+1]
            c[:,i] = V[:,li+2]
            li += 3

    return a, b, c
"""
