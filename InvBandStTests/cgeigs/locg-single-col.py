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
def ropt(n):
    r = empty(shape=(n, 1), dtype=complex)
    for i in range(0, n):
        r[i][0] = random() + IMAG*1j*random()
    r /= sqrt(conj(transpose(r))@r)
    return r

#ordinary grahm schmidt
import numpy
def ogs(n, a, b, c):
    V = empty(shape=(n, 3), dtype=complex)
    for i in range(0, n):
        V[i][0] = a[i]
        V[i][1] = b[i]
        V[i][2] = c[i]

    V = transpose(V)
    num_vecs = V.shape[0]
    B = numpy.zeros_like(V)

    for j in range(0, num_vecs):
        temp = V[j]
        for k in range(0, j):
            temp = temp - conj(transpose(B[k]))@V[j]*B[k]
        B[j] = temp / sqrt(conj(transpose(temp))@temp)
    V = deepcopy(transpose(B))

    return V[:,0], V[:,1], V[:,2]

def matmake(n):
    R = empty(shape=(n, n), dtype=complex)

    for i in range(0, n):
        R[i][i] = 100*(2*random() - 1) + 0.0j
        for j in range(i+1, n):
            R[i][j] = 1 * ( (2*random() - 1) + IMAG*1j*(2*random() - 1))
            R[j][i] = R[i][j].real - 1j*R[i][j].imag

    return R


def checkcondition(n, a, b, c):
    V = empty(shape=(n, 3), dtype=complex)
    for i in range(0, n):
        V[i][0] = a[i]
        V[i][1] = b[i]
        V[i][2] = c[i]

    return numpy.linalg.cond(conj(transpose(V))@A@V, 2)

def ritz(n, A, a, b, c):
    V = empty(shape=(n, 3), dtype=complex)
    for i in range(0, n):
        V[i][0] = a[i]
        V[i][1] = b[i]
        V[i][2] = c[i]
    #V[:,0] = deepcopy(reshape(a, newshape=(n, 1)))
    #V[:,1] = deepcopy(b)
    #V[:,2] = deepcopy(c)

    m = 1 ##1 eigenvalue
    H = conj(transpose(V))@A@V
    #print("cond ", numpy.linalg.cond(H, 2))
    e, u = eig(H)
    sind = e.argsort()[:m]
    e = e[sind]
    u = u[:,sind]
    v = V@u
    return e, v

def locg1(A, tol):
    npw = A.shape[0]

    w, xim1, xi = ropt(npw), ropt(npw), ropt(npw)
    xi /= sqrt(conj(transpose(xi)) @ xi)
    T = eye(npw, dtype=complex)
    #for i in range(0, npw):
    #    T[i][i] = 1/A[i][i]

    e, v = None, None
    eold = 2*tol
    for i in range(0, 10000):
        lam = conj(transpose(xi))@(A@xi)  #/ (conj(transpose(xi))@xi)
        r = xi - A@xi/lam
        w = T@r
        print(numpy.linalg.norm(w))

        w, xi, xim1 = ogs(npw, w, xi, xim1)
        e, v = ritz(npw, A, w, xi, xim1)

        #print(i, abs(e[0] - eold))
        if(abs(e[0] - eold) < tol):
            break
        eold = e[0]

        ##convert xi -> xip1
        xim1 = deepcopy(xi)
        xi = deepcopy(v)

    print("finished in", i, "steps")
    return e, v


"""
from scipy.linalg import lstsq
def locg2(A, tol):
    npw = A.shape[0]

    w, xi = ropt(npw), ropt(npw)
    pi = zeros(shape=(npw, 1), dtype=complex)
    T = eye(npw, dtype=complex)
    #for i in range(0, npw):
    #    T[i][i] = 1/A[i][i]

    e, v = None, None
    eold = 2*tol
    for i in range(0, 10000):
        lam = conj(transpose(xi))@(A@xi) #/ (conj(transpose(xi))@xi)
        r = xi - A@xi/lam
        w = T@r

        #w /= sqrt(conj(transpose(w))@w)
        xi /= sqrt(conj(transpose(xi))@xi)
        e, v = ritz(npw, A, w, xi, pi)

        #print(i, abs(e[0] - eold))
        if(abs(e[0] - eold) < tol):
            break
        eold = e[0]


        M = empty(shape=(npw, 2), dtype=complex)
        for i in range(0, npw):
            M[i][0] = xi[i]
            M[i][1] = pi[i]
        res = lstsq(a=M, b=v-w)[0]

        pi = w + res[1]*pi
        #pi /= sqrt(conj(transpose(pi)) @ pi)
        xi = deepcopy(v)

    print("finished in", i, "steps")
    return e, v
"""
seed(267)
neig = 1
A = matmake(123)
#print(A)

print("actual ...")
ereal, vreal = eig(A)
print(sorted(ereal)[:neig])

print("locg1 ...")
ecg, vcg = locg1(A=A, tol=1e-8)
print(sorted(ecg))
#print("locg2 ...")
#ecg, vcg = locg2(A=A, tol=1e-8)
#print(sorted(ecg))
