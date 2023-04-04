from random import random, seed
from numpy import array, transpose, allclose, conj, dot, sqrt
from copy import deepcopy

def SysMake(size, min, max):
    #Make coefficient matrix A
    mat = [[None for i in range(0, size)] for j in range(0, size)]

    for i in range(0, size):
        mat[i][i] = min + random()*(max-min)
        for j in range(i + 1, size):
            re = min + random()*(max-min)
            im = min + random()*(max-min)
            mat[i][j] = re + 1j*im
            mat[j][i] = re - 1j*im

    ##and check if it is really hermitian
    if(not allclose(array(mat), conj(transpose(mat)))):
        print("I suck")
        exit(1)

    #Make resultant vector b
    res = [None for i in range(0, size)]
    for i in range(0, size):
        re = min + random() * (max - min)
        im = min + random() * (max - min)
        res[i] = re + 1j*im

    return mat, res

def CAV(z): ##absolute value of a complex number
    return sqrt(z.real*z.real + z.imag*z.imag)



from numpy import array
#Based on https://github.com/fredrik-johansson/mpmath/blob/master/mpmath/matrices/eigen_symmetric.py
def QR(n, A, giveVecs=False, smallTol=1e-8, itrLim=30):
    def IndTrns(n, i, j):
        return n*i + j

    a_ = [None for i in range(0, n*n)]
    for i in range(0, n):
        for j in range(0, n):
            a_[IndTrns(n, i, j)] = A[i][j]

    ##allocate tmp arrays
    A = array(a_)
    d = array([0. + 0j for i in range(0, n)])
    e = array([0. + 0j for i in range(0, n)])
    t = array([0. + 0j for i in range(0, n)])


    ##################################
    ### TRIDIAG POTION OF THE ALGO ###
    ##################################
    t[n-1] = 1.
    for i in range(n-1, 0, -1):
        l = i - 1

        ##vector scaling
        sc = 0.
        for j in range(0, i):
            ji = IndTrns(n, j, i)
            sc += abs(A[ji].real) + abs(A[ji].imag)

        ###early term if scale is small - no need for useless, expensive calculations
        if(abs(sc) < smallTol):
            e[i] = 0.
            d[i] = 0.
            t[l] = 1.
            continue

        scinv = 1./sc


        ##stopping criteria for last iteration
        if(i == 1):
            f0 = A[IndTrns(n, l, i)]
            f1 = abs(f0)
            e[i] = f1
            d[i] = 0.
            if(f1 > smallTol):
                t[l] = t[i]*f0/f1
            else:
                t[l] = t[i]
            continue


        ##householder transformation setup
        h = 0.
        for j in range(0, i):
            ji = IndTrns(n, j, i)
            A[ji] *= scinv
            h += A[ji].real*A[ji].real + A[ji].imag*A[ji].imag

        f0 = A[IndTrns(n, l, i)]
        f1 = abs(f0) ##f1 = CAV(f0) ???
        g = sqrt(h)
        h += g*f1
        e[i] = sc*g
        ttmp = None
        if(f1 > smallTol):
            f0 /= f1
            ttmp = -t[i]*f0
            g *= f0
        else:
            ttmp = -t[i]
        A[IndTrns(n, l, i)] += g
        f0 = 0.


        ##householder transformation application
        for j in range(0, i):
            if(giveVecs):
                A[IndTrns(n, i, j)] = A[IndTrns(n, j, i)] / h

            ###A dot U
            g = 0 + 0j
            for k in range(0, j+1):
                g += conj(A[IndTrns(n, k, j)]) * A[IndTrns(n, k, i)]
            for k in range(j+1, i):
                g += A[IndTrns(n, j, k)] * A[IndTrns(n, k, i)]

            ###P
            t[j] = g/h
            f0 += conj(t[j])*A[IndTrns(n, j, i)]

        hhparam = 0.5*f0/h

        ###reduce A, get Q
        for j in range(0, i):
            f0 = A[IndTrns(n, j, i)]
            g = t[j] - hhparam*f0
            t[j] = g

            for k in range(0, j + 1):
                A[IndTrns(n, k, j)] -= conj(f0)*t[k] + conj(g)*A[IndTrns(n, k, i)]

        t[l] = ttmp
        d[i] = h


    #################################
    ### EIGVAL POTION OF THE ALGO ###
    #################################
    def safehypot(a, b): ##aux function for (a^2 + b^2)^(1/2) w/o under/overflow
        #return sqrt(a*a + b*b)
        absa, absb = abs(a), abs(b)
        if(absa > absb):
            return absa*(1. + (absb/absa)**(2))**(1/2)
        if(absb == 0.0):
            return 0.0
        return absb*(1. + (absa/absb)**(2))**(1/2)

    #setup arrays: shift {e} to the left, make d explicitly real-valued
    for i in range(1, n):
        e[i-1] = e[i]
    e[n-1] = 0.

    d[0] = 0.
    for i in range(0, n):
        ii = IndTrns(n, i, i)
        tmp = d[i]
        d[i] = A[ii].real
        A[ii] = tmp
    #note that at this point, all values are strictly real

    B = [[0. + 0j for j in range(0, n)] for i in range(0, n)]
    for i in range(0, n):
        B[i][i] = 1. + 0j
    #from copy import deepcopy
    #B = deepcopy(A)
    B = array(B)

    for l in range(0, n):
        itrCount = 0


        while(1):
            ##grab a small off-diagonal element
            m = l
            while(1):
                if(m + 1 == n):
                    break
                if(  abs(e[m]) < smallTol*(abs(d[m]) + abs(d[m+1]))  ):
                    break
                m += 1
            if(m == l):
                break

            if(itrCount >= itrLim):
                print("I suck")
                exit(2)

            itrCount += 1

            ##shift
            g = 0.5*(d[l+1] - d[l])/e[l]
            r = safehypot(g, 1.0)
            s = g - r if g < 0.0 else g + r
            g = d[m] - d[l] + e[l] / s

            ##plane-into-givens rotations to get back to tridiagonal form
            s, c, p = 1., 1., 0.
            for i in range(m-1, l-1, -1):
                f = s*e[i]
                b = c*e[i]

                ###slightly better than the standard QL algo
                if(abs(f) > abs(g)):
                    c = g/f
                    r = safehypot(c, 1.)
                    e[i+1] = f*r
                    s = 1/r
                    c *= s
                else:
                    s = f/g
                    r = safehypot(s, 1.)
                    e[i+1] = g*r
                    c = 1/r
                    s *= c

                g = d[i+1] - p
                r = (d[i] - g)*s + 2.*c*b
                p = s*r
                d[i+1] = g + p
                g = c*r - b

                ##Get eigenvectors if necessary
                if(giveVecs):
                    for w in range(0, n):
                        f = B[w][i+1]
                        B[w][i+1] = s*B[w][i] + c*f
                        B[w][i]   = c*B[w][i] - s*f

            #if(abs(r) <= smallTol and i >= l):
            #    continue
            d[l] -= p
            e[l] = g
            e[m] = 0.

    #sorting: NOTE: BUBBLE SORT - DON'T USE THIS GARBAGE
    for ii in range(1, n):
        i = ii - 1
        k = i
        p = d[i]
        for j in range(ii, n):
            if(d[j] >= p):
                continue
            k = j
            p = d[k]
        if(k == i):
            continue
        d[k] = d[i]
        d[i] = p

        if(giveVecs):
            for w in range(0, n):
                p = B[w][i]
                B[w][i] = B[w][k]
                B[w][k] = p

    if(not giveVecs):
        return d ##d holds the (unsorted) eigenvalues

    for i in range(0, n):
        for k in range(0, n):
            B[k][i] *= t[k]

    for i in range(0, n):
        if(A[IndTrns(n, i, i)] != 0):
            for j in range(0, n):
                g = 0
                for k in range(0, i):
                    g += conj(A[IndTrns(n, i, k)]) * B[k][j]
                for k in range(0, i):
                    B[k][j] -= g*A[IndTrns(n, k, i)]

    return array(d), array(B)

"""
**************************************************************************************************************
*****  TESTING ***********************************************************************************************
**************************************************************************************************************
"""

from numpy import round, dot, array, allclose
import time

SIZE = 3
MIN, MAX = -10, +10
seed(69101)
A, b = SysMake(SIZE, MIN, MAX)
A, b = array(A), array(b)

from numpy.linalg import eig, eigh, norm
print("EIGH")
eh, vh = eigh(A)
print(eh)
print(vh)
print(allclose(dot(A, vh) - eh*vh, 0.0 + 0.0j, 1e-6, 1e-6))


print("EIG")
e, v = eig(A)
print(e.real)
print(v)
print(allclose(dot(A, v) - e*v, 0.0 + 0.0j, 1e-6, 1e-6))


print("QR")
eigs, vecs = QR(SIZE, A, True)
print(eigs.real)
print(vecs)
print(allclose(dot(A, vecs) - eigs*array(vecs), 0.0 + 0.0j, 1e-6, 1e-6))
