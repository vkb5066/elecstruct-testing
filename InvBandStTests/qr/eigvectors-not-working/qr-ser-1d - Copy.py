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


def IndexTransform(n, row, col):
    return row*n + col
def MatrixTransform(A):
    n = len(A)
    ret = [None for i in range(0, n*n)]
    for i in range(0, n):
        for j in range(0, n):
            ret[IndexTransform(n=n, row=i, col=j)] = A[i][j]
    return ret


#Based on https://github.com/fredrik-johansson/mpmath/blob/master/mpmath/matrices/eigen_symmetric.py
def QR(n, A, giveVecs=False, smallTol=1e-8, itrLim=30):
    from numpy import array

    #Transform A into new A
    A = array(MatrixTransform(A))



    ##allocate tmp arrays
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
            ij = IndexTransform(n, j, i)
            sc += abs(A[ij].real) + abs(A[ij].imag)

        ###early term if scale is small - no need for useless, expensive calculations
        if(abs(sc) < smallTol):
            e[i] = 0.
            d[i] = 0.
            t[l] = 1.
            continue

        scinv = 1./sc


        ##stopping criteria for last iteration
        if(i == 1):
            f0 = A[IndexTransform(n, l, i)]
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
            ij = IndexTransform(n, j, i)
            A[ij] *= scinv
            h += A[ij].real*A[ij].real + A[ij].imag*A[ij].imag

        il = IndexTransform(n, l, i)
        f0 = A[il]
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
        A[il] += g
        f0 = 0.


        ##householder transformation application
        for j in range(0, i):
            if(giveVecs):
                A[IndexTransform(n, i, j)] = A[IndexTransform(n, j, i)] / h

            ###A dot U
            g = 0 + 0j
            for k in range(0, j+1):
                g += conj(A[IndexTransform(n, k, j)]) * A[IndexTransform(n, k, i)]
            for k in range(j+1, i):
                g += A[IndexTransform(n, j, k)] * A[IndexTransform(n, k, i)]

            ###P
            t[j] = g/h
            f0 += conj(t[j])*A[IndexTransform(n, j, i)]

        hhparam = 0.5*f0/h

        ###reduce A, get Q
        for j in range(0, i):
            f0 = A[IndexTransform(n, j, i)]
            g = t[j] - hhparam*f0
            t[j] = g

            for k in range(0, j + 1):
                A[IndexTransform(n, k, j)] -= conj(f0)*t[k] + conj(g)*A[IndexTransform(n, k, i)]

        t[l] = ttmp
        d[i] = h



    #################################
    ## INTERMED POTION OF THE ALGO ##
    #################################
    #This portion is responsible for convient shifts and, optionally,
    #the expensive portion associated with returning the eigenvectors

    ##shift
    for i in range(1, n):
        e[i-1] = e[i]
    e[n-1] = 0.

    ##make d real-valued
    d[0] = 0.
    for i in range(0, n):
        ii = IndexTransform(n, i, i)
        tmp = d[i]
        d[i] = A[ii].real
        A[ii] = tmp

    ##setup eigenvectors if necessary
    B = None
    if(giveVecs):
        B = [0. for i in range(0, n*n)]
        for i in range(0, n):
            B[IndexTransform(n, i, i)] = 1. + 0.j



    #################################
    ### EIGVAL POTION OF THE ALGO ###
    #################################
    # note that at this point, all values are strictly real
    def safehypot(a, b): ##aux function for (a^2 + b^2)^(1/2) w/o under/overflow
        #return sqrt(a*a + b*b)
        absa, absb = abs(a), abs(b)
        if(absa > absb):
            return absa*(1. + (absb/absa)**(2))**(1/2)
        if(absb == 0.0):
            return 0.0
        return absb*(1. + (absa/absb)**(2))**(1/2)


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
                ##TODO: this sets eigenvectors as columns, not rows ...
                ## ... bad array accesses and such
                if(giveVecs):
                    for w in range(0, n):
                        wi = IndexTransform(n, w, i)
                        wip1 = IndexTransform(n, w, i+1)
                        #wi = IndexTransform(n, i, w)
                        #wip1 = IndexTransform(n, i+1, w)
                        f = B[wip1]
                        B[wip1] = s*B[wi] + c*f
                        B[wi]   = c*B[wi] - s*f

            #if(abs(r) <= smallTol and i >= l):
            #    continue
            d[l] -= p
            e[l] = g
            e[m] = 0.

    if(not giveVecs):
        return d ##d holds the (unsorted) eigenvalues



    #################################
    ### EIGVEC POTION OF THE ALGO ###
    #################################
    #if we're here, we want the eigenvectors
    if(giveVecs):
        for i in range(0, n):
            for k in range(0, n):
                B[IndexTransform(n, k, i)] *= t[k]

        for i in range(0, n):
            if(abs(A[IndexTransform(n, i, i)]) > smallTol):
                for j in range(0, n):
                    g = 0. + 0j
                    for k in range(0, i):
                        g += conj(A[IndexTransform(n, i, k)])*B[IndexTransform(n, k, j)]
                    for k in range(0, i):
                        B[IndexTransform(n, k, j)] -= g*A[IndexTransform(n, k, i)]

    return d, B



"""
**************************************************************************************************************
*****  TESTING ***********************************************************************************************
**************************************************************************************************************
"""

from numpy import dot, array, round
def Check(n, A, v, e, tol=1e-4):
    res = dot(A, v) - e*array(v)
    return abs(res) < tol

SIZE = 3
MIN, MAX = -10, +10
seed(6910)
A, b = SysMake(SIZE, MIN, MAX)
A, b = array(A), array(b)

from numpy.linalg import eigh
print("numpy...")
eValsNumpy, eVecsNumpy = eigh(a=A)
for i in range(0, SIZE):
    v = [None for j in range(0, SIZE)]
    for j in range(0, SIZE):
        v[j] = eVecsNumpy[j][i]
    v = array(v)
    print(round(eValsNumpy[i], 2), (round(v, 3)))
    print(Check(SIZE, A, v, eValsNumpy[i]))

print("\n\nmine...")
eVals, eVecs = QR(n=SIZE, A=A, giveVecs=True)
for i in range(0, SIZE):
    v = [None for j in range(0, SIZE)]
    for j in range(0, SIZE):
        v[j] = eVecs[IndexTransform(SIZE, i, j)]
    v = array(v)
    print(round(eVals[i], 2), round(v, 3))
    print(Check(SIZE, A, v, eVals[i]))

#print(round(eVecsNumpy, 2))
