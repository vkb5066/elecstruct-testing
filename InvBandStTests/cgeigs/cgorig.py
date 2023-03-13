from numpy import array, empty, zeros, cos, sin, arctan, sqrt
from numpy import transpose, eye
from numpy.linalg import eig
from random import seed, random
from copy import deepcopy

#this shit dont work yo


def matmake(n):
    R = empty(shape=(n, n), dtype=complex)

    for i in range(0, n):
        R[i][i] = (i+8)**(2/3) + 10*random()
        for j in range(i+1, n):
            R[i][j] = sqrt(i+j+2) - int(sqrt(i+j+2)) - 0.5 + 0j#+ 10j*(2*random() - 1)
            R[j][i] = R[i][j].real  + 0j#- R[i][j].imag

    return R

##original grahm schmidt
import numpy
def ogs(V):
    V = transpose(V)
    num_vecs = V.shape[0]
    B = numpy.zeros_like(V)

    for j in range(0, num_vecs):
        temp = V[j]
        for k in range(0, j):
            temp = temp - B[k].T @ V[j] * B[k]
        B[j] = temp / numpy.linalg.norm(temp)

    return transpose(B)

#orthogonalizes column vector @ index i to the (assumed to be orthonormal) column vectors indixed as 0 -> i-1
def ortho(V, i):
    for j in range(0, i):
        of = transpose(V[:,j]) @ V[:,i]
        V[:,i] = V[:,i] - of*V[:,j]
    V[:,i] /= transpose(V[:,i]) @ V[:,i]

    return V[:,i]

def ritz(A, V):
    m = V.shape[1]
    H = transpose(V)@A@V
    e, u = eig(H)
    sind = e.argsort()[:m]
    e = e[sind]
    u = u[:,sind]
    v = V@u
    return e, v

def pcg(A, niter, nblck, nline, linetol):
    npw = A.shape[0]


    lam = empty(shape=nblck, dtype=float)
    X = empty(shape=(npw, nblck), dtype=complex)
    rj =   empty(shape=npw, dtype=complex) ##complex??
    rjp1 = empty(shape=npw, dtype=complex) ##complex??
    dj =   empty(shape=npw, dtype=complex) ##complex??
    djp1 = empty(shape=npw, dtype=complex) ##complex??
    beta = 0.0 + 0.0j ##complex??

    #init
    for i in range(0, npw):
        for j in range(0, nblck):
            X[i][j] = random() + 0j#+ 1j*random()
    X = ogs(X)

    P = eye(npw, npw)
    #P = empty(shape=(npw, npw), dtype=complex)
    #for i in range(0, npw):
    #    P[i][i] = 1/A[i][i]
    #    for j in range(i+1, npw):
    #        P[i][j] = 0.0 + 0.0j
    #        P[j][i] = 0.0 + 0.0j
        #print(P[i][i])

    #outer loop
    for i in range(0, niter):
        #block optimization
        for m in range(0, nblck):
            Xm = deepcopy(ogs(X[:,0:m+1])[:,m])
            #Xm = ortho(V=deepcopy(X), i=m)
            ax = A@Xm
            #inner loop
            for j in range(0, nline):
                lamm = transpose(Xm)@A@Xm
                tsttol = transpose(A@Xm - lamm*Xm)@(A@Xm - lamm*Xm)
                #print(i, m, j, tsttol)
                if(tsttol < linetol):
                    break
                rjp1 = (eye(npw) - Xm@transpose(Xm)) @ ax
                if(j > 0):
                    beta = (rjp1@P@rjp1) / (rj@P@rj)
                else:
                    beta = 0.0 + 0.0j ##complex??
                djp1 = -(P@rjp1) + beta*dj
                djp1 = (eye(npw) - Xm@transpose(Xm))@djp1
                gamma = 1/sqrt(transpose(djp1)@djp1)
                theta = 0.5*abs(arctan((2*gamma*djp1@ax)/(lamm-gamma*gamma*transpose(djp1)@A@djp1)))
                Xm = cos(theta)*Xm + sin(theta)*gamma*djp1
                ax = cos(theta)*ax + sin(theta)*gamma*A@djp1

                rj = deepcopy(rjp1)
                dj = deepcopy(djp1)
            X[:,m] = deepcopy(Xm)

        lam, X = ritz(A=A, V=deepcopy(X))
        print(lam[0])
    return lam, X

seed(26)
neig = 1
A = matmake(50)
#print(A)


ereal, vreal = eig(A)
print(sorted(ereal)[:neig])

ecg, vcg = pcg(A=A, niter=100, nblck=neig, nline=20, linetol=0.001)
print(sorted(ecg))
