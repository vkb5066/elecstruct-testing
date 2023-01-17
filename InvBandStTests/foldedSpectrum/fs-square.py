from random import random, seed
from numpy import array, transpose, allclose, conj, dot, sqrt, vdot, zeros, round, eye
from numpy.linalg import inv, eig
from copy import deepcopy




"""
bro
"""

def IsHerm(A):
    return allclose(conj(transpose(A)), A, 1e-6, 1e-6)

def SysMake(size, min, max, mino, maxo):
    #Make coefficient matrix A
    mat = [[None for i in range(0, size)] for j in range(0, size)]

    for i in range(0, size):
        mat[i][i] = min + random()*(max-min) + 0.j
        for j in range(i + 1, size):
            re = mino + random()*(maxo-mino)
            im = mino + random()*(maxo-mino)
            mat[i][j] = re + 1j*im
            mat[j][i] = re - 1j*im

    ##and check if it is really hermitian
    if(not allclose(array(mat), conj(transpose(mat)))):
        print("I suck")
        exit(1)

    return mat







"""
**************************************************************************************************************
*****  TESTING ***********************************************************************************************
**************************************************************************************************************
"""


N_EIGS = 3
SIZE = 4
MIN, MAX = 10, 150
MINO, MAXO = -3, 2
seed(111)


A = SysMake(size=SIZE, min=MIN, max=MAX, mino=MINO, maxo=MAXO)
print("orig herm: ", IsHerm(A))
A = array(A)

#Base, original eigenvalues
va, ve = eig(A)
sind = va.argsort()[:N_EIGS + SIZE - N_EIGS]
va = va[sind]
ve = ve[:,sind]

div = 2

#Folded Spectrum ...
eref = 100
I = eye(SIZE)
AF = dot(A - eref*I, A - eref*I)
#AF = dot(transpose(conj(A - eref*I)), A - eref*I)
#print(dot(A - (eref+0.1)*I, A - (eref+0.1)*I))
#print(dot(A - (eref-0.1)*I, A - (eref-0.1)*I))
from numpy import ceil
print(round(A, 0))
print(round(AF, 0))

print("folded herm:", IsHerm(A))
#... unshifted eigenvalues
vafu, vefu = eig(AF)
sind = vafu.argsort()[:N_EIGS]
vafu = vafu[sind]
vefu = vefu[:,sind]
#... shifted eigenvalues
vaf, vef = deepcopy(vafu), deepcopy(vefu)
vaf2 = deepcopy(vaf)
vef2 = deepcopy(vef)
for i in range(0, N_EIGS):
    vaf[i] = +(abs(vaf[i].real))**(1/2)  + eref
    vaf2[i]= -(abs(vaf2[i].real))**(1/2) + eref



print("orig eigs:")
for e in range(0, N_EIGS + SIZE - N_EIGS):
    print(round(va[e], 5))#, '(', round(abs(va[e].real - eref), 5), ')')

print("folded eigs (u), (folded eigs)^(1/div) (u):")
for e in range(0, N_EIGS):
    print(round(vafu[e], 5), round((vafu[e])**(1/div), 5))

print("folded eigs (s):")
for e in range(0, N_EIGS):
    print(round(vaf[e], 5), "or", round(vaf2[e], 5))

print("Diagonals of A:")
for i in range(0, SIZE):
    print(round(A[i][i], 0))


#print("\n\n\n")
#from numpy.linalg import norm
#print("orig vecs:")                                  #\
#print(round(ve, 2))                                  # \
#print("folded vecs:") #\                             #  \ seem to be consistent, just out of order here
#print(round(vef, 2))  # \ seem to always be equal?   #  / obviously
#print("or")           # / by which i mean i've       # /
#print(round(vef2, 2)) #/  checked one (1) matrix     #/

#We can use the eigenvectors to determine which eigenvalue is the right one
#and hope that theres somehow a better way
#it actually might be better to just do folded spectrum on eref +/- eps with eps an order of magnitude
#larger than the tolerance for the davidson method
#for i in range(0, N_EIGS):
#    print(round(A@vef[:,i] - vaf[i]*vef[:,i], 2), "vs", round(A@vef[:,i] - vaf2[i]*vef[:,i], 2))


#Optimization note: for checking which of the two eigenvalues is the right one, you'd naively
#just check A@x - lambda@x ~ 0.  But you really just need to check the first entry of A@x - lambda@x
#instead of making the entire vector (so long as the first entry of lambda1@x !~ lambda2@x, in which case
#you'd just move on to the next entry, etc.).
#Basically, compute and store [(lambda 1) @ x] -> l1x as well as [(lambda 2) @ x] -> l2x
#then, [(A row i) @ (x)] - l1x -> eps1 and [(A row i) @ (x)] - l2x -> eps2
#followed by a check that eps1 !~ eps2 (it is unlikley that they will be close).  If they're not
#close, then chooce the eigenvalue that is associated with the smaller eps and quit.  If they are
#close, increment i and try again from the step on the line beginning with "then, [".
print(N_EIGS, "true eigs, as determined by fs, closest to", str(eref) + ":")
for e in range(0, N_EIGS):
    eps1, eps2 = sum(abs(A@vef[:,e] - vaf[e]*vef[:,e])), sum(abs(A@vef[:,e] - vaf2[e]*vef[:,e]))
    if(eps1 < eps2):
        print(round(vaf[e], 5))
    else:
        print(round(vaf2[e], 5))
