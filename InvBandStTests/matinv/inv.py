from numpy import allclose, conj, array, transpose
from random import random

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


from numpy import eye, zeros, dot
from numpy.linalg import norm
def ApproxInvDiag1(n, A, nitr):
    I = eye(n)
    Vi = zeros((n, n), dtype=complex) ##initial guess: if A is diagonally dominant, then inv(A) ~ 1/diag(A)

    for i in range(0, n):
        Vi[i][i] = 1/A[i][i]

    #Loop: use newton's method with a (hopefully!) good initial guess V0
    for i in range(0, nitr):
        print(i)
        Vi = 2*Vi - Vi@A@Vi

    return Vi

from numpy.linalg import pinv
def ApproxInvDiag2(n, A):
    #_a = A(ii)
    #_b = A(ij) for i!=j else 0
    _a, _b = zeros((n, n), dtype=complex), zeros((n, n), dtype=complex)
    for i in range(0, n):
        _a[i][i] = A[i][i]
        for j in range(i+1, n):
            _b[i][j] = A[i][j]
            _b[j][i] = conj(A[i][j])
    _an1 = pinv(_a)

    print(norm(_an1 @ _b) < 1)

    return _an1 - _an1@_b@_an1 + _an1@_b@_an1@_b@_an1



from numpy import round
from random import seed
seed(25)
SIZE = 60
from matplotlib import pyplot as plt
H = array(SysMake(SIZE, 1, 10, -2, 2))
plt.matshow(abs(H))
plt.colorbar()
plt.show()
HinvTrue = pinv(H)
print("beg")
HinvTrial = ApproxInvDiag1(SIZE, H, 3)
#HinvTrial = ApproxInvDiag2(SIZE, H)


print((H@HinvTrue -  eye(SIZE)).max())
print((H@HinvTrial - eye(SIZE)).max())
