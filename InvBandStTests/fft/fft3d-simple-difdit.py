#Convolution of two functions using fft


from numpy import empty, linspace, sin, cos, pi, transpose, exp
from scipy.fft import fft, ifft, fftn, ifftn

def func(x, y, z):
    return sin(50.0*2.0*pi*x*z) + 0.5*sin(80.0*2.0*pi*y) + 10.47*cos(24.0*2.0*pi*x*y*z)*1.j

PX, PY, PZ = 6, 3, 5
NX, NY, NZ = 2**PX, 2**PY, 2**PZ ##12 = 4096
LOX, HIX, LOY, HIY, LOZ, HIZ = -1.23, 3.54, 3.47, 6.21, -0.56, 8.80
BASE = empty(shape=(NX, NY, NZ), dtype=complex).tolist()
for nx, x in enumerate(linspace(LOX, HIX, NX)):
    for ny, y in enumerate(linspace(LOY, HIY, NY)):
        for nz, z in enumerate(linspace(LOZ, HIZ, NZ)):
            BASE[nx][ny][nz] = func(x, y, z)

#Algos for bit reversal---------------------------------------------------------------------------------------
def revbinupdate(a, b):
    ## a ^= b means a = a ^ b (^ is XOR)
    ## used to be 'revbinupdate(a, b); while(!(a ^= b) & b); {b >>= 1}; return a;'
    a = a^b
    while(not((a) & b)):
        b >>= 1
        a = a^b
    return a

#Permutation of arr's indices (arr is length n) that undo the bit-reversal that the dif/dit in-place fft algos
#do.  also needed is the half length of the array, nh = n >> 1
def _revbinpermute(arr, n, nh):
    if(n <= 2):
        return arr

    i = 0
    j = 1

    while(j < nh):
        #odd j portion
        i += nh
        arr[j], arr[i] = arr[i], arr[j]
        j += 1

        #even j portion
        i = revbinupdate(i, nh)
        if(i > j):
            arr[j], arr[i] = arr[i], arr[j]
            arr[n-j-1], arr[n-i-1] = arr[n-i-1], arr[n-j-1]
        j += 1

    return arr

#FFT Algos----------------------------------------------------------------------------------------------------
#1D DIT FFT algo with no bit reversal section
#this one is arranged for good memory access, but is most efficient when a trig table is used due to the
#increased number of calls to complex exp
#Needs the array to edit, the log_2(its size), its size, and x, the value in exp(ixj) (may be neg for ifft)
#Note1: permute is done BEFORE the dit algo ... NOTE2: x is often just isign * pi
def _fftbasedit(arr, l2n, n, x):
    ##explicitly do the trivial 1+0i multiplications
    for i in range(0, n, 2):
        tmpip0 = arr[i] + arr[i+1]
        tmpip1 = arr[i] - arr[i+1]
        arr[i]   = tmpip0
        arr[i+1] = tmpip1
    ##do the rest of the transform
    for ldm in range(2, l2n+1):
        m = 1<<ldm
        mh = m>>1
        phi = x / float(mh)

        for i in range(0, n, m):
            for j in range(0, mh):
                ij = i + j
                ijmh = ij + mh

                u = arr[ij]
                v = arr[ijmh] * exp(phi*float(j)*1.0j) ##call to trig table with index j
                arr[ij]   = u + v
                arr[ijmh] = u - v

    return arr

#1D DIF FFT algo with no bit reversal section
#this one is arranged for good memory access, but is most efficient when a trig table is used due to the
#increased number of calls to complex exp
#Needs the array to edit, the log_2(its size), its size, and x, the value in exp(ixj) (may be neg for ifft)
#Note1: permute is done AFTER the dif algo ... NOTE2: x is often just isign * pi
def _fftbasedif(arr, l2n, n, x):
    ##do most of the transform
    for ldm in range(l2n, 1, -1): ##ldm=l2n; ldm >= 2; --ldm
        m = 1<<ldm
        mh = m>>1
        phi = x / float(mh)

        for i in range(0, n, m):
            for j in range(0, mh):
                ij = i + j
                ijmh = ij + mh

                u = arr[ij]
                v = arr[ijmh]
                arr[ij]   =  u + v
                arr[ijmh] = (u - v) * exp(phi*float(j)*1.0j) ##call to trig table with index j
    ##explicitly do the trivial 1+0i multiplications
    for i in range(0, n, 2):
        tmpip0 = arr[i] + arr[i+1]
        tmpip1 = arr[i] - arr[i+1]
        arr[i]   = tmpip0
        arr[i+1] = tmpip1

    return arr

from numpy import round
def fft3dit(arr, pows, sizes, isign):
    ##permute---------------------------------------------------------
    for i in range(0, sizes[0]):
        for j in range(0, sizes[1]):
            arr[i][j] = _revbinpermute(arr=arr[i][j], n=sizes[2], nh=sizes[2]//2)
    arr = transpose(arr, axes=(1, 2, 0))
    for i in range(0, sizes[1]):
        for j in range(0, sizes[2]):
            arr[i][j] = _revbinpermute(arr=arr[i][j], n=sizes[0], nh=sizes[0]//2)
    arr = transpose(a=arr, axes=(1, 2, 0))
    for i in range(0, sizes[2]):
        for j in range(0, sizes[0]):
            arr[i][j] = _revbinpermute(arr=arr[i][j], n=sizes[1], nh=sizes[1]//2)
    arr = transpose(a=arr, axes=(1, 2, 0))

    ##fft------------------------------------------------------------
    for i in range(0, sizes[0]):
        for j in range(0, sizes[1]):
            arr[i][j] = _fftbasedit(arr=arr[i][j], l2n=pows[2], n=sizes[2], x=isign*pi)
    arr = transpose(arr, axes=(1, 2, 0))
    for i in range(0, sizes[1]):
        for j in range(0, sizes[2]):
            arr[i][j] = _fftbasedit(arr=arr[i][j], l2n=pows[0], n=sizes[0], x=isign*pi)
    arr = transpose(a=arr, axes=(1, 2, 0))
    for i in range(0, sizes[2]):
        for j in range(0, sizes[0]):
            arr[i][j] = _fftbasedit(arr=arr[i][j], l2n=pows[1], n=sizes[1], x=isign*pi)
    arr = transpose(a=arr, axes=(1, 2, 0))

    return arr

from numpy import array
def fft3dif(arr, pows, sizes, isign):
    arr = array(arr)
    ##fft------------------------------------------------------------
    for i in range(0, sizes[0]):
        for j in range(0, sizes[1]):
            arr[i,j,:] = _fftbasedif(arr=arr[i,j,:], l2n=pows[2], n=sizes[2], x=isign*pi)
    for i in range(0, sizes[0]):
        for k in range(0, sizes[2]):
            arr[i,:,k] = _fftbasedif(arr=arr[i,:,k], l2n=pows[1], n=sizes[1], x=isign*pi)
    for j in range(0, sizes[1]):
        for k in range(0, sizes[2]):
            arr[:,j,k] = _fftbasedif(arr=arr[:,j,k], l2n=pows[0], n=sizes[0], x=isign*pi)


    ##permute---------------------------------------------------------
    for i in range(0, sizes[0]):
        for j in range(0, sizes[1]):
            arr[i][j] = _revbinpermute(arr=arr[i][j], n=sizes[2], nh=sizes[2]//2)
    arr = transpose(arr, axes=(1, 2, 0))
    for i in range(0, sizes[1]):
        for j in range(0, sizes[2]):
            arr[i][j] = _revbinpermute(arr=arr[i][j], n=sizes[0], nh=sizes[0]//2)
    arr = transpose(a=arr, axes=(1, 2, 0))
    for i in range(0, sizes[2]):
        for j in range(0, sizes[0]):
            arr[i][j] = _revbinpermute(arr=arr[i][j], n=sizes[1], nh=sizes[1]//2)
    arr = transpose(a=arr, axes=(1, 2, 0))

    return arr




#Testing------------------------------------------------------------------------------------------------------
from copy import deepcopy

#The FFT Tests--------------------------------------------------------
res1 = fftn(x=deepcopy(BASE), s=(NX, NY, NZ))
buffer = empty(shape=max(NX, NY, NZ), dtype=complex)
res2 = fft3dit(arr=deepcopy(BASE), pows=[PX, PY, PZ], sizes=[NX, NY, NZ], isign=-1)
res3 = fft3dif(arr=deepcopy(BASE), pows=[PX, PY, PZ], sizes=[NX, NY, NZ], isign=-1)
from numpy import max
print(max(res1 - res2))
print(max(res1 - res3))
print(max(res2 - res3))

#The Convolution Tests-----------------------------------------------
