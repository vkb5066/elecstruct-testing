#Testing that the 2d FFT is just repeated applications of the 1D FFT

from numpy import empty, linspace, sin, cos, pi, transpose
from scipy.fft import fft, ifft, fftn, ifftn

def func(x):
    return sin(50.0*2.0*pi*x*x) + 0.5*sin(80.0*2.0*pi*x) + 10.47*cos(24.0*2.0*pi*x*x*x)*1.j

POW = 6
NX = 2**POW ##12 = 4096
LOX, HIX = -1.23, 3.54
BASE = empty(shape=(NX), dtype=complex).tolist()
for nx, x in enumerate(linspace(LOX, HIX, NX)):
    BASE[nx] = func(x)

def revbinupdate(a, b):
    ## a ^= b means a = a ^ b (^ is XOR)
    ## used to be 'while(!(a ^= b) & b)'
    a = a^b
    while(not((a) & b)):
        b >>= 1
        a = a^b
    return a

def revbinpermute(arr, n):
    if(n <= 2):
        return arr

    nh = n>>1
    r = 0
    x = 1

    while(x < nh):
        #odd x portion
        r += nh
        arr[x], arr[r] = arr[r], arr[x]
        x += 1

        #even x portion
        r = revbinupdate(r, n//2)
        if(r > x):
            arr[x], arr[r] = arr[r], arr[x]
            arr[n-x-1], arr[n-r-1] = arr[n-r-1], arr[n-x-1]
        x += 1

    return arr


from numpy import exp

#This one works with worse memory locality, but lower # of trig computations
#The improvement on page 414 is not implemented
def fft_dit_rad2(arr, log2dim, si):
    n = 1<<log2dim
    pis = si*pi

    arr = revbinpermute(arr, n)
    for ldm in range(1, log2dim+1):
        m = 1<<ldm
        mh = m>>1
        phi = pis / float(mh)

        for i in range(0, mh):
            e = exp(phi*float(i)*1j)

            for r in range(0, n, m):
                i0 = r + i
                i1 = i0 + mh

                u = arr[i0]
                v = arr[i1] * e
                arr[i0] = u + v
                arr[i1] = u - v

    return arr


from numpy import round
from copy import deepcopy
res1 = fft(x=deepcopy(BASE), n=NX)
res2 = fft_dit_rad2(arr=deepcopy(BASE), log2dim=POW, si=-1) ##note that numpy's sign
from numpy import max                                       ##convention is fucked up
print(max(res1 - res2))
