#Testing that the 2d FFT is just repeated applications of the 1D FFT
#except now, all of the bit reversals are done first

from numpy import empty, linspace, sin, cos, pi, transpose, exp
from scipy.fft import fft, ifft, fftn, ifftn

def func(x, y):
    return sin(50.0*2.0*pi*x) + 0.5*sin(80.0*2.0*pi*y) + 10.47*cos(24.0*2.0*pi*x*y)*1.j

POWX, POWY = 7, 6
NX, NY = 2**POWX, 2**POWY ##12 = 4096
LOX, HIX, LOY, HIY = -1.23, 3.54, 3.47, 6.21
BASE = empty(shape=(NX, NY), dtype=complex).tolist()
for nx, x in enumerate(linspace(LOX, HIX, NX)):
    for ny, y in enumerate(linspace(LOY, HIY, NY)):
        BASE[nx][ny] = func(x, y)


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
        r = revbinupdate(r, nh)
        if(r > x):
            arr[x], arr[r] = arr[r], arr[x]
            arr[n-x-1], arr[n-r-1] = arr[n-r-1], arr[n-x-1]
        x += 1

    return arr


#less trig computations
def fft_dit_rad2_nobr1(arr, log2dim, si):
    n = 1<<log2dim
    pis = si*pi

    #arr = revbinpermute(arr, n)
    ##explicitly do the trivial 1+0i multiplications
    for r in range(0, n, 2):
        tmprp0 = arr[r] + arr[r+1]
        tmprp1 = arr[r] - arr[r+1]
        arr[r]   = tmprp0
        arr[r+1] = tmprp1
    ##do the rest of the transform
    for ldm in range(2, log2dim+1):
        m = 1<<ldm
        mh = m>>1
        phi = pis / float(mh)

        for i in range(0, mh):
            e = exp(phi*float(i)*1j)

            for r in range(0, n, m):
                i0 = r + i
                i1 = i0 + mh
                print("good trig", i0, i1)
                u = arr[i0]
                v = arr[i1] * e
                arr[i0] = u + v
                arr[i1] = u - v

    return arr


#better memory access patterns (maybe better than above when paired with a precomputed trig table)
def fft_dit_rad2_nobr2(arr, log2dim, si):
    n = 1<<log2dim
    pis = si*pi

    #arr = revbinpermute(arr, n)
    ##explicitly do the trivial 1+0i multiplications
    for r in range(0, n, 2):
        tmprp0 = arr[r] + arr[r+1]
        tmprp1 = arr[r] - arr[r+1]
        arr[r]   = tmprp0
        arr[r+1] = tmprp1
    ##do the rest of the transform
    for ldm in range(2, log2dim+1):
        m = 1<<ldm
        mh = m>>1
        phi = pis / float(mh)

        for r in range(0, n, m):
            for i in range(0, mh):
                e = exp(phi*float(i)*1j)
                i0 = r + i
                i1 = i0 + mh
                print("good mem", i0, i1)
                u = arr[i0]
                v = arr[i1] * e
                arr[i0] = u + v
                arr[i1] = u - v

    return arr





def fft2d(arr, log2dims, si):
    dims = [1<<log2dims[0], 1<<log2dims[1]]


    #Bit reversal
    ##Go over rows
    for i in range(0, dims[0]):
        arr[i] = revbinpermute(arr[i], n=dims[1])

    ##Transpose so that rows -> columns (better memory access)
    arr = transpose(a=arr, axes=(1, 0))

    ##Go over columns
    for i in range(0, dims[1]):
        arr[i] = revbinpermute(arr[i], n=dims[0])

    ##Get back to normal order
    arr = transpose(a=arr, axes=(1, 0))



    #FFT
    ##Go over rows
    for i in range(0, dims[0]):
        arr[i] = fft_dit_rad2_nobr2(arr=arr[i], log2dim=log2dims[1], si=si)
        exit(1)
    ##Transpose so that rows -> columns (better memory access)
    arr = transpose(a=arr, axes=(1, 0))

    ##Go over columns
    for i in range(0, dims[1]):
        arr[i] = fft_dit_rad2_nobr2(arr=arr[i], log2dim=log2dims[0], si=si)

    ##Get back to normal order
    arr = transpose(a=arr, axes=(1, 0))

    return arr


from copy import deepcopy
res1 = fftn(x=deepcopy(BASE), s=(NX, NY))
res2 = fft2d(arr=deepcopy(BASE), log2dims=[POWX, POWY], si=-1)

from numpy import max
print(max(res1 - res2))

##TODO: do for 3d, implement our own DIF and DIT FFT with plans, no bit reversal sections, etc
