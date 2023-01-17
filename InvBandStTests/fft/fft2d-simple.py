#Testing that the 2d FFT is just repeated applications of the 1D FFT

from numpy import empty, linspace, sin, cos, pi, transpose
from scipy.fft import fft, ifft, fftn, ifftn

def func(x, y):
    return sin(50.0*2.0*pi*x) + 0.5*sin(80.0*2.0*pi*y) + 10.47*cos(24.0*2.0*pi*x*y)*1.j

NX, NY = 2**6, 2**5 ##12 = 4096
LOX, HIX, LOY, HIY = -1.23, 3.54, 3.47, 6.21
BASE = empty(shape=(NX, NY), dtype=complex).tolist()
for nx, x in enumerate(linspace(LOX, HIX, NX)):
    for ny, y in enumerate(linspace(LOY, HIY, NY)):
        BASE[nx][ny] = func(x, y)

def fft2d(arr, dims):
    #Go over rows
    for i in range(0, dims[0]):
        arr[i] = fft(x=arr[i], n=dims[1])

    #Transpose so that rows -> columns (better memory access)
    arr = transpose(a=arr, axes=(1, 0))

    #Go over columns
    for i in range(0, dims[1]):
        arr[i] = fft(x=arr[i], n=dims[0])

    #Get back to normal order
    arr = transpose(a=arr, axes=(1, 0))

    return arr

from copy import deepcopy
res1 = fftn(x=deepcopy(BASE), s=(NX, NY))
res2 = fft2d(arr=deepcopy(BASE), dims=[NX, NY])

from numpy import max
print(max(res1 - res2))

##TODO: do for 3d, implement our own DIF and DIT FFT with plans, no bit reversal sections, etc
