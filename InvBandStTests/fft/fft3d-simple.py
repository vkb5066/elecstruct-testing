#Testing that the 2d FFT is just repeated applications of the 1D FFT

from numpy import empty, linspace, sin, cos, pi, transpose
from scipy.fft import fft, ifft, fftn, ifftn

def func(x, y, z):
    return sin(50.0*2.0*pi*x*z) + 0.5*sin(80.0*2.0*pi*y) + 10.47*cos(24.0*2.0*pi*x*y*z)*1.j

NX, NY, NZ = 2**6, 2**3, 2**5 ##12 = 4096
LOX, HIX, LOY, HIY, LOZ, HIZ = -1.23, 3.54, 3.47, 6.21, -0.56, 8.80
BASE = empty(shape=(NX, NY, NZ), dtype=complex).tolist()
for nx, x in enumerate(linspace(LOX, HIX, NX)):
    for ny, y in enumerate(linspace(LOY, HIY, NY)):
        for nz, z in enumerate(linspace(LOZ, HIZ, NZ)):
            BASE[nx][ny][nz] = func(x, y, z)

def fft3d(arr, dims):

    #fft dim 0
    for i in range(0, dims[0]):
        for j in range(0, dims[1]):
            arr[i][j] = fft(x=arr[i][j], n=dims[2])

    #Permute for better memory access
    arr = transpose(arr, axes=(1, 2, 0))

    #fft dim 1
    for i in range(0, dims[1]):
        for j in range(0, dims[2]):
            arr[i][j] = fft(x=arr[i][j], n=dims[0])

    #Permute for better memory access
    arr = transpose(a=arr, axes=(1, 2, 0))

    #fft dim 2
    for i in range(0, dims[2]):
        for j in range(0, dims[0]):
            arr[i][j] = fft(x=arr[i][j], n=dims[1])

    #Get back to normal order
    arr = transpose(a=arr, axes=(1, 2, 0))
    return arr

"""
    print(arr.shape)
    for i in range(0, dims[0]):
        for j in range(0, dims[1]):
            arr[i,j,:] = fft(x=arr[i,j,:], n=dims[2])
    for i in range(0, dims[1]):
        for j in range(0, dims[2]):
            arr[:,i,j] = fft(x=arr[:,i,j], n=dims[0])
    for i in range(0, dims[0]):
        for j in range(0, dims[2]):
            arr[i,:,j] = fft(x=arr[i,:,j], n=dims[1])

    return arr
"""

from copy import deepcopy
res1 = fftn(x=deepcopy(BASE), s=(NX, NY, NZ))
res2 = fft3d(arr=deepcopy(BASE), dims=[NX, NY, NZ])

from numpy import max
print(max(res1 - res2))

##TODO: implement our own DIF and DIT FFT with plans, no bit reversal sections, etc
