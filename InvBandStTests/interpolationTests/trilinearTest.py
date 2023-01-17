import numpy as np
from scipy.interpolate import interpn

XLO, XHI = 0, 5
YLO, YHI = 0, 5
ZLO, ZHI = 0, np.pi
N_PTS = 27

def value_func_3d(x, y, z):
    return 2.8*x + 3.1*y*y - 7.16*z*z*z + 4.1*z*z - 3.8*z

tpoints = np.array([[3.21,  0.12,  3.13],
                    [3.48,  1.49,  2.15]])

#Numpy's stuff
print("NUMPY")
x = np.linspace(XLO, XHI, N_PTS)
y = np.linspace(YLO, YHI, N_PTS)
z = np.linspace(ZLO, ZHI, N_PTS)

points = (x, y, z)
values = value_func_3d(*np.meshgrid(*points, indexing='ij'))

print(interpn(points, values, tpoints))

#My Stuff
#assumes (1) we have an equal number of sample pts in all directions
#        (2) XLO = YLO, XHI = YHI
print("ME")
dx = (XHI - XLO) / (N_PTS - 1) ##also equal to dy
dz = (ZHI - ZLO) / (N_PTS - 1)

samples = [None for i in range(0, N_PTS*N_PTS*N_PTS)]
for xi in range(0, N_PTS):
    for yi in range(0, N_PTS):
        for zi in range(0, N_PTS):
            samples[N_PTS*N_PTS*xi + N_PTS*yi + zi] = value_func_3d(XLO + dx*xi, YLO + dx*yi, ZLO + dz*zi)
print(np.allclose(values.flatten() - np.array(samples), 0.0, 1e-8, 1e-8))

def ijktoi(n, i, j, k):
    return n*n*i + n*j + k

for tpt in tpoints:
    xloind = int(  (tpt[0]-XLO)/dx  )
    xhiind = xloind + 1
    yloind = int(  (tpt[1]-YLO)/dx  )
    yhiind = yloind + 1
    zloind = int(  (tpt[2]-ZLO)/dz  )
    zhiind = zloind + 1

    xd = (tpt[0] - (XLO+dx*xloind)) / dx
    yd = (tpt[1] - (YLO+dx*yloind)) / dx
    zd = (tpt[2] - (ZLO+dz*zloind)) / dz

    #get function values at all edges
    ##bottom face
    c000 = samples[ijktoi(N_PTS, xloind, yloind, zloind)]
    c100 = samples[ijktoi(N_PTS, xhiind, yloind, zloind)]
    c010 = samples[ijktoi(N_PTS, xloind, yhiind, zloind)]
    c110 = samples[ijktoi(N_PTS, xhiind, yhiind, zloind)]
    ##top face
    c001 = samples[ijktoi(N_PTS, xloind, yloind, zhiind)]
    c101 = samples[ijktoi(N_PTS, xhiind, yloind, zhiind)]
    c011 = samples[ijktoi(N_PTS, xloind, yhiind, zhiind)]
    c111 = samples[ijktoi(N_PTS, xhiind, yhiind, zhiind)]

    #interpolate along x
    c00 = c000*(1-xd) + c100*xd
    c01 = c001*(1-xd) + c101*xd
    c10 = c010*(1-xd) + c110*xd
    c11 = c011*(1-xd) + c111*xd

    #interpolate along y
    c0 = c00*(1-yd) + c10*yd
    c1 = c01*(1-yd) + c11*yd

    #interpolate along z
    c = c0*(1-zd) + c1*zd

    print(c)


#My Stuff optimized
#assumes (1) we have an equal number of sample pts in all directions
#        (2) XLO = YLO, XHI = YHI
#        (3) XLO = YLO = ZLO = 0, ZHI = pi
print("ME2")
npts2 = N_PTS*N_PTS
dx = (XHI - XLO) / (N_PTS - 1) ##also equal to dy
dz = (np.pi) / (N_PTS - 1)

samples = [None for i in range(0, N_PTS*N_PTS*N_PTS)]
for xi in range(0, N_PTS):
    for yi in range(0, N_PTS):
        for zi in range(0, N_PTS):
            samples[N_PTS*N_PTS*xi + N_PTS*yi + zi] = value_func_3d(XLO + dx*xi, YLO + dx*yi, ZLO + dz*zi)
print(np.allclose(values.flatten() - np.array(samples), 0.0, 1e-8, 1e-8))

for tpt in tpoints:
    ##x, y, z scaled simplified for xlo = 0
    xsc = tpt[0]/dx
    ysc = tpt[1]/dx
    zsc = tpt[2]/dz

    ##indexes of low points in the easy-to-understand format where samples is a 3d tensor
    ##i.e. the lower left-hand corner sample is samples[xloind][yloind][zloind] ...
    xloind = int(xsc)
    yloind = int(ysc)
    zloind = int(zsc)

    ##interpolation params m = (x - xlo) / (xhi - xlo) and 1 - m simplified for xlo = 0
    xd = xsc - xloind
    xdsub = 1.0 - xd
    yd = ysc - yloind
    ydsub = 1.0 - yd
    zd = zsc - zloind

    #... but i'm not interested in chasing pointers, so instead the tensor is stored as a 1D array
    #that is accessed by npts^2*i + npts*j + k, which works so long as the tensor's dim = npts x npts x npts
    #here are some helpful indices to make accessing that array a bit less disgusting
    i000 = npts2*xloind + N_PTS*yloind + zloind
    i010 = i000 + N_PTS
    i100 = i000 + npts2
    i110 = i100 + N_PTS

    #Do an interpolation
    #this is the same equation as is given on wikipedia:
    #https://en.wikipedia.org/wiki/Trilinear_interpolation
    #as c = c0(1-zd) + c1*zd   (where c is the interpolated point)
    print(
          (
                   (samples[i000]*(xdsub) + samples[i100]*xd)*(ydsub) + \
                   (samples[i010]*(xdsub) + samples[i110]*xd)*yd
          )*(1.0-zd) + \
          (
                  (samples[i000 + 1]*(xdsub) + samples[i100 + 1]*xd)*(ydsub) + \
                  (samples[i010 + 1]*(xdsub) + samples[i110 + 1]*xd)*yd
          )*zd
         )
