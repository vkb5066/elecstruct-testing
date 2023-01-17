#Implementation of a 3d fft <-> ifft from :
#Numerical Recipes in C (W. Press)

from numpy import empty, array, linspace
from numpy import sin, cos
from scipy.fft import fftn, ifftn

def f(x, y, z):
    return 4.81*sin(1.34*x) + 1.88*cos(7.02*y) + (0.47*sin(2.11*z) + 2.98*cos(1.47*x*y*z))*1.0j


NX, NY, NZ = 2**4, 2**4, 2**4 ##12 = 4096
LOX, HIX, LOY, HIY, LOZ, HIZ = -1.23, 3.54, 3.47, 6.21, -0.56, 8.80
BASE = empty(shape=(NX, NY, NZ), dtype=complex).tolist()
for nx, x in enumerate(linspace(LOX, HIX, NX)):
    for ny, y in enumerate(linspace(LOY, HIY, NY)):
        for nz, z in enumerate(linspace(LOZ, HIZ, NZ)):
            BASE[nx][ny][nz] = f(x, y, z)



#Scipy's FFT--------------------------------------------------------------------------------------------------
print("start scipy")
fftscipy = fftn(x=BASE, s=(NX, NY, NZ), overwrite_x=False)
print("end scipy")


#My (base) FFT------------------------------------------------------------------------------------------------
##Transforms 3d (numpy-style) array into a 1d representation suitable for fourn
def intrans3(data, dim):
    ret = empty(shape=(2*dim[0]*dim[1]*dim[2]), dtype=float).tolist() ##stored as [..., real, imag, ...]

    count = 0
    for nx in range(0, dim[0]):
        for ny in range(0, dim[1]):
            for nz in range(0, dim[2]):
                ret[count] = data[nx][ny][nz].real
                count += 1
                ret[count] = data[nx][ny][nz].imag
                count += 1

    return ret

#Transforms a 1d (my style) array into a 3d representation suitable for numpy
def outrans3(data, dim):
    ret = empty(shape=(dim[0], dim[1], dim[2]), dtype=complex).tolist()

    count = 0
    for nx in range(0, dim[0]):
        for ny in range(0, dim[1]):
            for nz in range(0, dim[2]):
                ret[nx][ny][nz] = data[count] + (data[count+1])*1.0j
                count += 2

    return ret

#Make sure that my functions make sense
#from numpy import allclose
#mattolong = intrans3(data=BASE, dim=[NX, NY, NZ])
#longtomat = outrans3(data=mattolong, dim=[NX, NY, NZ])
#print(allclose(BASE, longtomat, 1e-8, 1e-8))


def fft3(data, dim, sign):
    ##uints (seems like idim should be a NORMAL INT, NOT UINT)
    idim, i1, i2, i3, i2rev, i3rev, ip1, ip2, ip3, ifp1, ifp2 = None, None, None, None, None, None, None, \
                                                                None, None, None, None
    ibit, k1, k2, n, nprev, nrem, ntot = None, None, None, None, None, None, None
    ##floats / doubles
    tempi, tempr, theta, wi, wpi, wpr, wr, wtemp = None, None, None, None, None, None, None, None

    #initialization
    ntot = dim[0]*dim[1]*dim[2]
    assert (not(ntot < 2 and ntot&(ntot-1))) ##must be powers of 2
    nprev = 1

    #Go over each dimension
    for idim in range(2, -1, -1): ##idim = 2; idim >= 0; idim--
        ##loop prep
        n = dim[idim]
        nrem = ntot//(n*nprev)
        ip1 = nprev << 1
        ip2 = ip1*n
        ip3 = ip2*nrem
        i2rev = 0

        ##bit reversal
        for i2 in range(0, ip2, ip1):		# i2=0; i2<ip2; i2+=ip1
            if(i2 < i2rev):
                for i1 in range(i2, i2+ip1-1, 2): #i1=i2; i1<i2+ip1-1; i1+=2
                    for i3 in range(i1, ip3, ip2):                # i3=i1; i3<ip3; i3+=ip2
                        i3rev = i2rev + i3 - i2
                        data[i3], data[i3rev] = data[i3rev], data[i3]
                        data[i3+1], data[i3rev+1] = data[i3rev+1], data[i3+1]

            ibit = ip2 >> 1
            while (ibit >= ip1 and i2rev+1 > ibit):
                i2rev -= ibit
                ibit >>= 1

            i2rev += ibit
        ifp1 = ip1

        ##D-L section
        while (ifp1 < ip2):
            ##trig rec init
            ifp2 = ifp1 << 1
            theta = float(sign)*6.28318530717959 / float((ifp2//ip1))
            wtemp = sin(0.5*theta)
            wpr = -2.0*wtemp*wtemp
            wpi = sin(theta)

            ##actually apply the transformation
            wr=1.0
            wi=0.0
            for i3 in range(0, ifp1, ip1): ##i3=0; i3<ifp1; i3+=ip1
                for i1 in range(i3, i3+ip1-1, 2): ##i1=i3; i1<i3+ip1-1; i1+=2
                    for i2 in range(i1, ip3, ifp2): ##i2=i1; i2<ip3; i2+=ifp2
                        k1 = i2
                        k2 = k1 + ifp1

                        tempr = wr*data[k2]   - wi*data[k2+1]
                        tempi = wr*data[k2+1] + wi*data[k2]
                        data[k2] =   data[k1]   - tempr
                        data[k2+1] = data[k1+1] - tempi
                        data[k1]   += tempr
                        data[k1+1] += tempi

                wtemp = wr
                wr = wtemp*wpr - wi*wpi    + wr
                wi = wi*wpr   + wtemp*wpi + wi

            ifp1 = ifp2
        nprev *= n

    return data


from copy import deepcopy
base = intrans3(data=BASE, dim=[NX, NY, NZ])
print("start fft3")
fftmine = fft3(data=base, dim=[NX, NY, NZ], sign=-1) ##-1 corresponds to scipy's +1 ...
print("end fft3")
#fftmine = outrans3(data=fftmine, dim=[NX, NY, NZ])
fftscipy = intrans3(data=fftscipy, dim=[NX, NY, NZ])

#Compare the two for consistency
from numpy import max, allclose
print(max(array(fftscipy) - array(fftmine)))
