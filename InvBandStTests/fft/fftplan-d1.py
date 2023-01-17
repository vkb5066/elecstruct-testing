#Testing my understanding of simple fourier transforms ...

USE_SCIPY_CONVENTIONS = 1 ##changes a few factors of -1

from numpy import sin, cos, pi, linspace
from scipy.fft import fft, fftfreq
from matplotlib import pyplot as plt
fig, ax = plt.subplots(5, 1)


def func(x):
    return sin(50.0*2.0*pi*x) + 0.5*sin(80.0*2.0*pi*x) + 1.47*cos(24.0*2.0*pi*x)*1.j


#Constants
N_SAMPLE_PTS = 256
SAMPLE_LO = 0.0
SAMPLE_HI = 3/4
sspacing = (SAMPLE_HI - SAMPLE_LO) / N_SAMPLE_PTS

#Base x, y
##set
xBase = list(linspace(SAMPLE_LO, SAMPLE_HI, N_SAMPLE_PTS))
yBase = [func(x) for x in xBase]
yBaseRe = [y.real for y in yBase]
yBaseIm = [y.imag for y in yBase]
##plot
ax[0].plot(xBase, yBaseRe, 'r')
ax[0].plot(xBase, yBaseIm, 'b')






#   ---   custom fft   ---------------------------------------------------------------------------------------
print("custom...")
#Convert an array of [z0, z1, z2, ..., z(N-1)] to an  array of [z0.re, z0.im, z1.re, z1.im, ... ]
def GetAdjacentArr(arr):
    ret = [None for i in range(0, 2*len(arr))]
    if(type(arr[0]) == int):
        for i in range(0, len(arr)):
            ret[2*i] = arr[i]
            ret[2*i + 1] = 0.
    elif(type(arr[0] == complex)):
        for i in range(0, len(arr)):
            ret[2*i] = arr[i].real
            ret[2*i + 1] = arr[i].imag
    else:
        print("???")
        exit(1)

    return ret

##nSamples = number of complex data points (the actual array is twice this size since
##it's elements are [z0.re, z0.im, z1.re, z1.im, ...])
def MyFFtFreqs(nSamples, sampleSpacing):
    len = 2*nSamples
    mult = 1/(nSamples*sampleSpacing)

    ret = [None for i in range(0, 2*nSamples)]
    ret[0], ret[1] = 0., 0.
    for i in range(1, nSamples//2):
        indP = 2*i
        indN = len - indP
        val = i*mult

        ret[indP] = val
        ret[indP + 1] = val
        ret[indN] = -val
        ret[indN + 1] = -val
    ret[nSamples], ret[nSamples + 1] = ret[nSamples + 2] - mult, ret[nSamples + 2] - mult

    return ret


#From Numerical Recipes's 'four1' example ...
##nSamples = number of complex data points (the actual array is twice this size since
##it's elements are [z0.re, z0.im, z1.re, z1.im, ...])
##nSamples must be a power of 2
##type is -1 or +1, for inverse (times nSamples) or forwards dft respectivly
from numpy import pi
def MyFFT(nSamples, arr, type):
    ##check for nSamples a power of 2 - the one and only error checking that I'll do
    if(nSamples < 2 or nSamples&(nSamples - 1)):
        print("not dealing with that number")
        exit(2)

    #uncomment for scipy convention
    if(USE_SCIPY_CONVENTIONS):
        type *= -1

    nn, mMax, i, j, m, iStep, count = None, None, None, None, None, None, None ##ints
    wTmp, wr, tmpR, wpr, wpi, wi, tmpI, theta = None, None, None, None, None, None, None, None ##floats
    nn = nSamples << 1

    ##bit reversal
    j = 1
    for i in range(1, nn, 2):
        if(j > i):
            arr[j - 1], arr[i - 1] = arr[i - 1], arr[j - 1]
            arr[j], arr[i] = arr[i], arr[j]

        m = nSamples
        while(m > 2 and j > m):
            j -= m
            m >>= 1

        j += m

    ##Danielson - Lanczos
    mMax = 2
    count = 0
    while(nn > mMax):
        iStep = mMax << 1
        theta = type*(2.0*pi/mMax) ###trig recurr
        wTmp = sin(0.5*theta)
        wpr = -2.0*wTmp*wTmp
        wpi = sin(theta)

        ##DELTE ME
        print("mmax", mMax, "->", count, "thetas/pi", theta/pi, 0.5*theta/pi)
        ##DELETE ME

        wr = 1.0
        wi = 0.0

        for m in range(1, mMax, 2):
            for i in range(m, nn + 1, iStep):
                j = i + mMax
                tmpR = wr*arr[j - 1] - wi*arr[j]
                tmpI = wr*arr[j] + wi*arr[j - 1]

                arr[j - 1] = arr[i - 1] - tmpR
                arr[j] = arr[i] - tmpI
                arr[i - 1] += tmpR
                arr[i] += tmpI

            wTmp = wr
            wr = wTmp*wpr - wi*wpi + wr
            wi = wi*wpr + wTmp*wpi + wi

        mMax = iStep
        count += 1

    return arr


#-------------------------------------------------------------------------------------------------------------

##set 1
yBase_ = GetAdjacentArr(yBase)
yTran = MyFFT(N_SAMPLE_PTS, yBase_, 1)
xTran = MyFFtFreqs(N_SAMPLE_PTS, sspacing)
xTranReduced, yTranRe, yTranIm = [], [], []
for i in range(0, N_SAMPLE_PTS):
    xTranReduced.append(xTran[2*i])
    yTranRe.append(yTran[2*i])
    yTranIm.append(yTran[2*i + 1])
##plot
ax[1].plot(xTranReduced, yTranRe, 'r')
ax[1].plot(xTranReduced, yTranIm, 'b')


#set 2
yBase_ = GetAdjacentArr(yBase)
yTran = MyFFT(N_SAMPLE_PTS, yTran, 1)
xTran = MyFFtFreqs(N_SAMPLE_PTS, sspacing)
xTranReduced, yTranRe, yTranIm = [], [], []
for i in range(0, N_SAMPLE_PTS):
    xTranReduced.append(xTran[2*i])
    yTranRe.append(yTran[2*i] / N_SAMPLE_PTS)
    yTranIm.append(yTran[2*i + 1] / N_SAMPLE_PTS)
##plot
ax[2].plot(xBase, yTranRe, 'r')
ax[2].plot(xBase, yTranIm, 'b')



#   ---   End   ----------------------------------------------------------------------------------------------
ax[0].set_xlim(SAMPLE_LO, SAMPLE_HI)
ax[2].set_xlim(SAMPLE_LO, SAMPLE_HI)
plt.show()
