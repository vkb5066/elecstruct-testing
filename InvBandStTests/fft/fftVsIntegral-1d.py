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
##plot
ax[0].set_xlim(SAMPLE_LO, SAMPLE_HI)
ax[0].plot(xBase, yBase, 'k')



#   ---   scipy   --------------------------------------------------------------------------------------------
print("scipy...")
#Transformed x, y
##set
yTran = list(fft(yBase)*sspacing)
yTranCopy = yTran[:]
#yTran = list(fft(yBase)/N_SAMPLE_PTS) ##directly from scipy docs - seems to be wrong?
yTranRe, yTranIm = [y.real for y in yTran], [y.imag for y in yTran]
xTran = list(fftfreq(N_SAMPLE_PTS, sspacing))
##plot
ax[1].plot(xTran, yTranRe, 'r')
ax[1].plot(xTran, yTranIm, 'b')

#delete me
if(0):
    ax[1].plot(yTranRe, 'r')
    ax[1].plot(yTranIm, 'b')
    from numpy import conj
    from numpy import array
    yTran_ = list(fft(conj(yBase))*sspacing)
    yTranRe_, yTranIm_ = [y.real for y in yTran_], [y.imag for y in yTran_]
    #for i in range(0, N_SAMPLE_PTS//4):
    #    yTranRe_[i] = -yTranRe_[i]
    #    yTranIm_[i] = -yTranIm_[i]
    #for i in range(3*N_SAMPLE_PTS//4, N_SAMPLE_PTS):
    #    yTranRe_[i] = -yTranRe_[i]
    #    yTranIm_[i] = -yTranIm_[i]
    ax[2].plot(yTranRe_, 'r')
    ax[2].plot(yTranIm_, 'b')
    from numpy import allclose
    print(allclose(conj(yTran_), yTran, 1e-6, 1e-6))
    ax[3].plot(array(yTranRe_) - array(yTranRe), 'r.')
    ax[3].plot(array(yTranIm_) - array(yTranIm), 'b.')
    plt.show()
    exit(3)


#   ---   naive   --------------------------------------------------------------------------------------------
print("naive...")
def TrapIntegrate(dx, y0, len):
    su = (y0[0] + y0[-1])/2
    for i in range(1, len - 1):
        su += y0[i]
    return dx*su

def NaiveFT(nSamplePts, xOrig, yOrig, freqMin, freqMax):
    spacing = (freqMax - freqMin)/(nSamplePts - 1)
    dx = xOrig[1] - xOrig[0]

    freqs = [None for i in range(0, nSamplePts)]
    transfRe, transfIm = freqs[:], freqs[:]
    for i in range(0, nSamplePts):
        sampleFreq = freqMin + i*spacing
        transformedReIntegrand = [yOrig[j]*cos(2*pi*sampleFreq*xOrig[j]) for j in range(0, nSamplePts)]
        if(USE_SCIPY_CONVENTIONS):
            transformedImIntegrand = [-yOrig[j]*sin(2*pi*sampleFreq*xOrig[j]) for j in range(0, nSamplePts)]
        else:
            transformedImIntegrand = [yOrig[j]*sin(2*pi*sampleFreq*xOrig[j]) for j in range(0, nSamplePts)]


        freqs[i] = sampleFreq
        transfRe[i] = TrapIntegrate(dx, transformedReIntegrand, nSamplePts)
        transfIm[i] = TrapIntegrate(dx, transformedImIntegrand, nSamplePts)

    return freqs, transfRe, transfIm

#Other constants
freqSampleMin = -(N_SAMPLE_PTS/2) / (N_SAMPLE_PTS*sspacing) ##only works for even numbers of n sample pts
freqSampleMax = (N_SAMPLE_PTS/2 - 1) / (N_SAMPLE_PTS*sspacing) ##only works for even numbers of n sample pts
#Transformed x, y
##set
xTran, yTranRe, yTranIm = NaiveFT(N_SAMPLE_PTS, xBase, yBase, freqSampleMin, freqSampleMax)

##plot
ax[2].plot(xTran, yTranRe, 'r')
ax[2].plot(xTran, yTranIm, 'b')



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

NORMALIZE = 1
##set
yBase_ = GetAdjacentArr(yBase)
yTran = MyFFT(N_SAMPLE_PTS, yBase_, 1)
if(NORMALIZE):
    yTran = [y*sspacing for y in yTran]
xTran = MyFFtFreqs(N_SAMPLE_PTS, sspacing)
xTranReduced, yTranRe, yTranIm = [], [], []
for i in range(0, N_SAMPLE_PTS):
    xTranReduced.append(xTran[2*i])
    yTranRe.append(yTran[2*i])
    yTranIm.append(yTran[2*i + 1])
##plot
ax[3].plot(xTranReduced, yTranRe, 'r')
ax[3].plot(xTranReduced, yTranIm, 'b')


#   ---   scipy back transform   -----------------------------------------------------------------------------
from scipy.fft import ifft
yBase = list(ifft(yTranCopy).real/sspacing)
ax[4].plot(xBase, yBase)


#   ---   End   ----------------------------------------------------------------------------------------------
for i in range(1, 4):
    ax[i].set_xlim(-100, +100)
    ax[i].set_ylim(-.25, +.25)
plt.show()
