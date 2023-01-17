#Testing my understanding of simple fourier transforms ...

USE_SCIPY_CONVENTIONS = 1 ##changes a few factors of -1

from numpy import sin, cos, pi, linspace
from scipy.fft import fft, fftfreq
from matplotlib import pyplot as plt
fig, ax = plt.subplots(5, 1)


def func(x):
    return sin(50.0*2.0*pi*x) + 0.5*sin(80.0*2.0*pi*x) + 1.47*cos(24.0*2.0*pi*x)*1.j


#Constants
N_SAMPLE_PTS = 64
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


#delete me
N_SAMPLE_PTS = 4
yBase = [-3.45-3.211j, -3.45-3.158j, -3.45-1.664j, -3.45+0.561j]
#end



#   ---   scipy   --------------------------------------------------------------------------------------------
print("scipy...")
#Transformed x, y
##set
yTran = list(fft(yBase))
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




#   ---   custom fft   ---------------------------------------------------------------------------------------
class plan:
    brtablelen = None
    brtable = None
    ttable = None ##sin(theta) given the denominator of

    def __init__(self, n, type): ##n = number of COMPLEX values
        nn = n << 1
        self.brtable = [None for i in range(0, n)] ###will be reduced
        self.ttable = [None for i in range(0, nn - 2)]

        #Get the bit reversal map ...rmap[i] = [j] means that index 2i should be swapped with index 2j
        j = 1
        count = 0
        for i in range(1, nn, 2):
            if(j > i):
                self.brtable[count] = i - 1
                self.brtable[count+1] = j - 1
                count += 2

            m = n
            while (m > 2 and j > m):
                j -= m
                m >>= 1

            j += m
        self.brtablelen = count
        self.brtable = self.brtable[:count]

        #Get the trig table w/ Danielson - Lanczos: ttable[j] = wr_j, ttable[j+1] = (type)*wi_j
        mMax = 2
        count = 0
        while (nn > mMax):
            iStep = mMax << 1
            theta = type*(2.0*pi/mMax)  ###trig recurr init
            wTmp = sin(0.5 * theta)
            wpr = -2.0 * wTmp*wTmp
            wpi = sin(theta)

            wr = 1.0
            wi = 0.0
            from numpy import arcsin
            from fractions import Fraction as fraction
            for m in range(1, mMax, 2):
                wTmp = wr ##trig recurr fin
                wr = wTmp*wpr - wi*wpi   + wr
                wi = wi*wpr   + wTmp*wpi + wi

                self.ttable[count] = wr
                self.ttable[count + 1] = type*wi

                count += 2

            mMax = iStep

        return










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
from numpy import pi, array, round
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
        #print("mmax", mMax, "->", count, "thetas/pi", theta/pi, 0.5*theta/pi, round(wpr, 3), round(wpi, 3))
        ##DELETE ME

        wr = 1.0
        wi = 0.0
        from numpy import arcsin
        from fractions import Fraction as fraction
        for m in range(1, mMax, 2):
            for i in range(m, nn + 1, iStep):
                j = i + mMax
                tmpR = wr*arr[j - 1] - wi*arr[j]
                tmpI = wr*arr[j] + wi*arr[j - 1]

                arr[j - 1] = arr[i - 1] - tmpR
                arr[j] = arr[i] - tmpI
                arr[i - 1] += tmpR
                arr[i] += tmpI
            print("WRWIo", wr, wi)
            wTmp = wr
            wr = wTmp*wpr - wi*wpi + wr
            wi = wi*wpr + wTmp*wpi + wi
            print("WRWIn", wr, wi)
            count += 1

        mMax = iStep


    return arr





def MyFFT_wplan(nSamples, arr, type, plan):
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
    for i in range(0, pl.brtablelen, 2):
        im, jm = pl.brtable[i], pl.brtable[i+1]
        arr[jm], arr[im] = arr[im], arr[jm]
        im += 1
        jm += 1
        arr[jm], arr[im] = arr[im], arr[jm]



    ##Danielson - Lanczos
    mMax = 2
    count = 0
    while(nn > mMax):
        iStep = mMax << 1

        wr = 1.0
        wi = 0.0
        from numpy import arcsin
        from fractions import Fraction as fraction
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
            wr = plan.ttable[count]
            wi = type*plan.ttable[count+1]

            count += 2

        mMax = iStep


    return arr






##set
yBase_ = GetAdjacentArr(yBase)
yTran = MyFFT(N_SAMPLE_PTS, yBase_, 1)
print(yTran)
xTran = MyFFtFreqs(N_SAMPLE_PTS, sspacing)
xTranReduced, yTranRe, yTranIm = [], [], []
for i in range(0, N_SAMPLE_PTS):
    xTranReduced.append(xTran[2*i])
    yTranRe.append(yTran[2*i])
    yTranIm.append(yTran[2*i + 1])
##plot
ax[2].plot(xTranReduced, yTranRe, 'r')
ax[2].plot(xTranReduced, yTranIm, 'b')



#set plan version
pl = plan(N_SAMPLE_PTS, type=+1) ##plan for +1
#print(pl.brtable)
#exit(3)
yBase_ = GetAdjacentArr(yBase)
yTran = MyFFT_wplan(N_SAMPLE_PTS, yBase_, 1, pl)
xTran = MyFFtFreqs(N_SAMPLE_PTS, sspacing)
xTranReduced, yTranRe, yTranIm = [], [], []
for i in range(0, N_SAMPLE_PTS):
    xTranReduced.append(xTran[2*i])
    yTranRe.append(yTran[2*i])
    yTranIm.append(yTran[2*i + 1])
##plot
ax[3].plot(xTranReduced, yTranRe, 'r')
ax[3].plot(xTranReduced, yTranIm, 'b')

#   ---   End   ----------------------------------------------------------------------------------------------
from numpy import max
print(max(array(yTranRe) - array(yTranCopy).real))
print(max(array(yTranIm) - array(yTranCopy).imag))
#plt.show()
