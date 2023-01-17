#Testing my understanding of simple fourier transforms ...

USE_SCIPY_CONVENTIONS = 1 ##changes a few factors of -1

from numpy import sin, cos, pi, linspace
from scipy.fft import fft, fftfreq, ifft
from matplotlib import pyplot as plt
fig, ax = plt.subplots(3, 1)


def func(x):
    return sin(50.0*2.0*pi*x) + 0.5*sin(80.0*2.0*pi*x)


#Constants
N_SAMPLE_PTS = 512
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



#forward transform
print("-> ...")
#Transformed x, y
##set
yTran = list(fft(yBase))
yTranRe, yTranIm = [y.real for y in yTran], [y.imag for y in yTran]
xTran = list(fftfreq(N_SAMPLE_PTS, sspacing))
##plot
ax[1].plot(xTran, yTranRe, 'r')
ax[1].plot(xTran, yTranIm, 'b')



#backwards transform
print("<- ...")
yTran = list(ifft(yTran))
yTranRe, yTranIm = [y.real for y in yTran], [y.imag for y in yTran]
xTran = xBase

ax[2].plot(xTran, yTranRe, 'r')
ax[2].plot(xTran, yTranIm, 'b')

plt.show()
