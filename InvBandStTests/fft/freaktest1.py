#Testing my understanding of simple fourier transforms ...

USE_SCIPY_CONVENTIONS = 1 ##changes a few factors of -1

from numpy import sin, cos, pi, linspace
from scipy.fft import fft, fftfreq, ifft, fftshift
from matplotlib import pyplot as plt
fig, ax = plt.subplots(3, 1)


def func(x):
    return sin(50.0*2.0*pi*x) + 0.5*sin(80.0*2.0*pi*x)


def IsContained(base, test, tol):
    for bas in test:
        res = False
        for tes in base:
            if(abs(bas - tes) < tol):
                res = True
                break

        if(not res):
            return False
    return True


"""
SAMPLE UNO
"""
#Constants
N_SAMPLE_PTS1 = 16
SAMPLE_LO1 = 12.7
SAMPLE_HI1 = 26.4
sspacing1 = (SAMPLE_HI1 - SAMPLE_LO1) / N_SAMPLE_PTS1

#Base x, y
xBase1 = list(linspace(SAMPLE_LO1, SAMPLE_HI1, N_SAMPLE_PTS1))
yBase1 = [func(x) for x in xBase1]

#Trans x, y
xTrans1 = fftfreq(n=N_SAMPLE_PTS1, d=sspacing1)
yTrans1 = fft(x=yBase1, n=N_SAMPLE_PTS1)

print(xTrans1)


"""
SAMPLE DOS
"""
#Constants
N_SAMPLE_PTS2 = 11
SAMPLE_LO2 = 16.7/2
SAMPLE_HI2 = 30.4/2
sspacing2 = (SAMPLE_HI2 - SAMPLE_LO2) / N_SAMPLE_PTS2

#Base x, y
xBase2 = list(linspace(SAMPLE_LO2, SAMPLE_HI2, N_SAMPLE_PTS2))
yBase2 = [func(x) for x in xBase2]

#Trans x, y
xTrans2 = fftfreq(n=N_SAMPLE_PTS2, d=sspacing2)
yTrans2 = fft(x=yBase2, n=N_SAMPLE_PTS2)

print(xTrans2)


print(IsContained(xTrans1, xTrans2, 1e-6))
