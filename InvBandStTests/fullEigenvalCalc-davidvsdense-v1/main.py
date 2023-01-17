#This is an example of how the main portion of solving for eigenvals/vecs will work
#in the c code
#It compares the results of ...
#  (1) Explicitly building the hamiltonian and using a dense eigensolver on it
#  to
#  (2) Using a combination of FFTs and Davidson's algorithm for minimum eigenpairs
#The implementations will eventually all be my own manual versions, so don't pay any attention to
#the speed of the program, only that the results are nearly identical

import headerdavidson as hdav
import headerfft as hfft
import headerrunparams as hrps ## definitions for the input file, hidden to avoid making this file too complex

N_EIGS = 8

#Finds maximum sampling index given a Gcut^2, the corresponding recip lattice vector b,
#and a k-point's fractional distance along b
from numpy import sqrt, dot
def GiveSampleMaxInd(gcut2, b, k):
    x = sqrt(gcut2/dot(b,b))
    return max(   abs(  int(+x - k)  ),
                  abs(  int(-x - k)  )   )


#Grab FFT(V*(G)) -> V(r)
from numpy import zeros
from scipy.fft import fftn, ifftn
sizesv = [hfft.NPO2(2*GiveSampleMaxInd(hrps.gcut2loc, hrps.B[0], hrps.KPT[0]) + 1), ##npo2 of 2hmax+1
          hfft.NPO2(2*GiveSampleMaxInd(hrps.gcut2loc, hrps.B[1], hrps.KPT[1]) + 1), ##npo2 of 2kmax+1
          hfft.NPO2(2*GiveSampleMaxInd(hrps.gcut2loc, hrps.B[2], hrps.KPT[2]) + 1)] ##npo2 of 2lmax+1
vggrid = zeros(shape=(sizesv[0], sizesv[1], sizesv[2]), dtype=complex)
vggrid = hfft.GetVg(sizes=sizesv, vg=vggrid,
                    atomsites=hrps.REAL_SITES_LOC, potparams=hrps.PPARAMS_LOC, B=hrps.B,
                    conj_=False) ##note! if conj = true, return conj(X) from DavidFFT()
vrgrid = ifftn(x=vggrid, s=(sizesv[0], sizesv[1], sizesv[2]),
               overwrite_x=False) * sizesv[0]*sizesv[1]*sizesv[2]

#Get the G vectors for this k point and energy cutoff
sizesp = [GiveSampleMaxInd(hrps.gcut2, hrps.B[0], hrps.KPT[0]), ##hmax
          GiveSampleMaxInd(hrps.gcut2, hrps.B[1], hrps.KPT[1]), ##kmax
          GiveSampleMaxInd(hrps.gcut2, hrps.B[2], hrps.KPT[2])] ##lmax

mills, gs = hrps.GetPsiList(B=hrps.B, kpt=hrps.KPT, millmax=sizesp, gcut2=hrps.gcut2)
npw = len(mills)

#Get eigenpairs the smart way
vas, ves = hdav.DavidFFT(n=npw, mills=mills, gs=gs, kpt=hrps.KPT, vrgrid=vrgrid, gridsizes=sizesv,
                         nEigs=N_EIGS, maxBasisSize=None, v0m=0, v0n=0, V0=None, eTol=1e-3)
#vas, ves = hdav.DavidCnv(n=npw, mills=mills, gs=gs, kpt=hrps.KPT, vggrid=vggrid, gridsizes=sizesv,
#                         nEigs=N_EIGS, maxBasisSize=None, v0m=0, v0n=0, V0=None, eTol=1e-3)

#Get eigenpairs the dumb way
from numpy import allclose, transpose, conj, max
from numpy.linalg import eig
A = hrps.BuildHamil(k=hrps.KPT, npw=npw, gs=gs, mils=mills)
vad, ved = eig(A)
sind = vad.argsort()[:N_EIGS]
vad = vad[sind]
ved = ved[:,sind]

#And compare the results
print("\n\n")
valsok = allclose(vas, vad, 1e-3, 1e-3)
print(valsok)
if(not valsok):
    print(vas - vad)
ves = transpose(ves)
vecsok = allclose(ved, ves, 1e-3, 1e-3)
print(vecsok)
if(not(vecsok)):
    print(max(ved - ves))

okay = True
for i in range(0, N_EIGS):
    okay = allclose(A@ves[:,i] - vas[i]*ves[:,i], 0.0, 1e-3, 1e-3)
    if(not okay):
        print(i, max(A@ves[:,i] - vas[i]*ves[:,i]))
