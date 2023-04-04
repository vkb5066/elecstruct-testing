#This is an example of how the main portion of solving for eigenvals/vecs will work
#in the c code
#It compares the results of ...
#  (1) Explicitly building the hamiltonian and using a dense eigensolver on it
#  to
#  (2) Using a combination of FFTs and Davidson's algorithm for minimum eigenpairs
#The implementations will eventually all be my own manual versions, so don't pay any attention to
#the speed of the program, only that the results are nearly identical

import headerlobcg as hlcg
import headerfft as hfft
import headerrunparams as hrps ## definitions for the input file, hidden to avoid making this file too complex
from copy import deepcopy
from random import seed



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
sizesv = [max(2, n) for n in sizesv]
print(sizesv)
vggrid = zeros(shape=(sizesv[0]*sizesv[1]*sizesv[2]), dtype=complex)
vggrid = hfft.GetVg(sizes=sizesv, vg=vggrid,
                    atomsites=hrps.REAL_SITES_LOC, potparams=hrps.PPARAMS_LOC, B=hrps.B,
                    conj_=False) ##note! if conj = true, return conj(X) from DavidFFT()

from numpy import empty
buff = empty(shape=max([sizesv[0], sizesv[1], sizesv[2]]), dtype=complex)
powsv = [hfft.NLO2(n) for n in sizesv]
vrgrid = hfft._fftconv3basedif(arr=deepcopy(vggrid), buff=buff, pows=powsv, sizes=sizesv)
#Get the G vectors for this k point and energy cutoff
sizesp = [GiveSampleMaxInd(hrps.gcut2, hrps.B[0], hrps.KPT[0]), ##hmax
          GiveSampleMaxInd(hrps.gcut2, hrps.B[1], hrps.KPT[1]), ##kmax
          GiveSampleMaxInd(hrps.gcut2, hrps.B[2], hrps.KPT[2])] ##lmax

mills, gs = hrps.GetPsiList(B=hrps.B, kpt=hrps.KPT, millmax=sizesp, gcut2=hrps.gcut2)
npw = len(mills)

N_EIGS = 2
seed(0xBEEFBABE)
TEST_FOLDSPEC = 1
eref = -4
if(not TEST_FOLDSPEC):
    #Get eigenpairs the smart way
    vas, ves = hlcg.locg(npw=npw, mills=mills, gs=gs, kpt=hrps.KPT, vrgrid=vrgrid, gridsizes=sizesv,
                         neigs=N_EIGS, v0m=0, v0n=0, V0=None, tol=1e-3,
                         fsm=TEST_FOLDSPEC, eref=0.0, stab=1)
    print(vas)

    #Get eigenpairs the dumb way
    from numpy import sort
    from numpy.linalg import eig
    from scipy.sparse.linalg import eigs
    #A = hrps.BuildHamil(k=hrps.KPT, npw=npw, gs=gs, mils=mills)
    A = hrps.BuildHamil(k=hrps.KPT, npw=npw, gs=gs, mils=mills, uvgg=0, vgg=vggrid, sizes=sizesv)
    #with open("out", 'w') as outfile:
    #    outfile.write(str(npw) + "\n")
    #    for i in range(0, npw):
    #        for j in range(0, npw):
    #            outfile.write(str(i) + ' ' + str(j) + ' ' + str(A[i][j].real) + ' ' + str(A[i][j].imag) + "\n")
    #    exit(3)
    vad, ved = eig(A)
    sind = vad.argsort()[:N_EIGS]
    vad = vad[sind]
    ved = ved[:,sind]

    #from scipy.sparse.linalg import lobpcg
    #from random import random
    #from numpy import conj, transpose
    #X = empty(shape=(npw, N_EIGS), dtype=complex)
    #for i in range(0, N_EIGS):
    #    for j in range(0, npw):
    #        X[j][i] = random() + 1j * random()
    #for j in range(0, N_EIGS):
    #    X[:, j] /= sqrt(conj(transpose(X[:, j])) @ X[:, j])
    #res = lobpcg(A@A, X, verbosityLevel=0, largest=False, maxiter=50)
    #print("lobcg: ", res[0])
    #print("actual:", vad.real)

    #print("\ndifferences in sparse vecs vs dumb vecs")
    #vad2, ved2 = eigs(A, k=N_EIGS, which='SR')
    #for i in range(0, N_EIGS):
    #    diff = min(max(ved[:,i] - ved2[:,i]), max(ved[:,i] + ved2[:,i])) ##even normalized, may differ in sign
    #    print(i, abs(diff))

    #And compare the results
    from numpy import allclose, transpose, conj, round
    from numpy.linalg import norm
    print("\n\n")
    valsok = allclose(vas, vad, 1e-3, 1e-3)
    print("dav vals match with dumb vals?", valsok)
    if(not valsok):
        print(vas - vad)

    ves = transpose(ves)
    okay = True
    print("\nerrors in H|psi> = e|psi> from dav")
    for i in range(0, N_EIGS):
        okay = allclose(A@ves[:,i] - vas[i]*ves[:,i], 0.0, 1e-3, 1e-3)
        if(not okay):
            print(i, max(A@ves[:,i] - vas[i]*ves[:,i]))

    print("\nerrors in H|psi> = e|psi> from dumb method")
    for i in range(0, N_EIGS):
        okay = allclose(A@ved[:,i] - vad[i]*ved[:,i], 0.0, 1e-3, 1e-3)
        if(not okay):
            print(i, max(A@ved[:,i] - vad[i]*ved[:,i]))

    #this one is slightly worrying, but maybe shouldn't be ... I mean, Hx - ex ~ 0, and x are normalized
    #so what gives?  Maybe repeated eigenvalues?
    #print("\ndifferences in dav vecs vs dumb vecs")
    #for i in range(0, N_EIGS):
    #    diff = min(max(ved[:,i] - ves[:,i]), max(ved[:,i] + ves[:,i])) ##even normalized, may differ in sign
    #    print(i, abs(diff))

    #print("\nnorms (dumb, dav)")
    #for i in range(0, N_EIGS):
    #    print(i, norm(ved[:,i]), norm([ves[:,i]]))

if(TEST_FOLDSPEC):
    # Get eigenpairs the smart way
    A = hrps.BuildHamil(k=hrps.KPT, npw=npw, gs=gs, mils=mills)
    vas, ves = hlcg.locg(npw=npw, mills=mills, gs=gs, kpt=hrps.KPT, vrgrid=vrgrid, gridsizes=sizesv,
                         neigs=N_EIGS, v0m=0, v0n=0, V0=None, tol=1e-3,
                         fsm=TEST_FOLDSPEC, eref=eref, stab=1)

    # Get eigenpairs the dumb way
    from numpy import sort
    from numpy.linalg import eig, eigh

    A = hrps.BuildHamil(k=hrps.KPT, npw=npw, gs=gs, mils=mills)
    vad, ved = eigh(A)
    ###get those closest to eref
    distsfromeref = []
    for i in range(0, npw):
        distsfromeref.append(abs(vad[i].real - eref))
    vad = vad.tolist()
    distsfromeref, vad = (list(t) for t in zip(*sorted(zip(distsfromeref, vad))))
    vad = vad[:N_EIGS]
    vad = sorted([v.real for v in vad])

    #And finially compare eigenvalues
    print("My eigs, closest to", eref)
    print(vas.real)
    print("dumb eigs closest to", eref)
    print(vad)
    print("max diff =", max([abs(v.real - q) for v, q in zip(vas, vad)]))

    print("\nerrors in H|psi> = e|psi> from dav")
    from numpy import allclose, transpose
    ves = transpose(ves)
    for i in range(0, N_EIGS):
        okay = allclose(A@ves[:,i] - vas[i]*ves[:,i], 0.0, 1e-3, 1e-3)
        if(not okay):
            print(i, max(A@ves[:,i] - vas[i]*ves[:,i]))

    from scipy.sparse.linalg import eigs as seigs
    res = seigs(A=A, k=1, sigma=7.5)
    print(res[0])
