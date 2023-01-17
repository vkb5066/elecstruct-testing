#Testing my understanding of applying the action of a wavefunction to a real-space local potential

USE_SCIPY_CONVENTIONS = 1 ##changes a few factors of -1
ENCUT = 115 ##eV
FFT_GCUT_MULT = 2.50 ##between 1 (a rough approximation) and 2 (100% match w/ naive multiplication)
                     ##VASP uses 1.75 by default

from numpy import array, dot, cross, transpose, sqrt, conj, pi, zeros, empty, exp
from scipy.fft import fft, fftfreq, ifft

def NPO2(n):
    p = 1 ##uint
    if(n and not (n&(n-1))):
        return n

    while (p < n):
        p <<= 1

    return p


class gv:
    h, k, l = None, None, None
    G  = None##h*b1 + k*b2 + l*b3

    def __init__(self, _h, _k, _l, _g):
        self.h = _h
        self.k = _k
        self.l = _l
        self.G = _g


#Real lattice
A1 = array([4.2,  0.4,  0.2])
A2 = array([-0.1, 4.6,  0.3])
A3 = array([0.3,  0.2,  3.8])
#Reciprocl lattice
scvol = dot(A1, cross(A2, A3)) / (2*pi)
B1 = cross(A2, A3)/scvol
B2 = cross(A3, A1)/scvol
B3 = cross(A1, A2)/scvol


#Gcut is prop to sqrt(encut)
hbar2over2m = 3.81
gcut2 = ENCUT/hbar2over2m
gcut2loc = FFT_GCUT_MULT*FFT_GCUT_MULT*gcut2 ##multiplied twice since we only keep gcut^2

#A sample K point
KPT = array([0.2, -0.1, 0.3]) ##fractional, along b1, b2, b3

#Calculate maximum h, k, l needed to describe V in real space (taken as the maximum over all k-points)
hmaxloc = max(abs(int(+sqrt(gcut2loc/dot(B1,B1)) - KPT[0])),
              abs(int(-sqrt(gcut2loc/dot(B1,B1)) - KPT[0])))
kmaxloc = max(abs(int(+sqrt(gcut2loc/dot(B2,B2)) - KPT[1])),
              abs(int(-sqrt(gcut2loc/dot(B2,B2)) - KPT[1])))
lmaxloc = max(abs(int(+sqrt(gcut2loc/dot(B3,B3)) - KPT[2])),
              abs(int(-sqrt(gcut2loc/dot(B3,B3)) - KPT[2])))

#Calculate maximum h, k, l needed to grab all possible G vectors for this specific k point
hmax = max(abs(int(+sqrt(gcut2/dot(B1,B1)) - KPT[0])),
           abs(int(-sqrt(gcut2/dot(B1,B1)) - KPT[0])))
kmax = max(abs(int(+sqrt(gcut2/dot(B2,B2)) - KPT[1])),
           abs(int(-sqrt(gcut2/dot(B2,B2)) - KPT[1])))
lmax = max(abs(int(+sqrt(gcut2/dot(B3,B3)) - KPT[2])),
           abs(int(-sqrt(gcut2/dot(B3,B3)) - KPT[2])))

#Get the nearest ceil power of 2 for fft sizes (always >= minimum needed sizes 2*(hkl)+1) ...
fftsh, fftsk, fftsl = NPO2(2*hmaxloc+1), NPO2(2*kmaxloc+1), NPO2(2*lmaxloc+1)
# ... these should be at least 4 (size 0 is an error, size 1 is odd, size 2 only includes -h, 0, and
#size 3 isn't a power of 2)
assert fftsh > 3
assert fftsk > 3
assert fftsl > 3

#Get the spacings for the fft grids
spach, spack, spacl = sqrt(dot(B1, B1)), sqrt(dot(B2, B2)), sqrt(dot(B3, B3))

#Setup the fft grid for V
#vgrid = zeros(shape=(2*hmaxloc+1, 2*kmaxloc+1, 2*lmaxloc+1), dtype=complex)
vgrid = zeros(shape=(fftsh, fftsk, fftsl), dtype=complex)

#Setup the field of G vectors needed to describe |psi> for this specific k point ...
psis = []
for h in range(-hmax, +hmax+1):
    for k in range(-kmax, +kmax+1):
        for l in range(-lmax, +lmax+1):
            g = h*B1 + k*B2 + l*B3
            if(dot(KPT+g, KPT+g) < gcut2):
                psis.append(gv(h, k, l, g))
npw = len(psis)

#... and make a fft grid for |psi> (for now, have it in-line with V(r))
psigrid = zeros(shape=(fftsh, fftsk, fftsl), dtype=complex)

#Make a random trial |psi> to compute its action on the hamiltonian
from random import random, seed
seed(69420)
PSI_TRIAL = array([random() + random()*1j for i in range(0, npw)])
## ... we like the wavefunction to be normalized.  Not sure if its 100% mandatory here, but jic
PSI_TRIAL = PSI_TRIAL / sqrt(dot(conj(PSI_TRIAL), PSI_TRIAL))


# --- HERE BEGINS THE STUFF NEEDED TO SETUP A PHYSICALLY REALISTIC HAMILTONIAN - SPECIFICALLY, THE ---
# --- ATOM COORDINATES ARE NEEDED FOR STRUCTURE FACTORS AND V(G) ARE NEEDED TO CALCULATE THE GRID  ---
# --- V(R)                                                                                         ---
def FracToCart(A1, A2, A3, crds):
    A = array([A1, A2, A3])
    return list(dot(transpose(A), transpose(crds)))

REAL_SITES_LOC = {
                    "Ga": [FracToCart(A1, A2, A3, [0.6666, 0.3033, 0.4000]),
                           FracToCart(A1, A2, A3, [0.4363, 0.6666, 0.0000])],
                    "As": [FracToCart(A1, A2, A3, [0.6566, 0.3213, 0.8740]),
                           FracToCart(A1, A2, A3, [0.3233, 0.5666, 0.3600])]
                 }
#                     a1,    a2,    a3,    a4    in Ryd
PPARAMS_LOC = {"Ga": [1.22/2,  2.45,  0.54, -2.71],
               "As": [0.35/2,  2.62,  0.93,  1.57 ]}

def LocKin(k, Gi, Gj):
    return hbar2over2m*dot(k + Gi.G, k + Gi.G)

##Local potential stuff
def StructFact(G, elem):
    S = 0.0 + 0.0j
    for i in range(0, len(REAL_SITES_LOC[elem])):
        gdottau = dot(G, REAL_SITES_LOC[elem][i])
        S += exp(-1j * gdottau)

    return 1/len(REAL_SITES_LOC[elem]) * S

def LocAtomPot(q, elem):
    q2 = dot(q, q)
    if(q2 < 1e-7):
        return 0.0 + 0.0j

    ##mind units for real implementation! no reason here since im just comparing hamiltonians
    ##but these are usually given as a function of bohr radaii, and the output in Ry or Ha
    a1, a2, a3, a4 = PPARAMS_LOC[elem][0], PPARAMS_LOC[elem][1], PPARAMS_LOC[elem][2], PPARAMS_LOC[elem][3]
    x = a3*(q2-a4)
    if(x > 500.):
        return 0.0 + 0.0j
    return a1*(q2-a2) / (1+exp(x))*50

def LocPot(Gi, Gj):
    q = sqrt(dot(Gi.G - Gj.G, Gi.G - Gj.G))
    ret = 0.0 + 0.0j
    for elem in REAL_SITES_LOC.keys():
        ret += StructFact(Gi.G - Gj.G, elem) * LocAtomPot(q, elem)
    return ret



# ---------------------------------------------------------------------------------------------------
# ------------------------------- THE ACTION H |psi> ... THE DUMB WAY -------------------------------
# ---------------------------------------------------------------------------------------------------
from numpy import round
print("easy")

#The full hamiltonian (only explicitly created for testing - never actually do this unless
#nrows < 100 (FFT_GCUT_MULT=1) to 700 (FFT_GCUT_MULT=2) according to martins1998) has the same
#dimensions as psi
npw = len(psis)
hamil = empty(shape=(npw, npw), dtype=complex)
Teasy, VEasy = zeros(shape=(npw, npw), dtype=complex), zeros(shape=(npw, npw), dtype=complex)
for i in range(0, npw):
    hamil[i][i] = LocKin(KPT, psis[i], psis[i])
    Teasy[i][i] = hamil[i][i]

    for j in range(i+1, npw):
        hamil[i][j] = LocPot(psis[i], psis[j])
        hamil[j][i] = conj(hamil[i][j])
        VEasy[i][j] = hamil[i][j]
        VEasy[j][i] = hamil[j][i]

#Compute H |psi>
actiondumb = dot(hamil, PSI_TRIAL)
tparteasy = dot(Teasy, PSI_TRIAL)
vparteasy = dot(VEasy, PSI_TRIAL)
actiondumbSep = tparteasy + vparteasy

# ---------------------------------------------------------------------------------------------------
# ----------------------------- THE ACTION H |psi> ... THE NOT DUMB WAY -----------------------------
# ---------------------------------------------------------------------------------------------------
print("smart")

#Set V on the recip grid ... in reality, this should be done once at the beginning of the program
#and then kept constant for the rest ...
for hi in range(0, fftsh):
    for ki in range(0, fftsk):
        for li in range(0, fftsl):
            ##Transform indices from 0 -> fft size to -hmax, ..., 0, ..., +hmax, etc
            ##really, these should be in their respective loops and ffts*//2 set in local variables
            h = hi - fftsh//2
            k = ki - fftsk//2
            l = li - fftsl//2

            ##really, this corresponds to Gi - Gj
            G = h*B1 + k*B2 + l*B3
            q = sqrt(dot(G, G))

            #vgrid[hi][ki][li] = 0.0 + 0.0j ##already memset to zero
            for elem in REAL_SITES_LOC.keys():
                vgrid[hi][ki][li] += StructFact(G, elem) * LocAtomPot(q, elem)

# ... and transform it to real space and keep it there
from scipy.fft import fftn, ifftn
vgrid = ifftn(x=vgrid, s=(fftsh, fftsk, fftsl), overwrite_x=False) * fftsh*fftsk*fftsl

#The diagonal (in recip space) kinetic energy term
Tpsi = empty(shape=npw, dtype=complex) ##pure real
for i in range(0, npw):
    Tpsi[i] = LocKin(KPT, psis[i], psis[i]) * PSI_TRIAL[i]

#Place the wavefunction on the recip space grid
for i in range(0, len(psis)):
    h, k, l = psis[i].h, psis[i].k, psis[i].l

    ##Transform from -hmax, ..., 0, ... +hmax -> h index in the fft array, etc
    ##really, ffts*//2 should be set in local variables
    hi = h + fftsh//2
    ki = k + fftsk//2
    li = l + fftsl//2

    psigrid[hi][ki][li] = PSI_TRIAL[i]

#Transform the wavefunction into real space
psigrid = ifftn(x=psigrid, s=(fftsh, fftsk, fftsl), overwrite_x=False) * fftsh*fftsk*fftsl

#And compute the action V |psi> in real space -> set into psigrid
#Note that (I'M PRETTY SURE) that V(r) is 100% real
for hi in range(0, fftsh):
    for ki in range(0, fftsk):
        for li in range(0, fftsl):
            psigrid[hi][ki][li] = psigrid[hi][ki][li] * vgrid[hi][ki][li]

#Transform the product V |psi> back into reciprocal space ...
psigrid = fftn(x=psigrid, s=(fftsh, fftsk, fftsl), overwrite_x=False) / (fftsh*fftsk*fftsl)

SI = 6
print("fft size", fftsh, fftsk, fftsl)
print("Searching recip space grid ... ")
from numpy import allclose
val = vparteasy[SI]
for hi in range(0, fftsh):
    for ki in range(0, fftsk):
        for li in range(0, fftsl):
            h = hi - fftsh//2
            k = ki - fftsk//2
            l = li - fftsl//2

            if(allclose(psigrid[hi][ki][li], val, 1e-6, 1e-6)):
                print(h, k, l, "->", hi, ki, li)
                print(round(psigrid[hi][ki][li], 5), round(val, 5))
                print("\n")
print("done\n\n")
print("The coordinates that I'm grabbing for V|psi> ...")
# ... and put it back into the recip space vector of psis
for i in range(0, len(psis)):
    h, k, l = psis[i].h, psis[i].k, psis[i].l

    ##Transform from -hmax, ..., 0, ... +hmax -> h index in the fft array, etc
    ##really, ffts*//2 should be set in local variables
    hi = h + fftsh//2
    ki = k + fftsk//2
    li = l + fftsl//2
    if(i == SI):
        print(i, "...", h, k, l, "->", hi, ki, li)
    PSI_TRIAL[i] = psigrid[hi][ki][li] #* fftsh*fftsk*fftsl
print("done")
#for i in range(0, npw):
#    print(psis[i].h, psis[i].k, psis[i].l, round(vparteasy[i], 5), round(PSI_TRIAL[i], 5))

print("the coordinates that i shoiuld be (???) grabbing for V|psi> ...")
for i in range(0, len(psis)):
    h, k, l = psis[i].h, psis[i].k, psis[i].l

    ##Transform from -hmax, ..., 0, ... +hmax -> h index in the fft array, etc
    ##really, ffts*//2 should be set in local variables
    hi = (h + fftsh)%fftsh
    ki = (k + fftsk)%fftsk
    li = (l + fftsl)%fftsl
    #Alternatevly, ffts* are guarenteed to be powers of 2 ... so we can do some
    #magic to speed up the calculation by an unnoticeable amount while also making
    #the code much harder to understand.  Great idea!
    hi = (h+fftsh) & (fftsh-1)
    ki = (k+fftsk) & (fftsk-1)
    li = (l+fftsl) & (fftsl-1)
    if(i == SI):
        print(i, "...", h, k, l, "->", hi, ki, li)
    PSI_TRIAL[i] = psigrid[hi][ki][li] #* fftsh*fftsk*fftsl
print("done")


#Now add the kinetic and potential energy terms for the final result
actionsmart = Tpsi + PSI_TRIAL

from numpy import max
print(max(actiondumb - actionsmart))
#print(actiondumb)
#print(actionsmart)
#print(actiondumb - actiondumbSep)
