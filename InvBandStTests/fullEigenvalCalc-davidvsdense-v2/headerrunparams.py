#Input params like POSCAR, energy cutoff, ... etc
#Hidden in this file to avoid overcomplicating main

#######################
# LATTICE DEFINITIONS #
#######################
from numpy import array, dot, cross, pi, transpose, exp, sqrt
hbar2over2m = 3.81
#Real lattice
a = 5.65
A1 = array([0.0, a/2, a/2])
A2 = array([a/2, 0.0, a/2])
A3 = array([a/2, a/2, 0.0])
del a

A = array([A1, A2, A3])
#Reciprocl lattice
scvol = dot(A1, cross(A2, A3)) / (2*pi)
B1 = cross(A2, A3)/scvol
B2 = cross(A3, A1)/scvol
B3 = cross(A1, A2)/scvol
B = array([B1, B2, B3])
#Just a sample k-point
KPT = array([0.0, 0.0, 0.0]) ##fractional, along b1, b2, b3  3333333333333333333333333333333333333333333333333333

def FracToCart(A1, A2, A3, crds):
    A = array([A1, A2, A3])
    return list(dot(transpose(A), transpose(crds)))

REAL_SITES_LOC = {
                    "Ga": [FracToCart(A1, A2, A3, [0.00, 0.00, 0.00])],
                    "As": [FracToCart(A1, A2, A3, [0.25, 0.25, 0.25])]
                 }
#                     a1,    a2,    a3,    a4    in Ryd
PPARAMS_LOC = {"Ga": [1.22/2,  2.45,  0.54, -2.71],
               "As": [0.35/2,  2.62,  0.93,  1.57 ]}

#def LocKin(k, Gi, Gj):
#    k, Gi, Gj = array(k), array(Gi), array(Gj)
#    return hbar2over2m*dot(k + Gi, k + Gi)
def LocKin2(k, hm, km, lm): ##k in fractional coords
    kpg = empty(shape=3)
    hpk, kpk, lpk = k[0] + hm, k[1] + km, k[2] + lm
    kpg[0] = hpk*B[0][0] + kpk*B[1][0] + lpk*B[2][0]
    kpg[1] = hpk*B[0][1] + kpk*B[1][1] + lpk*B[2][1]
    kpg[2] = hpk*B[0][2] + kpk*B[1][2] + lpk*B[2][2]

    return hbar2over2m*dot(kpg, kpg)

##Local potential stuff
def StructFact(G, elem):
    S = 0.0 + 0.0j
    for i in range(0, len(REAL_SITES_LOC[elem])):
        gdottau = dot(G, REAL_SITES_LOC[elem][i])
        S += exp(-1j * gdottau)

    return 1/len(REAL_SITES_LOC[elem]) * S

def LocAtomPot(q, elem):
    q2 = dot(q, q)
    if(q2 < 1e-10):
        #print("uh oh!!!", q)
        return 0.0 + 0.0j
    b2a = 0.529177
    r2e = 13.61
    q2 *= b2a*b2a

    ##mind units for real implementation! no reason here since im just comparing hamiltonians
    ##but these are usually given as a function of bohr radaii, and the output in Ry or Ha
    a1, a2, a3, a4 = PPARAMS_LOC[elem][0], PPARAMS_LOC[elem][1], PPARAMS_LOC[elem][2], PPARAMS_LOC[elem][3]
    x = a3*(q2-a4)
    return a1*(q2-a2) / (1+exp(x)) * r2e

def LocPot(Gi, Gj):
    Gi, Gj = array(Gi), array(Gj)
    q = sqrt(dot(Gi - Gj, Gi - Gj))
    ret = 0.0 + 0.0j
    for elem in REAL_SITES_LOC.keys():
        ret += StructFact(Gi - Gj, elem) * LocAtomPot(q, elem)
    return ret


####################
# CUTOFF CONSTANTS #
####################
ENCUT = 150 ##eV
FFT_GCUT_MULT = 1.00 ##between 1 (a rough approximation) and 2 (100% match w/ naive multiplication)
                     ##VASP uses 1.75 by default
#Gcut is prop to sqrt(encut)
gcut2 = ENCUT/hbar2over2m
gcut2loc = FFT_GCUT_MULT*FFT_GCUT_MULT*gcut2 ##multiplied twice since we only keep gcut^2

def _acc3(i, j, k, lx, ly, lz):
    #return i*(ly*lz) + j*(lz) + k
    return (i*ly + j)*lz + k

#########################
# HAMILTONIAN FUNCTIONS #
#########################
from numpy import empty, conj
def BuildHamil(k, npw, gs, mils, uvgg=0, vgg=None, sizes=None):
    hamil = empty(shape=(npw, npw), dtype=complex)
    if(not uvgg):
        for i in range(0, npw):
            #hamil[i][i] = LocKin(k=k, Gi=gs[i], Gj=gs[i])
            hamil[i][i] = LocKin2(k=k, hm=mils[i][0], km=mils[i][1], lm=mils[i][2])

            for j in range(i+1, npw):
                hamil[i][j] = LocPot(Gi=gs[i], Gj=gs[j])
                hamil[j][i] = conj(hamil[i][j])
    if(uvgg):
        for i in range(0, npw):
           # hamil[i][i] = LocKin(k=k, Gi=gs[i], Gj=gs[i])
            hamil[i][i] = LocKin2(k=k, hm=mils[i][0], km=mils[i][1], lm=mils[i][2])

            for j in range(i + 1, npw):
                hkls = array(mils[i]) - array(mils[j])
                inh = hkls[0] + (sizes[0] >> 1)
                ink = hkls[1] + (sizes[1] >> 1)
                inl = hkls[2] + (sizes[2] >> 1)
                #inh = (hkls[0] + sizes[0]) & (sizes[0] - 1)
                #ink = (hkls[1] + sizes[1]) & (sizes[1] - 1)
                #inl = (hkls[2] + sizes[2]) & (sizes[2] - 1)
                if(inh < 0 or ink < 0 or inl < 0):
                    # print(inh, ink, inl, _acc3(inh, ink, inl, sizes[0], sizes[1], sizes[2]))
                    #print(hkls)
                    hamil[i][j] = 0. + 0.j
                    hamil[j][i] = 0. + 0.j
                else:
                    #print(inh, ink, inl, _acc3(inh, ink, inl, sizes[0], sizes[1], sizes[2]))
                    hamil[i][j] = vgg[_acc3(inh, ink, inl, sizes[0], sizes[1], sizes[2])]
                    hamil[j][i] = conj(hamil[i][j])




    return hamil


###################
# OTHER FUNCTIONS #
###################

#Retruns a list of G vectors for all |k+G| < Gcut in line with their miller indices
#i.e. hkls, gs in-line arrays s.t. hkls[i] = [h_i, k_i, l_i] and gs[i] = h_i*b1 + k_i*b2 + l_i*b3
#B = [[b1x b1y b1z], [b2x, b2y, b2z], [b3x, b3y, b3z]]
#kpt = in fractional coordinates along b
#millmax[3] = [habsmax, kabsmax, labsmax]
def GetPsiList(B, kpt, millmax, gcut2):
    size = (2*millmax[0]+1) * (2*millmax[1]+1) * (2*millmax[2]+1)
    hkls = [[None, None, None] for i in range(0, size)]
    gs = [[None, None, None] for i in range(0, size)] ##CHECK: do I ever actually use this?

    counter = 0
    for h in range(-millmax[0], +millmax[0]+1):
        for k in range(-millmax[1], +millmax[1]+1):
            for l in range(-millmax[2], +millmax[2]+1):
                #print(h, k, l)
                #k+G = kpt*b1 + h*b1 + kpt*b2 + k*b2 + kpt*b3 + l*b3
                hpk, kpk, lpk = float(h) + kpt[0], float(k) + kpt[1], float(l) + kpt[2]
                gs[counter][0] = hpk*B[0][0] + kpk*B[1][0] + lpk*B[2][0]
                gs[counter][1] = hpk*B[0][1] + kpk*B[1][1] + lpk*B[2][1]
                gs[counter][2] = hpk*B[0][2] + kpk*B[1][2] + lpk*B[2][2]

                if(dot(gs[counter], gs[counter]) < gcut2):
                    hkls[counter][0] = h
                    hkls[counter][1] = k
                    hkls[counter][2] = l
                    gs[counter][0] = float(h)*B[0][0] + float(k)*B[1][0] + float(l)*B[2][0]
                    gs[counter][1] = float(h)*B[0][1] + float(k)*B[1][1] + float(l)*B[2][1]
                    gs[counter][2] = float(h)*B[0][2] + float(k)*B[1][2] + float(l)*B[2][2]

                    counter += 1

    hkls = hkls[:counter]
    gs = gs[:counter]
    return hkls, gs
