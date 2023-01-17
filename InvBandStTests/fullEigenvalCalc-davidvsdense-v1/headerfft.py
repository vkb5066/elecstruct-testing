#Returns the power of 2 closest to unsigned int n that is >= n
def NPO2(n):
    p = 1 ##uint
    if(n and not (n&(n-1))):
        return n

    while (p < n):
        p <<= 1

    return p

#Returns the size of the fft grid given a miller index m that satisfies m*|b| >= Gcut, m=|m|
def GiveFFTSize(millermaxabsind, B):
    return max(4, NPO2(2*m+1)) ##always must be at least 4 to avoid problems with fft algo

#Gets V on the reciprocal space grid, given sizes[3] are powers of 2 sizes for the fft grid
#V should already be memset to zero
#atomsites and potparams are the locations and potential params as dictionaries whose keys are identical
from numpy import dot, sqrt, conj
import headerrunparams as hrps
def GetVg(sizes, vg, atomsites, potparams, B, conj_=False):
    halfsizeh = sizes[0] >> 1
    halfsizek = sizes[1] >> 1
    halfsizel = sizes[2] >> 1
    g = [None, None, None]

    for hi in range(0, sizes[0]):
        h = hi - halfsizeh
        for ki in range(0, sizes[1]):
            k = ki - halfsizek
            for li in range(0, sizes[2]):
                l = li - halfsizel

                #Compute the local potential for this h, k, l
                g[0] = float(h)*B[0][0] + float(k)*B[1][0] + float(l)*B[2][0] #+ kpt[0]
                g[1] = float(h)*B[0][1] + float(k)*B[1][1] + float(l)*B[2][1] #+ kpt[1]
                g[2] = float(h)*B[0][2] + float(k)*B[1][2] + float(l)*B[2][2] #+ kpt[2]
                q = sqrt(dot(g, g))

                for elem in atomsites.keys():
                    vg[hi][ki][li] += hrps.StructFact(g, elem) * hrps.LocAtomPot(q, elem)
                if(conj_):
                    vg[hi][ki][li] = conj(vg[hi][ki][li])

    return vg

#Gets psi on the reciprocal space grid
#coeffs are in-line with mills
#psigrid should already be memset to zero
def GetPsig(npw, mills, coeffs, sizes, psigrid):
    for i in range(0, npw):
        hi = mills[i][0] + (sizes[0] >> 1)
        ki = mills[i][1] + (sizes[1] >> 1)
        li = mills[i][2] + (sizes[2] >> 1)

        psigrid[hi][ki][li] = coeffs[i]

    return psigrid


#Transforms psi on the reciprocal grid back into the coefficients of the coefficient vector
def GetPsiv(npw, mills, coeffs, psigrid, sizes):
    hmod = sizes[0] - 1
    kmod = sizes[1] - 1
    lmod = sizes[2] - 1

    for i in range(0, npw):
        hi = (mills[i][0] + sizes[0]) & hmod
        ki = (mills[i][1] + sizes[1]) & kmod
        li = (mills[i][2] + sizes[2]) & lmod

        coeffs[i] = psigrid[hi][ki][li]

    return coeffs
