#Returns the power of 2 closest to unsigned int n that is >= n
def NPO2(n):
    p = 1 ##uint
    if(n and not (n&(n-1))):
        return n

    while (p < n):
        p <<= 1

    return p

#Returns the power that is given to 2 to yield n (i.e. NLO2(8) = 3 since 2^3 = 8)
def NLO2(n):
    r = 0
    n >>= 1
    while(n): ##r=0; while(n >>= 1){ r++; } return r
        n >>= 1
        r += 1
    return r


def _acc3(i, j, k, lx, ly, lz):
    #return i*(ly*lz) + j*(lz) + k
    return (i*ly + j)*lz + k


#Returns the size of the fft grid given a miller index m that satisfies m*|b| >= Gcut, m=|m|
def GiveFFTSize(millermaxabsind, B):
    return max(4, NPO2(4*m+1)) ##always must be at least 4 to avoid problems with fft algo

#Gets V on the reciprocal space grid, given sizes[3] are powers of 2 sizes for the fft grid
#V should already be memset to zero
#atomsites and potparams are the locations and potential params as dictionaries whose keys are identical
from numpy import dot, sqrt, conj, pi, exp, round
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
                #print(h, k, l)

                #Compute the local potential for this h, k, l
                g[0] = float(h)*B[0][0] + float(k)*B[1][0] + float(l)*B[2][0]
                g[1] = float(h)*B[0][1] + float(k)*B[1][1] + float(l)*B[2][1]
                g[2] = float(h)*B[0][2] + float(k)*B[1][2] + float(l)*B[2][2]
                q = sqrt(dot(g, g))

                ind = _acc3(hi, ki, li, sizes[0], sizes[1], sizes[2])
                #print(ind, h, k, l, hi, ki, li, round(q, 3), ":")
                for elem in atomsites.keys():
                    s = hrps.StructFact(g, elem)
                    va = hrps.LocAtomPot(q, elem)
                #    print("  ", elem, round(s, 5), round(va, 5))
                    vg[ind] += s*va
                #print("|-->", round(vg[ind], 5), "\n")
                if(conj_):
                    vg[ind] = conj(vg[ind])

    #exit(2)
    return vg

#Gets psi on the reciprocal space grid
#coeffs are in-line with mills
#psigrid should already be memset to zero
def GetPsig(npw, mills, coeffs, sizes, psigrid):
    for i in range(0, npw):
        hi = mills[i][0] + (sizes[0] >> 1)
        ki = mills[i][1] + (sizes[1] >> 1)
        li = mills[i][2] + (sizes[2] >> 1)

        ind = _acc3(hi, ki, li, sizes[0], sizes[1], sizes[2])
        psigrid[ind] = coeffs[i]

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

        ind = _acc3(hi, ki, li, sizes[0], sizes[1], sizes[2])
        coeffs[i] = psigrid[ind]

    return coeffs



#main fft stuff ----------------------------------------------------------------------------------------------


#Algos for bit reversal---------------------------------------
def revbinupdate(a, b):
    ## a ^= b means a = a ^ b (^ is XOR)
    ## used to be 'revbinupdate(a, b); while(!(a ^= b) & b); {b >>= 1}; return a;'
    a = a^b
    while(not((a) & b)):
        b >>= 1
        a = a^b
    return a

#Permutation of arr's indices (arr is length n) that undo the bit-reversal that the dif/dit in-place fft algos
#do.  also needed is the half length of the array, nh = n >> 1
def _revbinpermute(arr, n, nh):
    if(n <= 2):
        return arr

    i = 0
    j = 1

    while(j < nh):
        #odd j portion
        i += nh
        arr[j], arr[i] = arr[i], arr[j]
        j += 1

        #even j portion
        i = revbinupdate(i, nh)
        if(i > j):
            arr[j], arr[i] = arr[i], arr[j]
            arr[n-j-1], arr[n-i-1] = arr[n-i-1], arr[n-j-1]
        j += 1

    return arr



#Cos/Sin tables for recurrence.  Declare these static, const, maybe extern, and hard code them in
L2M_MAX = 32 ##all fft grid sizes must be <= 2^(L2M_MAX)
CT = [None for i in range(0, L2M_MAX)] ##covers all the way up to 2^32
ST = [None for i in range(0, L2M_MAX)] ##might reasonably only go up to 2^16 or so ...

from numpy import cos, sin
for i in range(2, L2M_MAX):
    d = (1 << i) >> 1
    CT[i] = cos(pi/float(d)) ##note that the first 2 entries (arr[0], arr[1]) are completly unused
    ST[i] = sin(pi/float(d)) ##maybe store some useful constants in there?


#FFT Algos---------------------------------------------------
#1D DIT FFT algo with no bit reversal section
#this one is arranged for good memory access, but is most efficient when a trig table is used due to the
#increased number of calls to complex exp
#Needs the array to edit, the log_2(its size), its size, and x, the value in exp(ixj) (may be neg for ifft)
#Note1: permute is done BEFORE the dit algo ... NOTE2: x is often just isign * pi
def _fftbasedit(arr, l2n, n, x):
    isign = -1 if x < 0. else +1 ##really, just consider sending in isign to this function then setting
                                 ## ipi = isign*pi

    ##explicitly do the trivial 1+0i multiplications
    for i in range(0, n, 2):
        u = arr[i] + arr[i+1]
        v = arr[i] - arr[i+1]
        arr[i]   = u
        arr[i+1] = v
    ##do the rest of the transform
    for ldm in range(2, l2n+1):
        m = 1<<ldm
        mh = m>>1
        phi = x / float(mh)

        for i in range(0, n, m):
            ###again, explicit 1+0i multiplications
            ijmh = i + mh
            u = arr[i] + arr[ijmh]
            v = arr[i] - arr[ijmh]
            arr[i]    = u
            arr[ijmh] = v

            ###trig recurr init
            ##see https://en.wikipedia.org/wiki/Trigonometric_tables if we need less prec loss
            wr = CT[ldm]
            wi = isign*ST[ldm]
            cn = wr*1 - wi*0
            sn = wi*1 + wr*0
            for j in range(1, mh):
                ij = i + j
                ijmh = ij + mh

                u = arr[ij]
                #v = arr[ijmh] * (cn + 1.0j*sn) #exp(phi*float(j)*1.0j)
                v = (arr[ijmh].real*cn - arr[ijmh].imag*sn) + 1j*(arr[ijmh].real*sn + arr[ijmh].imag*cn)
                arr[ij]   = u + v
                arr[ijmh] = u - v

                ##update trig rec
                cnt = cn
                cn = wr*cn - wi*sn
                sn = wi*cnt + wr*sn

    return arr

#1D DIF FFT algo with no bit reversal section
#this one is arranged for good memory access, but is most efficient when a trig table is used due to the
#increased number of calls to complex exp
#Needs the array to edit, the log_2(its size), its size, and x, the value in exp(ixj) (may be neg for ifft)
#Note1: permute is done AFTER the dif algo ... NOTE2: x is often just isign * pi
def _fftbasedif(arr, l2n, n, x, p=0):
    isign = -1 if x < 0. else +1 ##really, just consider sending in isign to this function then setting
                                 ## ipi = isign*pi
    #if(p):
    #    print(arr, "\n\n")
    ##do most of the transform
    for ldm in range(l2n, 1, -1): ##ldm=l2n; ldm >= 2; --ldm
        m = 1<<ldm
        mh = m>>1
        phi = x / float(mh)

        #if(p):
        #    print(ldm, m, mh)

        for i in range(0, n, m):
            ###again, explicit 1+0i multiplications
            ijmh = i + mh
            u = arr[i] + arr[ijmh]
            v = arr[i] - arr[ijmh]
            arr[i]    = u
            arr[ijmh] = v

            ###trig recurr init
            ##see https://en.wikipedia.org/wiki/Trigonometric_tables if we need less prec loss
            wr = CT[ldm]
            wi = isign*ST[ldm]
            cn = wr*1 - wi*0
            sn = wi*1 + wr*0
            for j in range(1, mh):
                ij = i + j
                ijmh = ij + mh

                u = arr[ij]
                v = arr[ijmh]
                umv = u - v
                arr[ij]   =  u + v
                #arr[ijmh] = (umv) * (cn + 1.0j*sn) #exp(phi*float(j)*1.0j)
                arr[ijmh] = (umv.real*cn - umv.imag*sn) + 1j*(umv.real*sn + umv.imag*cn)

                ##update trig rec
                cnt = cn
                cn = wr*cn - wi*sn
                sn = wi*cnt + wr*sn


    ##explicitly do the trivial 1+0i multiplications
    for i in range(0, n, 2):
        #if(p):
        #    print(i, u, v, arr[i], arr[i+1])
        u = arr[i] + arr[i+1]
        v = arr[i] - arr[i+1]
        arr[i]   = u
        arr[i+1] = v
        #if (p):
            #print(i, u, v, arr[i], arr[i + 1], "\n")


    #if(p):
        #print("\n\n", arr)
        #exit(30)
    return arr






"""
This stuff won't work now that I've changed indices to be c-friendly.  
See 'fft3d-simple-difdit-1dinput-reorg.py' vs. the same title w/o '-input' for how to deal with this, if it
becomes necessary (it prob. wont)


def fft3dit(arr, buff, pows, sizes, isign):
    ##permute---------------------------------------------------------
    for i in range(0, sizes[0]):
        for j in range(0, sizes[1]):
            for k in range(0, sizes[2]):
                buff[k] = arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])]
            buff[0:sizes[2]] = _revbinpermute(arr=buff[0:sizes[2]], n=sizes[2], nh=sizes[2]//2)
            for k in range(0, sizes[2]):
                arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])] = buff[k]
    for i in range(0, sizes[0]):
        for k in range(0, sizes[2]):
            for j in range(0, sizes[1]):
                buff[j] = arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])]
            buff[0:sizes[1]] = _revbinpermute(arr=buff[0:sizes[1]], n=sizes[1], nh=sizes[1]//2)
            for j in range(0, sizes[1]):
                arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])] = buff[j]
    for j in range(0, sizes[1]): ##we actually don't need a buffer for this one
        for k in range(0, sizes[2]):
            lo = _acc3(0, j, k,        sizes[0], sizes[1], sizes[2])
            hi = _acc3(sizes[0], j, k, sizes[0], sizes[1], sizes[2])
            arr[lo:hi] = _revbinpermute(arr=arr[lo:hi], n=sizes[0], nh=sizes[0]//2)

    ##fft------------------------------------------------------------
    for i in range(0, sizes[0]):
        for j in range(0, sizes[1]):
            for k in range(0, sizes[2]):
                buff[k] = arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])]
            buff[0:sizes[2]] = _fftbasedit(arr=buff[0:sizes[2]], l2n=pows[2], n=sizes[2], x=isign*pi)
            for k in range(0, sizes[2]):
                arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])] = buff[k]
    for i in range(0, sizes[0]):
        for k in range(0, sizes[2]):
            for j in range(0, sizes[1]):
                buff[j] = arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])]
            buff[0:sizes[1]] = _fftbasedit(arr=buff[0:sizes[1]], l2n=pows[1], n=sizes[1], x=isign*pi)
            for j in range(0, sizes[1]):
                arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])] = buff[j]
    for j in range(0, sizes[1]): ##we actually don't need a buffer for this one
        for k in range(0, sizes[2]):
            lo = _acc3(0, j, k,        sizes[0], sizes[1], sizes[2])
            hi = _acc3(sizes[0], j, k, sizes[0], sizes[1], sizes[2])
            arr[lo:hi] = _fftbasedit(arr=arr[lo:hi], l2n=pows[0], n=sizes[0], x=isign*pi)

    return arr


def fft3dif(arr, buff, pows, sizes, isign):
    ##fft------------------------------------------------------------
    for i in range(0, sizes[0]):
        for j in range(0, sizes[1]):
            for k in range(0, sizes[2]):
                buff[k] = arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])]
            buff[0:sizes[2]] = _fftbasedif(arr=buff[0:sizes[2]], l2n=pows[2], n=sizes[2], x=isign*pi)
            for k in range(0, sizes[2]):
                arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])] = buff[k]
    for i in range(0, sizes[0]):
        for k in range(0, sizes[2]):
            for j in range(0, sizes[1]):
                buff[j] = arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])]
            buff[0:sizes[1]] = _fftbasedif(arr=buff[0:sizes[1]], l2n=pows[1], n=sizes[1], x=isign*pi)
            for j in range(0, sizes[1]):
                arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])] = buff[j]
    for j in range(0, sizes[1]): ##we actually don't need a buffer for this one
        for k in range(0, sizes[2]):
            lo = _acc3(0, j, k,        sizes[0], sizes[1], sizes[2])
            hi = _acc3(sizes[0], j, k, sizes[0], sizes[1], sizes[2])
            arr[lo:hi] = _fftbasedif(arr=arr[lo:hi], l2n=pows[0], n=sizes[0], x=isign*pi)

    ##permute---------------------------------------------------------
    for i in range(0, sizes[0]):
        for j in range(0, sizes[1]):
            for k in range(0, sizes[2]):
                buff[k] = arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])]
            buff[0:sizes[2]] = _revbinpermute(arr=buff[0:sizes[2]], n=sizes[2], nh=sizes[2]//2)
            for k in range(0, sizes[2]):
                arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])] = buff[k]
    for i in range(0, sizes[0]):
        for k in range(0, sizes[2]):
            for j in range(0, sizes[1]):
                buff[j] = arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])]
            buff[0:sizes[1]] = _revbinpermute(arr=buff[0:sizes[1]], n=sizes[1], nh=sizes[1]//2)
            for j in range(0, sizes[1]):
                arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])] = buff[j]
    for j in range(0, sizes[1]): ##we actually don't need a buffer for this one
        for k in range(0, sizes[2]):
            lo = _acc3(0, j, k,        sizes[0], sizes[1], sizes[2])
            hi = _acc3(sizes[0], j, k, sizes[0], sizes[1], sizes[2])
            arr[lo:hi] = _revbinpermute(arr=arr[lo:hi], n=sizes[0], nh=sizes[0]//2)

    return arr




#This is a bit better than the above one at the cost of twice (!) the memory
def fft3dif_(arr, pows, sizes, isign):
    tmp = empty(shape=sizes[0]*sizes[1]*sizes[2], dtype=complex)

    ##fft------------------------------------------------------------
    for j in range(0, sizes[1]):
        for k in range(0, sizes[2]):
            lo = _acc3(0, j, k, sizes[0], sizes[1], sizes[2])
            arr[lo:lo+sizes[0]] = _fftbasedif(arr=arr[lo:lo+sizes[0]], l2n=pows[0], n=sizes[0], x=isign*pi)
            ##exchange ijk -> jki
            for i in range(0, sizes[0]):
                tmp[_acc3(j, k, i, sizes[1], sizes[2], sizes[0])] = arr[lo + i]
    for k in range(0, sizes[2]):
        for i in range(0, sizes[0]):
            lo = _acc3(0, k, i, sizes[1], sizes[2], sizes[0])
            tmp[lo:lo+sizes[1]] = _fftbasedif(arr=tmp[lo:lo+sizes[1]], l2n=pows[1], n=sizes[1], x=isign*pi)
            ##exchange jki -> kij
            for j in range(0, sizes[1]):
                arr[_acc3(k, i, j, sizes[2], sizes[0], sizes[1])] = tmp[lo + j]
    for i in range(0, sizes[0]):
        for j in range(0, sizes[1]):
            lo = _acc3(0, i, j, sizes[2], sizes[0], sizes[1])
            arr[lo:lo+sizes[2]] = _fftbasedif(arr=arr[lo:lo+sizes[2]], l2n=pows[2], n=sizes[2], x=isign*pi)
            ##exchange kij -> ijk
            for k in range(0, sizes[2]):
                tmp[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])] = arr[lo + k]


    ##permute---------------------------------------------------------
    for j in range(0, sizes[1]):
        for k in range(0, sizes[2]):
            lo = _acc3(0, j, k, sizes[0], sizes[1], sizes[2])
            tmp[lo:lo+sizes[0]] = _revbinpermute(arr=tmp[lo:lo+sizes[0]], n=sizes[0], nh=sizes[0]//2)
            ##exchange ijk -> jki
            for i in range(0, sizes[0]):
                arr[_acc3(j, k, i, sizes[1], sizes[2], sizes[0])] = tmp[lo + i]
    for k in range(0, sizes[2]):
        for i in range(0, sizes[0]):
            lo = _acc3(0, k, i, sizes[1], sizes[2], sizes[0])
            arr[lo:lo+sizes[1]] = _revbinpermute(arr=arr[lo:lo+sizes[1]], n=sizes[1], nh=sizes[1]//2)
            ##exchange jki -> kij
            for j in range(0, sizes[1]):
                tmp[_acc3(k, i, j, sizes[2], sizes[0], sizes[1])] = arr[lo + j]
    for i in range(0, sizes[0]):
        for j in range(0, sizes[1]):
            lo = _acc3(0, i, j, sizes[2], sizes[0], sizes[1])
            tmp[lo:lo+sizes[2]] = _revbinpermute(arr=tmp[lo:lo+sizes[2]], n=sizes[2], nh=sizes[2]//2)
            ##exchange kij -> ijk
            for k in range(0, sizes[2]):
                arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])] = tmp[lo + k]

    return arr




#This is a bit better than the above one at the cost of twice (!) the memory
def fft3dit_(arr, pows, sizes, isign):
    tmp = empty(shape=sizes[0]*sizes[1]*sizes[2], dtype=complex)

    ##permute---------------------------------------------------------
    for j in range(0, sizes[1]):
        for k in range(0, sizes[2]):
            lo = _acc3(0, j, k, sizes[0], sizes[1], sizes[2])
            arr[lo:lo+sizes[0]] = _revbinpermute(arr=arr[lo:lo+sizes[0]], n=sizes[0], nh=sizes[0]//2)
            ##exchange ijk -> jki
            for i in range(0, sizes[0]):
                tmp[_acc3(j, k, i, sizes[1], sizes[2], sizes[0])] = arr[lo + i]
    for k in range(0, sizes[2]):
        for i in range(0, sizes[0]):
            lo = _acc3(0, k, i, sizes[1], sizes[2], sizes[0])
            tmp[lo:lo+sizes[1]] = _revbinpermute(arr=tmp[lo:lo+sizes[1]], n=sizes[1], nh=sizes[1]//2)
            ##exchange jki -> kij
            for j in range(0, sizes[1]):
                arr[_acc3(k, i, j, sizes[2], sizes[0], sizes[1])] = tmp[lo + j]
    for i in range(0, sizes[0]):
        for j in range(0, sizes[1]):
            lo = _acc3(0, i, j, sizes[2], sizes[0], sizes[1])
            arr[lo:lo+sizes[2]] = _revbinpermute(arr=arr[lo:lo+sizes[2]], n=sizes[2], nh=sizes[2]//2)
            ##exchange kij -> ijk
            for k in range(0, sizes[2]):
                tmp[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])] = arr[lo + k]

    ##fft------------------------------------------------------------
    for j in range(0, sizes[1]):
        for k in range(0, sizes[2]):
            lo = _acc3(0, j, k, sizes[0], sizes[1], sizes[2])
            tmp[lo:lo+sizes[0]] = _fftbasedit(arr=tmp[lo:lo+sizes[0]], l2n=pows[0], n=sizes[0], x=isign*pi)
            ##exchange ijk -> jki
            for i in range(0, sizes[0]):
                arr[_acc3(j, k, i, sizes[1], sizes[2], sizes[0])] = tmp[lo + i]
    for k in range(0, sizes[2]):
        for i in range(0, sizes[0]):
            lo = _acc3(0, k, i, sizes[1], sizes[2], sizes[0])
            arr[lo:lo+sizes[1]] = _fftbasedit(arr=arr[lo:lo+sizes[1]], l2n=pows[1], n=sizes[1], x=isign*pi)
            ##exchange jki -> kij
            for j in range(0, sizes[1]):
                tmp[_acc3(k, i, j, sizes[2], sizes[0], sizes[1])] = arr[lo + j]
    for i in range(0, sizes[0]):
        for j in range(0, sizes[1]):
            lo = _acc3(0, i, j, sizes[2], sizes[0], sizes[1])
            tmp[lo:lo+sizes[2]] = _fftbasedit(arr=tmp[lo:lo+sizes[2]], l2n=pows[2], n=sizes[2], x=isign*pi)
            ##exchange kij -> ijk
            for k in range(0, sizes[2]):
                arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])] = tmp[lo + k]

    return arr





#Very special cases ------------------------------------------------------------------------------------------






#The first part of a fft convolution, acts on one input, transforming it w/o bit reversal by forward dif fft
def _fftconv3basedif_old(arr, buff, pows, sizes):
    for i in range(0, sizes[0]):
        for j in range(0, sizes[1]):
            for k in range(0, sizes[2]):
                buff[k] = arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])]
            buff[0:sizes[2]] = _fftbasedif(arr=buff[0:sizes[2]], l2n=pows[2], n=sizes[2], x=+pi)
            for k in range(0, sizes[2]):
                arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])] = buff[k]
    for i in range(0, sizes[0]):
        for k in range(0, sizes[2]):
            for j in range(0, sizes[1]):
                buff[j] = arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])]
            buff[0:sizes[1]] = _fftbasedif(arr=buff[0:sizes[1]], l2n=pows[1], n=sizes[1], x=+pi)
            for j in range(0, sizes[1]):
                arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])] = buff[j]
    for j in range(0, sizes[1]): ##we actually don't need a buffer for this one
        for k in range(0, sizes[2]):
            lo = _acc3(0, j, k,        sizes[0], sizes[1], sizes[2])
            hi = _acc3(sizes[0], j, k, sizes[0], sizes[1], sizes[2])
            arr[lo:hi] = _fftbasedif(arr=arr[lo:hi], l2n=pows[0], n=sizes[0], x=+pi)

    return arr

#The second part of a fft convolution, acts on one input, transforming it w/o bit reversal by inverse dit fft
def _fftconv3basedit_old(arr, buff, pows, sizes):
    for i in range(0, sizes[0]):
        for j in range(0, sizes[1]):
            for k in range(0, sizes[2]):
                buff[k] = arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])]
            buff[0:sizes[2]] = _fftbasedit(arr=buff[0:sizes[2]], l2n=pows[2], n=sizes[2], x=-pi)
            for k in range(0, sizes[2]):
                arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])] = buff[k]
    for i in range(0, sizes[0]):
        for k in range(0, sizes[2]):
            for j in range(0, sizes[1]):
                buff[j] = arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])]
            buff[0:sizes[1]] = _fftbasedit(arr=buff[0:sizes[1]], l2n=pows[1], n=sizes[1], x=-pi)
            for j in range(0, sizes[1]):
                arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])] = buff[j]
    for j in range(0, sizes[1]): ##we actually don't need a buffer for this one
        for k in range(0, sizes[2]):
            lo = _acc3(0, j, k,        sizes[0], sizes[1], sizes[2])
            hi = _acc3(sizes[0], j, k, sizes[0], sizes[1], sizes[2])
            arr[lo:hi] = _fftbasedit(arr=arr[lo:hi], l2n=pows[0], n=sizes[0], x=-pi)

    return arr
"""



# --- VERY VERY SPECIAL CASES --------------------------------------------------------------------------------
from numpy import empty
#The first part of a fft convolution, acts on one input, transforming it w/o bit reversal by forward dif fft
#Better than above at the cost of more memory
#TODO: allocate tmp outside of loop, watch out for memory leaks! this (maybe) returns tmp, not arr!
#(you could just swap ptrs i.e. SWAP(tmp, arr) before returning, then you can always free tmp w/o worrying)
def _fftconv3basedif(arr, pows, sizes,
                     buff=None): ###buff isn't used, but i kept it here to be compatible with old fun calls
    tmp = empty(shape=sizes[0]*sizes[1]*sizes[2], dtype=complex)
    #print("IJK")
    for i in range(0, sizes[0]): ##ijk loop
        for j in range(0, sizes[1]):
            lo = _acc3(i, j, 0, sizes[0], sizes[1], sizes[2])
            arr[lo:lo+sizes[2]] = _fftbasedif(arr=arr[lo:lo+sizes[2]], l2n=pows[2], n=sizes[2], x=+pi)
            for k in range(0, sizes[2]): ##permute ijk -> jki
                tmp[_acc3(j, k, i, sizes[1], sizes[2], sizes[0])] = arr[lo + k]
    #print("JKI")
    for j in range(0, sizes[1]): ##jki loop
        for k in range(0, sizes[2]):
            lo = _acc3(j, k, 0, sizes[1], sizes[2], sizes[0])
            tmp[lo:lo+sizes[0]] = _fftbasedif(arr=tmp[lo:lo+sizes[0]], l2n=pows[0], n=sizes[0], x=+pi)
            for i in range(0, sizes[0]): ##permute jki -> kij
                #print(lo, _acc3(k, i, j, sizes[2], sizes[0], sizes[1]), lo+i)
                arr[_acc3(k, i, j, sizes[2], sizes[0], sizes[1])] = tmp[lo + i]
    #print("KIJ")
    for k in range(0, sizes[2]): ##kij loop
        for i in range(0, sizes[0]):
            lo = _acc3(k, i, 0, sizes[2], sizes[0], sizes[1])
            #print(lo, "b4", arr[lo])
            arr[lo:lo+sizes[1]] = _fftbasedif(arr=arr[lo:lo+sizes[1]], l2n=pows[1], n=sizes[1], x=+pi, p=1)
            ##we can skip this so long as we're careful to do the dit algo backwards
            ##for j in range(0, sizes[1]): ##permute kij -> ijk
            ##    tmp[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])] = arr[lo + j]
    #        print(lo, "af", arr[lo])
    return arr



#The first part of a fft convolution, acts on one input, transforming it w/o bit reversal by forward dif fft
#Better than above at the cost of more memory
#TODO: allocate tmp outside of loop, watch out for memory leaks! this (maybe) returns tmp, not arr!
#(you could just swap ptrs i.e. SWAP(tmp, arr) before returning, then you can always free tmp w/o worrying)
##NOTE: ASSUMES THAT THIS CAME FROM THE ABOVE DIF ALGO W/O THE LAST TRANSPOSE I.E. THE INPUT IS COMING IN
##      KIJ ORDER.  THIS RETURNS BACK TO NORMAL ORDERING
def _fftconv3basedit(arr, pows, sizes,
                     buff=None): ###buff isn't used, but i kept it here to be compatible with old code
    tmp = empty(shape=sizes[0]*sizes[1]*sizes[2], dtype=complex)

    for k in range(0, sizes[2]): ##kij loop
        for i in range(0, sizes[0]):
            lo = _acc3(k, i, 0, sizes[2], sizes[0], sizes[1])
            arr[lo:lo+sizes[1]] = _fftbasedit(arr=arr[lo:lo+sizes[1]], l2n=pows[1], n=sizes[1], x=-pi)
            for j in range(0, sizes[1]): ##permute kij -> jki
                tmp[_acc3(j, k, i, sizes[1], sizes[2], sizes[0])] = arr[lo + j]
    for j in range(0, sizes[1]): ##jki loop
        for k in range(0, sizes[2]):
            lo = _acc3(j, k, 0, sizes[1], sizes[2], sizes[0])
            tmp[lo:lo+sizes[0]] = _fftbasedit(arr=tmp[lo:lo+sizes[0]], l2n=pows[0], n=sizes[0], x=-pi)
            for i in range(0, sizes[0]): ##permute jki -> ijk
                arr[_acc3(i, j, k, sizes[0], sizes[1], sizes[2])] = tmp[lo + i]
    for i in range(0, sizes[0]): ##ijk loop
        for j in range(0, sizes[1]):
            lo = _acc3(i, j, 0, sizes[0], sizes[1], sizes[2])
            arr[lo:lo+sizes[2]] = _fftbasedit(arr=arr[lo:lo+sizes[2]], l2n=pows[2], n=sizes[2], x=-pi)
            ##if I did this correctly, we're back at ijk (the original) ordering.  No need to permute again
            ##for k in range(0, sizes[2]): ##permute ijk -> kij
            ##    tmp[_acc3(k, i, j, sizes[2], sizes[0], sizes[1])] = arr[lo + k]

    return arr
