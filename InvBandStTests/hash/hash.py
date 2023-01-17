
##Hash testing
MAX_SIZE = 999999*8
MAX_HKL = 75


if(1):
    P, Q, R = 2971, 6277, 7907
    def hash(a, b, c):
        return (a*P + b*Q + c*R) % MAX_SIZE

    li = [0 for i in range(0, MAX_SIZE)]
    colls = 0
    for h in range(0, MAX_HKL):
        print(h)
        for k in range(0, MAX_HKL):
            for l in range(0, MAX_HKL):
                ke = hash(h, k, l)
                if(li[ke]): ##not used
                    colls += 1
                li[ke] = 1 ##used

    print("\n")
    print(colls / MAX_SIZE)
    exit(1)


##How often things near zero testing
hkls = []
for h in range(-MAX_HKL, MAX_HKL + 1):
    for k in range(-MAX_HKL, MAX_HKL + 1):
        for l in range(-MAX_HKL, MAX_HKL + 1):
            hkls.append([h, k, l])

md = (3*2*MAX_HKL)**3
dists2, counts = [i for i in range(0, md + 1)], [0 for i in range(0, md + 1)]
for n, gi in enumerate(hkls):
    print(n)
    for gj in hkls:
        d = (gi[0]-gj[0])**2 + (gi[1]-gj[1])**2 + (gi[2]-gj[2])**2
        counts[d] += 1
