#Testing how to choose params of the fft such that we correctly sample all necessary frequencies
from numpy import round

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



def NP2(u):
    pass


b1 = 0.7548
def MakeFreks(hm):
    res = [0.0]
    for i in range(1, hm+1):
        res.append(  1/(2*b1) * (i/hm)  )
    for i in range(1, hm+1):
        res.append(  1/(2*b1) * (i/hm - 1/(hm) - 1)  )
    return res

print(round(sorted(MakeFreks(4)), 3))
