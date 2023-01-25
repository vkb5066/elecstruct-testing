#Stuff for linear interpolation of discrete pseudopotential points
#avoids a bunch of calls to some potentially pretty expensive functions

from numpy import sin, cos, linspace

#hidden function of q^2 that we don't know about
def f(q2):
    return 2.4 + 1.27*sin(5.6*q2) + 0.448*cos(1.9*q2)

#inputs: {V(q^2)} = the local atomic potential as a function of q^2 = |Gi - Gj|
#                   (this is spherically symmetric, pure real)
#        qlo = lowest sampling point of q.  Anything lower is set to 0
#        qhi = highest sampling point of q.  Anything higher is set to 0
#        nq = number of q sample points.  signifies that {q} is an array from qlo to qhi of nq sample pts
q2lo = 0.42
q2hi = 2.48
nq2 = 25
#the true samples v(q^2) given as input
q2arr = linspace(start=q2lo, stop=q2hi, num=nq2)
vq2s = [f(q2) for q2 in q2arr]

#create a linear interpolation for any requested input from q2lo to q2hi
dq2inv = (nq2-1) / (q2hi - q2lo) ##inverse of q sample spacing i.e. 1 / (q2(i+1) - q2(i))

#for i in range(0, nq2):
#    print(i, q2arr[i])
#print("---")
def getlinterp(q2req):
    ##sample indices (low, high)
    ssam = (q2req - q2lo)*dq2inv
    loi = int(ssam)
    hii = loi + 1

    if(loi < 0 or hii >= nq2):
        return 0.

    #lin interp
    fra = ssam - float(loi)
    return vq2s[loi]*(1.0-fra) + vq2s[hii]*(fra)

#now sample the linear interpolation more densly than the number of sample points
q2ds = list(linspace(start=q2lo, stop=q2hi-0.001, num=3*nq2))
q2ds += [0.5, 0.55, 0.9, 0.95, 1.2, 1.3, 1.4, 2.08, 2.09, 2.10]
vq2ds = [getlinterp(s) for s in q2ds]


#plot
from matplotlib import pyplot as plt
plt.plot(q2arr,
         vq2s,
         color='k', marker='o', linewidth=0., label="true vq2 samples")
#a fine interpolation of v(q^2) to compare our linear interpolation to
plt.plot(linspace(start=q2lo, stop=q2hi, num=1000),
         [f(q2) for q2 in linspace(start=q2lo, stop=q2hi, num=1000)],
         color='b', label="exact vq2 function")
#numpy's linear interpolation
plt.plot(linspace(start=q2lo, stop=q2hi, num=nq2),
         [f(q2) for q2 in linspace(start=q2lo, stop=q2hi, num=nq2)],
         color='g', label="exact vq2 lin interp")
#our linear interpolated dense points
plt.plot(q2ds,
         vq2ds,
         color='r', marker='.', linewidth=0., label="approximate q2 by lin interp")


plt.legend()
plt.show()
