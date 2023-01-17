#Just doing this in 2D
from numpy import array, dot
from numpy.linalg import norm


#           x    y
b1 = array([2.1, 0.3])
b2 = array([0.1, 1.8])

ks = array([[0.0, 0.0], ##f0
            [0.6, -0.5], ##f1
            [0.2, 0.1], ##f2
            [0.8, 0.9], ##f3
            [-0.1, -0.1], ##f4
            [0.2, 0.3], ##f5
            [0.8, 0.1], ##f6
            [-0.9, -0.6]])##f7
Gcut = 10.6

one = Gcut/((dot(b1+b2, b1+b2))**(1/2)) ##low accuracy (best average)
two = 1/2 * (Gcut/((dot(b1, b1))**(1/2)) + Gcut/((dot(b2, b2))**(1/2))) ##high accuracy (larg val dominates)
print('', one, int(one + 0.5), "\n", two, int(two + 0.5)) ##int(float + 0.5) rounds float instead of flooring
                                                          ##beware of negatives tho! (not a problem here)

def Field1InField2(f1, f2, tol=1e-6):
    if(len(f1) > len(f2)): ##if num vecs in f1 > f2, then theres no way f1 is contained in f2
        return 0

    for i in range(0, len(f1)):
        thisv = f1[i]
        thisvInF2 = False
        for j in range(0, len(f2)):
            tthisv = f2[j]
            if(norm(array(thisv) - array(tthisv)) < tol):
                thisvInF2 = True
                break

        if(not thisvInF2):
            print(i, thisv, norm(array(thisv)))
            return 0

    return 1

def PlotDots(f, c, ms):
    x, y = [], []
    for i in range(0, len(f)):
        x.append(f[i][0])
        y.append(f[i][1])
    plt.plot(x, y, color=c, marker='o', markersize=ms, linewidth=0.)

#List all individual fields - dependent on k
allFields = [None for i in range(0, len(ks))]
for i in range(0, len(ks)):
    thisField = []
    k = ks[i]
    for hm in range(-30, 30 + 1):
        for km in range(-30, 30 + 1):
            G = hm*b1 + km*b2
            if(norm(k + G) < Gcut):
                thisField.append(G)
    allFields[i] = thisField[:]
#Now, we have all of the fields for each k point
for i in range(0, len(ks)):
    print("kpt", i, "  ", len(allFields[i]))


f0, f1, f2, f3, f4, f5, f6, f7 = allFields[0], allFields[1], allFields[2], allFields[3], allFields[4],\
                                 allFields[5], allFields[6], allFields[7]


#Get one large field that holds all subfields
#k points should be reduced fractional (|k| < 1.0).  Let the user do this - anyone using this should know
#better than to put k points outside of the BZ
habsmax, kabsmax = 0, 0
for i in range(0, len(ks)):
    k = ks[i] ##as long as ks are given in fractional coordinates (i.e. the basis of B), this works

    hmtmp = max(abs(int( (Gcut*Gcut/dot(b1, b1))**(1/2) - k[0])),
                abs(int(-(Gcut*Gcut/dot(b1, b1))**(1/2) - k[0])))
    kmtmp = max(abs(int( (Gcut*Gcut/dot(b2, b2))**(1/2) - k[1])),
                abs(int(-(Gcut*Gcut/dot(b2, b2))**(1/2) - k[1])))

    if(hmtmp > habsmax):
        habsmax = hmtmp
    if(kmtmp > kabsmax):
        kabsmax = kmtmp
print(habsmax, kabsmax)
maxField = []
for hm in range(-habsmax, habsmax+1):
    for km in range(-kabsmax, kabsmax+1):
        maxField.append(hm*b1 + km*b2)


#Check that the max field encloses all subfields
#for i in range(0, len(ks)):
#    print(Field1InField2(allFields[0], maxField))

from matplotlib import pyplot as plt
PlotDots(maxField, 'b', 10)
PlotDots(f0, 'g', 8)
PlotDots(f3, 'k', 6)
PlotDots(f7, 'r', 2.5)
plt.grid(alpha=0.5)
plt.plot([0.0], [0.0], marker='+', markersize=6, linewidth=0., color="blue")

plt.show()
