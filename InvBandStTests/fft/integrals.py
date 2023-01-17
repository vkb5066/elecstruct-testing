a1 = [4.0, 0.0, 2.0]
a2 = [0.0, 3.0, 0.0]
a3 = [0.0, 1.0, 5.0]

A = [a1, a2, a3]


N_GP_A1 = 175
N_GP_A2 = 175
N_GP_A3 = 175


#--- FUNCTIONS -----------------------------------------------------------------------------------------------
class gridpoint:
    crystal = None
    cartesian = None
    val = None

    def __init__(self):
        self.crystal = []
        self.cartesian = []
        self.val = 0

def DirectToCartesian(v, A):
    ret = [Dot([A[0][0], A[1][0], A[2][0]], v),
           Dot([A[0][1], A[1][1], A[2][1]], v),
           Dot([A[0][2], A[1][2], A[2][2]], v)]
    return ret

def Dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def Cross(a, b):
    return [+(a[1]*b[2] - a[2]*b[1]), -(a[0]*b[2] - a[2]*b[0]), +(a[0]*b[1] - a[1]*b[0])]

#--- END -----------------------------------------------------------------------------------------------------


#Fill grid
print("init grid...")
a1m = (Dot(a1, a1))**(1/2)
a2m = (Dot(a2, a2))**(1/2)
a3m = (Dot(a3, a3))**(1/2)
from numpy import linspace
GRID = []
for n1, a1itr in enumerate(linspace(0, a1, N_GP_A1)):
    for n2, a2itr in enumerate(linspace(0, a2, N_GP_A2)):
        for n3, a3itr in enumerate(linspace(0, a3, N_GP_A3)):
            g = gridpoint()
            g.crystal = [n1/(N_GP_A1 - 1), n2/(N_GP_A2 - 1), n3/(N_GP_A3 - 1)]
            #g.cartesian = list(a1itr + a2itr + a3itr)
            g.cartesian = DirectToCartesian(g.crystal, A)
            GRID.append(g)
print("done")


def TrueVol(A):
    return Dot(A[0], Cross(A[1], A[2]))
dADirect = [[1/N_GP_A1, 0., 0.],
            [0., 1/N_GP_A2, 0.],
            [0., 0., 1/N_GP_A3]]
dV = Dot(DirectToCartesian(dADirect[0], A), Cross(DirectToCartesian(dADirect[1], A),
                                                  DirectToCartesian(dADirect[2], A)))
print("dV:", dV)



print("True Vol:", TrueVol(A))
print("Approx Vol:", sum([1 for i in range(0, len(GRID))])*dV)


#--- BEGIN ACTUAL INTEGRALS ----------------------------------------------------------------------------------
##set
from numpy import sin
for g in GRID:
    x, y, z = g.cartesian[0], g.cartesian[1], g.cartesian[2]
    g.val = 5.2*sin(4.1*x) + 1.37*sin(9-y) + 2/(z + 1)
##integrate
su = 0.
for g in GRID:
    su += g.val
su = su*dV
print("Approx Integral:", su)

#--- END ACTUAL INTEGRALS ------------------------------------------------------------------------------------




#Plot
print("plotting...")
if(0):
    from mpl_toolkits import mplot3d
    from matplotlib import pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    xpcr, ypcr, zpcr = [], [], []
    xpca, ypca, zpca = [], [], []
    for g in GRID:
        xpcr.append(g.crystal[0])
        ypcr.append(g.crystal[1])
        zpcr.append(g.crystal[2])
        xpca.append(g.cartesian[0])
        ypca.append(g.cartesian[1])
        zpca.append(g.cartesian[2])
    ax.scatter(xpcr, ypcr, zpcr, color='k', s=5)
    ax.scatter(xpca, ypca, zpca, color='r', s=2.5)

    plt.show()
print("done")
