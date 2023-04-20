from matplotlib import pyplot as plt
from matplotlib import rc, rcParams
from numpy import linspace as ls
from numpy import polyfit as pf
from numpy import array as ar
from scipy.interpolate import CubicSpline as cs

"""
A good reference:
https://matplotlib.org/1.4.2/users/customizing.html
"""

# figure options
rcParams["font.family"] = "calibri"
rcParams["font.weight"] = "bold" ##use \mathbf{} or \bf{}
rcParams['axes.linewidth'] = 1.5
#rcParams["font.family"] = "times new roman"
rcParams["mathtext.fontset"] = "dejavusans"
#rcParams["mathtext.fontset"] = "dejavuserif"
plt.tight_layout = True
rcParams["figure.autolayout"] = False


#common constants
def WIDTH(ncol=1): ##inches
    if(ncol == 1):
        return 3.5 ##~ 90mm
    if(ncol == 1.5):
        return 5.5 ##~ 140mm
    if(ncol == 2):
        return 7.2 ##~ 180mm
    return none
def HEIGHT(nrow=1): ##inches, max is about 8.5in for a page
    if(nrow == 1):
        return 3.5 ##~ 90mm
    if(nrow == 1.5):
        return 5.5 ##~ 140mm
    if(nrow == 2):
        return 7.2 ##~ 180mm
    return none


GRID_ALPHA = 0.5

FTSIZE_SML = 10
FTSIZE_MED = 12
FTSIZE_BIG = 14

MKSIZE_SML = 1.50
MKSIZE_MED = 3.25
MKSIZE_BIG = 5.00

LNSIZE_SML = 0.8
LNSIZE_MED = 1.2
LNSIZE_BIG = 1.5

DPI = 1000

#useful functions
def FloatFormat(d, dec=3):
    return f'{d:.{dec}f}'

def RmAxTicks(ax, x=True, y=True):
    if(x):
        ax.tick_params(axis='x', size=0., width=0., color="white", pad=0., labelsize=0.)
    if(y):
        ax.tick_params(axis='y', size=0., width=0., color="white", pad=0., labelsize=0.)
    return ax

def SetTickSize(ax, fontsize=FTSIZE_SML, x=True, y=True):
    if(x):
        ax.tick_params(axis='x', labelsize=fontsize)
    if(y):
        ax.tick_params(axis='y', labelsize=fontsize)
    return ax

def PolyInterp(x, y, deg=1, npts=1000):
    xi, yi = (list(a) for a in zip(*sorted(zip(x, y))))
    fit = pf(xi, yi, deg=deg)

    xi = ls(start=xi[0], stop=xi[-1], num=npts, endpoint=True)
    yi = []
    for x_ in xi:
        sum = 0.
        for i in range(0, deg + 1):
            sum += fit[i]*x_**(deg - i)
        yi.append(sum)

    return xi, yi

def SplineInterp(x, y, npts=1000):
    spline = cs(x, y)
    xg = ls(x[0], x[-1], npts)
    yg = ar(spline(xg))

    return list(xg), list(yg)

#common colors
COLORS_DICT =  {"red":    "#ff0000",  # red
                "orange": "#eb7c00",  # orange
                "lime":   "#93ff27",  # lime
                "dgreen": "#00b423",  # d green
                "black":  "black",    # black
                "cyan":   "#00fdea",  # cyan
                "blue":   "#0058fd",  # blue
                "purple": "#cb34ff",  # purple
                "pink":   "#ff04b7",  # pink
                "white": "white"}     # white

#               red        d green       blue    purple    orange     purple    lime        pink
COLORS_LIST = ["#ff0000", "#00b423", "#0058fd", "#cb34ff", "#eb7c00", "black", "#93ff27", "#ff04b7",
#               cyan
               "#00fdea"]
#               red        d green       blue    orange    purple    purple    lime        pink
COLORS_LIST = ["#ff0000", "#00b423", "#0058fd", "#eb7c00", "#cb34ff" , "black", "#93ff27", "#ff04b7",
#               cyan
               "#00fdea"]
COLORS_LIST = COLORS_LIST + COLORS_LIST

MK_LIST = ['o', 's', '^', '|', '+', 'd', 'x', '1']
MK_LIST = MK_LIST + MK_LIST

LNSTYLE_LIST = ['-', '-.', '--', ':']
LNSTYLE_LIST = LNSTYLE_LIST + LNSTYLE_LIST + LNSTYLE_LIST + LNSTYLE_LIST

#useful symbols
ANGSTROM = 'Å'


#specific to ag-bi-i
NAME_DICT = {"i4": r"AgBiI$_4$",
             "i5": "Ag$_2$BiI$_5$",
             "i6": "Ag$_3$BiI$_6$",
             "i7": "AgBi$_2$I$_7$"}
