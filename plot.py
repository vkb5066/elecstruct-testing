#!/usr/bin/python3.11

from globalfig import *

INFILE_LOC = "data.csv"
N_ATOMS = 6 + 6 + 24 ## 2x2x1 supercell of AgBiI4

#Reads csv and returns [energy], [bandgap]
def ReadData(infileLoc):
    nrg, bg = [], []
    with open(infileLoc, 'r') as infile:
        for n, lin in enumerate(infile):
            if(n == 0):
                continue
            line = lin.split(',')
            print(line)
            nrg.append(float(line[2]))
            bg.append(float(line[14]))

        infile.close()
    return nrg, bg

#read, transform data: energy -> meV/atom above min
x, y = ReadData(INFILE_LOC)
x = [1000*(x_ - min(x))/N_ATOMS for x_ in x]

#plotting junk
from matplotlib import pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(WIDTH(1), HEIGHT(1)))

ax.set_xticks([480/4*i for i in range(0, 5)])
ax.set_yticks([1.8/6*i for i in range(0, 7)])
ax.set_ylim(0.0, 1.6)
ax.set_xlabel(r"$E-E_\mathrm{min}$ (meV/atom)", fontsize=FTSIZE_MED, fontweight="bold")
ax.set_ylabel("Band Gap (eV)", fontsize=FTSIZE_MED, fontweight="bold")

ax = SetTickSize(ax, FTSIZE_MED)
ax.xaxis.set_tick_params(width=1.5)
ax.yaxis.set_tick_params(width=1.5)

ax.grid(alpha=GRID_ALPHA)

ax.plot(x, y, color=COLORS_DICT["black"], marker='.', markersize=MKSIZE_MED, linewidth=0.)


plt.subplots_adjust(left=0.180, right=0.980, bottom=0.143, top=0.995, hspace=0.2, wspace=0.2)
#plt.show()
plt.savefig("bgvsnrg.png", dpi=DPI)
