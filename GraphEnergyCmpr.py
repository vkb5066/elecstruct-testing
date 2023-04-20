from headerSimAnn import *
from globalFig import *

FOLDER = "simann-energyCmpr-nn1//"
BASE_NAME_TST = FOLDER + "tst" ##name of split files without the final extensions
BASE_NAME_TRN = FOLDER + "trn" ##name of split files without the final extensions
LATTICE_FILE_NAME = FOLDER + "lattice.tsam"
N_ATOMS = 6 + 6 + 24 ## 2x2x1 supercell of AgBiI4

#Reads csv and returns a dict of training and testing data
#The format of the returns are {idNum: [DFT energy, None]} (none is meant to be filled later with the
#predictions)
#
def ReadData(infileLoc):
    trn, tst = dict(), dict()
    with open(infileLoc, 'r') as infile:
        for n, lin in enumerate(infile):
            if(n == 0):
                continue
            line = lin.split(',')

            path = line[0]
            id = int(line[1])
            nrg = float(line[2])

            ##Decide if train or test by string in first line
            if("all_train" in path):
                trn[id] = [nrg, None]
            elif("all_test" in path):
                tst[id] = [nrg, None]
            else:
                print("???")
                exit(1)
        infile.close()
    return trn, tst


#Testing data input
#tstA, tstSites, tstMap = ReadLatticeFile(LATTICE_FILE_NAME)
#tstEnvs = ReadEnvFile(infileLoc=BASE_NAME_TST + ".env")
tstDec = ReadDecompFile(BASE_NAME_TST + ".dec")
#tstOccs = ReadOccFile(BASE_NAME_TST + ".occ")

#Training data input
trnA, trnSites, trnMap = ReadLatticeFile(LATTICE_FILE_NAME)
trnEnvs = ReadEnvFile(infileLoc=BASE_NAME_TRN + ".env")
trnDec = ReadDecompFile(BASE_NAME_TRN + ".dec")
trnOccs = ReadOccFile(BASE_NAME_TRN + ".occ")

#Prep training data
trnDat, tstDat = ReadData(FOLDER + "agbii4TrnTst.csv")
##align training data decompositions with DFT energies
trnB = [trnDat[i][0] for i in range(0, len(trnDec))] ###[0] for DFT energies
##do the training on the training set only
from numpy import array
from scipy.sparse.linalg import lsqr
x0 = [sum(trnB)/len(trnB)/len(trnOccs[0]) for i in range(0, len(trnDec[0]))]  ##init guess: avg embed nrg/site
x = list( lsqr(A=array(trnDec), b=array(trnB), x0=array(x0))[0] )

#Use x to re-predict the training data energies (but don't be surprised when they fit much better than the
#testing set!)
from numpy import dot
trnB = list(dot(trnDec, x))
##fill the easier to use dictionary with predictions
for i in range(0, len(trnB)):
    if(i not in trnDat.keys()):
        print("missing training data:", i)
        continue
    trnDat[i][1] = trnB[i] ###[1] for predicted energies

#Use x to predict the testing data energies
tstB = list(dot(tstDec, x))
##fill the easier to use dictionary with the predictions
for i in range(0, len(tstB)):
    if(i not in tstDat.keys()):
        print("missing testing data:", i)
        continue
    tstDat[i][1] = tstB[i] ###[1] for predicted energies

#Give the mean absolute error of testing data
#also the rmse
mae = 0.
rmse = 0.
for va in tstDat.values():
    mae += abs(va[1] - va[0])
    rmse += (va[1] - va[0])**2
mae = mae/len(tstDat.keys())
rmse = (((1000./N_ATOMS)**2)*rmse/len(tstDat.keys()))**(1/2)


#Append the number of unique sites for the testing data
for i in range(0, len(tstDec)):
    if(i not in tstDat.keys()):
        continue
    nUnq = 0
    for j in range(0, len(tstDec[i])):
        if(tstDec[i][j] != 0):
            nUnq += 1
    tstDat[i].append(nUnq)


#Plot the data
from matplotlib import pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(WIDTH(1), HEIGHT(1)))
#ax2 = ax.twinx()

##for more "readable" code later
nrgsTstDFT = [t[0] for t in tstDat.values()]
nrgsTstPred = [t[1] for t in tstDat.values()]
nrgsTrnDFT = [t[0] for t in trnDat.values()]
nrgsTrnPred = [t[1] for t in trnDat.values()]

#Get axes scales: absolute difference from minimum energy of both predicted, actual energies (meV/site)
absMin = min([min(nrgsTstDFT), min(nrgsTstPred), min(nrgsTrnDFT), min(nrgsTrnPred)])
minDFT = min([min(nrgsTstDFT), min(nrgsTrnDFT)])
minPred = min([min(nrgsTstPred), min(nrgsTrnPred)])
nSites = len(trnOccs[0])
##and then scale everything
if(1): ###scale by the minimum of BOTH the DFT and predicted energies
    nrgsTstDFT = [1000.*(e - absMin)/N_ATOMS for e in nrgsTstDFT]
    nrgsTstPred = [1000.*(e - absMin)/N_ATOMS for e in nrgsTstPred]
    nrgsTrnDFT = [1000.*(e - absMin)/N_ATOMS for e in nrgsTrnDFT]
    nrgsTrnPred = [1000.*(e - absMin)/N_ATOMS for e in nrgsTrnPred]
if(0): ###scale the DFT energies by the minimum of the DFT energies, and the pred energies by their minimum
    nrgsTstDFT = [1000.*(e - minDFT)/N_ATOMS for e in nrgsTstDFT]
    nrgsTstPred = [1000.*(e - minPred)/N_ATOMS for e in nrgsTstPred]
    nrgsTrnDFT = [1000.*(e - minDFT)/N_ATOMS for e in nrgsTrnDFT]
    nrgsTrnPred = [1000.*(e - minPred)/N_ATOMS for e in nrgsTrnPred]
if(0): ###no scaling by minimum energy whatsoever
    nrgsTstDFT = [1000.*(e)/N_ATOMS for e in nrgsTstDFT]
    nrgsTstPred = [1000.*(e)/N_ATOMS for e in nrgsTstPred]
    nrgsTrnDFT = [1000.*(e)/N_ATOMS for e in nrgsTrnDFT]
    nrgsTrnPred = [1000.*(e)/N_ATOMS for e in nrgsTrnPred]

#Convert mae into mev/atom
mae = 1000.*mae/N_ATOMS
print(mae)
print(rmse)
##and show it on the plot
ax.text(0.4, 0.05, s="mae = " + FloatFormat(mae, 1) + r"$\,$meV/atom", transform=ax.transAxes,
        fontsize=FTSIZE_MED)

#Get axes limits: max, min of scaled data for x (DFT) axis and y (predicted) axis
xLo = min([min(nrgsTstDFT), min(nrgsTrnDFT)])
xHi = max([max(nrgsTstDFT), max(nrgsTrnDFT)])
yLo = min([min(nrgsTstPred), min(nrgsTrnPred)])
yHi = max([max(nrgsTstPred), max(nrgsTrnPred)])
##make the picture square
xLo, yLo = min(xLo, yLo), min(xLo, yLo)
xHi, yHi = max(xHi, yHi), max(xHi, yHi)

#Axis labels
ax.set_xlabel("DFT Energy (meV/atom)", fontsize=FTSIZE_MED, fontweight="bold")
ax.set_ylabel("SCM Energy (meV/atom)", fontsize=FTSIZE_MED, fontweight="bold")

#Consistent x, y ticks
tks = []
for i in range(int(min(xLo, yLo)), int(max(xHi, yHi)), 100): ##every 100 meV/atom
    tks.append(i)
ax.set_xticks(tks)
ax.set_yticks(tks)
ax = SetTickSize(ax, FTSIZE_MED)

#Plot the testing data as black dots
ax.plot(nrgsTstDFT, nrgsTstPred, color=COLORS_DICT["black"], marker='.', markersize=MKSIZE_SML, linewidth=0.,
        label="Test")
#Plot the training data as red dots
#ax.plot(nrgsTrnDFT, nrgsTrnPred, color=COLORS_DICT["red"], marker='.', markersize=MKSIZE_SML, linewidth=0.,
#        label="Train")
#Plot the ideal fit as a blue bar
ax.plot([xLo, xHi], [yLo, yHi], color=COLORS_DICT["blue"], linewidth=LNSIZE_BIG, label="Ideal")

ax.xaxis.set_tick_params(width=1.5)
ax.yaxis.set_tick_params(width=1.5)

ax.grid(alpha=GRID_ALPHA)
ax.legend(frameon=True, fancybox=False, facecolor=COLORS_DICT["white"], edgecolor=COLORS_DICT["black"],
          framealpha=1., fontsize=FTSIZE_MED)
plt.subplots_adjust(left=0.171, right=0.995, bottom=0.143, top=0.995, hspace=0.2, wspace=0.2)

print(len(nrgsTstDFT), len(nrgsTrnDFT))
plt.savefig("nrgCmprCEDFT.pdf")
plt.savefig("nrgCmprCEDFT.png", dpi=DPI)
plt.show()
