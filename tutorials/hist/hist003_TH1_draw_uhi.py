## \file
## \ingroup tutorial_hist
## \notebook
## Draw a 1D histogram to a canvas.
##
## \note When using graphics inside a ROOT macro the objects must be created with `new`.
##
## \macro_image
## \macro_code
##
## \date July 2025
## \author Rene Brun, Giacomo Parolini, Nursena Bitirgen

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import ROOT

# Create and fill the histogram.
# See hist002_TH1_fillrandom_userfunc.C for more information about this section.
form1 = ROOT.TFormula("form1", "abs(sin(x)/x)")
rangeMin = 0.0
rangeMax = 10.0
sqroot = ROOT.TF1("sqroot", "x*gaus(0) + [3]*form1", rangeMin, rangeMax)
sqroot.SetParameters(10.0, 4.0, 1.0, 20.0)

nBins = 200
h1d = ROOT.TH1D("h1d", "Test random numbers", nBins, rangeMin, rangeMax)

random_numbers = np.array([sqroot.GetRandom() for _ in range(10000)])
h1d[...] = np.histogram(np.array([sqroot.GetRandom() for _ in range(10000)]), bins=nBins, range=(rangeMin, rangeMax))[0]

# Create a canvas and draw the histogram
plt.figure(figsize=(7, 9))

# Split the canvas into two sections to plot both the function and the histogram
plt.axes([0.05, 0.55, 0.90, 0.40])
x = np.linspace(rangeMin, rangeMax, 500)
plt.plot(x, [sqroot.Eval(xi) for xi in x], "b-", lw=5)
plt.grid()
plt.title("x*gaus(0) + [3]*form1")
plt.text(5, 40, "The sqroot function", fontsize=18, weight="bold", bbox=dict(facecolor="white", edgecolor="black"))
plt.xlim(rangeMin, rangeMax)
plt.ylim(bottom=0)

plt.axes([0.05, 0.05, 0.90, 0.40])
hep.histplot(h1d, yerr=False, histtype="fill", color="brown", alpha=0.7, edgecolor="blue", linewidth=1)

plt.title("Test random numbers")
plt.xlabel("x")
plt.ylabel("Entries")
plt.xlim(rangeMin, rangeMax)
plt.ylim(bottom=0)

stats_text = (
    f"Entries = {len(random_numbers)}\nMean = {(np.mean(random_numbers)):.3f}\nStd Dev = {(np.std(random_numbers)):.2f}"
)
plt.text(0.90, 0.90, stats_text, transform=plt.gca().transAxes, ha="right", va="top", bbox=dict(facecolor="white"))
plt.show()
