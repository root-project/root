## \file
## \ingroup tutorial_hist
## \notebook
## Fill a 1D histogram from a user-defined parametric function.
##
## \macro_image
## \macro_code
##
## \date November 2024
## \author Rene Brun, Giacomo Parolini
import ROOT

# Create a user-defined formula.
# A function (any dimension) or a formula may reference an already defined formula
form1 = ROOT.TFormula("form1", "abs(sin(x)/x)")

# Create a 1D function using the formula defined above and the predefined "gaus" formula.
rangeMin = 0.0
rangeMax = 10.0
sqroot = ROOT.TF1("sqroot", "x*gaus(0) + [3]*form1", rangeMin, rangeMax)
sqroot.SetLineColor(4)
sqroot.SetLineWidth(6)
# Set parameters to the functions "gaus" and "form1".
gausScale = 10.0  # [0]
gausMean = 4.0    # [1]
gausVar = 1.0     # [2]
form1Scale = 20.0 # [3]
sqroot.SetParameters(gausScale, gausMean, gausVar, form1Scale)

# Create a one dimensional histogram and fill it following the distribution in function sqroot.
h1d = ROOT.TH1D("h1d", "Test random numbers", 200, rangeMin, rangeMax)

# Use our user-defined function to fill the histogram with random values sampled from it.
h1d.FillRandom("sqroot", 10000)

# Open a ROOT file and save the formula, function and histogram
with ROOT.TFile.Open("fillrandom_userfunc_py.root", "RECREATE") as myFile:
   myFile.WriteObject(form1, form1.GetName())
   myFile.WriteObject(sqroot, sqroot.GetName())
   myFile.WriteObject(h1d, h1d.GetName())
