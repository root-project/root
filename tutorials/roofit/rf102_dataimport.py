## \file
## \ingroup tutorial_roofit
## \notebook
## 'BASIC FUNCTIONALITY' RooFit tutorial macro #102
## Importing data from ROOT TTrees and THx histograms
##
## \macro_code
##
## \date February 2018
## \author Clemens Lange
## \author Wouter Verkerke (C version)

import ROOT
from array import array


def makeTH1():

    # Create ROOT ROOT.TH1 filled with a Gaussian distribution

    hh = ROOT.TH1D("hh", "hh", 25, -10, 10)
    for i in range(100):
        hh.Fill(ROOT.gRandom.Gaus(0, 3))
    return hh


def makeTTree():
    # Create ROOT ROOT.TTree filled with a Gaussian distribution in x and a
    # uniform distribution in y

    tree = ROOT.TTree("tree", "tree")
    px = array("d", [0])
    py = array("d", [0])
    tree.Branch("x", px, "x/D")
    tree.Branch("y", py, "y/D")
    for i in range(100):
        px[0] = ROOT.gRandom.Gaus(0, 3)
        py[0] = ROOT.gRandom.Uniform() * 30 - 15
        tree.Fill()
    return tree


############################
# Importing ROOT histograms
############################
# Import ROOT TH1 into a RooDataHist
# ---------------------------------------------------------
# Create a ROOT TH1 histogram
hh = makeTH1()

# Declare observable x
x = ROOT.RooRealVar("x", "x", -10, 10)

# Create a binned dataset that imports contents of ROOT.TH1 and associates
# its contents to observable 'x'
dh = ROOT.RooDataHist("dh", "dh", ROOT.RooArgList(x), ROOT.RooFit.Import(hh))

# Plot and fit a RooDataHist
# ---------------------------------------------------
# Make plot of binned dataset showing Poisson error bars (ROOT.RooFit
# default)
frame = x.frame(ROOT.RooFit.Title("Imported ROOT.TH1 with Poisson error bars"))
dh.plotOn(frame)

# Fit a Gaussian p.d.f to the data
mean = ROOT.RooRealVar("mean", "mean", 0, -10, 10)
sigma = ROOT.RooRealVar("sigma", "sigma", 3, 0.1, 10)
gauss = ROOT.RooGaussian("gauss", "gauss", x, mean, sigma)
gauss.fitTo(dh)
gauss.plotOn(frame)

# Plot and fit a RooDataHist with internal errors
# ---------------------------------------------------------------------------------------------

# If histogram has custom error (i.e. its contents is does not originate from a Poisson process
# but e.g. is a sum of weighted events) you can data with symmetric 'sum-of-weights' error instead
# (same error bars as shown by ROOT)
frame2 = x.frame(ROOT.RooFit.Title("Imported ROOT.TH1 with internal errors"))
dh.plotOn(frame2, ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2))
gauss.plotOn(frame2)

# Please note that error bars shown (Poisson or SumW2) are for visualization only, the are NOT used
# in a maximum likelihood fit
#
# A (binned) ML fit will ALWAYS assume the Poisson error interpretation of data (the mathematical definition
# of likelihood does not take any external definition of errors). Data with non-unit weights can only be correctly
# fitted with a chi^2 fit (see rf602_chi2fit.py)
#
# Importing ROOT TTrees
# -----------------------------------------------------------
# Import ROOT TTree into a RooDataSet

tree = makeTTree()

# Define 2nd observable y
y = ROOT.RooRealVar("y", "y", -10, 10)

# Construct unbinned dataset importing tree branches x and y matching between branches and ROOT.RooRealVars
# is done by name of the branch/RRV
#
# Note that ONLY entries for which x,y have values within their allowed ranges as defined in
# ROOT.RooRealVar x and y are imported. Since the y values in the import tree are in the range [-15,15]
# and RRV y defines a range [-10,10] this means that the ROOT.RooDataSet
# below will have less entries than the ROOT.TTree 'tree'

ds = ROOT.RooDataSet("ds", "ds", ROOT.RooArgSet(x, y), ROOT.RooFit.Import(tree))

# Plot data set with multiple binning choices
# ------------------------------------------------------------------------------------
# Print number of events in dataset
ds.Print()

# Print unbinned dataset with default frame binning (100 bins)
frame3 = y.frame(ROOT.RooFit.Title("Unbinned data shown in default frame binning"))
ds.plotOn(frame3)

# Print unbinned dataset with custom binning choice (20 bins)
frame4 = y.frame(ROOT.RooFit.Title("Unbinned data shown with custom binning"))
ds.plotOn(frame4, ROOT.RooFit.Binning(20))

# Draw all frames on a canvas
c = ROOT.TCanvas("rf102_dataimport", "rf102_dataimport", 800, 800)
c.Divide(2, 2)
c.cd(1)
ROOT.gPad.SetLeftMargin(0.15)
frame.GetYaxis().SetTitleOffset(1.4)
frame.Draw()
c.cd(2)
ROOT.gPad.SetLeftMargin(0.15)
frame2.GetYaxis().SetTitleOffset(1.4)
frame2.Draw()
c.cd(3)
ROOT.gPad.SetLeftMargin(0.15)
frame3.GetYaxis().SetTitleOffset(1.4)
frame3.Draw()
c.cd(4)
ROOT.gPad.SetLeftMargin(0.15)
frame4.GetYaxis().SetTitleOffset(1.4)
frame4.Draw()

c.SaveAs("rf102_dataimport.png")
