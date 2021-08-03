## \file
## \ingroup tutorial_roofit
## \notebook
##
## 'LIKELIHOOD AND MINIMIZATION' RooFit tutorial macro #605
##
## Working with the profile likelihood estimator
##
## \macro_code
##
## \date February 2018
## \author Clemens Lange
## \author Wouter Verkerke (C version)


import ROOT


# Create model and dataset
# -----------------------------------------------

# Observable
x = ROOT.RooRealVar("x", "x", -20, 20)

# Model (intentional strong correlations)
mean = ROOT.RooRealVar("mean", "mean of g1 and g2", 0, -10, 10)
sigma_g1 = ROOT.RooRealVar("sigma_g1", "width of g1", 3)
g1 = ROOT.RooGaussian("g1", "g1", x, mean, sigma_g1)

sigma_g2 = ROOT.RooRealVar("sigma_g2", "width of g2", 4, 3.0, 6.0)
g2 = ROOT.RooGaussian("g2", "g2", x, mean, sigma_g2)

frac = ROOT.RooRealVar("frac", "frac", 0.5, 0.0, 1.0)
model = ROOT.RooAddPdf("model", "model", [g1, g2], [frac])

# Generate 1000 events
data = model.generate({x}, 1000)

# Construct plain likelihood
# ---------------------------------------------------

# Construct unbinned likelihood
nll = model.createNLL(data, NumCPU=2)

# Minimize likelihood w.r.t all parameters before making plots
ROOT.RooMinimizer(nll).migrad()

# Plot likelihood scan frac
frame1 = frac.frame(Bins=10, Range=(0.01, 0.95), Title="LL and profileLL in frac")
nll.plotOn(frame1, ShiftToZero=True)

# Plot likelihood scan in sigma_g2
frame2 = sigma_g2.frame(Bins=10, Range=(3.3, 5.0), Title="LL and profileLL in sigma_g2")
nll.plotOn(frame2, ShiftToZero=True)

# Construct profile likelihood in frac
# -----------------------------------------------------------------------

# The profile likelihood estimator on nll for frac will minimize nll w.r.t
# all floating parameters except frac for each evaluation

pll_frac = nll.createProfile({frac})

# Plot the profile likelihood in frac
pll_frac.plotOn(frame1, LineColor="r")

# Adjust frame maximum for visual clarity
frame1.SetMinimum(0)
frame1.SetMaximum(3)

# Construct profile likelihood in sigma_g2
# -------------------------------------------------------------------------------

# The profile likelihood estimator on nll for sigma_g2 will minimize nll
# w.r.t all floating parameters except sigma_g2 for each evaluation
pll_sigmag2 = nll.createProfile({sigma_g2})

# Plot the profile likelihood in sigma_g2
pll_sigmag2.plotOn(frame2, LineColor="r")

# Adjust frame maximum for visual clarity
frame2.SetMinimum(0)
frame2.SetMaximum(3)

# Make canvas and draw ROOT.RooPlots
c = ROOT.TCanvas("rf605_profilell", "rf605_profilell", 800, 400)
c.Divide(2)
c.cd(1)
ROOT.gPad.SetLeftMargin(0.15)
frame1.GetYaxis().SetTitleOffset(1.4)
frame1.Draw()
c.cd(2)
ROOT.gPad.SetLeftMargin(0.15)
frame2.GetYaxis().SetTitleOffset(1.4)
frame2.Draw()

c.SaveAs("rf605_profilell.png")
