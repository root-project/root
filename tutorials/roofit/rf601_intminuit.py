## \file
## \ingroup tutorial_roofit
## \notebook
##
## 'LIKELIHOOD AND MINIMIZATION' RooFit tutorial macro #601
##
## Interactive minimization with MINUIT
##
## \macro_code
##
## \date February 2018
## \author Clemens Lange
## \author Wouter Verkerke (C version)


import ROOT


# Setup pdf and likelihood
# -----------------------------------------------

# Observable
x = ROOT.RooRealVar("x", "x", -20, 20)

# Model (intentional strong correlations)
mean = ROOT.RooRealVar("mean", "mean of g1 and g2", 0)
sigma_g1 = ROOT.RooRealVar("sigma_g1", "width of g1", 3)
g1 = ROOT.RooGaussian("g1", "g1", x, mean, sigma_g1)

sigma_g2 = ROOT.RooRealVar("sigma_g2", "width of g2", 4, 3.0, 6.0)
g2 = ROOT.RooGaussian("g2", "g2", x, mean, sigma_g2)

frac = ROOT.RooRealVar("frac", "frac", 0.5, 0.0, 1.0)
model = ROOT.RooAddPdf("model", "model", [g1, g2], [frac])

# Generate 1000 events
data = model.generate({x}, 1000)

# Construct unbinned likelihood of model w.r.t. data
nll = model.createNLL(data)

# Interactive minimization, error analysis
# -------------------------------------------------------------------------------

# Create MINUIT interface object
m = ROOT.RooMinimizer(nll)

# Activate verbose logging of MINUIT parameter space stepping
m.setVerbose(True)

# Call MIGRAD to minimize the likelihood
m.migrad()

# Print values of all parameters, reflect values (and error estimates)
# that are back propagated from MINUIT
model.getParameters({x}).Print("s")

# Disable verbose logging
m.setVerbose(False)

# Run HESSE to calculate errors from d2L/dp2
m.hesse()

# Print value (and error) of sigma_g2 parameter, reflects
# value and error back propagated from MINUIT
sigma_g2.Print()

# Run MINOS on sigma_g2 parameter only
m.minos({sigma_g2})

# Print value (and error) of sigma_g2 parameter, reflects
# value and error back propagated from MINUIT
sigma_g2.Print()

# Saving results, contour plots
# ---------------------------------------------------------

# Save a snapshot of the fit result. ROOT.This object contains the initial
# fit parameters, final fit parameters, complete correlation
# matrix, EDM, minimized FCN , last MINUIT status code and
# the number of times the ROOT.RooFit function object has indicated evaluation
# problems (e.g. zero probabilities during likelihood evaluation)
r = m.save()

# Make contour plot of mx vs sx at 1,2, sigma
frame = m.contour(frac, sigma_g2, 1, 2, 3)
frame.SetTitle("Contour plot")

# Print the fit result snapshot
r.Print("v")

# Change parameter values, plotting
# -----------------------------------------------------------------

# At any moment you can manually change the value of a (constant)
# parameter
mean.setVal(0.3)

# Rerun MIGRAD,HESSE
m.migrad()
m.hesse()
frac.Print()

# Now fix sigma_g2
sigma_g2.setConstant(True)

# Rerun MIGRAD,HESSE
m.migrad()
m.hesse()
frac.Print()

c = ROOT.TCanvas("rf601_intminuit", "rf601_intminuit", 600, 600)
ROOT.gPad.SetLeftMargin(0.15)
frame.GetYaxis().SetTitleOffset(1.4)
frame.Draw()

c.SaveAs("rf601_intminuit.png")
