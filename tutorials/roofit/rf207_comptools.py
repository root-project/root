## \file
## \ingroup tutorial_roofit
## \notebook
## 'ADDITION AND CONVOLUTION' RooFit tutorial macro #207
## Tools and utilities for manipulation of composite objects
##
## \macro_code
##
## \date February 2018
## \authors Clemens Lange, Wouter Verkerke (C version)

import ROOT

# Set up composite pdf dataset
# --------------------------------------------------------

# Declare observable x
x = ROOT.RooRealVar("x", "x", 0, 10)

# Create two Gaussian PDFs g1(x,mean1,sigma) anf g2(x,mean2,sigma) and
# their parameters
mean = ROOT.RooRealVar("mean", "mean of gaussians", 5)
sigma = ROOT.RooRealVar("sigma", "width of gaussians", 0.5)
sig = ROOT.RooGaussian("sig", "Signal component 1", x, mean, sigma)

# Build Chebychev polynomial p.d.f.
a0 = ROOT.RooRealVar("a0", "a0", 0.5, 0.0, 1.0)
a1 = ROOT.RooRealVar("a1", "a1", 0.2, 0.0, 1.0)
bkg1 = ROOT.RooChebychev("bkg1", "Background 1", x, [a0, a1])

# Build expontential pdf
alpha = ROOT.RooRealVar("alpha", "alpha", -1)
bkg2 = ROOT.RooExponential("bkg2", "Background 2", x, alpha)

# Sum the background components into a composite background p.d.f.
bkg1frac = ROOT.RooRealVar("bkg1frac", "fraction of component 1 in background", 0.2, 0.0, 1.0)
bkg = ROOT.RooAddPdf("bkg", "Signal", [bkg1, bkg2], [bkg1frac])

# Sum the composite signal and background
bkgfrac = ROOT.RooRealVar("bkgfrac", "fraction of background", 0.5, 0.0, 1.0)
model = ROOT.RooAddPdf("model", "g1+g2+a", [bkg, sig], [bkgfrac])

# Create dummy dataset that has more observables than the above pdf
y = ROOT.RooRealVar("y", "y", -10, 10)
data = ROOT.RooDataSet("data", "data", {x, y})

# Basic information requests
# ---------------------------------------------


# Get list of observables
# ---------------------------------------------

# Get list of observables of pdf in context of a dataset
#
# Observables are define each context as the variables
# shared between a model and a dataset. In self case
# that is the variable 'x'

model_obs = model.getObservables(data)
model_obs.Print("v")

# Get list of parameters
# -------------------------------------------

# Get list of parameters, list of observables
model_params = model.getParameters({x})
model_params.Print("v")

# Get list of parameters, a dataset
# (Gives identical results to operation above)
model_params2 = model.getParameters(data)
model_params2.Print()

# Get list of components
# -------------------------------------------

# Get list of component objects, top-level node
model_comps = model.getComponents()
model_comps.Print("v")

# Modifications to structure of composites
# -------------------------------------------

# Create a second Gaussian
sigma2 = ROOT.RooRealVar("sigma2", "width of gaussians", 1)
sig2 = ROOT.RooGaussian("sig2", "Signal component 1", x, mean, sigma2)

# Create a sum of the original Gaussian plus the second Gaussian
sig1frac = ROOT.RooRealVar("sig1frac", "fraction of component 1 in signal", 0.8, 0.0, 1.0)
sigsum = ROOT.RooAddPdf("sigsum", "sig+sig2", [sig, sig2], [sig1frac])

# Construct a customizer utility to customize model
cust = ROOT.RooCustomizer(model, "cust")

# Instruct the customizer to replace node 'sig' with node 'sigsum'
cust.replaceArg(sig, sigsum)

# Build a clone of the input pdf according to the above customization
# instructions. Each node that requires modified is clone so that the
# original pdf remained untouched. The name of each cloned node is that
# of the original node suffixed by the name of the customizer object
#
# The returned head node own all nodes that were cloned as part of
# the build process so when cust_clone is deleted so will all other
# nodes that were created in the process.
cust_clone = cust.build(ROOT.kTRUE)

# Print structure of clone of model with sig.sigsum replacement.
cust_clone.Print("t")

# The RooCustomizer has the be deleted first.
# Otherwise, it might happen that `sig` or `sigsum` are deleted first, in which
# case the internal TLists in the RooCustomizer will complain about deleted
# objects.
del cust
