## \file
## \ingroup tutorial_roofit
## \notebook
##
## 'ORGANIZATION AND SIMULTANEOUS FITS' RooFit tutorial macro #510
##
## Working with named parameter sets and parameter snapshots in
## workspaces
##
## \macro_code
##
## \date February 2018
## \author Clemens Lange
## \author Wouter Verkerke (C version)


import ROOT


def fillWorkspace(w):
    # Create model
    # -----------------------

    # Declare observable x
    x = ROOT.RooRealVar("x", "x", 0, 10)

    # Create two Gaussian PDFs g1(x,mean1,sigma) anf g2(x,mean2,sigma) and
    # their parameters
    mean = ROOT.RooRealVar("mean", "mean of gaussians", 5, 0, 10)
    sigma1 = ROOT.RooRealVar("sigma1", "width of gaussians", 0.5)
    sigma2 = ROOT.RooRealVar("sigma2", "width of gaussians", 1)

    sig1 = ROOT.RooGaussian("sig1", "Signal component 1", x, mean, sigma1)
    sig2 = ROOT.RooGaussian("sig2", "Signal component 2", x, mean, sigma2)

    # Build Chebychev polynomial p.d.f.
    a0 = ROOT.RooRealVar("a0", "a0", 0.5, 0.0, 1.0)
    a1 = ROOT.RooRealVar("a1", "a1", -0.2, 0.0, 1.0)
    bkg = ROOT.RooChebychev("bkg", "Background", x, [a0, a1])

    # Sum the signal components into a composite signal p.d.f.
    sig1frac = ROOT.RooRealVar("sig1frac", "fraction of component 1 in signal", 0.8, 0.0, 1.0)
    sig = ROOT.RooAddPdf("sig", "Signal", [sig1, sig2], [sig1frac])

    # Sum the composite signal and background
    bkgfrac = ROOT.RooRealVar("bkgfrac", "fraction of background", 0.5, 0.0, 1.0)
    model = ROOT.RooAddPdf("model", "g1+g2+a", [bkg, sig], [bkgfrac])

    # Import model into p.d.f.
    w.Import(model)

    # Encode definition of parameters in workspace
    # ---------------------------------------------------------------------------------------

    # Define named sets "parameters" and "observables", list which variables should be considered
    # parameters and observables by the users convention
    #
    # Variables appearing in sets _must_ live in the workspace already, the autoImport flag
    # of defineSet must be set to import them on the fly. Named sets contain only references
    # to the original variables, the value of observables in named sets already
    # reflect their 'current' value
    params = model.getParameters({x})
    w.defineSet("parameters", params)
    w.defineSet("observables", {x})

    # Encode reference value for parameters in workspace
    # ---------------------------------------------------------------------------------------------------

    # Define a parameter 'snapshot' in the p.d.f.
    # Unlike a named set, parameter snapshot stores an independent set of values for
    # a given set of variables in the workspace. The values can be stored and reloaded
    # into the workspace variable objects using the loadSnapshot() and saveSnapshot()
    # methods. A snapshot saves the value of each variable, errors that are stored
    # with it as well as the 'Constant' flag that is used in fits to determine if a
    # parameter is kept fixed or not.

    # Do a dummy fit to a (supposedly) reference dataset here and store the results
    # of that fit into a snapshot
    refData = model.generate({x}, 10000)
    model.fitTo(refData, PrintLevel=-1)

    # The kTRUE flag imports the values of the objects in (*params) into the workspace
    # If not set, present values of the workspace parameters objects are stored
    w.saveSnapshot("reference_fit", params, True)

    # Make another fit with the signal componentforced to zero
    # and save those parameters too

    bkgfrac.setVal(1)
    bkgfrac.setConstant(True)
    bkgfrac.removeError()
    model.fitTo(refData, PrintLevel=-1)

    w.saveSnapshot("reference_fit_bkgonly", params, True)


# Create model and dataset
# -----------------------------------------------

w = ROOT.RooWorkspace("w")
fillWorkspace(w)

# Exploit convention encoded in named set "parameters" and "observables"
# to use workspace contents w/o need for introspected
model = w["model"]

# Generate data from p.d.f. in given observables
data = model.generate(w.set("observables"), 1000)

# Fit model to data
model.fitTo(data)

# Plot fitted model and data on frame of first (only) observable
frame = (w.set("observables").first()).frame()
data.plotOn(frame)
model.plotOn(frame)

# Overlay plot with model with reference parameters as stored in snapshots
w.loadSnapshot("reference_fit")
model.plotOn(frame, LineColor="r")
w.loadSnapshot("reference_fit_bkgonly")
model.plotOn(frame, LineColor="r", LineStyle="--")

# Draw the frame on the canvas
c = ROOT.TCanvas("rf510_wsnamedsets", "rf503_wsnamedsets", 600, 600)
ROOT.gPad.SetLeftMargin(0.15)
frame.GetYaxis().SetTitleOffset(1.4)
frame.Draw()

c.SaveAs("rf510_wsnamedsets.png")

# Print workspace contents
w.Print()

# Workspace will remain in memory after macro finishes
ROOT.gDirectory.Add(w)
