## \file
## \ingroup tutorial_roofit
## \notebook -nodraw
## Numeric algorithm tuning: configuration and customization of how MC sampling algorithms
## on specific pdfs are executed
##
## \macro_code
##
## \date February 2018
## \authors Clemens Lange, Wouter Verkerke (C++ version)

import ROOT


# Adjust global MC sampling strategy
# ------------------------------------------------------------------

# Example pdf for use below
x = ROOT.RooRealVar("x", "x", 0, 10)
model = ROOT.RooChebychev("model", "model", x, [0.0, 0.5, -0.1])

# Change global strategy for 1D sampling problems without conditional observable
# (1st kFALSE) and without discrete observable (2nd kFALSE) from ROOT.RooFoamGenerator,
# ( an interface to the ROOT.TFoam MC generator with adaptive subdivisioning strategy ) to ROOT.RooAcceptReject,
# a plain accept/reject sampling algorithm [ ROOT.RooFit default before
# ROOT 5.23/04 ]
ROOT.RooAbsPdf.defaultGeneratorConfig().method1D(False, False).setLabel("RooAcceptReject")

# Generate 10Kevt using ROOT.RooAcceptReject
data_ar = model.generate({x}, 10000, Verbose=True)
data_ar.Print()

# Adjusting default config for a specific pdf
# -------------------------------------------------------------------------------------

# Another possibility: associate custom MC sampling configuration as default for object 'model'
# The kTRUE argument will install a clone of the default configuration as specialized configuration
# for self model if none existed so far
model.specialGeneratorConfig(True).method1D(False, False).setLabel("RooFoamGenerator")

# Adjusting parameters of a specific technique
# ---------------------------------------------------------------------------------------

# Adjust maximum number of steps of ROOT.RooIntegrator1D in the global
# default configuration
ROOT.RooAbsPdf.defaultGeneratorConfig().getConfigSection("RooAcceptReject").setRealValue("nTrial1D", 2000)

# Example of how to change the parameters of a numeric integrator
# (Each config section is a ROOT.RooArgSet with ROOT.RooRealVars holding real-valued parameters
#  and ROOT.RooCategories holding parameters with a finite set of options)
model.specialGeneratorConfig().getConfigSection("RooFoamGenerator").setRealValue("chatLevel", 1)

# Generate 10Kevt using ROOT.RooFoamGenerator (FOAM verbosity increased
# with above chatLevel adjustment for illustration purposes)
data_foam = model.generate({x}, 10000, ROOT.RooFit.Verbose())
data_foam.Print()
