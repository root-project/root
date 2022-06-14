## \file
## \ingroup tutorial_roofit
## \notebook -nodraw
## Organization and simultaneous fits: RooFit memory tracing debug tool
##
## \macro_code
##
## \date February 2018
## \authors Clemens Lange, Wouter Verkerke (C++ version)

import ROOT


# Activate ROOT.RooFit memory tracing
ROOT.RooTrace.active(True)

# Construct gauss(x,m,s)
x = ROOT.RooRealVar("x", "x", -10, 10)
m = ROOT.RooRealVar("m", "m", 0, -10, 10)
s = ROOT.RooRealVar("s", "s", 1, -10, 10)
gauss = ROOT.RooGaussian("g", "g", x, m, s)

# Show dump of all ROOT.RooFit object in memory
ROOT.RooTrace.dump()

# Activate verbose mode
ROOT.RooTrace.verbose(True)

# Construct poly(x,p0)
p0 = ROOT.RooRealVar("p0", "p0", 0.01, 0.0, 1.0)
poly = ROOT.RooPolynomial("p", "p", x, [p0])

# Put marker in trace list for future reference
ROOT.RooTrace.mark()

# model = f*gauss(x) + (1-f)*poly(x)
f = ROOT.RooRealVar("f", "f", 0.5, 0.0, 1.0)
model = ROOT.RooAddPdf("model", "model", [gauss, poly], [f])

# Show object added to memory since marker
ROOT.RooTrace.printObjectCounts()

# Since verbose mode is still on, will see messages
# pertaining to destructor calls of all RooFit objects
# made in self macro
#
# A call to RooTrace.dump() at the end of self macro
# should show that there a no RooFit object left in memory
