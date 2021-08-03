## \file
## \ingroup tutorial_roofit
## \notebook
##
## 'LIKELIHOOD AND MINIMIZATION' RooFit tutorial macro #606
##
## Understanding and customizing error handling in likelihood evaluations
##
## \macro_code
##
## \date February 2018
## \author Clemens Lange
## \author Wouter Verkerke (C version)


import ROOT


# Create model and dataset
# ----------------------------------------------

# Observable
m = ROOT.RooRealVar("m", "m", 5.20, 5.30)

# Parameters
m0 = ROOT.RooRealVar("m0", "m0", 5.291, 5.20, 5.30)
k = ROOT.RooRealVar("k", "k", -30, -50, -10)

# Pdf
argus = ROOT.RooArgusBG("argus", "argus", m, m0, k)

# Sample 1000 events in m from argus
data = argus.generate({m}, 1000)

# Plot model and data
# --------------------------------------

frame1 = m.frame(Bins=40, Title="Argus model and data")
data.plotOn(frame1)
argus.plotOn(frame1)

# Fit model to data
# ---------------------------------

# The ARGUS background shape has a sharp kinematic cutoff at m=m0
# and is prone to evaluation errors if the cutoff parameter m0
# is floated: when the pdf cutoff value is lower than that in data
# events with m>m0 will have zero probability

# Perform unbinned ML fit. Print detailed error messages for up to
# 10 events per likelihood evaluation. The default error handling strategy
# is to return a very high value of the likelihood to MINUIT if errors occur,
# which will force MINUIT to retreat from the problematic area

argus.fitTo(data, PrintEvalErrors=10)

# Peform another fit. In self configuration only the number of errors per
# likelihood evaluation is shown, it is greater than zero. The
# EvalErrorWall(kFALSE) arguments disables the default error handling strategy
# and will cause the actual (problematic) value of the likelihood to be passed
# to MINUIT.
#
# NB: Use of self option is NOT recommended as default strategt as broken -log(L) values
# can often be lower than 'good' ones because offending events are removed.
# ROOT.This may effectively create a False minimum in problem areas. ROOT.This is clearly
# illustrated in the second plot

m0.setError(0.1)
argus.fitTo(data, PrintEvalErrors=0, EvalErrorWall=False)

# Plot likelihood as function of m0
# ------------------------------------------------------------------

# Construct likelihood function of model and data
nll = ROOT.RooNLLVar("nll", "nll", argus, data)

# Plot likelihood in m0 in range that includes problematic values
# In self configuration no messages are printed for likelihood evaluation errors,
# but if an likelihood value evaluates with error, corresponding value
# on the curve will be set to the value given in EvalErrorValue().

frame2 = m0.frame(Range=(5.288, 5.293), Title="-log(L) scan vs m0, regions masked")
nll.plotOn(frame2, ShiftToZero=True, PrintEvalErrors=-1, EvalErrorValue=(nll.getVal() + 10), LineColor="r")
frame2.SetMaximum(15)
frame2.SetMinimum(0)

c = ROOT.TCanvas("rf606_nllerrorhandling", "rf606_nllerrorhandling", 1200, 400)
c.Divide(2)
c.cd(1)
ROOT.gPad.SetLeftMargin(0.15)
frame1.GetYaxis().SetTitleOffset(1.4)
frame1.Draw()
c.cd(2)
ROOT.gPad.SetLeftMargin(0.15)
frame2.GetYaxis().SetTitleOffset(1.4)
frame2.Draw()

c.SaveAs("rf606_nllerrorhandling.png")
