## \file
## \ingroup tutorial_roostats
## \notebook -js
## Example of the BernsteinCorrection utility in RooStats.
##
## The idea is that one has a distribution coming either from data or Monte Carlo
## (called "reality" in the macro) and a nominal model that is not sufficiently
## flexible to take into account the real distribution.  One wants to take into
## account the systematic associated with this imperfect modeling by augmenting
## the nominal model with some correction term (in this case a polynomial).
## The BernsteinCorrection utility will import into your workspace a corrected model
## given by nominal(x) * poly_N(x), where poly_N is an n-th order polynomial in
## the Bernstein basis.  The degree N of the polynomial is chosen by specifying the tolerance
## one has in adding an extra term to the polynomial.
## The Bernstein basis is nice because it only has positive-definite terms
## and works well with PDFs.
## Finally, the macro makes a plot of:
##  - the data (drawn from 'reality'),
##  - the best fit of the nominal model (blue)
##  - and the best fit corrected model.
##
## \macro_image
## \macro_output
## \macro_code
##
## \date June 2022
## \authors Artem Busorgin, Kyle Cranmer (C++ version)

import sys
import ROOT

# set range of observable
lowRange = -1
highRange = 5

# make a RooRealVar for the observable
x = ROOT.RooRealVar("x", "x", lowRange, highRange)

# true model
narrow = ROOT.RooGaussian("narrow", "", x, ROOT.RooFit.RooConst(0.0), ROOT.RooFit.RooConst(0.8))
wide = ROOT.RooGaussian("wide", "", x, ROOT.RooFit.RooConst(0.0), ROOT.RooFit.RooConst(2.0))
reality = ROOT.RooAddPdf("reality", "", [narrow, wide], ROOT.RooFit.RooConst(0.8))

data = reality.generate(x, 1000)

# nominal model
sigma = ROOT.RooRealVar("sigma", "", 1.0, 0, 10)
nominal = ROOT.RooGaussian("nominal", "", x, ROOT.RooFit.RooConst(0.0), sigma)

wks = ROOT.RooWorkspace("myWorksspace")

wks.Import(data, Rename="data")
wks.Import(nominal)

if ROOT.TClass.GetClass("ROOT::Minuit2::Minuit2Minimizer"):
    # use Minuit2 if ROOT was built with support for it:
    ROOT.Math.MinimizerOptions.SetDefaultMinimizer("Minuit2")

# The tolerance sets the probability to add an unnecessary term.
# lower tolerance will add fewer terms, while higher tolerance
# will add more terms and provide a more flexible function.
tolerance = 0.05
bernsteinCorrection = ROOT.RooStats.BernsteinCorrection(tolerance)
degree = bernsteinCorrection.ImportCorrectedPdf(wks, "nominal", "x", "data")

if degree < 0:
    ROOT.Error("rs_bernsteinCorrection", "Bernstein correction failed !")
    sys.exit()

print("Correction based on Bernstein Poly of degree ", degree)

frame = x.frame()
data.plotOn(frame)
# plot the best fit nominal model in blue
minimType = ROOT.Math.MinimizerOptions.DefaultMinimizerType()
nominal.fitTo(data, PrintLevel=0, Minimizer=minimType)
nominal.plotOn(frame)

# plot the best fit corrected model in red
corrected = wks.pdf("corrected")
if not corrected:
    sys.exit()

# fit corrected model
corrected.fitTo(data, PrintLevel=0, Minimizer=minimType)
corrected.plotOn(frame, LineColor=ROOT.kRed)

# plot the correction term (* norm constant) in dashed green
# should make norm constant just be 1, not depend on binning of data
poly = wks.pdf("poly")
if poly:
    poly.plotOn(frame, LineColor=ROOT.kGreen, LineStyle=ROOT.kDashed)

# this is a switch to check the sampling distribution
# of -2 log LR for two comparisons:
# the first is for n-1 vs. n degree polynomial corrections
# the second is for n vs. n+1 degree polynomial corrections
# Here we choose n to be the one chosen by the tolerance
# criterion above, eg. n = "degree" in the code.
# Setting this to true is takes about 10 min.
checkSamplingDist = True
numToyMC = 20  # increase this value for sensible results

c1 = ROOT.TCanvas()
if checkSamplingDist:
    c1.Divide(1, 2)
    c1.cd(1)

frame.Draw()
ROOT.gPad.Update()

if checkSamplingDist:
    # check sampling dist
    ROOT.Math.MinimizerOptions.SetDefaultPrintLevel(-1)
    samplingDist = ROOT.TH1F("samplingDist", "", 20, 0, 10)
    samplingDistExtra = ROOT.TH1F("samplingDistExtra", "", 20, 0, 10)
    bernsteinCorrection.CreateQSamplingDist(
        wks, "nominal", "x", "data", samplingDist, samplingDistExtra, degree, numToyMC
    )

    c1.cd(2)
    samplingDistExtra.SetLineColor(ROOT.kRed)
    samplingDistExtra.Draw()
    samplingDist.Draw("same")

c1.SaveAs("rs_bernsteinCorrection.png")
