## \file
## \ingroup tutorial_roofit
## \notebook
## 'BASIC FUNCTIONALITY' RooFit tutorial macro #105
## Demonstration of binding ROOT Math functions as RooFit functions
## and pdfs
##
## \macro_code
##
## \date February 2018
## \author Clemens Lange
## \author Wouter Verkerke (C version)

import ROOT

# Bind ROOT TMath::Erf C function
# ---------------------------------------------------

# Bind one-dimensional ROOT.TMath.Erf function as ROOT.RooAbsReal function
# Directly trying this in python doesn't work:
# x = ROOT.RooRealVar("x", "x", -3, 3)
# erf = ROOT.RooFit.bindFunction("erf", ROOT.TMath.Erf, x)
# Need to go through C interface
ROOT.gInterpreter.ProcessLine(
    'auto x = RooRealVar("x", "x", -3, 3); auto myerf = RooFit::bindFunction("erf", TMath::Erf, x)'
)
x = ROOT.x
erf = ROOT.myerf

# Print erf definition
erf.Print()

# Plot erf on frame
frame1 = x.frame(Title="TMath.Erf bound as ROOT.RooFit function")
erf.plotOn(frame1)

# Bind ROOT::Math::beta_pdf C function
# -----------------------------------------------------------------------

# Bind pdf ROOT.Math.Beta with three variables as ROOT.RooAbsPdf function
# As above, this does not work directly in python
# x2 = ROOT.RooRealVar("x2", "x2", 0, 0.999)
# a = ROOT.RooRealVar("a", "a", 5, 0, 10)
# b = ROOT.RooRealVar("b", "b", 2, 0, 10)
# beta = ROOT.RooFit.bindPdf("beta", ROOT.Math.beta_pdf, x2, a, b)
ROOT.gInterpreter.ProcessLine(
    'auto x2 = RooRealVar("x2", "x2", 0, 0.999);\
    auto a = RooRealVar("a", "a", 5, 0, 10);\
    auto b = RooRealVar("b", "b", 5, 0, 10);\
    auto beta = RooFit::bindPdf("beta", ROOT::Math::beta_pdf, x2, a, b)'
)
x2 = ROOT.x2
a = ROOT.a
b = ROOT.b
beta = ROOT.beta

# Perf beta definition
beta.Print()

# Generate some events and fit
data = beta.generate(ROOT.RooArgSet(x2), 10000)
beta.fitTo(data)

# Plot data and pdf on frame
frame2 = x2.frame(Title="ROOT.Math.Beta bound as ROOT.RooFit pdf")
data.plotOn(frame2)
beta.plotOn(frame2)

# Bind ROOT TF1 as RooFit function
# ---------------------------------------------------------------

# Create a ROOT TF1 function
fa1 = ROOT.TF1("fa1", "sin(x)/x", 0, 10)

# Create an observable
x3 = ROOT.RooRealVar("x3", "x3", 0.01, 20)

# Create binding of TF1 object to above observable
rfa1 = ROOT.RooFit.bindFunction(fa1, x3)

# Print rfa1 definition
rfa1.Print()

# Make plot frame in observable, TF1 binding function
frame3 = x3.frame(Title="TF1 bound as ROOT.RooFit function")
rfa1.plotOn(frame3)

c = ROOT.TCanvas("rf105_funcbinding", "rf105_funcbinding", 1200, 400)
c.Divide(3)
c.cd(1)
ROOT.gPad.SetLeftMargin(0.15)
frame1.GetYaxis().SetTitleOffset(1.6)
frame1.Draw()
c.cd(2)
ROOT.gPad.SetLeftMargin(0.15)
frame2.GetYaxis().SetTitleOffset(1.6)
frame2.Draw()
c.cd(3)
ROOT.gPad.SetLeftMargin(0.15)
frame3.GetYaxis().SetTitleOffset(1.6)
frame3.Draw()

c.SaveAs("rf105_funcbinding.png")
