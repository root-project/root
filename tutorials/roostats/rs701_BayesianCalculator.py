## \file
## \ingroup tutorial_roostats
## \notebook
## Bayesian calculator: basic exmple
##
## \macro_image
## \macro_output
## \macro_code
##
## \date July 2022
## \authors Artem Busorgin, Gregory Schott (C++ version)

import ROOT

useBkg = True
confLevel = 0.90

w = ROOT.RooWorkspace("w", True)
w.factory("SUM::pdf(s[0.001,15]*Uniform(x[0,1]),b[1,0,2]*Uniform(x))")
w.factory("Gaussian::prior_b(b,1,1)")
w.factory("PROD::model(pdf,prior_b)")
model = w["model"]  # pdf*priorNuisance
nuisanceParameters = ROOT.RooArgSet(w["b"])

POI = w["s"]
priorPOI = w.factory("Uniform::priorPOI(s)")
priorPOI2 = w.factory("GenericPdf::priorPOI2('1/sqrt(@0)',s)")

w.factory("n[3]")  # observed number of events
# create a data set with n observed events
data = ROOT.RooDataSet("data", "", {w["x"], w["n"]}, "n")
data.add({w["x"]}, w["n"].getVal())

# to suppress messages when pdf goes to zero
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.FATAL)

nuisPar = 0
if useBkg:
    nuisPar = nuisanceParameters
else:
    w["b"].setVal(0)

size = 1.0 - confLevel
print("\nBayesian Result using a Flat prior ")
bcalc = ROOT.RooStats.BayesianCalculator(data, model, {POI}, priorPOI, nuisPar)
bcalc.SetTestSize(size)
interval = bcalc.GetInterval()
cl = bcalc.ConfidenceLevel()
print(
    "{}% CL central interval: [ {} - {} ] or {}% CL limits\n".format(
        cl, interval.LowerLimit(), interval.UpperLimit(), cl + (1.0 - cl) / 2
    )
)
plot = bcalc.GetPosteriorPlot()
c1 = ROOT.TCanvas("c1", "Bayesian Calculator Result")
c1.Divide(1, 2)
c1.cd(1)
plot.Draw()
c1.Update()

print("\nBayesian Result using a 1/sqrt(s) prior  ")
bcalc2 = ROOT.RooStats.BayesianCalculator(data, model, {POI}, priorPOI2, nuisPar)
bcalc2.SetTestSize(size)
interval2 = bcalc2.GetInterval()
cl = bcalc2.ConfidenceLevel()
print(
    "{}% CL central interval: [ {} - {} ] or {}% CL limits\n".format(
        cl, interval2.LowerLimit(), interval2.UpperLimit(), cl + (1.0 - cl) / 2
    )
)
plot2 = bcalc2.GetPosteriorPlot()
c1.cd(2)
plot2.Draw()
ROOT.gPad.SetLogy()
c1.Update()

# observe one event while expecting one background event -> the 95% CL upper limit on s is 4.10
# observe one event while expecting zero background event -> the 95% CL upper limit on s is 4.74

c1.SaveAs("rs701_BayesianCalculator.png")

# TODO: The BayesianCalculator has to be destructed first. Otherwise, we can get
# segmentation faults depending on the destruction order, which is random in
# Python. Probably the issue is that some object has a non-owning pointer to
# another object, which it uses in its destructor. This should be fixed either
# in the design of RooStats in C++, or with phythonizations.
del bcalc, bcalc2
