## \file
## \ingroup tutorial_roostats
## \notebook
## Produces an interval on the mean signal in a number counting experiment with known background using the
## Feldman-Cousins technique.
##
## Using the RooStats FeldmanCousins tool with 200 bins
## it takes 1 min and the interval is [0.2625, 10.6125]
## with a step size of 0.075.
## The interval in Feldman & Cousins's original paper is [.29, 10.81] Phys.Rev.D57:3873-3889,1998.
##
## \macro_image
## \macro_output
## \macro_code
##
## \date July 2022
## \authors Artem Busorgin, Kyle Cranmer (C++ version)

import ROOT

# to time the macro... about 30 s
t = ROOT.TStopwatch()
t.Start()

# make a simple model
x = ROOT.RooRealVar("x", "", 1, 0, 50)
mu = ROOT.RooRealVar("mu", "", 2.5, 0, 15)  # with a limit on mu>=0
b = ROOT.RooConstVar("b", "", 3.0)
mean = ROOT.RooAddition("mean", "", [mu, b])
pois = ROOT.RooPoisson("pois", "", x, mean)
parameters = {mu}

# create a toy dataset
data = pois.generate({x}, 1)
data.Print("v")

dataCanvas = ROOT.TCanvas("dataCanvas")
frame = x.frame()
data.plotOn(frame)
frame.Draw()
dataCanvas.Update()

w = ROOT.RooWorkspace()
modelConfig = ROOT.RooStats.ModelConfig("poissonProblem", w)
modelConfig.SetPdf(pois)
modelConfig.SetParametersOfInterest(parameters)
modelConfig.SetObservables({x})
w.Print()

# show use of Feldman-Cousins
fc = ROOT.RooStats.FeldmanCousins(data, modelConfig)
fc.SetTestSize(0.05)  # set size of test
fc.UseAdaptiveSampling(True)
fc.FluctuateNumDataEntries(False)  # number counting analysis: dataset always has 1 entry with N events observed
fc.SetNBins(100)  # number of points to test per parameter

# use the Feldman-Cousins tool
interval = fc.GetInterval()

# make a canvas for plots
intervalCanvas = ROOT.TCanvas("intervalCanvas")

print("is this point in the interval? ", interval.IsInInterval(parameters))
print("interval is [{}, {}]".format(interval.LowerLimit(mu), interval.UpperLimit(mu)))

# using 200 bins it takes 1 min and the answer is
# interval is [0.2625, 10.6125] with a step size of .075
# The interval in Feldman & Cousins's original paper is [.29, 10.81]
# Phys.Rev.D57:3873-3889,1998.

# No dedicated plotting class yet, so do it by hand:
parameterScan = fc.GetPointsToScan()
hist = parameterScan.createHistogram("mu", ROOT.RooFit.Binning(30))
hist.Draw()

marks = []
# loop over points to test
for i in range(parameterScan.numEntries()):
    # get a parameter point from the list of points to test.
    tmpPoint = parameterScan.get(i).clone("temp")

    mark = ROOT.TMarker(tmpPoint.getRealValue("mu"), 1, 25)
    if interval.IsInInterval(tmpPoint):
        mark.SetMarkerColor(ROOT.kBlue)
    else:
        mark.SetMarkerColor(ROOT.kRed)

    mark.Draw("s")
    marks.append(mark)

t.Stop()
t.Print()

dataCanvas.SaveAs("rs401c_FeldmanCousins_data.png")
intervalCanvas.SaveAs("rs401c_FeldmanCousins_hist.png")
