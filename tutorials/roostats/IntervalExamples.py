## \file
## \ingroup tutorial_roostats
## \notebook
## Example showing confidence intervals with four techniques.
##
## An example that shows confidence intervals with four techniques.
## The model is a Normal Gaussian G(x|mu,sigma) with 100 samples of x.
## The answer is known analytically, so this is a good example to validate
## the RooStats tools.
##
##  - expected interval is [-0.162917, 0.229075]
##  - plc  interval is     [-0.162917, 0.229075]
##  - fc   interval is     [-0.17    , 0.23]        // stepsize is 0.01
##  - bc   interval is     [-0.162918, 0.229076]
##  - mcmc interval is     [-0.166999, 0.230224]
##
## \macro_image
## \macro_output
## \macro_code
##
## \date July 2022
## \authors Artem Busorgin, Kyle Cranmer (C++ version)

import ROOT

# Time this macro
t = ROOT.TStopwatch()
t.Start()

# set RooFit random seed for reproducible results
ROOT.RooRandom.randomGenerator().SetSeed(3001)

# make a simple model via the workspace factory
wspace = ROOT.RooWorkspace()
wspace.factory("Gaussian::normal(x[-10,10],mu[-1,1],sigma[1])")
wspace.defineSet("poi", "mu")
wspace.defineSet("obs", "x")

# specify components of model for statistical tools
modelConfig = ROOT.RooStats.ModelConfig("Example G(x|mu,1)")
modelConfig.SetWorkspace(wspace)
modelConfig.SetPdf(wspace["normal"])
modelConfig.SetParametersOfInterest(wspace.set("poi"))
modelConfig.SetObservables(wspace.set("obs"))

# create a toy dataset
data = wspace["normal"].generate(wspace.set("obs"), 100)
data.Print()

# for convenience later on
x = wspace["x"]
mu = wspace["mu"]

# set confidence level
confidenceLevel = 0.95

# example use profile likelihood calculator
plc = ROOT.RooStats.ProfileLikelihoodCalculator(data, modelConfig)
plc.SetConfidenceLevel(confidenceLevel)
plInt = plc.GetInterval()

# example use of Feldman-Cousins
fc = ROOT.RooStats.FeldmanCousins(data, modelConfig)
fc.SetConfidenceLevel(confidenceLevel)
fc.SetNBins(100)  # number of points to test per parameter
fc.UseAdaptiveSampling(True)  # make it go faster

# Here, we consider only ensembles with 100 events
# The PDF could be extended and this could be removed
fc.FluctuateNumDataEntries(False)

interval = fc.GetInterval()

# example use of BayesianCalculator
# now we also need to specify a prior in the ModelConfig
wspace.factory("Uniform::prior(mu)")
modelConfig.SetPriorPdf(wspace["prior"])

# example usage of BayesianCalculator
bc = ROOT.RooStats.BayesianCalculator(data, modelConfig)
bc.SetConfidenceLevel(confidenceLevel)
bcInt = bc.GetInterval()

# example use of MCMCInterval
mc = ROOT.RooStats.MCMCCalculator(data, modelConfig)
mc.SetConfidenceLevel(confidenceLevel)
# special options
mc.SetNumBins(200)  # bins used internally for representing posterior
mc.SetNumBurnInSteps(500)  # first N steps to be ignored as burn-in
mc.SetNumIters(100000)  # how long to run chain
mc.SetLeftSideTailFraction(0.5)  # for central interval
mcInt = mc.GetInterval()

# for this example we know the expected intervals
expectedLL = data.mean(x) + ROOT.Math.normal_quantile((1 - confidenceLevel) / 2, 1) / ROOT.sqrt(data.numEntries())
expectedUL = data.mean(x) + ROOT.Math.normal_quantile_c((1 - confidenceLevel) / 2, 1) / ROOT.sqrt(data.numEntries())

# Use the intervals
print("expected interval is [{}, {}]".format(expectedLL, expectedUL))
print("plc interval is [{}, {}]".format(plInt.LowerLimit(mu), plInt.UpperLimit(mu)))
print("fc interval is [{}, {}]".format(interval.LowerLimit(mu), interval.UpperLimit(mu)))
print("bc interval is [{}, {}]".format(bcInt.LowerLimit(), bcInt.UpperLimit()))
print("mc interval is [{}, {}]".format(mcInt.LowerLimit(mu), mcInt.UpperLimit(mu)))
mu.setVal(0)
print("is mu=0 in the interval? ", plInt.IsInInterval({mu}))

# make a reasonable style
ROOT.gStyle.SetCanvasColor(0)
ROOT.gStyle.SetCanvasBorderMode(0)
ROOT.gStyle.SetPadBorderMode(0)
ROOT.gStyle.SetPadColor(0)
ROOT.gStyle.SetCanvasColor(0)
ROOT.gStyle.SetTitleFillColor(0)
ROOT.gStyle.SetFillColor(0)
ROOT.gStyle.SetFrameFillColor(0)
ROOT.gStyle.SetStatColor(0)

# some plots
canvas = ROOT.TCanvas("canvas")
canvas.Divide(2, 2)

# plot the data
canvas.cd(1)
frame = x.frame()
data.plotOn(frame)
data.statOn(frame)
frame.Draw()

# plot the profile likelihood
canvas.cd(2)
plot = ROOT.RooStats.LikelihoodIntervalPlot(plInt)
plot.Draw()

# plot the MCMC interval
canvas.cd(3)
mcPlot = ROOT.RooStats.MCMCIntervalPlot(mcInt)
mcPlot.SetLineColor(ROOT.kGreen)
mcPlot.SetLineWidth(2)
mcPlot.Draw()

canvas.cd(4)
bcPlot = bc.GetPosteriorPlot()
bcPlot.Draw()

canvas.Update()

t.Stop()
t.Print()

canvas.SaveAs("IntervalExamples.png")

# TODO: The BayesianCalculator and MCMCCalculator have to be destructed first.
# Otherwise, we can get segmentation faults depending on the destruction order,
# which is random in Python. Probably the issue is that some object has a
# non-owning pointer to another object, which it uses in its destructor. This
# should be fixed either in the design of RooStats in C++, or with
# phythonizations.
del bc
del mc
