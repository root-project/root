## \file
## \ingroup tutorial_roostats
## \notebook
## Limits: number counting experiment with uncertainty on both the background rate and signal efficiency.
##
## The usage of a Confidence Interval Calculator to set a limit on the signal is illustrated
##
## \macro_image
## \macro_output
## \macro_code
##
## \date June 2022
## \authors Artem Busorgin, Kyle Cranmer (C++ version)

import ROOT

# --------------------------------------
# An example of setting a limit in a number counting experiment with uncertainty on background and signal

# to time the macro
t = ROOT.TStopwatch()
t.Start()

# --------------------------------------
# The Model building stage
# --------------------------------------
wspace = ROOT.RooWorkspace()
wspace.factory(
    "Poisson::countingModel(obs[150,0,300], " "sum(s[50,0,120]*ratioSigEff[1.,0,3.],b[100]*ratioBkgEff[1.,0.,3.]))"
)  # counting model
wspace.factory("Gaussian::sigConstraint(gSigEff[1,0,3],ratioSigEff,0.05)")  # 5% signal efficiency uncertainty
wspace.factory("Gaussian::bkgConstraint(gSigBkg[1,0,3],ratioBkgEff,0.2)")  # 10% background efficiency uncertainty
wspace.factory("PROD::modelWithConstraints(countingModel,sigConstraint,bkgConstraint)")  # product of terms
wspace.Print()

modelWithConstraints = wspace["modelWithConstraints"]  # get the model
obs = wspace["obs"]  # get the observable
s = wspace["s"]  # get the signal we care about
b = wspace["b"]  # get the background and set it to a constant.  Uncertainty included in ratioBkgEff
b.setConstant()

ratioSigEff = wspace["ratioSigEff"]  # get uncertain parameter to constrain
ratioBkgEff = wspace["ratioBkgEff"]  # get uncertain parameter to constrain
constrainedParams = {ratioSigEff, ratioBkgEff}  # need to constrain these in the fit (should change default behavior)

gSigEff = wspace["gSigEff"]  # global observables for signal efficiency
gSigBkg = wspace["gSigBkg"]  # global obs for background efficiency
gSigEff.setConstant()
gSigBkg.setConstant()

# Create an example dataset with 160 observed events
obs.setVal(160.0)
data = ROOT.RooDataSet("exampleData", "exampleData", {obs})
data.add(obs)

# not necessary
modelWithConstraints.fitTo(data, Constrain=constrainedParams)

# Now let's make some confidence intervals for s, our parameter of interest
modelConfig = ROOT.RooStats.ModelConfig(wspace)
modelConfig.SetPdf(modelWithConstraints)
modelConfig.SetParametersOfInterest({s})
modelConfig.SetNuisanceParameters(constrainedParams)
modelConfig.SetObservables(obs)
modelConfig.SetGlobalObservables({gSigEff, gSigBkg})
modelConfig.SetName("ModelConfig")
wspace.Import(modelConfig)
wspace.Import(data)
wspace.SetName("w")
wspace.writeToFile("rs101_ws.root")

# First, let's use a Calculator based on the Profile Likelihood Ratio
plc = ROOT.RooStats.ProfileLikelihoodCalculator(data, modelConfig)
plc.SetTestSize(0.05)
lrinterval = plc.GetInterval()  # that was easy.

# Let's make a plot
dataCanvas = ROOT.TCanvas("dataCanvas")
dataCanvas.Divide(2, 1)
dataCanvas.cd(1)

plotInt = ROOT.RooStats.LikelihoodIntervalPlot(lrinterval)
plotInt.SetTitle("Profile Likelihood Ratio and Posterior for S")
plotInt.Draw()

# Second, use a Calculator based on the Feldman Cousins technique
fc = ROOT.RooStats.FeldmanCousins(data, modelConfig)
fc.UseAdaptiveSampling(True)
fc.FluctuateNumDataEntries(False)  # number counting analysis: dataset always has 1 entry with N events observed
fc.SetNBins(100)  # number of points to test per parameter
fc.SetTestSize(0.05)
# fc.SaveBeltToFile(True) # optional
fcint = fc.GetInterval()  # that was easy

fit = modelWithConstraints.fitTo(data, Save=True)

# Third, use a Calculator based on Markov Chain monte carlo
# Before configuring the calculator, let's make a ProposalFunction
# that will achieve a high acceptance rate
ph = ROOT.RooStats.ProposalHelper()
ph.SetVariables(fit.floatParsFinal())
ph.SetCovMatrix(fit.covarianceMatrix())
ph.SetUpdateProposalParameters(True)
ph.SetCacheSize(100)
pdfProp = ph.GetProposalFunction()  # that was easy

mc = ROOT.RooStats.MCMCCalculator(data, modelConfig)
mc.SetNumIters(20000)  # steps to propose in the chain
mc.SetTestSize(0.05)  # 95% CL
mc.SetNumBurnInSteps(40)  # ignore first N steps in chain as "burn in"
mc.SetProposalFunction(pdfProp)
mc.SetLeftSideTailFraction(0.5)  # find a "central" interval
mcInt = mc.GetInterval()  # that was easy

# Get Lower and Upper limits from Profile Calculator
print("Profile lower limit on s = ", lrinterval.LowerLimit(s))
print("Profile upper limit on s = ", lrinterval.UpperLimit(s))

# Get Lower and Upper limits from FeldmanCousins with profile construction
if fcint:
    fcul = fcint.UpperLimit(s)
    fcll = fcint.LowerLimit(s)
    print("FC lower limit on s = ", fcll)
    print("FC upper limit on s = ", fcul)
    fcllLine = ROOT.TLine(fcll, 0, fcll, 1)
    fculLine = ROOT.TLine(fcul, 0, fcul, 1)
    fcllLine.SetLineColor(ROOT.kRed)
    fculLine.SetLineColor(ROOT.kRed)
    fcllLine.Draw("same")
    fculLine.Draw("same")
    dataCanvas.Update()

# Plot MCMC interval and print some statistics
mcPlot = ROOT.RooStats.MCMCIntervalPlot(mcInt)
mcPlot.SetLineColor(ROOT.kMagenta)
mcPlot.SetLineWidth(2)
mcPlot.Draw("same")

mcul = mcInt.UpperLimit(s)
mcll = mcInt.LowerLimit(s)
print("MCMC lower limit on s = ", mcll)
print("MCMC upper limit on s = ", mcul)
print("MCMC Actual confidence level: ", mcInt.GetActualConfidenceLevel())

# 3-d plot of the parameter points
dataCanvas.cd(2)

# also plot the points in the markov chain
chainData = mcInt.GetChainAsDataSet()

print("plotting the chain data - nentries = ", chainData.numEntries())
chain = ROOT.RooStats.GetAsTTree("chainTreeData", "chainTreeData", chainData)
chain.SetMarkerStyle(6)
chain.SetMarkerColor(ROOT.kRed)

chain.Draw("s:ratioSigEff:ratioBkgEff", "nll_MarkovChain_local_", "box")  # 3-d box proportional to posterior

# the points used in the profile construction
parScanData = fc.GetPointsToScan()
print("plotting the scanned points used in the frequentist construction - npoints = ", parScanData.numEntries())

gr = ROOT.TGraph2D(parScanData.numEntries())
for ievt in range(parScanData.numEntries()):
    evt = parScanData.get(ievt)
    x = evt.getRealValue("ratioBkgEff")
    y = evt.getRealValue("ratioSigEff")
    z = evt.getRealValue("s")
    gr.SetPoint(ievt, x, y, z)

gr.SetMarkerStyle(24)
gr.Draw("P SAME")

# print timing info
t.Stop()
t.Print()

dataCanvas.SaveAs("rs101_limitexample.png")

# TODO: The MCMCCalculator has to be destructed first. Otherwise, we can get
# segmentation faults depending on the destruction order, which is random in
# Python. Probably the issue is that some object has a non-owning pointer to
# another object, which it uses in its destructor. This should be fixed either
# in the design of RooStats in C++, or with phythonizations.
del mc
