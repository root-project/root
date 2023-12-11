## \file
## \ingroup tutorial_roostats
## \notebook
## Standard demo of the Profile Likelihood calculator
## StandardProfileLikelihoodDemo
##
## This is a standard demo that can be used with any ROOT file
## prepared in the standard way.  You specify:
##  - name for input ROOT file
##  - name of workspace inside ROOT file that holds model and data
##  - name of ModelConfig that specifies details for calculator tools
##  - name of dataset
## With the values provided below the macro will attempt to run the
## standard hist2workspace example and read the ROOT file
## that it produces.
##
## The actual heart of the demo is only about 10 lines long.
##
## The ProfileLikelihoodCalculator is based on Wilks's theorem
## and the asymptotic properties of the profile likelihood ratio
## (eg. that it is chi-square distributed for the true value).
##
## \macro_image
## \macro_output
## \macro_code
##
## \authors Akeem Hart, Kyle Cranmer (C++ Version)

import ROOT

workspaceName = "combined"
modelConfigName = "ModelConfig"
dataName = "obsData"
confLevel = 0.95
nScanPoints = 50
plotAsTF1 = False
poiXMin = 1
poiXMax = 0
doHypoTest = False
nullParamValue = 0
filename = "results/example_combined_GaussExample_model.root"
# if file does not exists generate with histfactory
if ROOT.gSystem.AccessPathName(filename):
    # Normally this would be run on the command line
    print("will run standard hist2workspace example")
    ROOT.gROOT.ProcessLine(".! prepareHistFactory .")
    ROOT.gROOT.ProcessLine(".! hist2workspace config/example.xml")
    print("\n\n---------------------")
    print("Done creating example input")
    print("---------------------\n\n")

file = ROOT.TFile.Open(filename)

# -------------------------------------------------------
# Tutorial starts here
# -------------------------------------------------------

# get the workspace out of the file

w = file.Get(workspaceName)

# get the modelConfig out of the file
mc = w[modelConfigName]

# get the modelConfig out of the file
data = w[dataName]

# ---------------------------------------------
# create and use the ProfileLikelihoodCalculator
# to find and plot the 95% confidence interval
# on the parameter of interest as specified
# in the model config

pl = ROOT.RooStats.ProfileLikelihoodCalculator(data, mc)
pl.SetConfidenceLevel(confLevel)
interval = pl.GetInterval()

# print out the interval on the first Parameter of Interest
firstPOI = mc.GetParametersOfInterest().first()
limit_lower, limit_upper = interval.LowerLimit(firstPOI), interval.UpperLimit(firstPOI)
print(f"\n>>>> RESULT : {confLevel * 100}% interval on {firstPOI.GetName()} is : [{limit_lower}, {limit_upper}]\n")

# make a plot

print(
    "making a plot of the profile likelihood function ....(if it is taking a lot of time use less points or the "
    "TF1 drawing option)\n"
)
plot = ROOT.RooStats.LikelihoodIntervalPlot(interval)
plot.SetNPoints(nScanPoints)  # do not use too many points, it could become very slow for some models
if poiXMin < poiXMax:
    plot.SetRange(poiXMin, poiXMax)
opt = ""
if plotAsTF1:
    opt += "tf1"
plot.Draw(opt)  # use option TF1 if too slow (plot.Draw("tf1")

# if requested perform also an hypothesis test for the significance
if doHypoTest:
    nullparams = ROOT.RooArgSet("nullparams")
    nullparams.addClone(firstPOI)
    nullparams.setRealValue(firstPOI.GetName(), nullParamValue)
    pl.SetNullParameters(nullparams)
    print("Perform Test of Hypothesis : null Hypothesis is " + firstPOI.GetName() + str(nullParamValue))
    result = pl.GetHypoTest()
    print("\n>>>> Hypotheis Test Result ")
    result.Print()
