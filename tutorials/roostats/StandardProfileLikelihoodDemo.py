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
import sys

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

# Try to open the file
try:
    file = ROOT.TFile.Open(filename)
except:
    # if input file was specified but not found, quit
    print("StandardRooStatsDemoMacro: Input file %s is not found" % filename)
    sys.exit()

# -------------------------------------------------------
# Tutorial starts here
# -------------------------------------------------------

# get the workspace out of the file

w = file.Get(workspaceName)
if not w:
    print("Workspace not found")
    sys.exit()

# get the modelConfig out of the file
mc = w.obj(modelConfigName)

# get the modelConfig out of the file
data = w.data(dataName)

# make sure ingredients are found
if not data or not mc:
    w.Print()
    print("data or ModelConfig was not found")
    sys.exit()

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
print(
    "\n>>>> RESULT : "
    + str(confLevel * 100)
    + "% interval on "
    + firstPOI.GetName()
    + " is : ["
    + str(interval.LowerLimit(firstPOI))
    + ", "
    + str(interval.UpperLimit(firstPOI))
    + "]\n\n"
)

# make a plot

print(
    "making a plot of the profile likelihood function ....(if it is taking a lot of time use less points or the "
    "TF1 drawing option)\n"
)
plot = ROOT.RooStats.LikelihoodIntervalPlot(interval)
plot.SetNPoints(nScanPoints)  # do not use too many points, it could become very slow for some models
if poiXMin < poiXMax:
    plot.SetRange(poiXMin, poiXMax)
opt = ROOT.TString("")
if plotAsTF1:
    opt += ROOT.TString("tf1")
plot.Draw(opt.Data())  # use option TF1 if too slow (plot.Draw("tf1")

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
