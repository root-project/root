# \file
# \ingroup tutorial_roostats
# \notebook -js
# TwoSidedFrequentistUpperLimitWithBands
#
#
# This is a standard demo that can be used with any ROOT file
# prepared in the standard way.  You specify:
#  - name for input ROOT file
#  - name of workspace inside ROOT file that holds model and data
#  - name of ModelConfig that specifies details for calculator tools
#  - name of dataset
#
# With default parameters the macro will attempt to run the
# standard hist2workspace example and read the ROOT file
# that it produces.
#
# You may want to control:
# ~~~{.cpp}
#   double confidenceLevel=0.95;
#   double additionalToysFac = 1.;
#   int nPointsToScan = 12;
#   int nToyMC = 200;
# ~~~
#
# This uses a modified version of the profile likelihood ratio as
# a test statistic for upper limits (eg. test stat = 0 if muhat>mu).
#
# Based on the observed data, one defines a set of parameter points
# to be tested based on the value of the parameter of interest
# and the conditional MLE (eg. profiled) values of the nuisance parameters.
#
# At each parameter point, pseudo-experiments are generated using this
# fixed reference model and then the test statistic is evaluated.
# The auxiliary measurements (global observables) associated with the
# constraint terms in nuisance parameters are also fluctuated in the
# process of generating the pseudo-experiments in a frequentist manner
# forming an 'unconditional ensemble'.  One could form a 'conditional'
# ensemble in which these auxiliary measurements are fixed.  Note that the
# nuisance parameters are not randomized, which is a Bayesian procedure.
# Note, the nuisance parameters are floating in the fits.  For each point,
# the threshold that defines the 95% acceptance region is found.  This
# forms a "Confidence Belt".
#
# After constructing the confidence belt, one can find the confidence
# interval for any particular dataset by finding the intersection
# of the observed test statistic and the confidence belt.  First
# this is done on the observed data to get an observed 1-sided upper limt.
#
# Finally, there expected limit and bands (from background-only) are
# formed by generating background-only data and finding the upper limit.
# The background-only is defined as such that the nuisance parameters are
# fixed to their best fit value based on the data with the signal rate fixed to 0.
# The bands are done by hand for now, will later be part of the RooStats tools.
#
# On a technical note, this technique IS the generalization of Feldman-Cousins
# with nuisance parameters.
#
# Building the confidence belt can be computationally expensive.
# Once it is built, one could save it to a file and use it in a separate step.
#
# We can use PROOF to speed things along in parallel, however,
# the test statistic has to be installed on the workers
# so either turn off PROOF or include the modified test statistic
# in your $ROOTSYS/roofit/roostats/inc directory,
# add the additional line to the LinkDef.h file,
# and recompile root.
#
# Note, if you have a boundary on the parameter of interest (eg. cross-section)
# the threshold on the two-sided test statistic starts off at moderate values and plateaus.
#
# [#0] PROGRESS:Generation -- generated toys: 500 / 999
# NeymanConstruction: Prog: 12/50 total MC = 39 this test stat = 0
#  SigXsecOverSM=0.69 alpha_syst1=0.136515 alpha_syst3=0.425415 beta_syst2=1.08496 [-1e+30, 0.011215]  in interval = 1
#
# this tells you the values of the parameters being used to generate the pseudo-experiments
# and the threshold in this case is 0.011215.  One would expect for 95% that the threshold
# would be ~1.35 once the cross-section is far enough away from 0 that it is essentially
# unaffected by the boundary.  As one reaches the last points in the scan, the
# threshold starts to get artificially high.  This is because the range of the parameter in
# the fit is the same as the range in the scan.  In the future, these should be independently
# controlled, but they are not now.  As a result the ~50% of pseudo-experiments that have an
# upward fluctuation end up with muhat = muMax.  Because of this, the upper range of the
# parameter should be well above the expected upper limit... but not too high or one will
# need a very large value of nPointsToScan to resolve the relevant region.  This can be
# improved, but this is the first version of this script.
#
# Important note: when the model includes external constraint terms, like a Gaussian
# constraint to a nuisance parameter centered around some nominal value there is
# a subtlety.  The asymptotic results are all based on the assumption that all the
# measurements fluctuate... including the nominal values from auxiliary measurements.
# If these do not fluctuate, this corresponds to an "conditional ensemble".  The
# result is that the distribution of the test statistic can become very non-chi^2.
# This results in thresholds that become very large.
#
# \macro_image
# \macro_output
# \macro_code
#
# \authors Kyle Cranmer, contributions from Aaron Armbruster, Haoshuang Ji, Haichen Wang, Daniel Whiteson, and Jolly Chen (Python translation)

import ROOT

# -------------------------------------------------------


# User configuration parameters
infile = ""
workspaceName = "combined"
modelConfigName = "ModelConfig"
dataName = "obsData"


confidenceLevel = 0.95
# degrade/improve number of pseudo-experiments used to define the confidence belt.
# value of 1 corresponds to default number of toys in the tail, which is 50/(1-confidenceLevel)
additionalToysFac = 0.5
nPointsToScan = 20  # number of steps in the parameter of interest
nToyMC = 200  # number of toys used to define the expected limit and band

# -------------------------------------------------------
# First part is just to access a user-defined file
# or create the standard example file if it doesn't exist
filename = ""
if not infile:
    filename = "results/example_combined_GaussExample_model.root"
    fileExist = not ROOT.gSystem.AccessPathName(filename)  # note opposite return code
    # if file does not exists generate with histfactory
    if not fileExist:
        # Normally this would be run on the command line
        print(f"will run standard hist2workspace example")
        ROOT.gROOT.ProcessLine(".!  prepareHistFactory .")
        ROOT.gROOT.ProcessLine(".!  hist2workspace config/example.xml")
        print("\n\n---------------------")
        print("Done creating example input")
        print("---------------------\n\n")
else:
    filename = infile

# Try to open the file
inputFile = ROOT.TFile.Open(filename)

# -------------------------------------------------------
# Now get the data and workspace

# get the workspace out of the file
w = inputFile.Get(workspaceName)

# get the modelConfig out of the file
mc = w.obj(modelConfigName)

# get the modelConfig out of the file
data = w.data(dataName)

print(f"Found data and ModelConfig:")
mc.Print()

# -------------------------------------------------------
# Now get the POI for convenience
# you may want to adjust the range of your POI
firstPOI = mc.GetParametersOfInterest().first()
# firstPOI.setMin(0)
# firstPOI.setMax(10)

# -------------------------------------------------------
# create and use the FeldmanCousins tool
# to find and plot the 95% confidence interval
# on the parameter of interest as specified
# in the model config
# REMEMBER, we will change the test statistic
# so this is NOT a Feldman-Cousins interval
fc = ROOT.RooStats.FeldmanCousins(data, mc)
fc.SetConfidenceLevel(confidenceLevel)
fc.AdditionalNToysFactor(additionalToysFac)  # improve sampling that defines confidence belt
# fc.UseAdaptiveSampling(True) # speed it up a bit, but don't use for expected limits
fc.SetNBins(nPointsToScan)  # set how many points per parameter of interest to scan
fc.CreateConfBelt(True)  # save the information in the belt for plotting

# -------------------------------------------------------
# Feldman-Cousins is a unified limit by definition
# but the tool takes care of a few things for us like which values
# of the nuisance parameters should be used to generate toys.
# so let's just change the test statistic and realize this is
# no longer "Feldman-Cousins" but is a fully frequentist Neyman-Construction.
# fc.GetTestStatSampler().SetTestStatistic(onesided)
# fc.GetTestStatSampler().SetGenerateBinned(True)
toymcsampler = fc.GetTestStatSampler()
testStat = toymcsampler.GetTestStatistic()

# Since this tool needs to throw toy MC the PDF needs to be
# extended or the tool needs to know how many entries in a dataset
# per pseudo experiment.
# In the 'number counting form' where the entries in the dataset
# are counts, and not values of discriminating variables, the
# datasets typically only have one entry and the PDF is not
# extended.
if not mc.GetPdf().canBeExtended():
    if data.numEntries() == 1:
        fc.FluctuateNumDataEntries(False)
    else:
        print(f"Not sure what to do about this model")

if mc.GetGlobalObservables():
    print(f"will use global observables for unconditional ensemble")
    mc.GetGlobalObservables().Print()
    toymcsampler.SetGlobalObservables(mc.GetGlobalObservables())

# Now get the interval
interval = fc.GetInterval()
belt = fc.GetConfidenceBelt()

# print out the interval on the first Parameter of Interest
print(
    f"\n95% interval on {firstPOI.GetName()} is : [{interval.LowerLimit(firstPOI)}, {interval.UpperLimit(firstPOI)}] "
)

# get observed UL and value of test statistic evaluated there
tmpPOI = ROOT.RooArgSet(firstPOI)
observedUL = interval.UpperLimit(firstPOI)
firstPOI.setVal(observedUL)
obsTSatObsUL = fc.GetTestStatSampler().EvaluateTestStatistic(data, tmpPOI)

# Ask the calculator which points were scanned
parameterScan = fc.GetPointsToScan()

# make a histogram of parameter vs. threshold
histOfThresholds = ROOT.TH1F("histOfThresholds", "", parameterScan.numEntries(), firstPOI.getMin(), firstPOI.getMax())
histOfThresholds.SetDirectory(ROOT.nullptr)  # so th histogram doesn't get attached to the file with the workspace
histOfThresholds.GetXaxis().SetTitle(firstPOI.GetName())
histOfThresholds.GetYaxis().SetTitle("Threshold")

# loop through the points that were tested and ask confidence belt
# what the upper/lower thresholds were.
# For FeldmanCousins, the lower cut off is always 0
for i in range(parameterScan.numEntries()):
    tmpPoint = parameterScan.get(i).clone("temp")
    # print(get threshold")
    arMax = belt.GetAcceptanceRegionMax(tmpPoint)
    poiVal = tmpPoint.getRealValue(firstPOI.GetName())
    histOfThresholds.Fill(poiVal, arMax)

c1 = ROOT.TCanvas()
c1.Divide(2)
c1.cd(1)
histOfThresholds.SetMinimum(0)
histOfThresholds.Draw()
c1.cd(2)

# -------------------------------------------------------
# Now we generate the expected bands and power-constraint

# First: find parameter point for mu=0, with conditional MLEs for nuisance parameters
nll = mc.GetPdf().createNLL(data)
profile = nll.createProfile(mc.GetParametersOfInterest())
firstPOI.setVal(0.0)
profile.getVal()  # this will do fit and set nuisance parameters to profiled values
poiAndNuisance = ROOT.RooArgSet()
if mc.GetNuisanceParameters():
    poiAndNuisance.add(mc.GetNuisanceParameters())
poiAndNuisance.add(mc.GetParametersOfInterest())
w.saveSnapshot("paramsToGenerateData", poiAndNuisance)
paramsToGenerateData = poiAndNuisance.snapshot()
print("\nWill use these parameter points to generate pseudo data for bkg only")
paramsToGenerateData.Print("v")

unconditionalObs = ROOT.RooArgSet()
unconditionalObs.add(mc.GetObservables())
unconditionalObs.add(mc.GetGlobalObservables())  # comment this out for the original conditional ensemble

CLb = 0
CLbinclusive = 0

# Now we generate background only and find distribution of upper limits
histOfUL = ROOT.TH1F("histOfUL", "", 100, 0, firstPOI.getMax())
histOfUL.SetDirectory(ROOT.nullptr)  # make sure the histogram doesn't get attached to the file with the workspace
histOfUL.GetXaxis().SetTitle("Upper Limit (background only)")
histOfUL.GetYaxis().SetTitle("Entries")
for imc in range(nToyMC):

    # set parameters back to values for generating pseudo data
    # print("\n get current nuis, set vals, print again")
    w.loadSnapshot("paramsToGenerateData")
    # poiAndNuisance.Print("v")

    # now generate a toy dataset for the main measurement
    if not mc.GetPdf().canBeExtended():
        if data.numEntries() == 1:
            toyData = mc.GetPdf().generate(mc.GetObservables(), 1)
        else:
            print(f"Not sure what to do about this model")
    else:
        # print("generating extended dataset")
        toyData = mc.GetPdf().generate(mc.GetObservables(), Extended=True)

    # generate global observables
    one = mc.GetPdf().generate(mc.GetGlobalObservables(), 1)
    values = one.get()
    allVars = mc.GetPdf().getVariables()
    allVars.assign(values)

    # get test stat at observed UL in observed data
    firstPOI.setVal(observedUL)
    toyTSatObsUL = fc.GetTestStatSampler().EvaluateTestStatistic(toyData, tmpPOI)
    # toyData.get().Print("v")
    # print("obsTSatObsUL ", obsTSatObsUL, "toyTS ", toyTSatObsUL)
    if obsTSatObsUL < toyTSatObsUL:  # not sure about <= part yet
        CLb += (1.0) / nToyMC
    if obsTSatObsUL <= toyTSatObsUL:  # not sure about <= part yet
        CLbinclusive += (1.0) / nToyMC

    # loop over points in belt to find upper limit for this toy data
    thisUL = 0
    for i in range(parameterScan.numEntries()):
        tmpPoint = parameterScan.get(i).clone("temp")
        arMax = belt.GetAcceptanceRegionMax(tmpPoint)
        firstPOI.setVal(tmpPoint.getRealValue(firstPOI.GetName()))
        # thisTS = profile.getVal()
        thisTS = fc.GetTestStatSampler().EvaluateTestStatistic(toyData, tmpPOI)

        # print(f"poi = {firstPOI.getVal()} max is {arMax} this profile = {thisTS}")
        # print("thisTS = ", thisTS)
        if thisTS <= arMax:
            thisUL = firstPOI.getVal()
        else:
            break

    histOfUL.Fill(thisUL)

    # for few events, data is often the same, and UL is often the same
    # print("thisUL = ", thisUL)

# At this point we can close the input file, since the RooWorkspace is not used
# anymore.
inputFile.Close()

histOfUL.Draw()
c1.SaveAs("two-sided_upper_limit_output.pdf")

# if you want to see a plot of the sampling distribution for a particular scan point:
#
# sampPlot = ROOT.RooStats.SamplingDistPlot()
# indexInScan = 0
# tmpPoint = parameterScan.get(indexInScan).clone("temp")
# firstPOI.setVal( tmpPoint.getRealValue(firstPOI.GetName()) )
# toymcsampler.SetParametersForTestStat(tmpPOI)
# samp = toymcsampler.GetSamplingDistribution(tmpPoint)
# sampPlot.AddSamplingDistribution(samp)
# sampPlot.Draw()

# Now find bands and power constraint
bins = histOfUL.GetIntegral()
cumulative = histOfUL.Clone("cumulative")
cumulative.SetContent(bins)
band2sigDown = 0
band1sigDown = 0
bandMedian = 0
band1sigUp = 0
band2sigUp = 0
for i in range(1, cumulative.GetNbinsX() + 1):
    if bins[i] < ROOT.RooStats.SignificanceToPValue(2):
        band2sigDown = cumulative.GetBinCenter(i)
    if bins[i] < ROOT.RooStats.SignificanceToPValue(1):
        band1sigDown = cumulative.GetBinCenter(i)
    if bins[i] < 0.5:
        bandMedian = cumulative.GetBinCenter(i)
    if bins[i] < ROOT.RooStats.SignificanceToPValue(-1):
        band1sigUp = cumulative.GetBinCenter(i)
    if bins[i] < ROOT.RooStats.SignificanceToPValue(-2):
        band2sigUp = cumulative.GetBinCenter(i)

print(f"-2 sigma  band {band2sigDown}")
print(f"-1 sigma  band {band1sigDown} [Power Constraint)]")
print(f"median of band {bandMedian}")
print(f"+1 sigma  band {band1sigUp}")
print(f"+2 sigma  band {band2sigUp}")

# print out the interval on the first Parameter of Interest
print(f"\nobserved 95% upper-limit {interval.UpperLimit(firstPOI)}")
print(f"CLb strict [P(toy>obs|0)] for observed 95% upper-limit {CLb}")
print(f"CLb inclusive [P(toy>=obs|0)] for observed 95% upper-limit {CLbinclusive}")
