# \file
# \ingroup tutorial_roostats
# \notebook
# OneSidedFrequentistUpperLimitWithBands
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
# The first ~100 lines define a new test statistic, then the main macro starts.
# You may want to control:
# ~~~{.cpp}
#   double confidenceLevel=0.95;
#   int nPointsToScan = 12;
#   int nToyMC = 150;
# ~~~
# This uses a modified version of the profile likelihood ratio as
# a test statistic for upper limits (eg. test stat = 0 if muhat>mu).
#
# Based on the observed data, one defines a set of parameter points
# to be tested based on the value of the parameter of interest
# and the conditional MLE (eg. profiled) values of the nuisance parameters.
#
# At each parameter point, pseudo-experiments are generated using this
# fixed reference model and then the test statistic is evaluated.
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
# This is done by hand for now, will later be part of the RooStats tools.
#
# On a technical note, this technique is NOT the Feldman-Cousins technique,
# because that is a 2-sided interval BY DEFINITION.  However, like the
# Feldman-Cousins technique this is a Neyman-Construction.  For technical
# reasons the easiest way to implement this right now is to use the
# FeldmanCousins tool and then change the test statistic that it is using.
#
# Building the confidence belt can be computationally expensive.  Once it is built,
# one could save it to a file and use it in a separate step.
#
# Note, if you have a boundary on the parameter of interest (eg. cross-section)
# the threshold on the one-sided test statistic starts off very small because we
# are only including downward fluctuations.  You can see the threshold in these printouts:
# ~~~{.cpp}
# [#0] PROGRESS:Generation -- generated toys: 500 / 999
# NeymanConstruction: Prog: 12/50 total MC = 39 this test stat = 0
#  SigXsecOverSM=0.69 alpha_syst1=0.136515 alpha_syst3=0.425415 beta_syst2=1.08496 [-1e+30, 0.011215]  in interval = 1
# ~~~
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
# This results in thresholds that become very large. This can be seen in the following
# thought experiment.  Say the model is
# \f$ Pois(N | s + b)G(b0|b,sigma) \f$
# where \f$ G(b0|b,sigma) \f$ is the external constraint and b0 is 100.  If N is also 100
# then the profiled value of b given s is going to be some trade off between 100-s and b0.
# If sigma is \f$ \sqrt(N) \f$, then the profiled value of b is probably 100 - s/2   So for
# s=60 we are going to have a profiled value of b~70.  Now when we generate pseudo-experiments
# for s=60, b=70 we will have N~130 and the average shat will be 30, not 60.  In practice,
# this is only an issue for values of s that are very excluded.  For values of s near the 95%
# limit this should not be a big effect.  This can be avoided if the nominal values of the constraints also fluctuate,
# but that requires that those parameters are RooRealVars in the model.
# This version does not deal with this issue, but it will be addressed in a future version.
#
# \macro_image
# \macro_output
# \macro_code
#
# \authors Kyle Cranmer (C++ version), Haichen Wang, Daniel Whiteson, and P. P. (Python translation)

import ROOT

# -------------------------------------------------------
# The actual macro


def OneSidedFrequentistUpperLimitWithBands(
    infile="", workspaceName="combined", modelConfigName="ModelConfig", dataName="obsData"
):

    confidenceLevel = 0.95
    nPointsToScan = 12
    nToyMC = 150

    # -------------------------------------------------------
    # First part is just to access a user-defined file
    # or create the standard example file if it doesn't exist
    filename = ""
    if infile == "":
        filename = "results/example_combined_GaussExample_model.root"
        fileExist = not ROOT.gSystem.AccessPathName(filename)  # note opposite return code
        # if file does not exists generate with histfactory
        if not fileExist:
            # Normally this would be run on the command line
            print(f"will run standard hist2workspace example")
            ROOT.gROOT.ProcessLine(".not  prepareHistFactory .")
            ROOT.gROOT.ProcessLine(".not  hist2workspace config/example.xml")
            print(f"\n\n---------------------")
            print(f"Done creating example input")
            print(f"---------------------\n\n")

    else:
        filename = infile

    # Try to open the file
    file = ROOT.TFile.Open(filename)

    # if input file was specified but not found, quit
    if not file:
        print(f"StandardRooStatsDemoMacro: Input file {filename} is not found")
        return

    # -------------------------------------------------------
    # Now get the data and workspace

    # get the workspace out of the file
    w = file.Get(workspaceName)
    global gw
    gw = w
    global gfile
    gfile = file

    if not w:
        print(f"workspace not found")
        return

    # get the modelConfig out of the file
    mc = w.obj(modelConfigName)

    # get the modelConfig out of the file
    data = w.data(dataName)

    # make sure ingredients are found
    if not data or not mc:
        w.Print()
        print(f"data or ModelConfig was not found")
        return

    # -------------------------------------------------------
    # Now get the POI for convenience
    # you may want to adjust the range of your POI

    firstPOI = mc.GetParametersOfInterest().first()
    #  firstPOI->setMin(0);
    #  firstPOI->setMax(10);

    # --------------------------------------------
    # Create and use the FeldmanCousins tool
    # to find and plot the 95% confidence interval
    # on the parameter of interest as specified
    # in the model config
    # REMEMBER, we will change the test statistic
    # so this is NOT a Feldman-Cousins interval
    fc = ROOT.RooStats.FeldmanCousins(data, mc)
    fc.SetConfidenceLevel(confidenceLevel)
    fc.AdditionalNToysFactor(
        0.5
    )  # degrade/improve sampling that defines confidence belt: in this case makes the example faster
    #  fc.UseAdaptiveSampling(True); # speed it up a bit, don't use for expected limits
    fc.SetNBins(nPointsToScan)  # set how many points per parameter of interest to scan
    fc.CreateConfBelt(True)  # save the information in the belt for plotting

    # -------------------------------------------------------
    # Feldman-Cousins is a unified limit by definition
    # but the tool takes care of a few things for us like which values
    # of the nuisance parameters should be used to generate toys.
    # so let's just change the test statistic and realize this is
    # no longer "Feldman-Cousins" but is a fully frequentist Neyman-Construction.
    #  ProfileLikelihoodTestStatModified onesided(*mc->GetPdf());
    #  fc.GetTestStatSampler()->SetTestStatistic(&onesided);
    # ((ToyMCSampler*) fc.GetTestStatSampler())->SetGenerateBinned(True);
    toymcsampler = fc.GetTestStatSampler()
    testStat = toymcsampler.GetTestStatistic()
    testStat.SetOneSided(True)

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
        f"\n95% interval on {firstPOI.GetName()} is : [{interval.LowerLimit(firstPOI)}, {interval.UpperLimit(firstPOI)} ]"
    )

    # get observed UL and value of test statistic evaluated there
    tmpPOI = ROOT.RooArgSet(firstPOI)
    observedUL = interval.UpperLimit(firstPOI)
    firstPOI.setVal(observedUL)
    obsTSatObsUL = fc.GetTestStatSampler().EvaluateTestStatistic(data, tmpPOI)

    # Ask the calculator which points were scanned
    parameterScan = fc.GetPointsToScan()
    tmpPoint = ROOT.RooArgSet()

    # make a histogram of parameter vs. threshold
    histOfThresholds = ROOT.TH1F(
        "histOfThresholds", "", parameterScan.numEntries(), firstPOI.getMin(), firstPOI.getMax()
    )
    histOfThresholds.GetXaxis().SetTitle(firstPOI.GetName())
    histOfThresholds.GetYaxis().SetTitle("Threshold")

    # loop through the points that were tested and ask confidence belt
    # what the upper/lower thresholds were.
    # For FeldmanCousins, the lower cut off is always 0
    for i in range(parameterScan.numEntries()):
        tmpPoint = parameterScan.get(i).clone("temp")
        # cout <<"get threshold"<<endl;
        arMax = belt.GetAcceptanceRegionMax(tmpPoint)
        poiVal = tmpPoint.getRealValue(firstPOI.GetName())
        histOfThresholds.Fill(poiVal, arMax)

    c1 = ROOT.TCanvas()
    c1.Divide(2)
    c1.cd(1)
    histOfThresholds.SetMinimum(0)
    histOfThresholds.Draw()
    c1.Update()
    c1.Draw()
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
    print(f"\nWill use these parameter points to generate pseudo data for bkg only")
    paramsToGenerateData.Print("v")

    unconditionalObs = ROOT.RooArgSet()
    unconditionalObs.add(mc.GetObservables())
    unconditionalObs.add(mc.GetGlobalObservables())  # comment this out for the original conditional ensemble

    CLb = 0
    CLbinclusive = 0

    # Now we generate background only and find distribution of upper limits
    histOfUL = ROOT.TH1F("histOfUL", "", 100, 0, firstPOI.getMax())
    histOfUL.GetXaxis().SetTitle("Upper Limit (background only)")
    histOfUL.GetYaxis().SetTitle("Entries")
    for imc in range(nToyMC):

        # set parameters back to values for generating pseudo data
        #    cout << "\n get current nuis, set vals, print again" << endl;
        w.loadSnapshot("paramsToGenerateData")
        #    poiAndNuisance->Print("v");

        toyData = ROOT.RooDataSet()
        # debugging
        global gmc
        gmc = mc
        # return
        # now generate a toy dataset
        if not mc.GetPdf().canBeExtended():
            if data.numEntries() == 1:
                toyData = mc.GetPdf().generate(mc.GetObservables(), 1)
            else:
                print(f"Not sure what to do about this model")
        else:
            # print("generating extended dataset")
            toyData = mc.GetPdf().generate(mc.GetObservables(), Extended=True)

        # generate global observables
        # need to be careful for simpdf
        #    RooDataSet* globalData = mc->GetPdf()->generate(*mc->GetGlobalObservables(),1);

        simPdf = mc.GetPdf()
        if not simPdf:
            one = mc.GetPdf().generate(mc.GetGlobalObservables(), 1)
            values = one.get()
            allVars = mc.GetPdf().getVariables()
            allVars.assign(values)
            # del values
            # del one
        else:

            # try fix for sim pdf
            for tt in simPdf.indexCat():
                catName = tt.first
                # global gcatName
                # gcatName = catName
                # return
                # Get pdf associated with state from simpdf
                pdftmp = simPdf.getPdf(str(catName))

                # Generate only global variables defined by the pdf associated with this state
                globtmp = pdftmp.getObservables(mc.GetGlobalObservables())
                tmp = pdftmp.generate(globtmp, 1)

                # Transfer values to output placeholder
                globtmp.assign(tmp.get(0))

        #    globalData->Print("v");
        #    unconditionalObs = *globalData->get();
        #    mc->GetGlobalObservables()->Print("v");
        #    delete globalData;
        #    cout << "toy data = " << endl;
        #    toyData->get()->Print("v");

        # get test stat at observed UL in observed data
        firstPOI.setVal(observedUL)
        toyTSatObsUL = fc.GetTestStatSampler().EvaluateTestStatistic(toyData, tmpPOI)
        #    toyData->get()->Print("v");
        #    cout <<"obsTSatObsUL " <<obsTSatObsUL << "toyTS " << toyTSatObsUL << endl;
        if obsTSatObsUL < toyTSatObsUL:  # not sure about <= part yet
            CLb += (1.0) / nToyMC
        if obsTSatObsUL <= toyTSatObsUL:  # not sure about <= part yet
            CLbinclusive += (1.0) / nToyMC

        # loop over points in belt to find upper limit for this toy data
        thisUL = ROOT.Double_t(0)
        for i in range(parameterScan.numEntries()):
            tmpPoint = parameterScan.get(i).clone("temp")
            arMax = belt.GetAcceptanceRegionMax(tmpPoint)
            firstPOI.setVal(tmpPoint.getRealValue(firstPOI.GetName()))
            #   double thisTS = profile->getVal();
            thisTS = fc.GetTestStatSampler().EvaluateTestStatistic(toyData, tmpPOI)

            #   cout << "poi = " << firstPOI->getVal()
            # << " max is " << arMax << " this profile = " << thisTS << endl;
            #      cout << "thisTS = " << thisTS<<endl;
            if thisTS <= arMax:
                thisUL = firstPOI.getVal()
            else:
                break

        """
      #
      # loop over points in belt to find upper limit for this toy data
      thisUL = 0
      for i in range(histOfThresholds.GetNbinsX() ++i)
         tmpPoint = (RooArgSet) parameterScan.get(i).clone("temp")
         print("----------------  ", i)
         tmpPoint.Print("v")
         print(f"from hist ", histOfThresholds.GetBinCenter(i+1) )
         arMax = histOfThresholds.GetBinContent(i+1)
         # cout << " threshold from Hist = aMax " << arMax<<endl;
         # double arMax2 = belt->GetAcceptanceRegionMax(*tmpPoint);
         # cout << "from scan arMax2 = "<< arMax2 << endl; # not the same due to TH1F not TH1D
         # cout << "scan - hist" << arMax2-arMax << endl;
         firstPOI.setVal( histOfThresholds.GetBinCenter(i+1))
         #   double thisTS = profile->getVal();
         thisTS = fc.GetTestStatSampler().EvaluateTestStatistic(toyData,tmpPOI)

         #   cout << "poi = " << firstPOI->getVal()
         # = ROOT.Double_t() << " max is " << arMax << " this profile = " << thisTS << endl;
         #      cout << "thisTS = " << thisTS<<endl;

         # NOTE: need to add a small epsilon term for single precision vs. double precision
#         if(thisTS<=arMax + 1e-7){
#            thisUL = firstPOI->getVal();
#         } else{
#            break;
#         }
#      }
#      */
#
      """

        histOfUL.Fill(thisUL)

        # for few events, data is often the same, and UL is often the same
        #    cout << "thisUL = " << thisUL<<endl;

        # delete toyData
    c1.cd(2)
    histOfUL.Draw()
    c1.Update()
    c1.Draw()
    c1.SaveAs("OneSidedFrequentistUpperLimitWithBands.png")

    # if you want to see a plot of the sampling distribution for a particular scan point:
    #
    """
   SamplingDistPlot sampPlot
   indexInScan = 0
   tmpPoint = (RooArgSet) parameterScan.get(indexInScan).clone("temp")
   firstPOI.setVal( tmpPoint.getRealValue(firstPOI.GetName()) )
   toymcsampler.SetParametersForTestStat(tmpPOI)
   samp = toymcsampler.GetSamplingDistribution(tmpPoint)
   sampPlot.AddSamplingDistribution(samp)
   sampPlot.Draw()
   """

    # Now find bands and power constraint
    bins = histOfUL.GetIntegral()
    cumulative = histOfUL.Clone("cumulative")
    cumulative.SetContent(bins)
    band2sigDown = band1sigDown = bandMedian = band1sigUp = band2sigUp = ROOT.Double_t()
    for i in range(cumulative.GetNbinsX()):
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

    print(f"-2 sigma  band ", band2sigDown)
    print(f"-1 sigma  band {band1sigDown} [Power Constraint)]")
    print(f"median of band ", bandMedian)
    print(f"+1 sigma  band ", band1sigUp)
    print(f"+2 sigma  band ", band2sigUp)

    # print out the interval on the first Parameter of Interest
    print(f"\nObserved 95% upper-limit ", interval.UpperLimit(firstPOI))
    print(f"CLb strict [P(toy>obs|0)] for observed 95% upper-limit ", CLb)
    print("inclusive [P(toy>=obs|0)] for observed 95% upper-limit ", CLbinclusive)


OneSidedFrequentistUpperLimitWithBands(
    infile="", workspaceName="combined", modelConfigName="ModelConfig", dataName="obsData"
)
