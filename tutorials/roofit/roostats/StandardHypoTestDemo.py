# \file
# \ingroup tutorial_roostats
# \notebook
# Standard tutorial macro for hypothesis test (for computing the discovery significance) using all
# RooStats hypothesis tests calculators and test statistics.
#
# Usage:
#
# ~~~{.cpp}
# root>.L StandardHypoTestDemo.C
# root> StandardHypoTestDemo("fileName","workspace name","S+B modelconfig name","B model name","data set
# name",calculator type, test statistic type, number of toys)
#
#  type = 0 Freq calculator
#  type = 1 Hybrid calculator
#  type = 2 Asymptotic calculator
#  type = 3 Asymptotic calculator using nominal Asimov data sets (not using fitted parameter values but nominal ones)
#
# testStatType = 0 LEP
#              = 1 Tevatron
#              = 2 Profile Likelihood
#              = 3 Profile Likelihood one sided (i.e. = 0 if mu_hat < 0)
# ~~~
#
# \macro_image
# \macro_output
# \macro_code
#
# \author Lorenzo Moneta (C++ version), and P. P. (Python translation)

import array
from gc import collect
from warnings import warn

import ROOT


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


HypoTestOptions = Struct(
    # force all systematics to be off (i.e. set all nuisance parameters as constant)
    noSystematics=False,  # ratio Ntoys Null/ntoys ALT
    nToysRatio=4,  # change poi snapshot value for S+B model (needed for expected p0 values)
    poiValue=-1,
    printLevel=0,
    generateBinned=False,  # for binned generation
    enableDetailedOutput=True,  # for detailed output
)

optHT = HypoTestOptions


def StandardHypoTestDemo(
    infile="",
    workspaceName="combined",
    modelSBName="ModelConfig",
    modelBName="",  # 0 freq, 1 hybrid, 2 asymptotic \
    dataName="obsData",
    calcType=0,  # 0 LEP, 1 TeV, 2 LHC, 3 LHC - one sided \
    testStatType=3,
    ntoys=5000,
    useNC=False,
    nuisPriorName=0,
):

    noSystematics = optHT.noSystematics
    nToysRatio = optHT.nToysRatio  # ratio Ntoys Null/ntoys ALT
    poiValue = optHT.poiValue  # change poi snapshot value for S+B model (needed for expected p0 values)
    printLevel = optHT.printLevel
    generateBinned = optHT.generateBinned  # for binned generation
    enableDetOutput = optHT.enableDetailedOutput

    # Other Parameter to pass in tutorial
    # apart from standard for filename, ws, modelconfig and data

    # type = 0 Freq calculator
    # type = 1 Hybrid calculator
    # type = 2 Asymptotic calculator

    # testStatType = 0 LEP
    # = 1 Tevatron
    # = 2 Profile Likelihood
    # = 3 Profile Likelihood one sided (i.e. = 0 if mu < mu_hat)

    # ntoys:         number of toys to use

    # useNumberCounting:  set to True when using number counting events

    # nuisPriorName:   name of prior for the nuisance. This is often expressed as constraint term in the global model
    # It is needed only when using the HybridCalculator (type=1)
    # If not given by default the prior pdf from ModelConfig is used.

    # extra options are available as global parameters of the macro. They major ones are:

    # generateBinned       generate binned data sets for toys (default is false) - be careful not to activate with
    # a too large (>=3) number of observables
    # nToyRatio            ratio of S+B/B toys (default is 2)
    # printLevel

    # disable - can cause some problems
    # ToyMCSampler::SetAlwaysUseMultiGen(True);

    ROOT.RooStats.SimpleLikelihoodRatioTestStat.SetAlwaysReuseNLL(True)
    ROOT.RooStats.ProfileLikelihoodTestStat.SetAlwaysReuseNLL(True)
    ROOT.RooStats.RatioOfProfiledLikelihoodsTestStat.SetAlwaysReuseNLL(True)

    # RooRandom::randomGenerator()->SetSeed(0);

    # to change minimizers
    # ~~~{.bash}
    # ROOT::Math::MinimizerOptions::SetDefaultStrategy(0);
    # ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2");
    # ROOT::Math::MinimizerOptions::SetDefaultTolerance(1);
    # ~~~

    # -------------------------------------------------------
    # First part is just to access a user-defined file
    # or create the standard example file if it doesn't exist
    filename = ""
    if infile == "":
        filename = "results/example_combined_GaussExample_model.root"
        fileExist = not ROOT.gSystem.AccessPathName(filename, ROOT.kFileExists)  # reverse-convention 'not'
        # if file does not exists generate with histfactory
        if not fileExist:
            # Normally this would be run on the command line
            print("will run standard hist2workspace example")
            ROOT.gROOT.ProcessLine(".!  prepareHistFactory .")
            ROOT.gROOT.ProcessLine(".! hist2workspace config/example.xml")
            print("\n\n---------------------")
            print("Done creating example input")
            print("---------------------\n\n")

    else:
        filename = infile

    # Try to open the file
    file = ROOT.TFile.Open(filename)

    # if input file was specified but not found, quit
    if not file:
        print(f"StandardRooStatsDemoMacro: Input file {filename} is not found")
        return

    # -------------------------------------------------------
    # Tutorial starts here
    # -------------------------------------------------------

    # get the workspace out of the file
    w = file.Get(workspaceName)
    if not w:
        print("workspace not found")
        return

    w.Print()

    # get the modelConfig out of the file
    sbModel = w.obj(modelSBName)

    # get the modelConfig out of the file
    data = w.data(dataName)

    # make sure ingredients are found
    if not data or not sbModel:
        w.Print()
        print("data or ModelConfig was not found")
        return

    # make b model
    bModel = w.obj(modelBName)

    # case of no systematics
    # remove nuisance parameters from model
    if noSystematics:
        nuisPar = sbModel.GetNuisanceParameters()
        if nuisPar and nuisPar.getSize() > 0:
            print("StandardHypoTestInvDemo")
            print("  -  Switch off all systematics by setting them constant to their initial values")
            ROOT.RooStats.SetAllConstant(nuisPar)

        if bModel:
            bnuisPar = bModel.GetNuisanceParameters()
            if bnuisPar:
                ROOT.RooStats.SetAllConstant(bnuisPar)

    if not bModel:
        ROOT.Info("StandardHypoTestInvDemo", "The background model {} does not exist".format(modelBName))
        ROOT.Info("StandardHypoTestInvDemo", "Copy it from ModelConfig {} and set POI to zero".format(modelSBName))
        bModel = sbModel.Clone()
        bModel.SetName(modelSBName + "B_only")
        var = bModel.GetParametersOfInterest().first()
        if not var:
            return
        oldval = var.getVal()
        var.setVal(0)
        # bModel->SetSnapshot( RooArgSet(*var, *w->var("lumi"))  );
        bModel.SetSnapshot(ROOT.RooArgSet(var))
        var.setVal(oldval)

    if not sbModel.GetSnapshot() or (poiValue > 0):
        ROOT.Info("StandardHypoTestDemo", "Model {} has no snapshot  - make one using model poi".format(modelSBName))
        var = sbModel.GetParametersOfInterest().first()
        if not var:
            return
        oldval = var.getVal()
        if poiValue > 0:
            var.setVal(poiValue)
        # sbModel->SetSnapshot( RooArgSet(*var, *w->var("lumi") ) );
        sbModel.SetSnapshot(ROOT.RooArgSet(var))
        if poiValue > 0:
            var.setVal(oldval)
        # sbModel->SetSnapshot( *sbModel->GetParametersOfInterest() );

    # part 1, hypothesis testing
    slrts = ROOT.RooStats.SimpleLikelihoodRatioTestStat(bModel.GetPdf(), sbModel.GetPdf())
    # null parameters must include snapshot of poi plus the nuisance values
    nullParams = ROOT.RooArgSet(bModel.GetSnapshot())
    if bModel.GetNuisanceParameters():
        nullParams.add(bModel.GetNuisanceParameters())

    slrts.SetNullParameters(nullParams)
    altParams = ROOT.RooArgSet(sbModel.GetSnapshot())
    if sbModel.GetNuisanceParameters():
        altParams.add(sbModel.GetNuisanceParameters())
    slrts.SetAltParameters(altParams)

    profll = ROOT.RooStats.ProfileLikelihoodTestStat(bModel.GetPdf())

    ropl = ROOT.RooStats.RatioOfProfiledLikelihoodsTestStat(bModel.GetPdf(), sbModel.GetPdf(), sbModel.GetSnapshot())
    ropl.SetSubtractMLE(False)

    if testStatType == 3:
        profll.SetOneSidedDiscovery(1)
        profll.SetPrintLevel(printLevel)

    if enableDetOutput:
        slrts.EnableDetailedOutput()
        profll.EnableDetailedOutput()
        ropl.EnableDetailedOutput()

    # profll.SetReuseNLL(mOptimize);
    # slrts.SetReuseNLL(mOptimize);
    # ropl.SetReuseNLL(mOptimize);

    ROOT.RooStats.AsymptoticCalculator.SetPrintLevel(printLevel)

    # hypoCalc = HypoTestCalculatorGeneric.__smartptr__ # unnecessary
    # note here Null is B and Alt is S+B
    if calcType == 0:
        hypoCalc = ROOT.RooStats.FrequentistCalculator(data, sbModel, bModel)
    elif calcType == 1:
        hypoCalc = ROOT.RooStats.HybridCalculator(data, sbModel, bModel)
    elif calcType == 2:
        hypoCalc = ROOT.RooStats.AsymptoticCalculator(data, sbModel, bModel)

    if calcType == 0:
        hypoCalc.SetToys(ntoys, int(ntoys / nToysRatio))
        if enableDetOutput:
            (hypoCalc).StoreFitInfo(True)

    elif calcType == 1:
        hypoCalc.SetToys(ntoys, ntoys / nToysRatio)
        # n. a. yetif (enableDetOutput) : ((HybridCalculator*) hypoCalc)->StoreFitInfo(True);

    elif calcType == 2:
        if testStatType == 3:
            hypoCalc.SetOneSidedDiscovery(True)
        elif testStatType != 2 and testStatType != 3:
            warn(
                "StandardHypoTestDemo",
                "Only the PL test statistic can be used with AsymptoticCalculator - use by default a two-sided PL",
            )

    # check for nuisance prior pdf in case of nuisance parameters
    if calcType == 1 and (bModel.GetNuisanceParameters() or sbModel.GetNuisanceParameters()):
        # nuisPdf = 0
        if nuisPriorName:
            nuisPdf = w.pdf(nuisPriorName)
        # use prior defined first in bModel (then in SbModel)
        if not nuisPdf:
            ROOT.Info(
                "StandardHypoTestDemo",
                "No nuisance pdf given for the HybridCalculator - try to deduce  pdf from the   model",
            )
            if bModel.GetPdf() and bModel.GetObservables():
                nuisPdf = ROOT.RooStats.MakeNuisancePdf(bModel, "nuisancePdf_bmodel")
            else:
                nuisPdf = ROOT.RooStats.MakeNuisancePdf(sbModel, "nuisancePdf_sbmodel")

        if not nuisPdf:
            if bModel.GetPriorPdf():
                nuisPdf = bModel.GetPriorPdf()
                ROOT.Info(
                    "StandardHypoTestDemo",
                    "No nuisance pdf given - try to use %s that is defined as a prior pdf in the B model",
                    nuisPdf.GetName(),
                )
            else:
                ROOT.Error(
                    "StandardHypoTestDemo",
                    "Cannot run Hybrid calculator because no prior on the nuisance parameter is "
                    "specified or can be derived",
                )
                return

        assert nuisPdf
        ROOT.Info("StandardHypoTestDemo", "Using as nuisance Pdf ... ")
        nuisPdf.Print()

        nuisParams = (
            bModel.GetNuisanceParameters() if bModel.GetNuisanceParameters() else sbModel.GetNuisanceParameters()
        )
        nuisParsInPdf = nuisPdf.getObservables(nuisParams)
        if nuisParsInPdf.getSize() == 0:
            warn(
                "StandardHypoTestDemo",
                "Prior nuisance does not depend on nuisance parameters. They will be smeared in their full range",
            )

        # HybridCalculator
        hypoCalc.ForcePriorNuisanceAlt(nuisPdf)
        hypoCalc.ForcePriorNuisanceNull(nuisPdf)

    # hypoCalc->ForcePriorNuisanceAlt(*sbModel->GetPriorPdf());
    # hypoCalc->ForcePriorNuisanceNull(*bModel->GetPriorPdf());

    sampler = hypoCalc.GetTestStatSampler()

    if sampler and (calcType == 0 or calcType == 1):

        # look if pdf is number counting or extended
        if sbModel.GetPdf().canBeExtended():
            if useNC:
                warn("StandardHypoTestDemo", "Pdf is extended: but number counting flag is set: ignore it ")
    else:
        # for not extended pdf
        if not useNC:
            nEvents = int(data.numEntries())
            ROOT.Info(
                "StandardHypoTestDemo",
                "Pdf is not extended: number of events to generate taken  from observed data set is {nEvents}",
            )
            sampler.SetNEventsPerToy(nEvents)
        else:
            ROOT.Info("StandardHypoTestDemo", "using a number counting pdf")
            sampler.SetNEventsPerToy(1)

    if data.isWeighted() and not generateBinned:
        msgfmt = "Data set is weighted, nentries = {} and sum of weights = {:8.1f} but toy ".format(
            data.numEntries(), data.sumEntries()
        )
        msgfmt += "generation is unbinned - it would be faster to set generateBinned to True\n"

        ROOT.Info("StandardHypoTestDemo", msgfmt)

    if generateBinned:
        sampler.SetGenerateBinned(generateBinned)

    # set the test statistic
    if testStatType == 0:
        sampler.SetTestStatistic(slrts)
    if testStatType == 1:
        sampler.SetTestStatistic(ropl)
    if testStatType == 2 or testStatType == 3:
        sampler.SetTestStatistic(profll)

    htr = hypoCalc.GetHypoTest()
    htr.SetPValueIsRightTail(True)
    htr.SetBackgroundAsAlt(False)
    htr.Print()  # how to get meaningful CLs at this point?

    del sampler
    del slrts
    del ropl
    del profll
    collect()  # Trigger the garbage collector gc.collector()

    if calcType != 2:
        c1 = ROOT.TCanvas("myc1", "myc1")
        plot = ROOT.RooStats.HypoTestPlot(htr, 100)
        plot.SetLogYaxis(True)
        plot.Draw()
        c1.Update()
        c1.Draw()
        c1.SaveAs("StandardHypoTestDemo.png")
    else:
        print("Asymptotic results ")

    # look at expected significances
    # found median of S+B distribution
    if calcType != 2:

        altDist = htr.GetAltDistribution()
        htExp = ROOT.RooStats.HypoTestResult("Expected Result")
        htExp.Append(htr)
        # find quantiles in alt (S+B) distribution
        p = array.array("d", [i for i in range(5)])
        q = array.array("d", [0.5 for i in range(5)])
        for i in range(5):
            sig = -2 + i
            p[i] = ROOT.Math.normal_cdf(sig, 1)

        values = altDist.GetSamplingDistribution()
        ROOT.TMath.Quantiles(values.size(), 5, values.data(), q, p, False)

        for i in range(5):
            htExp.SetTestStatisticData(q[i])
            sig = -2 + i
            print(
                " Expected p -value and significance at ",
                sig,
                " sigma = ",
                htExp.NullPValue(),
                " significance ",
                htExp.Significance(),
                "sigma ",
            )

        else:
            # case of asymptotic calculator
            for i in range(5):
                sig = -2 + i
                # sigma is inverted here
                pval = ROOT.RooStats.AsymptoticCalculator.GetExpectedPValues(
                    htr.NullPValue(), htr.AlternatePValue(), -sig, False
                )
                print(
                    " Expected p -value and significance at ",
                    sig,
                    " sigma = ",
                    pval,
                    " significance ",
                    ROOT.Math.normal_quantile_c(pval, 1),
                    " sigma ",
                )

    # write result in a file in case of toys
    writeResult = calcType != 2

    if enableDetOutput:
        writeResult = True
        ROOT.Info("StandardHypoTestDemo", "Detailed output will be written in output result file")

    if htr != ROOT.kNone and writeResult:

        # write to a file the results
        calcTypeName = "Freq" if (calcType == 0) else ("Hybr" if (calcType == 1) else ("Asym"))
        resultFileName = ROOT.TString.Format("{}_HypoTest_ts{}_".format(calcTypeName, testStatType))
        # strip the / from the filename

        name = ROOT.TString(infile)
        name.Replace(0, name.Last("/") + 1, "")
        resultFileName += name

        fileOut = ROOT.TFile(str(resultFileName), "RECREATE")

        htr.Write()

        ROOT.Info(
            "StandardHypoTestDemo", "HypoTestResult has been written in the file {}".format(resultFileName.Data())
        )

        fileOut.Close()


# Preparing Running ...
infile = ""
workspaceName = "combined"
modelSBName = "ModelConfig"
modelBName = ""
# 0 freq, 1 hybrid, 2 asymptotic
dataName = "obsData"
calcType = 0
# 0 LEP, 1 TeV 2 LHC, 3 LHC - one sided
testStatType = 3
ntoys = 5000
useNC = False
nuisPriorName = 0
# Running ...
StandardHypoTestDemo(
    infile, workspaceName, modelSBName, modelBName, dataName, calcType, testStatType, ntoys, useNC, nuisPriorName
)
