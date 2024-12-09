# \file
# \ingroup tutorial_roostats
# \notebook
# Standard tutorial macro for performing an inverted  hypothesis test for computing an interval
#
# This macro will perform a scan of the p-values for computing the interval or limit
#
# Usage:
#
# ~~~{.py}
# ipython3> %run StandardHypoTestInvDemo.C
# ipython3> StandardHypoTestInvDemo("fileName","workspace name","S+B modelconfig name","B model name","data set")
# name",calculator type, test statistic type, use CLS,
#                                number of points, xmin, xmax, number of toys, use number counting)
#
# type = 0 Freq calculator
# type = 1 Hybrid calculator
# type = 2 Asymptotic calculator
# type = 3 Asymptotic calculator using nominal Asimov data sets (not using fitted parameter values but nominal ones)
#
# testStatType = 0 LEP
#              = 1 Tevatron
#              = 2 Profile Likelihood two sided
#              = 3 Profile Likelihood one sided (i.e. = 0 if mu < mu_hat)
#              = 4 Profile Likelihood signed ( pll = -pll if mu < mu_hat)
#              = 5 Max Likelihood Estimate as test statistic
#              = 6 Number of observed event as test statistic
# ~~~
#
# \macro_image
# \macro_output
# \macro_code
#
# \author Lorenzo Moneta (C++ version), and P. P. (Python translation)

import ROOT
import os


# structure defining the options
class HypoTestInvOptions:

    plotHypoTestResult = True  # plot test statistic result at each point
    writeResult = True  # write HypoTestInverterResult in a file
    resultFileName = (
        ROOT.TString()
    )  # file with results (by default is built automatically using the workspace input file name)
    optimize = True  # optimize evaluation of test statistic
    useVectorStore = True  # convert data to use  roofit data store
    generateBinned = False  # generate binned data sets
    noSystematics = False  # force all systematics to be off (i.e. set all nuisance parameters as constat
    # to their nominal values)
    nToysRatio = 2  # ratio Ntoys S+b/ntoysB
    maxPOI = -1  # max value used of POI (in case of auto scan)
    enableDetailedOutput = (
        False  # enable detailed output with all fit information for each toys (output will be written in result file)
    )
    rebuild = False  # re-do extra toys for computing expected limits and rebuild test stat
    # distributions (N.B this requires much more CPU (factor is equivalent to nToyToRebuild)
    nToyToRebuild = 100  # number of toys used to rebuild
    rebuildParamValues = 0  # = 0 do a profile of all the parameters on the B (alt snapshot) before performing a
    # rebuild operation (default)
    # = 1   use initial workspace parameters with B snapshot values
    # = 2   use all initial workspace parameters with B
    # Otherwise the rebuild will be performed using
    initialFit = -1  # do a first fit to the model (-1 : default, 0 skip fit, 1 do always fit)
    randomSeed = -1  # random seed (if = -1: use default value, if = 0 always random )

    nAsimovBins = 0  # number of bins in observables used for Asimov data sets (0 is the default and it is given by
    # workspace, typically is 100)

    reuseAltToys = False  # reuse same toys for alternate hypothesis (if set one gets more stable bands)
    confLevel = 0.95  # confidence level value

    minimizerType = ""  # minimizer type (default is what is in ROOT.Math.MinimizerOptions.DefaultMinimizerType()
    massValue = ""  # extra string to tag output file of result
    printLevel = 0  # print level for debugging PL test statistics and calculators

    useNLLOffset = False  # use NLL offset when fitting (this increase stability of fits)


optHTInv = HypoTestInvOptions()

# internal class to run the inverter and more

large_declaration = """ \
namespace RooStats{

class HypoTestInvTool {

   public:
   HypoTestInvTool();
   ~HypoTestInvTool(){};

   HypoTestInverterResult *RunInverter(RooWorkspace *w, const char *modelSBName, const char *modelBName,
   const char *dataName, int type, int testStatType, bool useCLs, int npoints,
   double poimin, double poimax, int ntoys, bool useNumberCounting = false,
   const char *nuisPriorName = 0);

   void AnalyzeResult(HypoTestInverterResult *r, int calculatorType, int testStatType, bool useCLs, int npoints,
   const char *fileNameBase = 0);

   void SetParameter(const char *name, const char *value);
   void SetParameter(const char *name, bool value);
   void SetParameter(const char *name, int value);
   void SetParameter(const char *name, double value);

   //private:
   public:
   bool mPlotHypoTestResult;
   bool mWriteResult;
   bool mOptimize;
   bool mUseVectorStore;
   bool mGenerateBinned;
   bool mRebuild;
   bool mReuseAltToys;
   bool mEnableDetOutput;
   int mNToyToRebuild;
   int mRebuildParamValues;
   int mPrintLevel;
   int mInitialFit;
   int mRandomSeed;
   double mNToysRatio;
   double mMaxPoi;
   int mAsimovBins;
   std::string mMassValue;
   std::string
   mMinimizerType; // minimizer type (default is what is in ROOT::Math::MinimizerOptions::DefaultMinimizerType())
   TString mResultFileName;
};

} // end using RooStats namespace

RooStats::HypoTestInvTool::HypoTestInvTool()
   : mPlotHypoTestResult(true), mWriteResult(false), mOptimize(true), mUseVectorStore(true), mGenerateBinned(false),
     mEnableDetOutput(false), mRebuild(false), mReuseAltToys(false),
     mNToyToRebuild(100), mRebuildParamValues(0), mPrintLevel(0), mInitialFit(-1), mRandomSeed(-1), mNToysRatio(2),
     mMaxPoi(-1), mAsimovBins(0), mMassValue(""), mMinimizerType(""), mResultFileName()
{
}

"""
ROOT.gInterpreter.Declare(large_declaration)


# Expanding definitions of class: HypoTestInvTool into HypoTestInvTool_plus
class HypoTestInvTool_plus(ROOT.RooStats.HypoTestInvTool):
    # RooStats.HypoTestInvTool.SetParameter(name,  value):
    def SetParameter(self, name, value):  # SetParameter is polymorphic.
        if (type(name) is str) and (type(value) is str):
            #
            # set boolean parameters
            #

            s_name = ROOT.std.string(name)

            if s_name.find("PlotHypoTestResult") != ROOT.std.string.npos:
                self.mPlotHypoTestResult = value
            if s_name.find("WriteResult") != ROOT.std.string.npos:
                self.mWriteResult = value
            if s_name.find("Optimize") != ROOT.std.string.npos:
                self.mOptimize = value
            if s_name.find("UseVectorStore") != ROOT.std.string.npos:
                self.mUseVectorStore = value
            if s_name.find("GenerateBinned") != ROOT.std.string.npos:
                self.mGenerateBinned = value
            if s_name.find("EnableDetailedOutput") != ROOT.std.string.npos:
                self.mEnableDetOutput = value
            if s_name.find("Rebuild") != ROOT.std.string.npos:
                self.mRebuild = value
            if s_name.find("ReuseAltToys") != ROOT.std.string.npos:
                self.mReuseAltToys = value

            return
        # RooStats.HypoTestInvTool.SetParameter = SetParameter

        # RooStats.HypoTestInvTool.SetParameter(name, value):
        # def SetParameter(self, name, value):
        elif (type(name) is str) and (type(value) is bool):

            #
            # set integer parameters
            #

            s_name = ROOT.std.string(name)

            if s_name.find("NToyToRebuild") != ROOT.std.string.npos:
                self.mNToyToRebuild = value
            if s_name.find("RebuildParamValues") != ROOT.std.string.npos:
                self.mRebuildParamValues = value
            if s_name.find("PrintLevel") != ROOT.std.string.npos:
                self.mPrintLevel = value
            if s_name.find("InitialFit") != ROOT.std.string.npos:
                self.mInitialFit = value
            if s_name.find("RandomSeed") != ROOT.std.string.npos:
                self.mRandomSeed = value
            if s_name.find("AsimovBins") != ROOT.std.string.npos:
                self.mAsimovBins = value

            return
        # RooStats.HypoTestInvTool.SetParameter = SetParameter

        # RooStats.HypoTestInvTool.SetParameter(name, value):
        # def SetParameter(self, name, value):
        elif (type(name) is str) and (type(value) is int):

            #
            # set double precision parameters
            #

            s_name = ROOT.std.string(name)

            if s_name.find("NToysRatio") != ROOT.std.string.npos:
                self.mNToysRatio = value
            if s_name.find("MaxPOI") != ROOT.std.string.npos:
                self.mMaxPoi = value

            return
        # RooStats.HypoTestInvTool.SetParameter = SetParameter

        # RooStats.HypoTestInvTool.SetParameter(name, value):
        # def SetParameter(self, name, value):
        elif (type(name) is str) and (type(value) is (float or double)):

            #
            # set string parameters
            #

            s_name = ROOT.std.string(name)

            if s_name.find("MassValue") != ROOT.std.string.npos:
                global gselfmMassValue
                gselfmMassValue = self.mMassValue
                self.mMassValue = value  # (self.mMassValue).assign(value)
            if s_name.find("MinimizerType") != ROOT.std.string.npos:
                self.mMassValue = value  # self.mMinimizerType.assign(value)
            if s_name.find("ResultFileName") != ROOT.std.string.npos:
                self.mResultFileName = value

            return

    ROOT.RooStats.HypoTestInvTool.SetParameter = SetParameter

    # --piece of code moved forwards...-
    # ---

    # RooStats.HypoTestInvTool.AnalyzeResult
    def AnalyzeResult(self, r, calculatorType, testStatType, useCLs, npoints, fileNameBase):
        # type(r) is HypoTestInverterResult

        # analyze result produced by the inverter, optionally save it in a file

        lowerLimit = 0
        llError = 0
        # if defined ROOT_SVN_VERSION && ROOT_SVN_VERSION >= 44126
        if r.IsTwoSided():
            lowerLimit = r.LowerLimit()
            llError = r.LowerLimitEstimatedError()

        # else
        lowerLimit = r.LowerLimit()
        llError = r.LowerLimitEstimatedError()
        # endif

        upperLimit = r.UpperLimit()
        ulError = r.UpperLimitEstimatedError()

        # ROOT.std::cout << "DEBUG : [ " << lowerLimit << " , " << upperLimit << "  ] " << ROOT.std::endl;

        if lowerLimit < upperLimit * (1.0 - 1.0e-4) and lowerLimit != 0:
            print(f"The computed lower limit is: ", lowerLimit, " +/- ", llError)

        print(f"The computed upper limit is: {upperLimit} +/- ", ulError)

        # compute expected limit
        print(f"Expected upper limits, using the B (alternate) model : ")
        print(f" expected limit (median) ", r.GetExpectedUpperLimit(0))
        print(f" expected limit (-1 sig) ", r.GetExpectedUpperLimit(-1))
        print(f" expected limit (+1 sig) ", r.GetExpectedUpperLimit(1))
        print(f" expected limit (-2 sig) ", r.GetExpectedUpperLimit(-2))
        print(f" expected limit (+2 sig) ", r.GetExpectedUpperLimit(2))

        # detailed output
        if self.mEnableDetOutput:
            self.mWriteResult = True
            ROOT.Info("StandardHypoTestInvDemo", "detailed output will be written in output result file")

        # write result in a file
        if r != ROOT.kNone and self.mWriteResult:

            # write to a file the results
            calcType = "Freq" if (calculatorType == 0) else ("Hybr" if (calculatorType == 1) else "Asym")
            limitType = "CLs" if (useCLs) else "Cls+b"
            scanType = "auto" if (npoints < 0) else "grid"
            if self.mResultFileName.IsNull():
                self.mResultFileName = ROOT.TString.Format(
                    "%s_%s_%s_ts%d_", calcType, limitType, scanType, testStatType
                )
                # strip the / from the filename
                if self.mMassValue.size() > 0:
                    self.mResultFileName += self.mMassValue
                    self.mResultFileName += "_"

                name = fileNameBase
                name.Replace(0, name.Last("/") + 1, "")
                self.mResultFileName += name

            # get (if existing) rebuilt UL distribution
            uldistFile = "RULDist.root"
            ulDist = 0
            existULDist = not ROOT.gSystem.AccessPathName(uldistFile)
            if existULDist:
                fileULDist = TFile.Open(uldistFile)
                if fileULDist:
                    ulDist = fileULDist.Get("RULDist")

            fileOut = TFile(self.mResultFileName, "RECREATE")
            r.Write()
            if ulDist:
                ulDist.Write()
            ROOT.Info(
                "StandardHypoTestInvDemo",
                "HypoTestInverterResult has been written in the file %s",
                self.mResultFileName.Data(),
            )

            fileOut.Close()

        # plot the result ( p values vs scan points)
        typeName = ""
        if calculatorType == 0:
            typeName = "Frequentist"
        if calculatorType == 1:
            typeName = "Hybrid"
        elif calculatorType == 2 or calculatorType == 3:
            typeName = "Asymptotic"
            self.mPlotHypoTestResult = False

        resultName = r.GetName()
        plotTitle = "{} CL Scan for workspace {}".format(str(typeName), resultName)
        global gr
        gr = r
        plot = ROOT.RooStats.HypoTestInverterPlot("HTI_Result_Plot", plotTitle, r)

        # plot in a new canvas with style
        c1Name = "{}_Scan".format(typeName)
        c1 = ROOT.TCanvas(c1Name)
        c1.SetLogy(False)

        plot.Draw("CLb 2CL")  # plot all and Clb
        # if (useCLs):
        #    plot.Draw("CLb 2CL")  # plot all and Clb ???
        # else:
        #    plot.Draw("")  # plot all and Clb ???
        c1.Update()
        c1.Draw()
        c1.SaveAs("StandardHypoTestInvDemo.c1.png")

        nEntries = r.ArraySize()

        # plot test statistics distributions for the two hypothesis
        if self.mPlotHypoTestResult:
            c2 = ROOT.TCanvas("c2")
            if nEntries > 1:
                ny = ROOT.TMath.CeilNint(ROOT.TMath.Sqrt(nEntries))
                nx = ROOT.TMath.CeilNint(float(nEntries) / ny)
                c2.Divide(nx, ny)

            for i in range(nEntries):
                if nEntries > 1:
                    c2.cd(i + 1)
                pl = plot.MakeTestStatPlot(i)
                pl.SetLogYaxis(True)
                pl.Draw()
            c2.Update()
            c2.Draw()
            c2.SaveAs("StandardHypoTestInvDemo.c2.png")

        ROOT.gPad = c1

    ROOT.RooStats.HypoTestInvTool.AnalyzeResult = AnalyzeResult

    # internal routine to run the inverter
    # RooStats.HypoTestInvTool.RunInverter
    def RunInverter(
        self,
        w,
        modelSBName,
        modelBName,
        dataName,
        Type,
        testStatType,
        useCLs,
        npoints,
        poimin,
        poimax,
        ntoys,
        useNumberCounting,
        nuisPriorName,
    ):

        print(f"Running HypoTestInverter on the workspace ", w.GetName())

        w.Print()

        data = w.data(dataName)
        if not data:
            Error("StandardHypoTestDemo", "Not existing data {}".format(dataName))
            return 0
        else:
            print(f"Using data set ", dataName)

        if self.mUseVectorStore:
            ROOT.RooAbsData.setDefaultStorageType(ROOT.RooAbsData.Vector)
            data.convertToVectorStore()

        # get models from WS
        # get the modelConfig out of the file
        bModel = w.obj(modelBName)
        sbModel = w.obj(modelSBName)

        if not sbModel:
            Error("StandardHypoTestDemo", "Not existing ModelConfig %s", modelSBName)
            return 0

        # check the model
        if not sbModel.GetPdf():
            Error("StandardHypoTestDemo", "Model %s has no pdf ", modelSBName)
            return 0

        if not sbModel.GetParametersOfInterest():
            Error("StandardHypoTestDemo", "Model %s has no poi ", modelSBName)
            return 0

        if not sbModel.GetObservables():
            Error("StandardHypoTestInvDemo", "Model %s has no observables ", modelSBName)
            return 0

        if not sbModel.GetSnapshot():
            ROOT.Info(
                "StandardHypoTestInvDemo", "Model {} has no snapshot  - make one using model poi".format(modelSBName)
            )
            sbModel.SetSnapshot(sbModel.GetParametersOfInterest())

        # case of no systematics
        # remove nuisance parameters from model
        if optHTInv.noSystematics:
            nuisPar = sbModel.GetNuisanceParameters()
            if nuisPar and nuisPar.getSize() > 0:
                print(
                    f"StandardHypoTestInvDemo",
                    "  -  Switch off all systematics by setting them constant to their initial values",
                )
                ROOT.RooStats.SetAllConstant(nuisPar)

            if bModel:
                bnuisPar = bModel.GetNuisanceParameters()
                if bnuisPar:
                    ROOT.RooStats.SetAllConstant(bnuisPar)

        if (not bModel) or (bModel == sbModel):
            ROOT.Info("StandardHypoTestInvDemo", "The background model  {} does not exist".format(modelBName))
            ROOT.Info("StandardHypoTestInvDemo", "Copy it from ModelConfig {} and set POI to zero".format(modelSBName))
            bModel = sbModel.Clone()
            bModel.SetName(modelSBName + "_with_poi_0")
            var = bModel.GetParametersOfInterest().first()
            if not var:
                return 0
            oldval = var.getVal()
            var.setVal(0)
            bModel.SetSnapshot(ROOT.RooArgSet(var))
            var.setVal(oldval)
        else:
            if not bModel.GetSnapshot():
                ROOT.Info(
                    "StandardHypoTestInvDemo",
                    "Model %s has no snapshot  - make one using model poi and 0 values ",
                    modelBName,
                )
                var = bModel.GetParametersOfInterest().first()
                if var:
                    oldval = var.getVal()
                    var.setVal(0)
                    bModel.SetSnapshot(ROOT.RooArgSet(var))
                    var.setVal(oldval)
                else:
                    Error("StandardHypoTestInvDemo", "Model %s has no valid poi", modelBName)
                    return 0

        # check model  has global observables when there are nuisance pdf
        # for the hybrid case the globals are not needed
        if Type != 1:
            hasNuisParam = sbModel.GetNuisanceParameters() and sbModel.GetNuisanceParameters().getSize() > 0
            hasGlobalObs = sbModel.GetGlobalObservables() and sbModel.GetGlobalObservables().getSize() > 0
            if hasNuisParam and not hasGlobalObs:
                # try to see if model has nuisance parameters first
                constrPdf = RooStats.MakeNuisancePdf(sbModel, "nuisanceConstraintPdf_sbmodel")
                if constrPdf:
                    Warning(
                        "StandardHypoTestInvDemo",
                        "Model %s has nuisance parameters but no global observables associated",
                        sbModel.GetName(),
                    )
                    Warning(
                        "StandardHypoTestInvDemo",
                        "\tThe effect of the nuisance parameters will not be treated correctly ",
                    )

        # save all initial parameters of the model including the global observables
        initialParameters = ROOT.RooArgSet()
        allParams = sbModel.GetPdf().getParameters(data)
        allParams.snapshot(initialParameters)

        # run first a data fit

        poiSet = sbModel.GetParametersOfInterest()
        poi = poiSet.first()

        print("StandardHypoTestInvDemo : POI initial value:   ", poi.GetName(), " = ", poi.getVal())

        # fit the data first (need to use constraint )
        tw = ROOT.TStopwatch()

        doFit = self.mInitialFit
        if testStatType == 0 and self.mInitialFit == -1:
            doFit = False  # case of LEP test statistic
        if type == 3 and self.mInitialFit == -1:
            doFit = False  # case of Asymptoticcalculator with nominal Asimov
        poihat = 0

        if len(self.mMinimizerType) == 0:
            self.mMinimizerType = ROOT.Math.MinimizerOptions.DefaultMinimizerType()
        else:
            ROOT.Math.MinimizerOptions.SetDefaultMinimizer(str(self.mMinimizerType))

        ROOT.Info(
            "StandardHypoTestInvDemo",
            "Using {} as minimizer for computing the test statistic".format(
                str(ROOT.Math.MinimizerOptions.DefaultMinimizerType())
            ),
        )

        if doFit:

            # do the fit : By doing a fit the POI snapshot (for S+B)  is set to the fit value
            # and the nuisance parameters nominal values will be set to the fit value.
            # This is relevant when using LEP test statistics

            ROOT.Info("StandardHypoTestInvDemo", " Doing a first fit to the observed data ")
            constrainParams = RooArgSet()
            if sbModel.GetNuisanceParameters():
                constrainParams.add(sbModel.GetNuisanceParameters())
            RooStats.RemoveConstantParameters(constrainParams)
            tw.Start()
            fitres = sbModel.GetPdf().fitTo(
                data,
                InitialHesse(False),
                Hesse(False),
                Minimizer(str(self.mMinimizerType), "Migrad"),
                Strategy(0),
                PrintLevel(self.mPrintLevel),
                Constrain(constrainParams),
                Save(True),
                Offset(RooStats.IsNLLOffset()),
            )

            if fitres.status() != 0:
                Warning(
                    "StandardHypoTestInvDemo",
                    "Fit to the model failed - try with strategy 1 and perform first an Hesse computation",
                )
                fitres = sbModel.GetPdf().fitTo(
                    data,
                    InitialHesse(True),
                    Hesse(False),
                    Minimizer(self.mMinimizerType, "Migrad"),
                    Strategy(1),
                    PrintLevel(self.mPrintLevel + 1),
                    Constrain(constrainParams),
                    Save(True),
                    Offset(RooStats.IsNLLOffset()),
                )

            if fitres.status() != 0:
                Warning("StandardHypoTestInvDemo", " Fit still failed - continue anyway.....")

            print("StandardHypoTestInvDemo - Best Fit value : ", poi.GetName(), " = ", poihat, " +/- ", poi.getError())
            print(f"Time for fitting : ")
            tw.Print()

            # save best fit value in the poi snapshot
            sbModel.SetSnapshot(sbModel.GetParametersOfInterest())
            print(f"StandardHypoTestInvo: snapshot of S+B Model ", sbModel.GetName(), " is set to the best fit value")

        # print a message in case of LEP test statistics because it affects result by doing or not doing a fit
        if testStatType == 0:
            if not doFit:
                ROOT.Info(
                    "StandardHypoTestInvDemo",
                    "Using LEP test statistic - an initial fit is not done and the TS will use "
                    + "the nuisances at the model value",
                )
            else:
                ROOT.Info(
                    "StandardHypoTestInvDemo",
                    "Using LEP test statistic - an initial fit has been done and the TS will use "
                    + "the nuisances at the best fit value",
                )

        # build test statistics and hypotest calculators for running the inverter

        slrts = ROOT.RooStats.SimpleLikelihoodRatioTestStat(sbModel.GetPdf(), bModel.GetPdf())

        # null parameters must includes snapshot of poi plus the nuisance values
        nullParams = ROOT.RooArgSet(sbModel.GetSnapshot())
        if sbModel.GetNuisanceParameters():
            nullParams.add(sbModel.GetNuisanceParameters())
        if sbModel.GetSnapshot():
            slrts.SetNullParameters(nullParams)
            altParams = ROOT.RooArgSet(bModel.GetSnapshot())
        if bModel.GetNuisanceParameters():
            altParams.add(bModel.GetNuisanceParameters())
        if bModel.GetSnapshot():
            slrts.SetAltParameters(altParams)
        if self.mEnableDetOutput:
            slrts.EnableDetailedOutput()

        # ratio of profile likelihood - need to pass snapshot for the alt
        ropl = ROOT.RooStats.RatioOfProfiledLikelihoodsTestStat(sbModel.GetPdf(), bModel.GetPdf(), bModel.GetSnapshot())
        ropl.SetSubtractMLE(False)
        if testStatType == 11:
            ropl.SetSubtractMLE(True)
        ropl.SetPrintLevel(self.mPrintLevel)
        ropl.SetMinimizer(self.mMinimizerType.c_str())
        if self.mEnableDetOutput:
            ropl.EnableDetailedOutput()

        profll = ROOT.RooStats.ProfileLikelihoodTestStat(sbModel.GetPdf())
        if testStatType == 3:
            profll.SetOneSided(True)
        if testStatType == 4:
            profll.SetSigned(True)
        profll.SetMinimizer(self.mMinimizerType.c_str())
        profll.SetPrintLevel(self.mPrintLevel)
        if self.mEnableDetOutput:
            profll.EnableDetailedOutput()

        profll.SetReuseNLL(self.mOptimize)
        slrts.SetReuseNLL(self.mOptimize)
        ropl.SetReuseNLL(self.mOptimize)

        if self.mOptimize:
            profll.SetStrategy(0)
            ropl.SetStrategy(0)
            ROOT.Math.MinimizerOptions.SetDefaultStrategy(0)

        if self.mMaxPoi > 0:
            poi.setMax(self.mMaxPoi)  # increase limit

        maxll = ROOT.RooStats.MaxLikelihoodEstimateTestStat(sbModel.GetPdf(), poi)
        nevtts = ROOT.RooStats.NumEventsTestStat()

        ROOT.RooStats.AsymptoticCalculator.SetPrintLevel(self.mPrintLevel)

        # create the HypoTest calculator class
        hc = ROOT.nullptr
        if Type == 0:
            hc = ROOT.RooStats.FrequentistCalculator(data, bModel, sbModel)
        elif Type == 1:
            hc = ROOT.RooStats.HybridCalculator(data, bModel, sbModel)
        # elif (Type == 2 ):
        #    hc = AsymptoticCalculator(data, bModel, sbModel, false, self.mAsimovBins)
        # elif (Type == 3 ):
        #    hc = AsymptoticCalculator(data, bModel, sbModel, True, self.mAsimovBins)  # for using
        # Asimov data generated with nominal values
        elif Type == 2:
            hc = ROOT.RooStats.AsymptoticCalculator(data, bModel, sbModel, False)
        elif Type == 3:
            hc = ROOT.RooStats.AsymptoticCalculator(
                data, bModel, sbModel, True
            )  # for using Asimov data generated with nominal values
        else:
            Error(
                "StandardHypoTestInvDemo",
                "Invalid - calculator type = {Type}supported values are only :\n\t\t\t 0 "
                + "(Frequentist) , 1 (Hybrid) , 2 (Asymptotic) ",
            )

            return 0

        # set the test statistic
        testStat = 0
        if testStatType == 0:
            testStat = slrts
        if testStatType == 1 or testStatType == 11:
            testStat = ropl
        if testStatType == 2 or testStatType == 3 or testStatType == 4:
            testStat = profll
        if testStatType == 5:
            testStat = maxll
        if testStatType == 6:
            testStat = nevtts

        if testStat == 0:
            Error(
                "StandardHypoTestInvDemo",
                "Invalid - test statistic type = {testStatType} supported values are only :\n\t\t\t 0 (SLR) "
                + ", 1 (Tevatron) , 2 (PLR), 3 (PLR1), 4(MLE)",
            )

            return 0

        toymcs = hc.GetTestStatSampler()
        if toymcs and (Type == 0 or Type == 1):
            # look if pdf is number counting or extended
            if sbModel.GetPdf().canBeExtended():
                if useNumberCounting:
                    Warning("StandardHypoTestInvDemo", "Pdf is extended: but number counting flag is set: ignore it ")
            else:
                # for not extended pdf
                if not useNumberCounting:
                    nEvents = data.numEntries()
                    ROOT.Info(
                        "StandardHypoTestInvDemo",
                        "Pdf is not extended: number of events to generate taken  from observed data set is {nEvents}",
                    )
                    toymcs.SetNEventsPerToy(nEvents)
                else:
                    ROOT.Info("StandardHypoTestInvDemo", "using a number counting pdf")
                    toymcs.SetNEventsPerToy(1)

            toymcs.SetTestStatistic(testStat)

            if data.isWeighted() and not self.mGenerateBinned:
                ROOT.Info(
                    "StandardHypoTestInvDemo",
                    (
                        "Data set is weighted, nentries = {} and sum of weights = {:8.1f} but toy"
                        + "generation is unbinned - it would be faster to set self.mGenerateBinned to true\n"
                    ).format(data.numEntries(), data.sumEntries()),
                )

            toymcs.SetGenerateBinned(self.mGenerateBinned)

            toymcs.SetUseMultiGen(self.mOptimize)

            if self.mGenerateBinned and sbModel.GetObservables().getSize() > 2:
                Warning(
                    "StandardHypoTestInvDemo",
                    (
                        "generate binned is activated but the number of observable is {}. Too much "
                        + "memory could be needed for allocating all the bins"
                    ).format(sbModel.GetObservables().getSize()),
                )

            # set the random seed if needed
            if self.mRandomSeed >= 0:
                RooRandom.randomGenerator().SetSeed(self.mRandomSeed)

        # specify if need to re-use same toys
        if self.mReuseAltToys:
            hc.UseSameAltToys()

        if Type == 1:
            hhc = HybridCalculator(hc)
            assert hhc

            hhc.SetToys(ntoys, ntoys / self.mNToysRatio)  # can use less ntoys for b hypothesis

            # remove global observables from ModelConfig (this is probably not needed anymore in 5.32)
            bModel.SetGlobalObservables(RooArgSet())
            sbModel.SetGlobalObservables(RooArgSet())

            # check for nuisance prior pdf in case of nuisance parameters
            if bModel.GetNuisanceParameters() or sbModel.GetNuisanceParameters():

                # fix for using multigen (does not work in this case)
                toymcs.SetUseMultiGen(False)
                ToyMCSampler.SetAlwaysUseMultiGen(False)

                nuisPdf = 0
                if nuisPriorName:
                    nuisPdf = w.pdf(nuisPriorName)
                # use prior defined first in bModel (then in SbModel)
                if not nuisPdf:
                    ROOT.Info(
                        "StandardHypoTestInvDemo",
                        "No nuisance pdf given for the HybridCalculator - try to deduce  pdf from the model",
                    )
                    if bModel.GetPdf() and bModel.GetObservables():
                        nuisPdf = RooStats.MakeNuisancePdf(bModel, "nuisancePdf_bmodel")
                    else:
                        nuisPdf = RooStats.MakeNuisancePdf(sbModel, "nuisancePdf_sbmodel")

                if not nuisPdf:
                    if bModel.GetPriorPdf():
                        nuisPdf = bModel.GetPriorPdf()
                        ROOT.Info(
                            "StandardHypoTestInvDemo",
                            "No nuisance pdf given - try to use %s that is defined as a prior pdf in the B model",
                            nuisPdf.GetName(),
                        )
                    else:
                        Error(
                            "StandardHypoTestInvDemo",
                            "Cannot run Hybrid calculator because no prior on the nuisance "
                            "parameter is specified or can be derived",
                        )
                        return 0

                assert nuisPdf
                ROOT.Info("StandardHypoTestInvDemo", "Using as nuisance Pdf ... ")
                nuisPdf.Print()

                nuisParams = (
                    ibModel.GetNuisanceParameters()
                    if (bModel.GetNuisanceParameters())
                    else (sbModel.GetNuisanceParameters())
                )
                npnuisPdf.getObservables(nuisParams)
                if np.getSize() == 0:
                    Warning(
                        "StandardHypoTestInvDemo",
                        "Prior nuisance does not depend on nuisance parameters. They will be smeared in their full range",
                    )

                hhc.ForcePriorNuisanceAlt(nuisPdf)
                hhc.ForcePriorNuisanceNull(nuisPdf)

        elif type == 2 or type == 3:
            if testStatType == 3:
                hc.SetOneSided(True)
            if testStatType != 2 and testStatType != 3:
                Warning(
                    "StandardHypoTestInvDemo",
                    "Only the PL test statistic can be used with AsymptoticCalculator - use by default a two-sided PL",
                )
        elif type == 0:
            hc.SetToys(ntoys, ntoys / self.mNToysRatio)
            # store also the fit information for each poi point used by calculator based on toys
            if self.mEnableDetOutput:
                hc.StoreFitInfo(True)
        elif type == 1:
            hc.SetToys(ntoys, ntoys / self.mNToysRatio)
            # store also the fit information for each poi point used by calculator based on toys
            # if (self.mEnableDetOutput):
            #    hc.StoreFitInfo(True)

        # Get the result
        ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.NumIntegration)

        calc = ROOT.RooStats.HypoTestInverter(hc)
        calc.SetConfidenceLevel(optHTInv.confLevel)

        calc.UseCLs(useCLs)
        calc.SetVerbose(True)

        if npoints > 0:
            if poimin > poimax:
                # if no min/max given scan between MLE and +4 sigma
                poimin = int(poihat)
                poimax = int(poihat + 4 * poi.getError())

            print(f"Doing a fixed scan  in interval : {poimin}, {poimax} ")
            calc.SetFixedScan(npoints, poimin, poimax)
        else:
            # poi.setMax(10*int( (poihat+ 10 *poi.getError() )/10 ) )
            print(f"Doing an  automatic scan  in interval : {poi.getMin()} , ", poi.getMax())

        tw.Start()
        r = calc.GetInterval()
        print(f"Time to perform limit scan \n")
        tw.Print()

        if self.mRebuild:

            print(f"\n\n")
            print(
                f"Rebuild the upper limit distribution by re-generating new set of pseudo-experiment and re-compute "
                + "for each of them a new upper limit\n\n"
            )

            allParams = sbModel.GetPdf().getParameters(data)

            # define on which value of nuisance parameters to do the rebuild
            # default is best fit value for bmodel snapshot

            if self.mRebuildParamValues != 0:
                # set all parameters to their initial workspace values
                allParams.assign(initialParameters)

            if self.mRebuildParamValues == 0 or self.mRebuildParamValues == 1:
                constrainParams = RooArgSet()
                if sbModel.GetNuisanceParameters():
                    constrainParams.add(sbModel.GetNuisanceParameters())
                RooStats.RemoveConstantParameters(constrainParams)

                poiModel = sbModel.GetParametersOfInterest()
                bModel.LoadSnapshot()

                # do a profile using the B model snapshot
                if self.mRebuildParamValues == 0:

                    RooStats.SetAllConstant(poiModel, True)

                    sbModel.GetPdf().fitTo(
                        data,
                        InitialHesse(False),
                        Hesse(False),
                        Minimizer(self.mMinimizerType, "Migrad"),
                        Strategy(0),
                        PrintLevel(self.mPrintLevel),
                        Constrain(constrainParams),
                        Offset(RooStats.IsNLLOffset()),
                    )

                    print(f"rebuild using fitted parameter value for B-model snapshot")
                    constrainParams.Print("v")

                    RooStats.SetAllConstant(poiModel, False)

            print(f"StandardHypoTestInvDemo: Initial parameters used for rebuilding: ")
            RooStats.PrintListContent(allParams, ROOT.std.cout)

            tw.Start()
            limDist = calc.GetUpperLimitDistribution(True, self.mNToyToRebuild)
            print(f"Time to rebuild distributions :")
            tw.Print()

            if limDist:
                print(f"Expected limits after rebuild distribution ")
                print(f"expected upper limit  (median of limit distribution) ", limDist.InverseCDF(0.5))
                print(
                    f"expected -1 sig limit (0.16% quantile of limit dist) ",
                    limDist.InverseCDF(ROOT.Math.normal_cdf(-1)),
                )
                print(
                    f"expected +1 sig limit (0.84% quantile of limit dist) ",
                    limDist.InverseCDF(ROOT.Math.normal_cdf(1)),
                )
                print(
                    f"expected -2 sig limit (.025% quantile of limit dist) ",
                    limDist.InverseCDF(ROOT.Math.normal_cdf(-2)),
                )
                print(
                    f"expected +2 sig limit (.975% quantile of limit dist) ",
                    limDist.InverseCDF(ROOT.Math.normal_cdf(2)),
                )

                # Plot the upper limit distribution
                limPlot = SamplingDistPlot(50 if (self.mNToyToRebuild < 200) else 100)
                limPlot.AddSamplingDistribution(limDist)
                limPlot.GetTH1F().SetStats(True)  # display statistics
                limPlot.SetLineColor(kBlue)
                c1 = TCanvas("limPlot", "Upper Limit Distribution")
                limPlot.Draw()
                c1.Update()
                c1.Draw()
                c1.SaveAs("StandardHypoTestInvDemo.1.png")

                # save result in a file
                limDist.SetName("RULDist")
                fileOut = TFile("RULDist.root", "RECREATE")
                limDist.Write()
                fileOut.Close()

                # update r to a new updated result object containing the rebuilt expected p-values distributions
                # (it will not recompute the expected limit)
                if r:
                    del r  # need to delete previous object since GetInterval will return a cloned copy
                r = calc.GetInterval()

            else:
                print(f"ERROR : failed to re-build distributions ")

        return r

    ROOT.RooStats.HypoTestInvTool.RunInverter = RunInverter


# end class HypoTestInvTool
# Loading new definitions of HypoTestInvTool_plus into HypoTestInvTool ...
ROOT.RooStats.HypoTestInvTool = HypoTestInvTool_plus
# Could seem redundant but it would help you if you want expand new-new definitions.


# --------------------------------------------------
def StandardHypoTestInvDemo(
    infile=0,
    wsName="combined",
    modelSBName="ModelConfig",
    modelBName="",
    dataName="obsData",
    calculatorType=0,
    testStatType=0,
    useCLs=True,
    npoints=6,
    poimin=0,
    poimax=5,
    ntoys=1000,
    useNumberCounting=False,
    nuisPriorName=0,
):
    """

    Other Parameter to pass in tutorial
    apart from standard for filename, ws, modelconfig and data

    type = 0 Freq calculator
    type = 1 Hybrid calculator
    type = 2 Asymptotic calculator
    type = 3 Asymptotic calculator using nominal Asimov data sets (not using fitted parameter values but nominal ones)

    testStatType = 0 LEP
    Tevatron
    Likelihood
    mu_hat)
    mu_hat)
    statistic
    statistic

    useCLs          scan for CLs (otherwise for CLs+b)

    npoints = -1

    poimin,poimax:  min/max value to scan in case of fixed scans
    (if min >  max, try to find automatically)

    ntoys:         number of toys to use

    useNumberCounting:  set to True when using number counting events

    nuisPriorName:   name of prior for the nuisance. This is often expressed as constraint term in the global model
    HybridCalculator (type=1)
    If not given by default the prior pdf from ModelConfig is used.

    extra options are available as global parameters of the macro. They major ones are:

    plotHypoTestResult   plot result of tests at each point (TS distributions) (default is True)
    writeResult          write result of scan (default is True)
    rebuild              rebuild scan for expected limits (require extra toys) (default is False)
    generateBinned       generate binned data sets for toys (default is False) - be careful not to activate with
    large (>=3) number of observables
    nToyRatio            ratio of S+B/B toys (default is 2)
    """

    filename = ROOT.TString(infile)
    if filename.IsNull():
        filename = "results/example_combined_GaussExample_model.root"
        fileExist = not ROOT.gSystem.AccessPathName(filename)  # note opposite return code
        # if file does not exists generate with histfactory
        if not fileExist:
            if os.name == "nt":
                print("HistFactory file cannto be generated on Windows - exit")
                return
            # Normally this would be run on the command line
            print(f"will run standard hist2workspace example")
            ROOT.gROOT.ProcessLine(".! prepareHistFactory .")
            ROOT.gROOT.ProcessLine(".! hist2workspace config/example.xml")
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

    calc = ROOT.RooStats.HypoTestInvTool()  # instance of HypoTestInvTool
    # calc = HypoTestInvTool_plus()           # instance of HypoTestInvTool

    # set parameters
    calc.SetParameter("PlotHypoTestResult", optHTInv.plotHypoTestResult)
    calc.SetParameter("WriteResult", optHTInv.writeResult)
    calc.SetParameter("Optimize", optHTInv.optimize)
    calc.SetParameter("UseVectorStore", optHTInv.useVectorStore)
    calc.SetParameter("GenerateBinned", optHTInv.generateBinned)
    calc.SetParameter("NToysRatio", optHTInv.nToysRatio)
    calc.SetParameter("MaxPOI", optHTInv.maxPOI)
    calc.SetParameter("EnableDetailedOutput", optHTInv.enableDetailedOutput)
    calc.SetParameter("Rebuild", optHTInv.rebuild)
    calc.SetParameter("ReuseAltToys", optHTInv.reuseAltToys)
    calc.SetParameter("NToyToRebuild", optHTInv.nToyToRebuild)
    calc.SetParameter("RebuildParamValues", optHTInv.rebuildParamValues)
    calc.SetParameter("MassValue", optHTInv.massValue)
    calc.SetParameter("MinimizerType", optHTInv.minimizerType)
    calc.SetParameter("PrintLevel", optHTInv.printLevel)
    calc.SetParameter("InitialFit", optHTInv.initialFit)
    calc.SetParameter("ResultFileName", optHTInv.resultFileName)
    calc.SetParameter("RandomSeed", optHTInv.randomSeed)
    calc.SetParameter("AsimovBins", optHTInv.nAsimovBins)

    # enable offset for all roostats
    if optHTInv.useNLLOffset:
        RooStats.UseNLLOffset(True)

    w = file.Get(wsName)
    r = 0
    print(w, "\t", filename)
    if w != ROOT.kNone:
        r = calc.RunInverter(
            w,
            modelSBName,
            modelBName,
            dataName,
            calculatorType,
            testStatType,
            useCLs,
            npoints,
            poimin,
            poimax,
            ntoys,
            useNumberCounting,
            nuisPriorName,
        )
        if not r:
            raise RuntimeError("Error running the HypoTestInverter - Exit ")
            return

    else:
        # case workspace is not present look for the inverter result
        print(f"Reading an HypoTestInverterResult with name {wsName} from file " << filename)
        r = file.Get(wsName)  # dynamic_cast
        if not r:
            raise RuntimeError("File {filename} does not contain a workspace or an HypoTestInverterResult - Exit ")

            file.ls()
            return

    calc.AnalyzeResult(r, calculatorType, testStatType, useCLs, npoints, infile)

    return


# --------------------------------------------------


def ReadResult(fileName, resultName="", useCLs=True):
    # read a previous stored result from a file given the result name
    StandardHypoTestInvDemo(fileName, resultName, "", "", "", 0, 0, useCLs)


# ifdef USE_AS_MAIN
def main():
    StandardHypoTestInvDemo()


main()
# endif
