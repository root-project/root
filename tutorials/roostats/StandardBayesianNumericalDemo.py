# \file
# \ingroup tutorial_roostats
# \notebook -js
# Standard demo of the numerical Bayesian calculator
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
# The actual heart of the demo is only about 10 lines long.
#
# The BayesianCalculator is based on Bayes's theorem
# and performs the integration using ROOT's numeric integration utilities
#
# \macro_image
# \macro_output
# \macro_code
#
# \author Kyle Cranmer (C++ version), and P. P. (Python translation)

import ROOT
import time


class Struct:
    def __init__(self, **kargs):
        self.__dict__.update(kargs)


BayesianNumericalOptions = Struct(  # interval CL
    confLevel=0.95,  # integration Type (default is adaptive (numerical integration)
    # possible values are "TOYMC" (toy MC integration, work when nuisances have a constraints pdf)
    #  "VEGAS" , "MISER", or "PLAIN"  (these are all possible MC integration)
    integrationType="",  # number of toys used for the MC integrations - for Vegas should be probably set to an higher value
    nToys=10000,  # flag to compute interval by scanning posterior (it is more robust but maybe less precise)
    scanPosterior=False,  # plot posterior function after having computed the interval
    plotPosterior=True,  # number of points for scanning the posterior (if scanPosterior = False it is used only for
    # plotting). Use by default a low value to speed-up tutorial
    nScanPoints=50,  # type of interval (0 is shortest, 1 central, 2 upper limit)
    intervalType=1,  # force a different value of POI for doing the scan (default is given value)
    maxPOI=-999,  # force integration of nuisance parameters to be within nSigma of their error (do first
    # a model fit to find nuisance error)
    nSigmaNuisance=-1,
)

optBayes = BayesianNumericalOptions


def StandardBayesianNumericalDemo(
    infile="", workspaceName="combined", modelConfigName="ModelConfig", dataName="obsData"
):

    # Setting timer
    t0 = time.time()

    # option definitions
    confLevel = optBayes.confLevel
    integrationType = ROOT.TString(optBayes.integrationType)
    nToys = optBayes.nToys
    scanPosterior = optBayes.scanPosterior
    plotPosterior = optBayes.plotPosterior
    nScanPoints = optBayes.nScanPoints
    intervalType = optBayes.intervalType
    maxPOI = optBayes.maxPOI
    nSigmaNuisance = optBayes.nSigmaNuisance

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
            # print(f"will run standard hist2workspace example")
            ROOT.gROOT.ProcessLine(".!  prepareHistFactory .")
            ROOT.gROOT.ProcessLine(".!  hist2workspace config/example.xml")
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
    # Tutorial starts here
    # -------------------------------------------------------

    # get the workspace out of the file
    w = file.Get(workspaceName)
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

    # ------------------------------------------
    # create and use the BayesianCalculator
    # to find and plot the 95% credible interval
    # on the parameter of interest as specified
    # in the model config

    # before we do that, we must specify our prior
    # it belongs in the model config, but it may not have
    # been specified
    prior = ROOT.RooUniform("prior", "", mc.GetParametersOfInterest())
    w.Import(prior)
    mc.SetPriorPdf(w.pdf("prior"))

    # do without systematics
    # mc->SetNuisanceParameters(RooArgSet() );
    if nSigmaNuisance > 0:
        pdf = mc.GetPdf()
        assert pdf
        res(
            pdf.fitTo(
                data,
                Save=True,
                Minimizer=MinimizerOptions.DefaultMinimizerType(),
                Hesse=True,
                PrintLevel=MinimizerOptions.DefaultPrintLevel() - 1,
            )
        )

        res.Print()
        nuisPar = RooArgList(mc.GetNuisanceParameters())
        for i in range(nuisPar.getSize()):
            v = nuisPar[i]
            assert v
            v.setMin(TMath.Max(v.getMin(), v.getVal() - nSigmaNuisance * v.getError()))
            v.setMax(TMath.Min(v.getMax(), v.getVal() + nSigmaNuisance * v.getError()))
            print(f"setting interval for nuisance  {v.GetName()} : [ {v.getMin()} , {v.getMax()}  ] \n")

    bayesianCalc = ROOT.RooStats.BayesianCalculator(data, mc)
    bayesianCalc.SetConfidenceLevel(confLevel)  # 95% interval

    # default of the calculator is central interval.  here use shortest , central or upper limit depending on input
    # doing a shortest interval might require a longer time since it requires a scan of the posterior function
    if intervalType == 0:
        bayesianCalc.SetShortestInterval()  # for shortest interval
    if intervalType == 1:
        bayesianCalc.SetLeftSideTailFraction(0.5)  # for central interval
    if intervalType == 2:
        bayesianCalc.SetLeftSideTailFraction(0.0)  # for upper limit

    if not integrationType.IsNull():
        bayesianCalc.SetIntegrationType(integrationType)  # set integrationType
        bayesianCalc.SetNumIters(nToys)  # set number of iterations (i.e. number of toys for MC integrations)

    # in case of toyMC make a nuisance pdf
    if integrationType.Contains("TOYMC"):
        nuisPdf = RooStats.MakeNuisancePdf(mc, "nuisance_pdf")
        print(f"using TOYMC integration: make nuisance pdf from the model ")
        nuisPdf.Print()
        bayesianCalc.ForceNuisancePdf(nuisPdf)
        scanPosterior = True  # for ToyMC the posterior is scanned anyway so used given points

    # compute interval by scanning the posterior function
    if scanPosterior:
        bayesianCalc.SetScanOfPosterior(nScanPoints)

    poi = mc.GetParametersOfInterest().first()
    if maxPOI != -999 and maxPOI > poi.getMin():
        poi.setMax(maxPOI)

    interval = bayesianCalc.GetInterval()

    # print out the interval on the first Parameter of Interest
    print(
        f"\n>>>> RESULT : {confLevel*100}% interval on {poi.GetName()} is : ["
        + f"{interval.LowerLimit():f}, {interval.UpperLimit():f} ] "
    )

    # end in case plotting is not requested
    if not plotPosterior:
        return

    # make a plot
    # since plotting may take a long time (it requires evaluating
    # the posterior in many points) this command will speed up
    # by reducing the number of points to plot - do 50

    # ignore errors of PDF if is zero
    ROOT.RooAbsReal.setEvalErrorLoggingMode("Ignore")

    # Stop timer
    t1 = time.time()
    print("Standard Bayesian Numerical Algorithm was performed in :")
    print("{:2f} seconds. ".format(t1 - t0))
    print(f"\nDrawing plot of posterior function.....")
    c1 = ROOT.TCanvas()
    # always plot using number of scan points
    bayesianCalc.SetScanOfPosterior(nScanPoints)

    plot = bayesianCalc.GetPosteriorPlot()
    plot.Draw()
    c1.Update()
    c1.Draw()
    c1.SaveAs("StandardBayesianNumericalDemo.png")
    global gbayesianCalc, gplot
    gbayesianCalc = bayesianCalc
    gplot = plot


StandardBayesianNumericalDemo(infile="", workspaceName="combined", modelConfigName="ModelConfig", dataName="obsData")
