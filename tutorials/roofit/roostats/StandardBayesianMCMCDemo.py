# \file
# \ingroup tutorial_roostats
# \notebook -js
# Standard demo of the Bayesian MCMC calculator
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
# The MCMCCalculator is a Bayesian tool that uses
# the Metropolis-Hastings algorithm to efficiently integrate
# in many dimensions.  It is not as accurate as the BayesianCalculator
# for simple problems, but it scales to much more complicated cases.
#
# \macro_image
# \macro_output
# \macro_code
#
# \author Kyle Cranmer (C++ version), and P. P. (Python translation)

import ROOT


# general Structure definition
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


BayesianMCMCOptions = Struct(
    confLevel=0.95,  # type of interval (0 is shortest, 1 central, 2 upper limit)
    intervalType=2,  # force different values of POI for doing the scan (default is given value)
    maxPOI=-999,
    minPOI=-999,  # number of iterations
    numIters=100000,  # number of burn in steps to be ignored
    numBurnInSteps=100,
)

optMCMC = BayesianMCMCOptions


def StandardBayesianMCMCDemo(infile="", workspaceName="combined", modelConfigName="ModelConfig", dataName="obsData"):

    # -------------------------------------------------------
    # First part is just to access a user-defined file
    # or create the standard example file if it doesn't exist

    filename = ""
    if infile == "":
        filename = "results/example_combined_GaussExample_model.root"
        fileExist = not ROOT.gSystem.AccessPathName(filename) # note opposite return code
        # if file does not exists generate with histfactory
        if not fileExist:
            # Normally this would be run on the command line
            print(f"will run standard hist2workspace example")
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
    # w = (RooWorkspace )file.Get(workspaceName)
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

    # Want an efficient proposal function
    # default is uniform.
    """
   #
   # this one is based on the covariance matrix of fit
   fit = mc.GetPdf().fitTo(data,Save())
   ProposalHelper ph
   ph.SetVariables((RooArgSet&)fit.floatParsFinal())
   ph.SetCovMatrix(fit.covarianceMatrix())
   ph.SetUpdateProposalParameters(True) # auto-create mean vars and add mappings
   ph.SetCacheSize(100)
   pf = ph.GetProposalFunction()
   """

    # this proposal function seems fairly robust
    sp = ROOT.RooStats.SequentialProposal(0.1)
    # -------------------------------------------------------
    # create and use the MCMCCalculator
    # to find and plot the 95% credible interval
    # on the parameter of interest as specified
    # in the model config
    mcmc = ROOT.RooStats.MCMCCalculator(data, mc)
    mcmc.SetConfidenceLevel(optMCMC.confLevel)  # 95% interval
    #  mcmc.SetProposalFunction(*pf);
    mcmc.SetProposalFunction(sp)
    mcmc.SetNumIters(optMCMC.numIters)  # Metropolis-Hastings algorithm iterations
    mcmc.SetNumBurnInSteps(optMCMC.numBurnInSteps)  # first N steps to be ignored as burn-in

    # default is the shortest interval.
    (optMCMC.intervalType == 0)
    mcmc.SetIntervalType(ROOT.RooStats.MCMCInterval.kShortest)  # for shortest interval (not really needed)
    (optMCMC.intervalType == 1)
    mcmc.SetLeftSideTailFraction(0.5)  # for central interval
    (optMCMC.intervalType == 2)
    mcmc.SetLeftSideTailFraction(0.0)  # for upper limit

    firstPOI = mc.GetParametersOfInterest().first()
    if optMCMC.minPOI != -999:
        firstPOI.setMin(optMCMC.minPOI)
    if optMCMC.maxPOI != -999:
        firstPOI.setMax(optMCMC.maxPOI)

    interval = mcmc.GetInterval()

    # make a plot
    c1 = ROOT.TCanvas("IntervalPlot")
    plot = ROOT.RooStats.MCMCIntervalPlot(interval)
    plot.Draw()

    c2 = ROOT.TCanvas("extraPlots")
    list = mc.GetNuisanceParameters()
    if list.getSize() > 1:
        n = list.getSize()
        ny = ROOT.TMath.CeilNint(ROOT.sqrt(n))
        nx = ROOT.TMath.CeilNint(ROOT.double(n) / ny)
        c2.Divide(nx, ny)

    # draw a scatter plot of chain results for poi vs each nuisance parameters
    nuis = ROOT.kNone
    iPad = 1  # iPad, that's funny

    for nuis in mc.GetNuisanceParameters():
        c2.cd(iPad)
        iPad += 1
        plot.DrawChainScatter(firstPOI, nuis)

    # print out the interval on the first Parameter of Interest
    print("\n>>>> RESULT : ", optMCMC.confLevel * 100, "interval on ", firstPOI.GetName(), "is : [")
    print(interval.LowerLimit(firstPOI), interval.UpperLimit(firstPOI))
    print("] ")
    gPad = c1

    c1.SaveAs("StandardBayesianMCMCDemo.1.IntervalPlot.png")
    c2.SaveAs("StandardBayesianMCMCDemo.2.extraPlots.png")


StandardBayesianMCMCDemo("", "combined", "ModelConfig", "obsData")
