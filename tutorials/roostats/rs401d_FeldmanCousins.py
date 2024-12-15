# \file
# \ingroup tutorial_roostats
# \notebook
# Neutrino Oscillation Example from Feldman & Cousins
#
# This tutorial shows a more complex example using the FeldmanCousins utility
# to create a confidence interval for a toy neutrino oscillation experiment.
# The example attempts to faithfully reproduce the toy example described in Feldman & Cousins'
# original paper, Phys.Rev.D57:3873-3889,1998.
#
# \macro_image
# \macro_output
# \macro_code
#
# \author Kyle Cranmer (C++ version), and P. P. (Python translation)

import ROOT


def rs401d_FeldmanCousins(doFeldmanCousins=False, doMCMC=True):

    # to time the macro
    t = ROOT.TStopwatch()
    t.Start()

    # Taken from Feldman & Cousins paper, Phys.Rev.D57:3873-3889,1998.
    # e-Print: physics/9711021 (see page 13.)
    #
    # Quantum mechanics dictates that the probability of such a transformation is given by the formula
    # $P (\nu\mu \rightarrow \nu e ) = sin^2 (2\theta) sin^2 (1.27 \Delta m^2 L /E )$
    # where P is the probability for a $\nu\mu$ to transform into a $\nu e$ , L is the distance in km between
    # the creation of the neutrino from meson decay and its interaction in the detector, E is the
    # neutrino energy in GeV, and $\Delta m^2 = |m^2 - m^2 |$ in $(eV/c^2 )^2$ .
    #
    # To demonstrate how this works in practice, and how it compares to alternative approaches
    # that have been used, we consider a toy model of a typical neutrino oscillation experiment.
    # The toy model is defined by the following parameters: Mesons are assumed to decay to
    # neutrinos uniformly in a region 600 m to 1000 m from the detector. The expected background
    # from conventional $\nu e$ interactions and misidentified $\nu\mu$ interactions is assumed to be 100
    # events in each of 5 energy bins which span the region from 10 to 60 GeV. We assume that
    # the $\nu\mu$ flux is such that if $P (\nu\mu \rightarrow \nu e ) = 0.01$ averaged over any bin, then that bin
    # would
    # have an expected additional contribution of 100 events due to $\nu\mu \rightarrow \nu e$ oscillations.

    # Make signal model model
    E = ROOT.RooRealVar("E", "", 15, 10, 60, "GeV")
    L = ROOT.RooRealVar("L", "", 0.800, 0.600, 1.0, "km")  # need these units in formula
    deltaMSq = ROOT.RooRealVar("deltaMSq", "#Delta m^{2}", 40, 1, 300, "eV/c^{2}")
    sinSq2theta = ROOT.RooRealVar("sinSq2theta", "sin^{2}(2#theta)", 0.006, 0.0, 0.02)
    # RooRealVar deltaMSq("deltaMSq","#Delta m^{2}",40,20,70,"eV/c^{2}");
    #  RooRealVar sinSq2theta("sinSq2theta","sin^{2}(2#theta)", .006,.001,.01);
    # PDF for oscillation only describes deltaMSq dependence, sinSq2theta goes into sigNorm
    oscillationFormula = "std::pow(std::sin(1.27 * x[2] * x[0] / x[1]), 2)"
    PnmuTone = ROOT.RooGenericPdf("PnmuTone", "P(#nu_{#mu} #rightarrow #nu_{e}", oscillationFormula, [L, E, deltaMSq])

    # only E is observable, so create the signal model by integrating out L
    sigModel = PnmuTone.createProjection(L)

    # create  $ \int dE' dL' P(E',L' | \Delta m^2)$.
    # Given RooFit will renormalize the PDF in the range of the observables,
    # the average probability to oscillate in the experiment's acceptance
    # needs to be incorporated into the extended term in the likelihood.
    # Do this by creating a RooAbsReal representing the integral and divide by
    # the area in the E-L plane.
    # The integral should be over "primed" observables, so we need
    # an independent copy of PnmuTone not to interfere with the original.

    # Independent copy for Integral
    EPrime = ROOT.RooRealVar("EPrime", "", 15, 10, 60, "GeV")
    LPrime = ROOT.RooRealVar("LPrime", "", 0.800, 0.600, 1.0, "km")  # need these units in formula
    PnmuTonePrime = ROOT.RooGenericPdf(
        "PnmuTonePrime", "P(#nu_{#mu} #rightarrow #nu_{e}", oscillationFormula, [LPrime, EPrime, deltaMSq]
    )
    intProbToOscInExp = PnmuTonePrime.createIntegral([EPrime, LPrime])

    # Getting the flux is a bit tricky.  It is more clear to include a cross section term that is not
    # explicitly referred to in the text, eg.
    # number events in bin = flux * cross-section for nu_e interaction in E bin * average prob nu_mu osc. to nu_e in bin
    # let maxEventsInBin = flux * cross-section for nu_e interaction in E bin
    # maxEventsInBin * 1% chance per bin =  100 events / bin
    # therefore maxEventsInBin = 10,000.
    # for 5 bins, this means maxEventsTot = 50,000
    maxEventsTot = ROOT.RooConstVar("maxEventsTot", "maximum number of sinal events", 50000)
    inverseArea = ROOT.RooConstVar(
        "inverseArea",
        "1/(#Delta E #Delta L)",
        1.0 / (EPrime.getMax() - EPrime.getMin()) / (LPrime.getMax() - LPrime.getMin()),
    )

    # $sigNorm = maxEventsTot \cdot \int dE dL \frac{P_{oscillate\ in\ experiment}}{Area} \cdot {sin}^2(2\theta)$
    sigNorm = ROOT.RooProduct("sigNorm", "", [maxEventsTot, intProbToOscInExp, inverseArea, sinSq2theta])
    # bkg = 5 bins * 100 events / bin
    bkgNorm = ROOT.RooConstVar("bkgNorm", "normalization for background", 500)

    # flat background (0th order polynomial, so no arguments for coefficients)
    bkgEShape = ROOT.RooPolynomial("bkgEShape", "flat bkg shape", E)

    # total model
    model = ROOT.RooAddPdf("model", "", [sigModel, bkgEShape], [sigNorm, bkgNorm])

    # for debugging, check model tree
    #  model.printCompactTree();
    #  model.graphVizTree("model.dot");

    # turn off some messages
    ROOT.RooMsgService.instance().setStreamStatus(0, False)
    ROOT.RooMsgService.instance().setStreamStatus(1, False)
    ROOT.RooMsgService.instance().setStreamStatus(2, False)

    # --------------------------------------
    # n events in data to data, simply sum of sig+bkg
    nEventsData = bkgNorm.getVal() + sigNorm.getVal()
    print("generate toy data with nEvents = ", nEventsData)
    # adjust random seed to get a toy dataset similar to one in paper.
    # Found by trial and error (3 trials, so not very "fine tuned")
    ROOT.RooRandom.randomGenerator().SetSeed(3)
    # create a toy dataset
    data = model.generate(E, nEventsData)

    # --------------------------------------
    # make some plots
    dataCanvas = ROOT.TCanvas("dataCanvas")
    dataCanvas.Divide(2, 2)

    # plot the PDF
    dataCanvas.cd(1)
    hh = PnmuTone.createHistogram("hh", E, ROOT.RooFit.YVar(L, ROOT.RooFit.Binning(40)), Binning=40, Scaling=False)
    hh.SetLineColor(ROOT.kBlue)
    hh.SetTitle("True Signal Model")
    hh.Draw("surf")
    dataCanvas.Update()
    dataCanvas.Draw()
    # dataCanvas.SaveAs("rs.1.png")
    # plot the data with the best fit
    dataCanvas.cd(2)
    Eframe = E.frame()
    data.plotOn(Eframe)
    model.fitTo(data, Extended=True)
    model.plotOn(Eframe)
    model.plotOn(Eframe, Components=sigModel, LineColor="r")
    model.plotOn(Eframe, Components=bkgEShape, LineColor="g")
    model.plotOn(Eframe)
    Eframe.SetTitle("toy data with best fit model (and sig+bkg components)")
    Eframe.Draw()
    dataCanvas.Update()
    dataCanvas.Draw()
    # dataCanvas.SaveAs("rs.2.png")

    # plot the likelihood function
    dataCanvas.cd(3)
    nll = model.createNLL(data, Extended=True)
    pll = ROOT.RooProfileLL("pll", "", nll, [deltaMSq, sinSq2theta])
    # hhh = nll.createHistogram("hhh",sinSq2theta, Binning(40), YVar(deltaMSq,Binning(40)))
    hhh = pll.createHistogram(
        "hhh", sinSq2theta, ROOT.RooFit.YVar(deltaMSq, ROOT.RooFit.Binning(40)), Binning=40, Scaling=False
    )
    hhh.SetLineColor(ROOT.kBlue)
    hhh.SetTitle("Likelihood Function")
    hhh.Draw("surf")

    dataCanvas.Update()
    dataCanvas.Draw()
    dataCanvas.SaveAs("3.png")

    # --------------------------------------------------------------
    # show use of Feldman-Cousins utility in RooStats
    # set the distribution creator, which encodes the test statistic
    parameters = ROOT.RooArgSet(deltaMSq, sinSq2theta)
    w = ROOT.RooWorkspace()

    modelConfig = ROOT.RooFit.ModelConfig()
    modelConfig.SetWorkspace(w)
    modelConfig.SetPdf(model)
    modelConfig.SetParametersOfInterest(parameters)

    fc = RooStats.FeldmanCousins(data, modelConfig)
    fc.SetTestSize(0.1)  # set size of test
    fc.UseAdaptiveSampling(True)
    fc.SetNBins(10)  # number of points to test per parameter

    # use the Feldman-Cousins tool
    interval = 0
    if doFeldmanCousins:
        interval = fc.GetInterval()

    # ---------------------------------------------------------
    # show use of ProfileLikeihoodCalculator utility in RooStats
    plc = RooStats.ProfileLikelihoodCalculator(data, modelConfig)
    plc.SetTestSize(0.1)

    plcInterval = plc.GetInterval()

    # --------------------------------------------
    # show use of MCMCCalculator utility in RooStats
    mcInt = ROOT.kNone

    if doMCMC:
        # turn some messages back on
        RooMsgService.instance().setStreamStatus(0, True)
        RooMsgService.instance().setStreamStatus(1, True)

        mcmcWatch = ROOT.TStopwatch()
        mcmcWatch.Start()

        axisList = ROOT.RooArgList(deltaMSq, sinSq2theta)
        mc = ROOT.RooStats.MCMCCalculator(data, modelConfig)
        mc.SetNumIters(5000)
        mc.SetNumBurnInSteps(100)
        mc.SetUseKeys(True)
        mc.SetTestSize(0.1)
        mc.SetAxes(axisList)  # set which is x and y axis in posterior histogram
        # mc.SetNumBins(50);
        mcInt = mc.GetInterval()

        mcmcWatch.Stop()
        mcmcWatch.Print()

    # -------------------------------
    # make plot of resulting interval

    dataCanvas.cd(4)

    # first plot a small dot for every point tested
    if doFeldmanCousins:
        parameterScan = fc.GetPointsToScan()
        hist = parameterScan.createHistogram(deltaMSq, sinSq2theta, 30, 30)
        #  hist.Draw()
        forContour = hist.Clone()

        # now loop through the points and put a marker if it's in the interval
        tmpPoint = RooArgSet()
        # loop over points to test
        for i in range(parameterScan.numEntries):
            # get a parameter point from the list of points to test.
            tmpPoint = parameterScan.get(i).clone("temp")

            if interval:
                if interval.IsInInterval(tmpPoint):
                    forContour.SetBinContent(
                        hist.FindBin(tmpPoint.getRealValue("sinSq2theta"), tmpPoint.getRealValue("deltaMSq")), 1
                    )
                else:
                    forContour.SetBinContent(
                        hist.FindBin(tmpPoint.getRealValue("sinSq2theta"), tmpPoint.getRealValue("deltaMSq")), 0
                    )

            del tmpPoint

        if interval:
            level = 0.5
            forContour.SetContour(1, level)
            forContour.SetLineWidth(2)
            forContour.SetLineColor(ROOT.kRed)
            forContour.Draw("cont2,same")

    mcPlot = ROOT.kNone
    if mcInt:
        print(f"MCMC actual confidence level: ", mcInt.GetActualConfidenceLevel())
        mcPlot = MCMCIntervalPlot(mcInt)
        mcPlot.SetLineColor(kMagenta)
        mcPlot.Draw()

    dataCanvas.Update()

    plotInt = LikelihoodIntervalPlot(plcInterval)

    plotInt.SetTitle("90% Confidence Intervals")
    if mcInt:
        plotInt.Draw("same")
    else:
        plotInt.Draw()

    dataCanvas.Update()

    dataCanvas.SaveAs("rs401d_FeldmanCousins.1.pdf")

    # print timing info
    t.Stop()
    t.Print()


rs401d_FeldmanCousins(doFeldmanCousins=False, doMCMC=True)
