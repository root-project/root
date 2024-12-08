# \file
# \ingroup tutorial_roostats
# \notebook -js
# A typical search for a new particle by studying an invariant mass distribution
#
# The macro creates a simple signal model and two background models,
# which are added to a RooWorkspace.
# The macro creates a toy dataset, and then uses a RooStats
# ProfileLikleihoodCalculator to do a hypothesis test of the
# background-only and signal+background hypotheses.
# In this example, shape uncertainties are not taken into account, but
# normalization uncertainties are.
#
# \macro_image
# \macro_output
# \macro_code
#
# \author Kyle Cranmer (C++ version), and P. P. (Python translation)

# use this order for safety on library loading
import ROOT


# see below for implementation
# AddModel(RooWorkspace )
# AddData(RooWorkspace )
# DoHypothesisTest(RooWorkspace )
# MakePlots(RooWorkspace )


# ____________________________________
def AddModel(wks):
    # wks has to be a Workspace

    # Make models for signal (Higgs) and background (Z+jets and QCD)
    # In real life, this part requires an intelligent modeling
    # of signal and background -- this is only an example.

    # set range of observable
    lowRange = 60
    highRange = 200

    # make a RooRealVar for the observable
    invMass = ROOT.RooRealVar("invMass", "M_inv", lowRange, highRange, "GeV")

    # --------------------------------------
    # make a simple signal model.
    mH = ROOT.RooRealVar("mH", "Higgs Mass", 130, 90, 160)
    sigma1 = ROOT.RooRealVar("sigma1", "Width of Gaussian", 12.0, 2, 100)
    sigModel = ROOT.RooGaussian("sigModel", "Signal Model", invMass, mH, sigma1)
    # we will test this specific mass point for the signal
    mH.setConstant()
    # and we assume we know the mass resolution
    sigma1.setConstant()

    # --------------------------------------
    # make zjj model.  Just like signal model
    mZ = ROOT.RooRealVar("mZ", "Z Mass", 91.2, 0, 100)
    sigma1_z = ROOT.RooRealVar("sigma1_z", "Width of Gaussian", 10.0, 6, 100)
    zjjModel = ROOT.RooGaussian("zjjModel", "Z+jets Model", invMass, mZ, sigma1_z)
    # we know Z mass
    mZ.setConstant()
    # assume we know resolution too
    sigma1_z.setConstant()

    # --------------------------------------
    # make QCD model
    a0 = ROOT.RooRealVar("a0", "a0", 0.26, -1, 1)
    a1 = ROOT.RooRealVar("a1", "a1", -0.17596, -1, 1)
    a2 = ROOT.RooRealVar("a2", "a2", 0.018437, -1, 1)
    a3 = ROOT.RooRealVar("a3", "a3", 0.02, -1, 1)
    qcdModel = ROOT.RooChebychev("qcdModel", "A  Polynomial for QCD", invMass, [a0, a1, a2])

    # let's assume this shape is known, but the normalization is not
    a0.setConstant()
    a1.setConstant()
    a2.setConstant()

    # --------------------------------------
    # combined model

    # Setting the fraction of Zjj to be 40% for initial guess.
    fzjj = ROOT.RooRealVar("fzjj", "fraction of zjj background events", 0.4, 0.0, 1)

    # Set the expected fraction of signal to 20%.
    fsigExpected = ROOT.RooRealVar("fsigExpected", "expected fraction of signal events", 0.2, 0.0, 1)
    fsigExpected.setConstant()  # use mu as main parameter, so fix this.

    # Introduce mu: the signal strength in units of the expectation.
    # eg. mu = 1 is the SM, mu = 0 is no signal, mu=2 is 2x the SM
    mu = ROOT.RooRealVar("mu", "signal strength in units of SM expectation", 1, 0.0, 2)

    # Introduce ratio of signal efficiency to nominal signal efficiency.
    # This is useful if you want to do limits on cross section.
    ratioSigEff = ROOT.RooRealVar("ratioSigEff", "ratio of signal efficiency to nominal signal efficiency", 1.0, 0.0, 2)
    ratioSigEff.setConstant(True)

    # finally the signal fraction is the product of the terms above.
    fsig = ROOT.RooProduct("fsig", "fraction of signal events", [mu, ratioSigEff, fsigExpected])

    # full model
    model = ROOT.RooAddPdf("model", "sig+zjj+qcd background shapes", [sigModel, zjjModel, qcdModel], [fsig, fzjj])

    # interesting for debugging and visualizing the model
    #  model.printCompactTree("","fullModel.txt");
    #  model.graphVizTree("fullModel.dot");

    wks.Import(model)


# ____________________________________
def AddData(wks):
    # wks has to be a Workspace
    # Add a toy dataset

    nEvents = 150
    model = wks.pdf("model")
    invMass = wks.var("invMass")

    data = model.generate(invMass, nEvents)

    wks.Import(data, Rename="data")


# ____________________________________
def DoHypothesisTest(wks):

    # Use a RooStats ProfileLikleihoodCalculator to do the hypothesis test.
    model = ROOT.RooFit.ModelConfig()
    model.SetWorkspace(wks)
    model.SetPdf("model")

    # plc.SetData("data");

    plc = ROOT.RooStats.ProfileLikelihoodCalculator()
    plc.SetData((wks.data("data")))

    # here we explicitly set the value of the parameters for the null.
    # We want no signal contribution, eg. mu = 0
    mu = wks.var("mu")
    #   nullParams = RooArgSet("nullParams")
    #   nullParams.addClone(mu)
    poi = ROOT.RooArgSet(mu)
    nullParams = poi.snapshot()
    nullParams.setRealValue("mu", 0)

    # plc.SetNullParameters(*nullParams);
    plc.SetModel(model)
    # NOTE: using snapshot will import nullparams
    # in the WS and merge with existing "mu"
    # model.SetSnapshot(*nullParams);

    # use instead setNuisanceParameters
    plc.SetNullParameters(nullParams)

    # We get a HypoTestResult out of the calculator, and we can query it.
    htr = plc.GetHypoTest()
    print(f"-------------------------------------------------")
    print(f"The p-value for the null is ", htr.NullPValue())
    print(f"Corresponding to a significance of ", htr.Significance())
    print(f"-------------------------------------------------\n\n")


# ____________________________________
def MakePlots(wks):
    # wks has to be a RooWorkspace

    # Make plots of the data and the best fit model in two cases:
    # first the signal+background case
    # second the background-only case.

    # get some things out of workspace
    model = wks.pdf("model")
    sigModel = wks.pdf("sigModel")
    zjjModel = wks.pdf("zjjModel")
    qcdModel = wks.pdf("qcdModel")

    mu = wks.var("mu")
    invMass = wks.var("invMass")
    data = wks.data("data")

    # --------------------------------------
    # Make plots for the Alternate hypothesis, eg. let mu float

    mu.setConstant(False)

    model.fitTo(data, Save=True, Minos=False, Hesse=False, PrintLevel=-1)

    # plot sig candidates, full model, and individual components
    c1 = ROOT.TCanvas()
    frame = invMass.frame()
    data.plotOn(frame)
    model.plotOn(frame)
    model.plotOn(frame, Components=sigModel, LineStyle="--", LineColor="r")
    model.plotOn(frame, Components=zjjModel, LineStyle="--", LineColor="b")
    model.plotOn(frame, Components=qcdModel, LineStyle="--", LineColor="g")

    frame.SetTitle("An example fit to the signal + background model")
    frame.Draw()
    c1.Update()
    c1.Draw()
    c1.SaveAs("rs102_hypotestwithshapes.1.png")
    #  cdata.SaveAs("alternateFit.gif");

    # --------------------------------------
    # Do Fit to the Null hypothesis.  Eg. fix mu=0

    mu.setVal(0)  # set signal fraction to 0
    mu.setConstant(True)  # set constant

    model.fitTo(data, Save=True, Minos=False, Hesse=False, PrintLevel=-1)

    # plot signal candidates with background model and components
    c2 = ROOT.TCanvas()
    xframe2 = invMass.frame()
    data.plotOn(xframe2, DataError="SumW2")
    model.plotOn(xframe2)
    model.plotOn(xframe2, Components=zjjModel, LineStyle="--", LineColor="b")
    model.plotOn(xframe2, Components=qcdModel, LineStyle="--", LineColor="g")

    xframe2.SetTitle("An example fit to the background-only model")
    xframe2.Draw()
    c2.Update()
    c2.Draw()
    c2.SaveAs("rs102_hypotestwithshapes.1.png")
    #  cbkgonly.SaveAs("nullFit.gif");


# ____________________________________
def rs102_hypotestwithshapes():

    # The main macro.

    # Create a workspace to manage the project.
    wspace = ROOT.RooWorkspace("myWS")

    # add the signal and background models to the workspace
    AddModel(wspace)

    # add some toy data to the workspace
    AddData(wspace)

    # inspect the workspace if you wish
    #  wspace->Print();

    # do the hypothesis test
    DoHypothesisTest(wspace)

    # make some plots
    MakePlots(wspace)

    # cleanup
    del wspace


rs102_hypotestwithshapes()
