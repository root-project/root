# \file
# \ingroup tutorial_roostats
# \notebook -js
# SPlot tutorial
#
# This tutorial shows an example of using SPlot to unfold two distributions.
# The physics context for the example is that we want to know
# the isolation distribution for real electrons from Z events
# and fake electrons from QCD.  Isolation is our 'control' variable.
# To unfold them, we need a model for an uncorrelated variable that
# discriminates between Z and QCD.  To do this, we use the invariant
# mass of two electrons.  We model the Z with a Gaussian and the QCD
# with a falling exponential.
#
# Note, since we don't have real data in this tutorial, we need to generate
# toy data.  To do that we need a model for the isolation variable for
# both Z and QCD.  This is only used to generate the toy data, and would
# not be needed if we had real data.
#
# \macro_image
# \macro_code
# \macro_output
#
# \authors Kyle Cranmer (C++ version), and P. P. (Python translation)


import ROOT


def AddModel(wspace):

    # Make models for signal (Higgs) and background (Z+jets and QCD)
    # In real life, this part requires an intelligent modeling
    # of signal and background -- this is only an example.

    # set range of observable
    lowRange, highRange = 0.0, 200.0

    # make a ROOT.RooRealVar for the observables
    invMass = ROOT.RooRealVar("invMass", "M_inv", lowRange, highRange, "GeV")
    isolation = ROOT.RooRealVar("isolation", "isolation", 0.0, 20.0, "GeV")

    # --------------------------------------
    # make 2-d model for Z including the invariant mass
    # distribution  and an isolation distribution which we want to
    # unfold from QCD.
    print(f"make z model")
    # mass model for Z
    mZ = ROOT.RooRealVar("mZ", "Z Mass", 91.2, lowRange, highRange)
    sigmaZ = ROOT.RooRealVar("sigmaZ", "Width of Gaussian", 2, 0, 10, "GeV")
    mZModel = ROOT.RooGaussian("mZModel", "Z+jets Model", invMass, mZ, sigmaZ)
    # we know Z mass
    mZ.setConstant()
    # we leave the width of the Z free during the fit in this example.

    # isolation model for Z.  Only used to generate toy MC.
    # the exponential is of the form exp(c*x).  If we want
    # the isolation to decay an e-fold every R GeV, we use
    # c = -1/R.
    zIsolDecayConst = ROOT.RooConstVar("zIsolDecayConst", "z isolation decay  constant", -1)
    zIsolationModel = ROOT.RooExponential("zIsolationModel", "z isolation model", isolation, zIsolDecayConst)

    # make the combined Z model
    zModel = ROOT.RooProdPdf("zModel", "2-d model for Z", ROOT.RooArgSet(mZModel, zIsolationModel))

    # --------------------------------------
    # make QCD model

    print(f"make qcd model")
    # mass model for QCD.
    # the exponential is of the form exp(c*x).  If we want
    # the mass to decay an e-fold every R GeV, we use
    # c = -1/R.
    # We can leave this parameter free during the fit.
    qcdMassDecayConst = ROOT.RooRealVar(
        "qcdMassDecayConst", "Decay const for QCD mass spectrum", -0.01, -100, 100, "1/GeV"
    )
    qcdMassModel = ROOT.RooExponential("qcdMassModel", "qcd Mass Model", invMass, qcdMassDecayConst)

    # isolation model for QCD.  Only used to generate toy MC
    # the exponential is of the form exp(c*x).  If we want
    # the isolation to decay an e-fold every R GeV, we use
    # c = -1/R.
    qcdIsolDecayConst = ROOT.RooConstVar("qcdIsolDecayConst", "Et resolution constant", -0.1)
    qcdIsolationModel = ROOT.RooExponential("qcdIsolationModel", "QCD isolation model", isolation, qcdIsolDecayConst)

    # make the 2-d model
    qcdModel = ROOT.RooProdPdf("qcdModel", "2-d model for QCD", [qcdMassModel, qcdIsolationModel])

    # combined model
    # These variables represent the number of Z or QCD events
    # They will be fitted.
    zYield = ROOT.RooRealVar("zYield", "fitted yield for Z", 500, 0.0, 5000)
    qcdYield = ROOT.RooRealVar("qcdYield", "fitted yield for QCD", 1000, 0.0, 10000)

    # now make the combined models
    print(f"make full model")
    model = ROOT.RooAddPdf("model", "z+qcd background models", [zModel, qcdModel], [zYield, qcdYield])
    massModel = ROOT.RooAddPdf("massModel", "z+qcd invariant mass model", [mZModel, qcdMassModel], [zYield, qcdYield])

    # interesting for debugging and visualizing the model
    model.graphVizTree("fullModel.dot")

    print(f"import model: ")
    model.Print()

    wspace.Import(model)
    wspace.Import(massModel, RecycleConflictNodes=True)


# Add a toy dataset
def AddData(wspace):

    # get what we need out of the workspace to make toy data
    model = wspace["model"]
    invMass = wspace["invMass"]
    isolation = wspace["isolation"]

    # make the toy data
    print("make data set and import to workspace")
    wspace.Import(model.generate([invMass, isolation]), Rename="data")


# Perform the plot
def DoSPlot(wspace):

    print(f"Calculate sWeights")

    # get what we need out of the workspace to do the fit
    massModel = wspace["massModel"]
    zYield = wspace["zYield"]
    qcdYield = wspace["qcdYield"]
    data = wspace["data"]

    # The sPlot technique requires that we fix the parameters
    # of the model that are not yields after doing the fit.
    #
    # This *could* be done with the lines below, however this is taken care of
    # by the ROOT.RooStats.SPlot constructor (or more precisely the AddSWeight
    # method).
    #
    # sigmaZ = ws["sigmaZ")
    # qcdMassDecayConst = ws["qcdMassDecayConst")
    # sigmaZ.setConstant()
    # qcdMassDecayConst.setConstant()

    ROOT.RooMsgService.instance().setSilentMode(True)

    print(f"\n\n------------------------------------------\nThe dataset before creating sWeights:\n")
    data.Print()

    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)

    # Now we use the SPlot class to add SWeights for the isolation variable to
    # our data set based on fitting the yields to the invariant mass variable.
    # Any keyword arguments will be forwarded to the underlying call to RooAbsPdf::fitTo().
    sData = ROOT.RooStats.SPlot("sData", "An SPlot", data, massModel, [zYield, qcdYield], Strategy=0)

    print(f"\n\nThe dataset after creating sWeights:\n")
    data.Print()

    # Check that our weights have the desired properties

    print("\n\n------------------------------------------\n\nCheck SWeights:")
    print("Yield of Z is\t", zYield.getVal(), ".  From sWeights it is ")
    print(sData.GetYieldFromSWeight("zYield"))

    print("Yield of QCD is\t", qcdYield.getVal(), ".  From sWeights it is : ")
    print(sData.GetYieldFromSWeight("qcdYield"))

    for i in range(10):
        print("Weight for event: ", i, sData.GetSWeight(i, "zYield"))
        print("qcd Weight: ", sData.GetSWeight(i, "qcdYield"))
        print("Total Weight: ", sData.GetSumOfEventSWeight(i))

    # import this new dataset with sWeights
    wspace.Import(data, Rename="dataWithSWeights")

    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.INFO)


def MakePlots(wspace):

    # Here we make plots of the discriminating variable (invMass) after the fit
    # and of the control variable (isolation) after unfolding with sPlot.
    # make our canvas
    cdata = ROOT.TCanvas("sPlot", "sPlot demo", 400, 600)
    cdata.Divide(1, 3)

    # get what we need out of the workspace
    model = wspace["model"]
    zModel = wspace["zModel"]
    qcdModel = wspace["qcdModel"]

    isolation = wspace["isolation"]
    invMass = wspace["invMass"]

    # note, we get the dataset with sWeights
    data = wspace["dataWithSWeights"]

    # create weighted data sets
    dataw_qcd = ROOT.RooDataSet(data.GetName(), data.GetTitle(), data.get(), Import=data, WeightVar="qcdYield_sw")
    dataw_z = ROOT.RooDataSet(data.GetName(), data.GetTitle(), data.get(), Import=data, WeightVar="zYield_sw")

    # plot invMass for data with full model and individual components overlaid
    # cdata = TCanvas()
    cdata.cd(1)
    frame = invMass.frame(Title="Fit of model to discriminating variable")
    data.plotOn(frame)
    model.plotOn(frame, Name="FullModel")
    model.plotOn(frame, Components=zModel, LineStyle="--", LineColor="r", Name="ZModel")
    model.plotOn(frame, Components=qcdModel, LineStyle="--", LineColor="g", Name="QCDModel")

    leg = ROOT.TLegend(0.11, 0.5, 0.5, 0.8)
    leg.AddEntry(frame.findObject("FullModel"), "Full model", "L")
    leg.AddEntry(frame.findObject("ZModel"), "Z model", "L")
    leg.AddEntry(frame.findObject("QCDModel"), "QCD model", "L")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)

    frame.Draw()
    leg.DrawClone()

    # Now use the sWeights to show isolation distribution for Z and QCD.
    # The SPlot class can make this easier, but here we demonstrate in more
    # detail how the sWeights are used.  The SPlot class should make this
    # very easy and needs some more development.

    # Plot isolation for Z component.
    # Do this by plotting all events weighted by the sWeight for the Z component.
    # The SPlot class adds a new variable that has the name of the corresponding
    # yield + "_sw".
    cdata.cd(2)

    frame2 = isolation.frame(Title="Isolation distribution with s weights to project out Z")
    # Since the data are weighted, we use SumW2 to compute the errors.
    dataw_z.plotOn(frame2, DataError="SumW2")
    zModel.plotOn(frame2, LineStyle="--", LineColor="r")

    frame2.Draw()

    # Plot isolation for QCD component.
    # Eg. plot all events weighted by the sWeight for the QCD component.
    # The SPlot class adds a new variable that has the name of the corresponding
    # yield + "_sw".
    cdata.cd(3)
    frame3 = isolation.frame(Title="Isolation distribution with s weights to project out QCD")
    dataw_qcd.plotOn(frame3, DataError="SumW2")
    qcdModel.plotOn(frame3, LineStyle="--", LineColor="g")

    frame3.Draw()

    cdata.SaveAs("rs301_splot.png")


def rs301_splot():

    # Create a workspace to manage the project.
    wspace = ROOT.RooWorkspace("myWS")

    # add the signal and background models to the workspace.
    # Inside this function you will find a description of our model.
    AddModel(wspace)

    # add some toy data to the workspace
    AddData(wspace)

    # do sPlot.
    # This will make a new dataset with sWeights added for every event.
    DoSPlot(wspace)

    # Make some plots showing the discriminating variable and
    # the control variable after unfolding.
    MakePlots(wspace)


rs301_splot()
