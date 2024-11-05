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
# \author Kyle Cranmer


import ROOT
from ROOT import RooStats, RooFit

# use this order for safety on library loading
SPlot = RooStats.SPlot  
std =   ROOT.std
nullptr = ROOT.nullptr
kGreen = ROOT.kGreen
kRed = ROOT.kRed
kDashed = ROOT.kDashed
RooFormulaVar = ROOT.RooFormulaVar
 
Rename = RooFit.Rename
Title  = RooFit.Title
Name   = RooFit.Name
Components = RooFit.Components
LineStyle = RooFit.LineStyle
LineColor = RooFit.LineColor
DataError = RooFit.DataError
RecycleConflictNodes = RooFit.RecycleConflictNodes
RooMsgService = ROOT.RooMsgService
RooArgList = ROOT.RooArgList
RooAbsData = ROOT.RooAbsData
RooRealVar = 		 ROOT.RooRealVar
RooDataSet = 		 ROOT.RooDataSet
RooRealVar = 		 ROOT.RooRealVar
RooGaussian = 		 ROOT.RooGaussian
RooExponential = 		 ROOT.RooExponential
RooChebychev = 		 ROOT.RooChebychev
RooAddPdf = 		 ROOT.RooAddPdf
RooProdPdf = 		 ROOT.RooProdPdf
RooAddition = 		 ROOT.RooAddition
RooProduct = 		 ROOT.RooProduct
RooAbsPdf = 		 ROOT.RooAbsPdf
RooFitResult = 		 ROOT.RooFitResult
RooWorkspace = 		 ROOT.RooWorkspace
RooConstVar = 		 ROOT.RooConstVar

TCanvas = 		 ROOT.TCanvas
TLegend = 		 ROOT.TLegend

RooArgSet = ROOT.RooArgSet
#include <iomanip>


# see below for implementation
# for the next functions:
# AddModel(RooWorkspace)
# AddData(RooWorkspace)
# DoSPlot(RooWorkspace)
# MakePlots(RooWorkspace)


#____________________________________
def AddModel(object): 
   
   # Make models for signal (Higgs) and background (Z+jets and QCD)
   # In real life, this part requires an intelligent modeling
   # of signal and background -- this is only an example.
   
   # set range of observable
   lowRange, highRange = 0., 200.
   
   # make a RooRealVar for the observables
   invMass = RooRealVar("invMass", "M_inv", lowRange, highRange, "GeV")
   isolation = RooRealVar("isolation", "isolation", 0., 20., "GeV")
   
   # --------------------------------------
   # make 2-d model for Z including the invariant mass
   # distribution  and an isolation distribution which we want to
   # unfold from QCD.
   print(f"make z model")
   # mass model for Z
   mZ = RooRealVar("mZ", "Z Mass", 91.2, lowRange, highRange)
   sigmaZ = RooRealVar("sigmaZ", "Width of Gaussian", 2, 0, 10, "GeV")
   mZModel = RooGaussian("mZModel", "Z+jets Model", invMass, mZ, sigmaZ)
   # we know Z mass
   mZ.setConstant()
   # we leave the width of the Z free during the fit in this example.
   
   # isolation model for Z.  Only used to generate toy MC.
   # the exponential is of the form exp(c*x).  If we want
   # the isolation to decay an e-fold every R GeV, we use
   # c = -1/R.
   zIsolDecayConst = RooConstVar("zIsolDecayConst", "z isolation decay  constant", -1)
   zIsolationModel = RooExponential("zIsolationModel", "z isolation model", isolation, zIsolDecayConst)
   
   # make the combined Z model
   zModel = RooProdPdf("zModel", "2-d model for Z", RooArgSet(mZModel, zIsolationModel))
   
   # --------------------------------------
   # make QCD model
   
   print(f"make qcd model")
   # mass model for QCD.
   # the exponential is of the form exp(c*x).  If we want
   # the mass to decay an e-fold every R GeV, we use
   # c = -1/R.
   # We can leave this parameter free during the fit.
   qcdMassDecayConst = RooRealVar("qcdMassDecayConst", "Decay const for QCD mass spectrum", -0.01, -100, 100, "1/GeV")
   qcdMassModel = RooExponential("qcdMassModel", "qcd Mass Model", invMass, qcdMassDecayConst)
   
   # isolation model for QCD.  Only used to generate toy MC
   # the exponential is of the form exp(c*x).  If we want
   # the isolation to decay an e-fold every R GeV, we use
   # c = -1/R.
   qcdIsolDecayConst = RooConstVar("qcdIsolDecayConst", "Et resolution constant", -.1)
   qcdIsolationModel = RooExponential("qcdIsolationModel", "QCD isolation model", isolation, qcdIsolDecayConst)
   
   # make the 2-d model
   qcdModel = RooProdPdf("qcdModel", "2-d model for QCD", RooArgSet(qcdMassModel, qcdIsolationModel))
   
   # --------------------------------------
   gws = object
   # combined model
   
   # These variables represent the number of Z or QCD events
   # They will be fitted.
   zYield = RooRealVar("zYield", "fitted yield for Z", 500, 0., 5000)
   qcdYield = RooRealVar("qcdYield", "fitted yield for QCD", 1000, 0., 10000)
   
   # now make the combined models
   print(f"make full model")
   model = RooAddPdf("model", "z+qcd background models", (zModel, qcdModel), (zYield, qcdYield) ) 
   massModel = RooAddPdf("massModel", "z+qcd invariant mass model", (mZModel, qcdMassModel), (zYield, qcdYield) )
   
   # interesting for debugging and visualizing the model
   model.graphVizTree("fullModel.dot")
   
   print(f"import model: ")
   model.Print()

   object.Import(model)

   try: 
      #in C++ :
      #ws->import(massModel, ROOT::RooFit::RecycleConflictNodes())
      #in python3.12 :
      object.Import(massModel, RecycleConflictNodes())
      print("massModel was imported to workspace resolving ConflictNodes...")
   except AttributeError:
      print("Model name conflict! Choose an appropiate action!.") 

#____________________________________
def AddData(object):

   # Add a toy dataset
   
   # get what we need out of the workspace to make toy data
   model = object.pdf("model")
   invMass = object.var("invMass")
   isolation = object.var("isolation")
   
   # make the toy data
   print(f"make data set and import to workspace")
   data = RooDataSet( model.generate( (invMass, isolation) ) )
   
   # import data into workspace
   object.Import(data, Rename("data"))
   

#____________________________________
def DoSPlot(object):

   print(f"Calculate sWeights")
   
   # get what we need out of the workspace to do the fit
   model = object.pdf("model")
   massModel = object.pdf("massModel")
   zYield = object.var("zYield")
   qcdYield = object.var("qcdYield")
   data = object.data("data")
   
   # The sPlot technique requires that we fix the parameters
   # of the model that are not yields after doing the fit.
   #
   # This *could* be done with the lines below, however this is taken care of
   # by the RooStats::SPlot constructor (or more precisely the AddSWeight
   # method).
   #
   # RooRealVar* sigmaZ = ws.var("sigmaZ");
   # RooRealVar* qcdMassDecayConst = ws.var("qcdMassDecayConst");
   # sigmaZ->setConstant();
   # qcdMassDecayConst->setConstant();
   
   RooMsgService.instance().setSilentMode(True)
   
   print(f"\n\n------------------------------------------\nThe dataset before creating sWeights:\n")
   data.Print()
   
   RooMsgService.instance().setGlobalKillBelow(RooFit.ERROR)
   
   # Now we use the SPlot class to add SWeights for the isolation variable to
   # our data set based on fitting the yields to the invariant mass variable
   sData = SPlot("sData", "An SPlot", data, massModel, RooArgList(zYield, qcdYield))
   
   print(f"\n\nThe dataset after creating sWeights:\n")
   data.Print()
   
   # Check that our weights have the desired properties

   print("\n\n------------------------------------------\n\nCheck SWeights:")

   print("\n")
   print( "Yield of Z is\t", zYield.getVal(), ".  From sWeights it is ")
   print( sData.GetYieldFromSWeight("zYield") )

   #print("Yield of QCD is\t", qcdYield->getVal(), ".  From sWeights it is ")
   print( sData.GetYieldFromSWeight("qcdYield") )
             

   for i in range(10):
      print("z Weight for event ", i, std.right, std.setw(12), sData.GetSWeight(i, "zYield"), "  qcd Weight")
      print(std.setw(12), sData.GetSWeight(i, "qcdYield"), "  Total Weight", std.setw(12), sData.GetSumOfEventSWeight(i) )
      
      
   
   print("\n")
   
   # import this new dataset with sWeights
   print(f"import new dataset with sWeights")
   object.Import(data, Rename("dataWithSWeights"))
   
   RooMsgService.instance().setGlobalKillBelow(RooFit.INFO)
   

def MakePlots(object):

   
   # Here we make plots of the discriminating variable (invMass) after the fit
   # and of the control variable (isolation) after unfolding with sPlot.
   print(f"make plots")
   
   # make our canvas
   cdata =  TCanvas("sPlot", "sPlot demo", 400, 600)
   cdata.Divide(1, 3)
   
   # get what we need out of the workspace
   model = object.pdf("model")
   zModel = object.pdf("zModel")
   qcdModel = object.pdf("qcdModel")
   
   isolation = object.var("isolation")
   invMass = object.var("invMass")
   
   # note, we get the dataset with sWeights
   data = object.data("dataWithSWeights")
   
   # create weighted data sets
   #"""
   ##info for debuggin :
   ## ROOT.nullptr doesn't work for : const RooFormulaVar & cutvar
   ##                                 const char* wgtVarName = nullptr 
   ##                                 const char* cuts = nullptr
   ## not even using hex(id(nullptr)) 
   ## improve pythonzation...
   ##RooDataSet::RooDataSet(RooStringView name, RooStringView title, RooDataSet* data, const RooArgSet& vars, const RooFormulaVar&     cutVar, const char* wgtVarName = nullptr) 
   ##RooDataSet::RooDataSet(RooStringView name, RooStringView title, RooDataSet* data, const RooArgSet& vars, const char* cuts = nu    llptr, const char* wgtVarName = nullptr) 
   #"""
   cutVar0 = RooFormulaVar() # continues at ...
   #for the purpose of this tutorial we don't use cuts. 
   #however, it is possible to implement them as shown below:
   zYield = object.var("zYield")
   qcdYield = object.var("qcdYield")
   ral = RooArgList(zYield, qcdYield)
   cutVar = RooFormulaVar("zYield < 0 and qcdYield < 0.5", "Cuts", " @0 < 0 && @1 < 0.5", ral, True)
   #... continues here
   dataw_qcd = RooDataSet(data.GetName(), data.GetTitle(), data, data.get(), cutVar0, "qcdYield_sw")
   dataw_z   = RooDataSet(data.GetName(), data.GetTitle(), data, data.get(), cutVar0, "zYield_sw")
   
   
   # this shouldn't be necessary, need to fix something with workspace
   # do this to set parameters back to their fitted values.
   #   model->fitTo(*data, Extended());
   
   # plot invMass for data with full model and individual components overlaid
   #  TCanvas* cdata = new TCanvas();
   cdata.cd(1)
   frame = invMass.frame(Title("Fit of model to discriminating variable"))
   data.plotOn(frame)
   model.plotOn(frame, Name("FullModel"))
   model.plotOn(frame, Components(zModel), LineStyle(kDashed), LineColor(kRed), Name("ZModel"))
   model.plotOn(frame, Components(qcdModel), LineStyle(kDashed), LineColor(kGreen), Name("QCDModel"))
   
   leg = TLegend(0.11, 0.5, 0.5, 0.8)
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
   
   frame2 = isolation.frame(Title("Isolation distribution with s weights to project out Z"))
   # Since the data are weighted, we use SumW2 to compute the errors.
   dataw_z.plotOn(frame2, DataError(RooAbsData.SumW2))
   zModel.plotOn(frame2, LineStyle(kDashed), LineColor(kRed))
   
   frame2.Draw()
   
   # Plot isolation for QCD component.
   # Eg. plot all events weighted by the sWeight for the QCD component.
   # The SPlot class adds a new variable that has the name of the corresponding
   # yield + "_sw".
   cdata.cd(3)
   frame3 = isolation.frame(Title("Isolation distribution with s weights to project out QCD"))
   dataw_qcd.plotOn(frame3, DataError(RooAbsData.SumW2))
   qcdModel.plotOn(frame3, LineStyle(kDashed), LineColor(kGreen))
   
   frame3.Draw()
   
   cdata.SaveAs("rs301_splot.png");
   

def rs301_splot():

   
   # Create a workspace to manage the project.
   global wspace 
   wspace = RooWorkspace("myWS")
   
   # add the signal and background models to the workspace.
   # Inside this function you will find a description of our model.
   AddModel(wspace)
   
   # add some toy data to the workspace
   AddData(wspace)
   
   # inspect the workspace if you wish
   #  wspace->Print();
   
   # do sPlot.
   # This will make a new dataset with sWeights added for every event.
   DoSPlot(wspace)
   
   # Make some plots showing the discriminating variable and
   # the control variable after unfolding.
   MakePlots(wspace)
   

rs301_splot()
