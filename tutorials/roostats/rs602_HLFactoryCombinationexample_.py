# \file
# \ingroup tutorial_roostats
# \notebook
# High Level Factory: creation of a combined model
#
# \macro_image
# \macro_output
# \macro_code
#
# \author Danilo Piparo

import ROOT

from ROOT import RooFit, RooStats

TString = 		 ROOT.TString
TROOT = 		 ROOT.TROOT
RooWorkspace = 		 ROOT.RooWorkspace
RooRealVar = 		 ROOT.RooRealVar
RooAbsPdf = 		 ROOT.RooAbsPdf
RooDataSet = 		 ROOT.RooDataSet
RooPlot = 		 ROOT.RooPlot
TCanvas = 		 ROOT.TCanvas

# use this order for safety on library loading
HLFactory = RooStats.HLFactory
ofstream = ROOT.ofstream
RooArgSet = ROOT.RooArgSet
Extended = RooFit.Extended
Slice = RooFit.Slice  
ProjWData = RooFit.ProjWData 

def rs602_HLFactoryCombinationexample():

   
   
   # create a card
   card_name = "HLFactoryCombinationexample.rs"
   with open(card_name, "w") as ofile:
      ofile.write("// The simplest card for combination\n\n")
      ofile.write("gauss1 = Gaussian(x[0,100],mean1[50,0,100],4);\n")
      ofile.write("flat1 = Polynomial(x,0);\n")
      ofile.write("sb_model1 = SUM(nsig1[120,0,300]*gauss1 , nbkg1[100,0,1000]*flat1);\n")
      ofile.write("gauss2 = Gaussian(x,mean2[80,0,100],5);\n")
      ofile.write("flat2 = Polynomial(x,0);\n")
      ofile.write("sb_model2 = SUM(nsig2[90,0,400]*gauss2 , nbkg2[80,0,1000]*flat2);\n")
   
   hlf = HLFactory("HLFavtoryCombinationexample", card_name, False)
   hlf.Print()
    
   hlf.AddChannel("model1", "sb_model1", "flat1")
   hlf.AddChannel("model2", "sb_model2", "flat2")
   hlf.Print()
   
   pdf = hlf.GetTotSigBkgPdf()
   thecat = hlf.GetTotCategory()
   pdf.Print()
   thecat.Print()
   
   x = (hlf.GetWs().arg("x"))
   x.Print()
    
   data = pdf.generate(RooArgSet(x, thecat), Extended())
   
   # --- Perform extended ML fit of composite PDF to toy data ---
   pdf.fitTo(data)
   
   # --- Plot toy data and composite PDF overlaid ---
   c = TCanvas("c", "c")
   xframe = x.frame()
   
   data.plotOn(xframe)
   thecat.setIndex(0)
   pdf.plotOn(xframe, Slice(thecat), ProjWData(thecat, data))
   
   thecat.setIndex(1)
   pdf.plotOn(xframe, Slice(thecat), ProjWData(thecat, data))
   
   ROOT.gROOT.SetStyle("Plain")
   xframe.Draw()
   c.Update()
   c.Draw()
   
   c.SaveAs("rs602_HLFactoryCombinationexample.png")

rs602_HLFactoryCombinationexample()
