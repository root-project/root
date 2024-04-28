# \file
# \ingroup roostats_python_tutorials
# \notebook -js
# tutorial demonstrating and validates the RooJeffreysPrior class
#
# Jeffreys's prior is an 'objective prior' based on formal rules.
# It is calculated from the Fisher information matrix.
#
# Read more:
# http:#en.wikipedia.org/wiki/Jeffreys_prior
#
# The analytic form is not known for most PDFs, but it is for
# simple cases like the Poisson mean, Gaussian mean, Gaussian sigma.
#
# This class uses numerical tricks to calculate the Fisher Information Matrix
# efficiently.  In particular, it takes advantage of a property of the
# 'Asimov data' as described in
# Asymptotic formulae for likelihood-based tests of new physics
# Glen Cowan, Kyle Cranmer, Eilam Gross, Ofer Vitells
# http:#arxiv.org/abs/arXiv:1007.1727
#
# This Demo has four parts:
#  1. TestJeffreysPriorDemo -- validates Poisson mean case 1/sqrt(mu)
#  2. TestJeffreysGaussMean -- validates Gaussian mean case
#  3. TestJeffreysGaussSigma -- validates Gaussian sigma case 1/sigma
#  4. TestJeffreysGaussMeanAndSigma -- demonstrates 2-d example
#
# \macro_image
# \macro_output
# \macro_code
#
# \author Kyle Cranmer
# \translator P. P.

import ROOT 
from ROOT import RooStats, RooFit

RooWorkspace = 		 ROOT.RooWorkspace
RooDataHist = 		 ROOT.RooDataHist
RooGenericPdf = 		 ROOT.RooGenericPdf
RooPlot = 		 ROOT.RooPlot
RooFitResult = 		 ROOT.RooFitResult
RooRealVar = 		 ROOT.RooRealVar
RooAbsPdf = 		 ROOT.RooAbsPdf
RooNumIntConfig = 		 ROOT.RooNumIntConfig

TH1F = 		 ROOT.TH1F
TCanvas = 		 ROOT.TCanvas
TLegend = 		 ROOT.TLegend
TMatrixDSym = 		 ROOT.TMatrixDSym

RooWorkspace = ROOT.RooWorkspace 
ExpectedData = RooFit.ExpectedData
Save = RooFit.Save
SumW2Error = RooFit.SumW2Error
kTRUE = ROOT.kTRUE
sqrt = ROOT.sqrt 
RooJeffreysPrior = ROOT.RooJeffreysPrior
LineColor = RooFit.LineColor
LineStyle = RooFit.LineStyle
kRed = ROOT.kRed
kDashed = ROOT.kDashed
kDashDotted = ROOT.kDashDotted




def rs302_JeffreysPriorDemo():

   w = RooWorkspace("w")
   w.factory("Uniform::u(x[0,1])")
   w.factory("mu[100,1,200]")
   w.factory("ExtendPdf::p(u,mu)")
   
   asimov = w.pdf("p").generateBinned(w.var("x"), ExpectedData())
   
   res = w.pdf("p").fitTo(asimov, Save(), SumW2Error(kTRUE))
   
   asimov.Print()
   res.Print()
   cov = res.covarianceMatrix()
   print("variance = ", (cov.Determinant()))
   print("stdev = ", sqrt(cov.Determinant()))
   cov.Invert()
   print("jeffreys = ", sqrt(cov.Determinant()))
   
   w.defineSet("poi", "mu")
   w.defineSet("obs", "x")
   
   pi = RooJeffreysPrior("jeffreys", "jeffreys", w.pdf("p"), w.set("poi"), w.set("obs"))
   
   test =  RooGenericPdf("Expected", "Expected = 1/#sqrt#mu", "1./sqrt(mu)",
   w.set("poi"))
   
   c1 =  TCanvas()
   plot = w.var("mu").frame()
   pi.plotOn(plot)
   test.plotOn(plot, LineColor(kRed), LineStyle(kDashDotted))
   plot.Draw()
   
   legend = plot.BuildLegend()
   legend.DrawClone()

   c1.Update()
   c1.Draw()
   c1.SaveAs("rs302_JeffreysPriorDemo.1.png")
   

#_________________________________________________
def TestJeffreysGaussMean():

   w = RooWorkspace("w")
   w.factory("Gaussian.g(x[0,-20,20],mu[0,-5.,5],sigma[1,0,10])")
   w.factory("n[10,.1,200]")
   w.factory("ExtendPdf.p(g,n)")
   w.var("sigma").setConstant()
   w.var("n").setConstant()
   
   asimov = w.pdf("p").generateBinned(w.var("x"), ExpectedData())
   
   resw.pdf("p").fitTo(asimov, Save(), SumW2Error(kTRUE))
   
   asimov.Print()
   res.Print()
   cov = res.covarianceMatrix()
   print("variance = ", (cov.Determinant()))
   print("stdev = ", sqrt(cov.Determinant()))
   cov.Invert()
   print("jeffreys = ", sqrt(cov.Determinant()))
   
   w.defineSet("poi", "mu")
   w.defineSet("obs", "x")
   
   pi = RooJeffreysPrior("jeffreys", "jeffreys", w.pdf("p"), w.set("poi"), w.set("obs"))
   
   temp = w.set("poi")
   pi.getParameters(temp).Print()
   
   #  return
   test =  RooGenericPdf("constant", "Expected = constant", "1", w.set("poi"))
   
   c2 =  TCanvas()
   plot = w.var("mu").frame()
   pi.plotOn(plot)
   test.plotOn(plot, LineColor(kRed), LineStyle(kDashDotted))
   plot.Draw()
   
   legend = plot.BuildLegend()
   legend.DrawClone()
   c2.Update()
   c2.Draw()
   c2.SaveAs("rs302_JeffreysPriorDemo.2.png")
   

#_________________________________________________
def TestJeffreysGaussSigma():

   # this one is VERY sensitive
   # if the Gaussian is narrow ~ range(x)/nbins(x) then the peak isn't resolved
   #   and you get really bizarre shapes
   # if the Gaussian is too wide range(x) ~ sigma then PDF gets renormalized
   #   and the PDF falls off too fast at high sigma
   w = RooWorkspace("w")
   w.factory("Gaussian.g(x[0,-20,20],mu[0,-5,5],sigma[1,1,5])")
   w.factory("n[100,.1,2000]")
   w.factory("ExtendPdf.p(g,n)")
   #  w.var("sigma").setConstant()
   w.var("mu").setConstant()
   w.var("n").setConstant()
   w.var("x").setBins(301)
   
   asimov = w.pdf("p").generateBinned(w.var("x"), ExpectedData())
   
   resw.pdf("p").fitTo(asimov, Save(), SumW2Error(kTRUE))
   
   asimov.Print()
   res.Print()
   cov = res.covarianceMatrix()
   print("variance = ", (cov.Determinant()))
   print("stdev = ", sqrt(cov.Determinant()))
   cov.Invert()
   print("jeffreys = ", sqrt(cov.Determinant()))
   
   w.defineSet("poi", "sigma")
   w.defineSet("obs", "x")
   
   pi = RooJeffreysPrior("jeffreys", "jeffreys", w.pdf("p"), w.set("poi"), w.set("obs"))
   
   temp = w.set("poi")
   pi.getParameters(temp).Print()
   
   test =  RooGenericPdf("test", "Expected = #sqrt2/#sigma", "sqrt(2.)/sigma", w.set("poi"))
   
   c3 =  TCanvas()
   plot = w.var("sigma").frame()
   pi.plotOn(plot)
   test.plotOn(plot, LineColor(kRed), LineStyle(kDashDotted))
   plot.Draw()
   
   legend = plot.BuildLegend()
   legend.DrawClone()
   c3.Update()
   c3.Draw()
   c3.SaveAs("rs302_JeffreysPriorDemo.3.png") 

#_________________________________________________
def TestJeffreysGaussMeanAndSigma():

   # this one is VERY sensitive
   # if the Gaussian is narrow ~ range(x)/nbins(x) then the peak isn't resolved
   #   and you get really bizarre shapes
   # if the Gaussian is too wide range(x) ~ sigma then PDF gets renormalized
   #   and the PDF falls off too fast at high sigma
   w = RooWorkspace("w")
   w.factory("Gaussian.g(x[0,-20,20],mu[0,-5,5],sigma[1,1.,5.])")
   w.factory("n[100,.1,2000]")
   w.factory("ExtendPdf.p(g,n)")
   
   w.var("n").setConstant()
   w.var("x").setBins(301)
   
   asimov = w.pdf("p").generateBinned(w.var("x"), ExpectedData())
   
   resw.pdf("p").fitTo(asimov, Save(), SumW2Error(kTRUE))
   
   asimov.Print()
   res.Print()
   cov = res.covarianceMatrix()
   print("variance = ", (cov.Determinant()))
   print("stdev = ", sqrt(cov.Determinant()))
   cov.Invert()
   print("jeffreys = ", sqrt(cov.Determinant()))
   
   w.defineSet("poi", "mu,sigma")
   w.defineSet("obs", "x")
   
   pi = RooJeffreysPrior("jeffreys", "jeffreys", w.pdf("p"), w.set("poi"), w.set("obs"))
   
   temp = w.set("poi")
   pi.getParameters(temp).Print()
   #  return
   
   c4 =  TCanvas()
   Jeff2d = pi.createHistogram("2dJeffreys", w.var("mu"), Binning(10, -5., 5), YVar(w.var("sigma"), Binning(10, 1., 5.)))
   Jeff2d.Draw("surf")
   c4.Update()
   c4.Draw()
   c4.SaveAs("rs302_JeffreysPriorDemo.4.png")
   
rs302_JeffreysPriorDemo()
