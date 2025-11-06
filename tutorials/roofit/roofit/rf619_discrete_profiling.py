## \file
## \ingroup tutorial_roofit_main
## \notebook -js
## Basic functionality: demonstrate fitting multiple models using RooMultiPdf and selecting the best one via Discrete
## Profiling method.
##
## \macro_image
## \macro_code
## \macro_output
##
## \date July 2025
## \author Galin Bistrev

import ROOT

x = ROOT.RooRealVar("x", "Observable", 0, 50)


lambda1 = ROOT.RooRealVar("lambda1", "slope1", -0.025, -0.1, -0.02)
expo1 = ROOT.RooExponential("expo1", "Exponential 1", x, lambda1)

c0 = ROOT.RooRealVar("c0", "Cheby coeff 0", -1.0, -1.0, 1.0)
c1 = ROOT.RooRealVar("c1", "Cheby coeff 1", 0.4, 0.05, 0.5)
chebCoeffs = ROOT.RooArgList(c0, c1)
cheb = ROOT.RooChebychev("cheb", "Chebyshev PDF", x, chebCoeffs)

pdfIndex0 = ROOT.RooCategory("pdfIndex0", "pdf index 0")
multiPdf0 = ROOT.RooMultiPdf("multiPdf0", "multiPdf0", pdfIndex0, ROOT.RooArgList(expo1, cheb))


lambdaExtra = ROOT.RooRealVar("lambdaExtra", "extra slope", -0.05, -1.0, -0.01)
expoExtra = ROOT.RooExponential("expoExtra", "extra exponential", x, lambdaExtra)


mean = ROOT.RooRealVar("mean", "shared mean", 25, 0, 50)
sigmaG = ROOT.RooRealVar("sigmaG", "Gaussian width", 2.0, 0.0, 5.0)
sigmaL = ROOT.RooRealVar("sigmaL", "Landau width", 3.0, 1.0, 8.0)

gauss1 = ROOT.RooGaussian("gauss1", "Gaussian", x, mean, sigmaG)
landau1 = ROOT.RooLandau("landau1", "Landau", x, mean, sigmaL)

pdfIndex1 = ROOT.RooCategory("pdfIndex1", "pdf index 1")
multiPdf1 = ROOT.RooMultiPdf("multiPdf1", "multiPdf1", pdfIndex1, ROOT.RooArgList(gauss1, landau1))


sigmaExtra = ROOT.RooRealVar("sigmaExtra", "extra Gaussian width", 3.0, 1.0, 6.0)
gaussExtra = ROOT.RooGaussian("gaussExtra", "extra Gaussian", x, mean, sigmaExtra)


frac0 = ROOT.RooRealVar("frac0", "fraction for cat0", 0.7, 0.0, 1.0)
addPdf0 = ROOT.RooAddPdf("addPdf0", "multiPdf0 + extra expo", ROOT.RooArgList(multiPdf0, gaussExtra), frac0)

frac1 = ROOT.RooRealVar("frac1", "fraction for cat1", 0.5, 0.0, 1.0)
addPdf1 = ROOT.RooAddPdf("addPdf1", "multiPdf1 + extra gauss", ROOT.RooArgList(multiPdf1, expoExtra), frac1)
catIndex = ROOT.RooCategory("catIndex", "Category")
catIndex.defineType("cat0", 0)
catIndex.defineType("cat1", 1)

simPdf = ROOT.RooSimultaneous("simPdf", "simultaneous model", catIndex)
simPdf.addPdf(addPdf0, "cat0")
simPdf.addPdf(addPdf1, "cat1")


data0 = addPdf0.generate(ROOT.RooArgSet(x), 800)
data1 = addPdf1.generate(ROOT.RooArgSet(x), 1000)


frame0 = x.frame()
data0.plotOn(frame0)
addPdf0.plotOn(frame0)
pdfIndex0.setIndex(1)
addPdf0.plotOn(frame0, ROOT.RooFit.LineColor(ROOT.kRed))

frame0.SetTitle("")
frame0.GetXaxis().SetTitle("Observable")
frame0.GetYaxis().SetTitle("Events")

leg0 = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)
leg0.AddEntry(frame0.getObject(0), "Data", "lep")
leg0.AddEntry(frame0.getObject(1), "Expo", "l")
leg0.AddEntry(frame0.getObject(2), "Poly", "l")


frame1 = x.frame()
data1.plotOn(frame1)
addPdf1.plotOn(frame1)
pdfIndex1.setIndex(1)
addPdf1.plotOn(frame1, ROOT.RooFit.LineColor(ROOT.kRed))

frame1.SetTitle("")
frame1.GetXaxis().SetTitle("Observable")
frame1.GetYaxis().SetTitle("Events")

leg1 = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)
leg1.AddEntry(frame1.getObject(0), "Data", "lep")
leg1.AddEntry(frame1.getObject(1), "Gauss", "l")
leg1.AddEntry(frame1.getObject(2), "Landau", "l")


combined_data = ROOT.RooDataSet("data", "combined", ROOT.RooArgSet(x, catIndex))

combined_vars = ROOT.RooArgSet(x, catIndex)

for i in range(data0.numEntries()):
    x.setVal(data0.get(i).getRealValue("x"))
    catIndex.setLabel("cat0")
    combined_data.add(combined_vars)

for i in range(data1.numEntries()):
    x.setVal(data1.get(i).getRealValue("x"))
    catIndex.setLabel("cat1")
    combined_data.add(combined_vars)


nll = simPdf.createNLL(combined_data)
minim = ROOT.RooMinimizer(nll)
minim.setStrategy(1)
minim.setEps(1e-7)
minim.setPrintLevel(-1)


nMeanPoints = 40
meanMin = 17
meanMax = 33

combosToPlot = [[i, j] for i in range(pdfIndex0.numTypes()) for j in range(pdfIndex1.numTypes())]

colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen + 2, ROOT.kMagenta, ROOT.kOrange + 7]
markers = [20, 21, 22, 23, 33]

graphs = []
for idx in range(len(combosToPlot)):
    g = ROOT.TGraph(nMeanPoints)
    g.SetLineColor(colors[idx % 5])
    g.SetMarkerColor(colors[idx % 5])
    g.SetMarkerStyle(markers[idx % 5])
    g.SetTitle(f"Combo [{combosToPlot[idx][0]},{combosToPlot[idx][1]}]")
    graphs.append(g)

profileGraph = ROOT.TGraph(nMeanPoints)
profileGraph.SetLineColor(ROOT.kBlack)
profileGraph.SetLineWidth(4)
profileGraph.SetMarkerColor(ROOT.kBlack)
profileGraph.SetMarkerStyle(22)
profileGraph.SetTitle("Profile")
graphs.append(profileGraph)


for i in range(nMeanPoints):
    meanVal = meanMin + i * (meanMax - meanMin) / (nMeanPoints - 1)
    mean.setVal(meanVal)

    for comboIdx, combo in enumerate(combosToPlot):
        pdfIndex0.setIndex(combo[0])
        pdfIndex1.setIndex(combo[1])
        pdfIndex0.setConstant(True)
        pdfIndex1.setConstant(True)
        mean.setConstant(True)
        minim.minimize("Minuit2", "Migrad")
        graphs[comboIdx].SetPoint(i, meanVal, nll.getVal())
        pdfIndex0.setConstant(False)
        pdfIndex1.setConstant(False)
        mean.setConstant(False)

    mean.setConstant(True)
    minim.minimize("Minuit2", "Migrad")
    profileGraph.SetPoint(i, meanVal, nll.getVal())


c = ROOT.TCanvas("c_rf619", "NLL vs Mean for Different Discrete Combinations", 1200, 400)
c.Divide(3, 1)

c.cd(1)
ROOT.gPad.SetLeftMargin(0.15)
frame0.GetYaxis().SetTitleOffset(1.4)
frame0.Draw()
leg0.Draw()

c.cd(2)
ROOT.gPad.SetLeftMargin(0.15)
frame1.GetYaxis().SetTitleOffset(1.4)
frame1.Draw()
leg1.Draw()

c.cd(3)
ROOT.gPad.SetLeftMargin(0.15)
mg = ROOT.TMultiGraph()
for g in graphs:
    mg.Add(g, "PL")
mg.Draw("APL")
mg.GetXaxis().SetTitle("Mean")
mg.GetYaxis().SetTitle("NLL")
ROOT.gPad.BuildLegend()
