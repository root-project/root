## \file
## \ingroup tutorial_roofit
## \notebook -js
## Performing a simple fit with RooLagrangianMorphFunc
##
## \macro_image
## \macro_output
## \macro_code
##
## \date January 2022
## \author Rahul Balasubramanian

import ROOT

ROOT.gStyle.SetOptStat(0)
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(True)

# Create functions
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
observablename = "pTV"
obsvar = ROOT.RooRealVar(observablename, "observable of pTV", 10, 600)

# Setup three EFT coefficent and constant SM modifier
kSM = ROOT.RooRealVar("kSM", "sm modifier", 1.0)
cHq3 = ROOT.RooRealVar("cHq3", "EFT modifier", -10.0, 10.0)
cHq3.setAttribute("NewPhysics", True)
cHl3 = ROOT.RooRealVar("cHl3", "EFT modifier", -10.0, 10.0)
cHl3.setAttribute("NewPhysics", True)
cHDD = ROOT.RooRealVar("cHDD", "EFT modifier", -10.0, 10.0)
cHDD.setAttribute("NewPhysics", True)

# Inputs to setup config
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
infilename = ROOT.gROOT.GetTutorialDir().Data() + "/roofit/input_histos_rf_lagrangianmorph.root"
par = "cHq3"
samplelist = [
    "SM_NPsq0",
    "cHq3_NPsq1",
    "cHq3_NPsq2",
    "cHl3_NPsq1",
    "cHl3_NPsq2",
    "cHDD_NPsq1",
    "cHDD_NPsq2",
    "cHl3_cHDD_NPsq2",
    "cHq3_cHDD_NPsq2",
    "cHl3_cHq3_NPsq2",
]

# Set Config
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -

config = ROOT.RooLagrangianMorphFunc.Config()
config.fileName = infilename
config.observableName = observablename
config.folderNames = samplelist
config.couplings.add(cHq3)
config.couplings.add(cHDD)
config.couplings.add(cHl3)
config.couplings.add(kSM)


# Create morphing function
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -

morphfunc = ROOT.RooLagrangianMorphFunc("morphfunc", "morphed dist. of pTV", config)

# Create pseudo data histogram to fit at cHq3 = 0.01, cHl3 = 1.0, cHDD = 0.2
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
morphfunc.setParameter("cHq3", 0.01)
morphfunc.setParameter("cHl3", 1.0)
morphfunc.setParameter("cHDD", 0.2)

pseudo_hist = morphfunc.createTH1("pseudo_hist")
pseudo_dh = ROOT.RooDataHist("pseudo_dh", "pseudo_dh", [obsvar], pseudo_hist)

# reset parameters to zeros before fit
morphfunc.setParameter("cHq3", 0.0)
morphfunc.setParameter("cHl3", 0.0)
morphfunc.setParameter("cHDD", 0.0)

# set error to set initial step size in fit
cHq3.setError(0.1)
cHl3.setError(0.1)
cHDD.setError(0.1)

# Wrap pdf on morphfunc and fit to data histogram
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -

# wrapper pdf to normalise morphing function to a morphing pdf
model = ROOT.RooWrapperPdf("wrap_pdf", "wrap_pdf", morphfunc)
fitres = model.fitTo(pseudo_dh, SumW2Error=True, Optimize=False, Save=True)
# run the fit
# Get the correlation matrix
hcorr = fitres.correlationHist()

# Extract postfit distribution and plot with initial histogram
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -

postfit_hist = morphfunc.createTH1("morphing_postfit_hist")
postfit_dh = ROOT.RooDataHist("morphing_postfit_dh", "morphing_postfit_dh", [obsvar], postfit_hist)

frame0 = obsvar.frame(Title="Input templates for p_{T}^{V}")
postfit_dh.plotOn(
    frame0,
    Name="postfit_dist",
    DrawOption="C",
    LineColor="b",
    DataError=None,
    XErrorSize=0,
)
pseudo_dh.plotOn(frame0, Name="input")

# Draw plots on canvas
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -

c1 = ROOT.TCanvas("fig3", "fig3", 800, 400)
c1.Divide(2, 1)

c1.cd(1)
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.05)

model.paramOn(frame0, ROOT.RooFit.Layout(0.50, 0.75, 0.9))
frame0.GetXaxis().SetTitle("p_{T}^{V}")
frame0.Draw()

c1.cd(2)
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.15)
ROOT.gStyle.SetPaintTextFormat("4.1f")
ROOT.gStyle.SetOptStat(0)
hcorr.SetMarkerSize(3.0)
hcorr.SetTitle("correlation matrix")
hcorr.GetYaxis().SetTitleOffset(1.4)
hcorr.GetYaxis().SetLabelSize(0.1)
hcorr.GetXaxis().SetLabelSize(0.1)
hcorr.GetYaxis().SetBinLabel(1, "c_{HDD}")
hcorr.GetYaxis().SetBinLabel(2, "c_{Hl^{(3)}}")
hcorr.GetYaxis().SetBinLabel(3, "c_{Hq^{(3)}}")
hcorr.GetXaxis().SetBinLabel(3, "c_{HDD}")
hcorr.GetXaxis().SetBinLabel(2, "c_{Hl^{(3)}}")
hcorr.GetXaxis().SetBinLabel(1, "c_{Hq^{(3)}}")
hcorr.GetYaxis().SetTitleOffset(1.4)
hcorr.Draw("colz text")

c1.SaveAs("rf712_lagrangianmorphfit.png")
