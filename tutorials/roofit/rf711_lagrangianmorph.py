## \file
## \ingroup tutorial_roofit
## \notebook -js
## Morphing effective field theory distributions with RooLagrangianMorphFunc.
## A morphing function as a function of one coefficient is setup and can be used
## to obtain the distribution for any value of the coefficient.
##
## \macro_image
## \macro_output
## \macro_code
##
## \date January 2022
## \authors Rahul Balasubramanian

import ROOT

ROOT.gStyle.SetOptStat(0)
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(True)

# Create functions
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
observablename = "pTV"

# Setup observable that is to be morphed
obsvar = ROOT.RooRealVar(observablename, "p_{T}^{V}", 10, 600)

# Setup two couplings that enters the morphing function
# kSM -> SM coupling set to constant (1)
# cHq3 -> EFT parameter with NewPhysics attribute set to true
kSM = ROOT.RooRealVar("kSM", "sm modifier", 1.0)
cHq3 = ROOT.RooRealVar("cHq3", "EFT modifier", 0.0, 1.0)
cHq3.setAttribute("NewPhysics", True)

# Inputs to setup config
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
infilename = ROOT.gROOT.GetTutorialDir().Data() + "/roofit/input_histos_rf_lagrangianmorph.root"
par = "cHq3"
samplelist = ["SM_NPsq0", "cHq3_NPsq1", "cHq3_NPsq2"]

# Set Config
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -

config = ROOT.RooLagrangianMorphFunc.Config()
config.fileName = infilename
config.observableName = observablename
config.folderNames = samplelist
config.couplings.add(cHq3)
config.couplings.add(kSM)


# Create morphing function
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -

morphfunc = ROOT.RooLagrangianMorphFunc("morphfunc", "morphed dist. of pTV", config)

# Get morphed distribution at cHq3 = 0.01, 0.5
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
morphfunc.setParameter("cHq3", 0.01)
morph_hist_0p01 = morphfunc.createTH1("morph_cHq3=0.01")
morphfunc.setParameter("cHq3", 0.25)
morph_hist_0p25 = morphfunc.createTH1("morph_cHq3=0.25")
morphfunc.setParameter("cHq3", 0.5)
morph_hist_0p5 = morphfunc.createTH1("morph_cHq3=0.5")
morph_datahist_0p01 = ROOT.RooDataHist("morph_dh_cHq3=0.01", "", [obsvar], morph_hist_0p01)
morph_datahist_0p25 = ROOT.RooDataHist("morph_dh_cHq3=0.25", "", [obsvar], morph_hist_0p25)
morph_datahist_0p5 = ROOT.RooDataHist("morph_dh_cHq3=0.5", "", [obsvar], morph_hist_0p5)

# Extract input templates for plotting
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -

input_hists = {sample: ROOT.TFile.Open(infilename).Get(sample).FindObject(observablename) for sample in samplelist}
input_datahists = {
    sample: ROOT.RooDataHist("dh_" + sample, "dh_" + sample, [obsvar], input_hists[sample]) for sample in samplelist
}

# Plot input templates
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -

frame0 = obsvar.frame(Title="Input templates for p_{T}^{V}")
for sample, color in zip(samplelist, "krb"):
    input_datahists[sample].plotOn(frame0, Name=sample, LineColor=color, MarkerColor=color, MarkerSize=1)

# Plot morphed templates for cHq3=0.01,0.25,0.5
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -

frame1 = obsvar.frame(Title="Morphed templates for selected values")
plot_args = dict(
    DrawOption="C",
    DataError=None,
    XErrorSize=0,
)
morph_datahist_0p01.plotOn(frame1, Name="morph_dh_cHq3=0.01", LineColor="kGreen", **plot_args)
morph_datahist_0p25.plotOn(frame1, Name="morph_dh_cHq3=0.25", LineColor="kGreen+1", **plot_args)
morph_datahist_0p5.plotOn(frame1, Name="morph_dh_cHq3=0.5", LineColor="kGreen+2", **plot_args)

# Create wrapped pdf to generate 2D dataset of cHq3 as a function of pTV
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -

model = ROOT.RooWrapperPdf("wrap_pdf", "wrap_pdf", morphfunc)
data = model.generate({cHq3, obsvar}, 1000000)
hh_data = ROOT.RooAbsData.createHistogram(data, "x,y", obsvar, Binning=20, YVar=dict(var=cHq3, Binning=50))
hh_data.SetTitle("Morphing prediction")

# Draw plots on canvas
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -

c1 = ROOT.TCanvas("fig3", "fig3", 1200, 400)
c1.Divide(3, 1)

c1.cd(1)
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.05)

frame0.Draw()
leg1 = ROOT.TLegend(0.55, 0.65, 0.94, 0.87)
leg1.SetTextSize(0.04)
leg1.SetFillColor(ROOT.kWhite)
leg1.SetLineColor(ROOT.kWhite)
leg1.AddEntry("SM_NPsq0", "SM", "LP")
leg1.AddEntry(0, "", "")
leg1.AddEntry("cHq3_NPsq1", "c_{Hq^{(3)}}=1.0 at #Lambda^{-2}", "LP")
leg1.AddEntry(0, "", "")
leg1.AddEntry("cHq3_NPsq2", "c_{Hq^{(3)}}=1.0 at #Lambda^{-4}", "LP")
leg1.Draw()

c1.cd(2)
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.05)

frame1.Draw()

leg2 = ROOT.TLegend(0.62, 0.65, 0.94, 0.87)
leg2.SetTextSize(0.04)
leg2.SetFillColor(ROOT.kWhite)
leg2.SetLineColor(ROOT.kWhite)

leg2.AddEntry("morph_dh_cHq3=0.01", "c_{Hq^{(3)}}=0.01", "L")
leg2.AddEntry(0, "", "")
leg2.AddEntry("morph_dh_cHq3=0.025", "c_{Hq^{(3)}}=0.025", "L")
leg2.AddEntry(0, "", "")
leg2.AddEntry("morph_dh_cHq3=0.5", "c_{Hq^{(3)}}=0.5", "L")
leg2.AddEntry(0, "", "")
leg2.Draw()

c1.cd(3)
ROOT.gPad.SetLeftMargin(0.12)
ROOT.gPad.SetRightMargin(0.18)
ROOT.gStyle.SetNumberContours(255)
ROOT.gStyle.SetPalette(ROOT.kGreyScale)
ROOT.gStyle.SetOptStat(0)
ROOT.TColor.InvertPalette()
ROOT.gPad.SetLogz()
hh_data.GetYaxis().SetTitle("c_{Hq^{(3)}}")
hh_data.GetYaxis().SetRangeUser(0, 0.5)
hh_data.GetZaxis().SetTitleOffset(1.8)
hh_data.Draw("COLZ")
c1.SaveAs("rf711_lagrangianmorph.png")
