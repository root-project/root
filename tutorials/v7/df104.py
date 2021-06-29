## \file
## \ingroup tutorial_v7
## The Higgs to two photons analysis from the ATLAS Open Data 2020 release, with RDataFrame.
##
## This tutorial is the Higgs to two photons analysis from the ATLAS Open Data release in 2020
## (http://opendata.atlas.cern/release/2020/documentation/). The data was taken with the ATLAS detector
## during 2016 at a center-of-mass energy of 13 TeV. Although the Higgs to two photons decay is very rare,
## the contribution of the Higgs can be seen as a narrow peak around 125 GeV because of the excellent
## reconstruction and identification efficiency of photons at the ATLAS experiment.
##
## The analysis is translated to a RDataFrame workflow processing 1.7 GB of simulated events and data.
##
## This macro is replica of tutorials/dataframe/df104_HiggsToTwoPhotons.py, but with usage of ROOT7 graphics
## Run macro with python3 -i df104.py command to get interactive canvas
##
## \macro_image (rcanvas_js)
## \macro_code
##
## \date 2021-06-15
## \authors Stefan Wunsch (KIT, CERN) Sergey Linev (GSI)

import ROOT
import os
from ROOT.Experimental import RCanvas, RText, RLegend, RPadPos, RPadExtent, TObjectDrawable

# Enable multi-threading
ROOT.ROOT.EnableImplicitMT()

# Create a ROOT dataframe for each dataset
path = "root://eospublic.cern.ch//eos/opendata/atlas/OutreachDatasets/2020-01-22"
df = {}
df["data"] = ROOT.RDataFrame("mini", (os.path.join(path, "GamGam/Data/data_{}.GamGam.root".format(x)) for x in ("A", "B", "C", "D")))
df["ggH"] = ROOT.RDataFrame("mini", os.path.join(path, "GamGam/MC/mc_343981.ggH125_gamgam.GamGam.root"))
df["VBF"] = ROOT.RDataFrame("mini", os.path.join(path, "GamGam/MC/mc_345041.VBFH125_gamgam.GamGam.root"))
processes = list(df.keys())

# Apply scale factors and MC weight for simulated events and a weight of 1 for the data
for p in ["ggH", "VBF"]:
    df[p] = df[p].Define("weight",
            "scaleFactor_PHOTON * scaleFactor_PhotonTRIGGER * scaleFactor_PILEUP * mcWeight");
df["data"] = df["data"].Define("weight", "1.0")

# Select the events for the analysis
for p in processes:
    # Apply preselection cut on photon trigger
    df[p] = df[p].Filter("trigP")

    # Find two good muons with tight ID, pt > 25 GeV and not in the transition region between barrel and encap
    df[p] = df[p].Define("goodphotons", "photon_isTightID && (photon_pt > 25000) && (abs(photon_eta) < 2.37) && ((abs(photon_eta) < 1.37) || (abs(photon_eta) > 1.52))")\
                 .Filter("Sum(goodphotons) == 2")

    # Take only isolated photons
    df[p] = df[p].Filter("Sum(photon_ptcone30[goodphotons] / photon_pt[goodphotons] < 0.065) == 2")\
                 .Filter("Sum(photon_etcone20[goodphotons] / photon_pt[goodphotons] < 0.065) == 2")

# Compile a function to compute the invariant mass of the diphoton system
ROOT.gInterpreter.Declare(
"""
using Vec_t = const ROOT::VecOps::RVec<float>;
float ComputeInvariantMass(Vec_t& pt, Vec_t& eta, Vec_t& phi, Vec_t& e) {
    ROOT::Math::PtEtaPhiEVector p1(pt[0], eta[0], phi[0], e[0]);
    ROOT::Math::PtEtaPhiEVector p2(pt[1], eta[1], phi[1], e[1]);
    return (p1 + p2).mass() / 1000.0;
}
""")

# Define a new column with the invariant mass and perform final event selection
hists = {}
for p in processes:
    # Make four vectors and compute invariant mass
    df[p] = df[p].Define("m_yy", "ComputeInvariantMass(photon_pt[goodphotons], photon_eta[goodphotons], photon_phi[goodphotons], photon_E[goodphotons])")

    # Make additional kinematic cuts and select mass window
    df[p] = df[p].Filter("photon_pt[goodphotons][0] / 1000.0 / m_yy > 0.35")\
                 .Filter("photon_pt[goodphotons][1] / 1000.0 / m_yy > 0.25")\
                 .Filter("m_yy > 105 && m_yy < 160")

    # Book histogram of the invariant mass with this selection
    hists[p] = df[p].Histo1D(
            ROOT.RDF.TH1DModel(p, "Diphoton invariant mass; m_{#gamma#gamma} [GeV];Events", 30, 105, 160),
            "m_yy", "weight")

# Run the event loop

# RunGraphs allows to run the event loops of the separate RDataFrame graphs
# concurrently. This results in an improved usage of the available resources
# if each separate RDataFrame can not utilize all available resources, e.g.,
# because not enough data is available.
ROOT.RDF.RunGraphs([hists[s] for s in ["ggH", "VBF", "data"]])

ggh = hists["ggH"].GetValue()
vbf = hists["VBF"].GetValue()
data = hists["data"].GetValue()

# Create the plot

# Set styles - not yet available for v7
# ROOT.gROOT.SetStyle("ATLAS")

# Create canvas with pads for main plot and data/MC ratio
c = RCanvas.Create("df104_HiggsToTwoPhotons")

lower_pad = c.AddPad(RPadPos(0,0.65), RPadExtent(1, 0.35))
upper_pad = c.AddPad(RPadPos(0,0), RPadExtent(1, 0.65))

upper_frame = upper_pad.AddFrame()
upper_frame.margins.bottom = 0
upper_frame.margins.left = 0.14
upper_frame.margins.right = 0.05
upper_frame.x.labels.hide = True

lower_frame = lower_pad.AddFrame()
lower_frame.margins.top = 0
lower_frame.margins.left = 0.14
lower_frame.margins.right = 0.05
lower_frame.margins.bottom = 0.3

# Fit signal + background model to data
fit = ROOT.TF1("fit", "([0]+[1]*x+[2]*x^2+[3]*x^3)+[4]*exp(-0.5*((x-[5])/[6])^2)", 105, 160)
fit.FixParameter(5, 125.0)
fit.FixParameter(4, 119.1)
fit.FixParameter(6, 2.39)
fit.SetLineColor(2)
fit.SetLineStyle(1)
fit.SetLineWidth(2)
# do not draw fit function
data.Fit("fit", "0", "", 105, 160)

# Draw data
data.SetMarkerStyle(20)
data.SetMarkerSize(1.2)
data.SetLineWidth(2)
data.SetLineColor(ROOT.kBlack)
data.SetMinimum(1e-3)
data.SetMaximum(8e3)
data.GetYaxis().SetLabelSize(0.045)
data.GetYaxis().SetTitleSize(0.05)
data.SetStats(0)
data.SetTitle("")

data_drawable = upper_pad.Add[TObjectDrawable]()
data_drawable.Set(data, "E")

# Draw fit
fit_drawable = upper_pad.Add[TObjectDrawable]()
fit_drawable.Set(fit, "SAME")

# Draw background
bkg = ROOT.TF1("bkg", "([0]+[1]*x+[2]*x^2+[3]*x^3)", 105, 160)
for i in range(4):
    bkg.SetParameter(i, fit.GetParameter(i))
bkg.SetLineColor(4)
bkg.SetLineStyle(2)
bkg.SetLineWidth(2)
bkg_drawable = upper_pad.Add[TObjectDrawable]()
bkg_drawable.Set(bkg, "SAME")

# Scale simulated events with luminosity * cross-section / sum of weights
# and merge to single Higgs signal
lumi = 10064.0
ggh.Scale(lumi * 0.102 / 55922617.6297)
vbf.Scale(lumi * 0.008518764 / 3441426.13711)
higgs = ggh.Clone()
higgs.Add(vbf)
higgs_drawable = upper_pad.Add[TObjectDrawable]()
higgs_drawable.Set(higgs, "HIST SAME")

# Draw ratio

ratiobkg = ROOT.TH1I("zero", "", 10, 105, 160)
ratiobkg.SetLineColor(4)
ratiobkg.SetLineStyle(2)
ratiobkg.SetLineWidth(2)
ratiobkg.SetMinimum(-125)
ratiobkg.SetMaximum(250)
ratiobkg.GetXaxis().SetLabelSize(0.08)
ratiobkg.GetXaxis().SetTitleSize(0.12)
ratiobkg.GetXaxis().SetTitleOffset(1.0)
ratiobkg.GetYaxis().SetLabelSize(0.08)
ratiobkg.GetYaxis().SetTitleSize(0.09)
ratiobkg.GetYaxis().SetTitle("Data - Bkg.")
ratiobkg.GetYaxis().CenterTitle()
ratiobkg.GetYaxis().SetTitleOffset(0.7)
ratiobkg.GetYaxis().SetNdivisions(503, False)
ratiobkg.GetYaxis().ChangeLabel(-1, -1, 0)
ratiobkg.GetXaxis().SetTitle("m_{#gamma#gamma} [GeV]")
lower_pad.Add[TObjectDrawable]().Set(ratiobkg, "AXIS")

ratiosig = ROOT.TH1F("ratiosig", "ratiosig", 5500, 105, 160)
ratiosig.Eval(fit)
ratiosig.SetLineColor(2)
ratiosig.SetLineStyle(1)
ratiosig.SetLineWidth(2)
ratiosig.Add(bkg, -1)
lower_pad.Add[TObjectDrawable]().Set(ratiosig, "SAME")

ratiodata = data.Clone()
ratiodata.Add(bkg, -1)
for i in range(1, data.GetNbinsX()):
    ratiodata.SetBinError(i, data.GetBinError(i))

lower_pad.Add[TObjectDrawable]().Set(ratiodata, "E SAME")

# Add RLegend
legend = upper_pad.Draw[RLegend](RPadPos(-0.05, 0.05), RPadExtent(0.3, 0.4))
legend.text.size = 0.05
legend.text.align = 32
legend.text.font = 4
legend.border.style = 0
legend.border.width = 0
legend.fill.style = 0

legend.AddEntry(data_drawable, "Data")
legend.AddEntry(bkg_drawable, "Background")
legend.AddEntry(fit_drawable, "Signal + Bkg.")
legend.AddEntry(higgs_drawable, "Signal")

# Add ATLAS labels
lbl1 = upper_pad.Draw[RText](RPadPos(0.05, 0.88), "ATLAS")
lbl1.onframe = True
lbl1.text.font = 7
lbl1.text.size = 0.05
lbl1.text.align = 11
lbl2 = upper_pad.Draw[RText](RPadPos(0.05 + 0.16, 0.88), "Open Data")
lbl2.onframe = True
lbl2.text.font = 4
lbl2.text.size = 0.05
lbl2.text.align = 11
lbl3 = upper_pad.Draw[RText](RPadPos(0.05, 0.82), "#sqrt{s} = 13 TeV, 10 fb^{-1}")
lbl3.onframe = True
lbl3.text.font = 4
lbl3.text.size = 0.04
lbl3.text.align = 11

# show canvas finally
c.SetSize(700, 780)
c.Show()

# Save plot in PNG file
c.SaveAs("df104.png")
print("Saved figure to df104.png")
