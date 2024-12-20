## \file
## \ingroup tutorial_dataframe
## \notebook -draw
## The Higgs to four lepton analysis from the ATLAS Open Data release of 2020, with RDataFrame.
##
## This tutorial is the Higgs to four lepton analysis from the ATLAS Open Data release in 2020
## (http://opendata.atlas.cern/release/2020/documentation/). The data was taken with the ATLAS detector
## during 2016 at a center-of-mass energy of 13 TeV. The decay of the Standard Model Higgs boson
## to two Z bosons and subsequently to four leptons is called the "golden channel". The selection leads
## to a narrow invariant mass peak on top a relatively smooth and small background, revealing
## the Higgs at 125 GeV.
## Systematic errors for the MC scale factors are computed and the Vary function of RDataFrame is used for plotting.
## The analysis is translated to an RDataFrame workflow processing about 300 MB of simulated events and data.
##
## See the [corresponding spec json file](https://github.com/root-project/root/blob/master/tutorials/analysis/dataframe/df106_HiggsToFourLeptons_spec.json).
##
## \macro_image
## \macro_code
## \macro_output
##
## \date March 2020, August 2022, August 2023
## \authors Stefan Wunsch (KIT, CERN), Julia Mathe (CERN), Marta Czurylo (CERN)

import ROOT
import os

# Enable Multi-threaded mode
ROOT.EnableImplicitMT()

# Create the RDataFrame from the spec json file. The df106_HiggsToFourLeptons_spec.json is provided in the same folder as this tutorial
dataset_spec = os.path.join(ROOT.gROOT.GetTutorialsDir(), "analysis", "dataframe", "df106_HiggsToFourLeptons_spec.json")
df = ROOT.RDF.Experimental.FromSpec(dataset_spec)  # Creates a single dataframe for all the samples

# Add the ProgressBar feature
ROOT.RDF.Experimental.AddProgressBar(df)

# Access metadata information that is stored in the JSON config file of the RDataFrame.
# The metadata contained in the JSON file is accessible within a `DefinePerSample` call, through the `RSampleInfo` class.
df = df.DefinePerSample("xsecs", 'rdfsampleinfo_.GetD("xsecs")')
df = df.DefinePerSample("lumi", 'rdfsampleinfo_.GetD("lumi")')
df = df.DefinePerSample("sumws", 'rdfsampleinfo_.GetD("sumws")')
df = df.DefinePerSample("sample_category", 'rdfsampleinfo_.GetS("sample_category")')

# We must further apply an MC correction for the ZZ decay due to missing gg->ZZ processes.
ROOT.gInterpreter.Declare(
    """
float scale(unsigned int slot, const ROOT::RDF::RSampleInfo &id){
                return id.Contains("mc_363490.llll.4lep.root") ? 1.3f : 1.0f;
}
"""
)

df = df.DefinePerSample("scale", "scale(rdfslot_, rdfsampleinfo_)")

# Select events for the analysis
ROOT.gInterpreter.Declare(
    """
using ROOT::RVecF;
using ROOT::RVecI;
bool GoodElectronsAndMuons(const RVecI &type, const RVecF &pt, const RVecF &eta, const RVecF &phi, const RVecF &e, const RVecF &trackd0pv, const RVecF &tracksigd0pv, const RVecF &z0)
{
    for (size_t i = 0; i < type.size(); i++) {
        ROOT::Math::PtEtaPhiEVector p(0.001*pt[i], eta[i], phi[i], 0.001*e[i]);
        if (type[i] == 11) {
            if (pt[i] < 7000 || abs(eta[i]) > 2.47 || abs(trackd0pv[i] / tracksigd0pv[i]) > 5 || abs(z0[i] * sin(p.Theta())) > 0.5) return false;
        } else {
            if (abs(trackd0pv[i] / tracksigd0pv[i]) > 5 || abs(z0[i] * sin(p.Theta())) > 0.5) return false;
        }
    }
    return true;
}
"""
)

# Select electron or muon trigger
df = df.Filter("trigE || trigM")

# Select events with exactly four good leptons conserving charge and lepton numbers
# Note that all collections are RVecs and good_lep is the mask for the good leptons.
# The lepton types are PDG numbers and set to 11 or 13 for an electron or muon
# irrespective of the charge.

df = (
    df.Define(
        "good_lep",
        "abs(lep_eta) < 2.5 && lep_pt > 5000 && lep_ptcone30 / lep_pt < 0.3 && lep_etcone20 / lep_pt < 0.3",
    )
    .Filter("Sum(good_lep) == 4")
    .Filter("Sum(lep_charge[good_lep]) == 0")
    .Define("goodlep_sumtypes", "Sum(lep_type[good_lep])")
    .Filter("goodlep_sumtypes == 44 || goodlep_sumtypes == 52 || goodlep_sumtypes == 48")
)

# Apply additional cuts depending on lepton flavour
df = df.Filter(
    "GoodElectronsAndMuons(lep_type[good_lep], lep_pt[good_lep], lep_eta[good_lep], lep_phi[good_lep], lep_E[good_lep], lep_trackd0pvunbiased[good_lep], lep_tracksigd0pvunbiased[good_lep], lep_z0[good_lep])"
)

# Create new columns with the kinematics of good leptons
df = (
    df.Define("goodlep_pt", "lep_pt[good_lep]")
    .Define("goodlep_eta", "lep_eta[good_lep]")
    .Define("goodlep_phi", "lep_phi[good_lep]")
    .Define("goodlep_E", "lep_E[good_lep]")
    .Define("goodlep_type", "lep_type[good_lep]")
)

# Select leptons with high transverse momentum
df = df.Filter("goodlep_pt[0] > 25000 && goodlep_pt[1] > 15000 && goodlep_pt[2] > 10000")

# Reweighting of the samples is different for "data" and "MC". This is the function to add reweighting for MC samples
ROOT.gInterpreter.Declare(
    """
double weights(float scaleFactor_1, float scaleFactor_2, float scaleFactor_3, float scaleFactor_4, float scale, float mcWeight, double xsecs, double sumws, double lumi)
{
    return scaleFactor_1 * scaleFactor_2 * scaleFactor_3 * scaleFactor_4 * scale * mcWeight * xsecs / sumws * lumi;
}
"""
)

# Use DefinePerSample to define which samples are MC and hence need reweighting
df = df.DefinePerSample("isMC", 'rdfsampleinfo_.Contains("mc")')
df = df.Define(
    "weight",
    "double x; return isMC ? weights(scaleFactor_ELE, scaleFactor_MUON, scaleFactor_LepTRIGGER, scaleFactor_PILEUP, scale, mcWeight, xsecs, sumws, lumi) :  1.;",
)

# Compute invariant mass of the four lepton system
ROOT.gInterpreter.Declare(
    """
float ComputeInvariantMass(RVecF pt, RVecF eta, RVecF phi, RVecF e)
{
    ROOT::Math::PtEtaPhiEVector p1{pt[0], eta[0], phi[0], e[0]};
    ROOT::Math::PtEtaPhiEVector p2{pt[1], eta[1], phi[1], e[1]};
    ROOT::Math::PtEtaPhiEVector p3{pt[2], eta[2], phi[2], e[2]};
    ROOT::Math::PtEtaPhiEVector p4{pt[3], eta[3], phi[3], e[3]};
    return 0.001 * (p1 + p2 + p3 + p4).M();
}
"""
)

df = df.Define("m4l", "ComputeInvariantMass(goodlep_pt, goodlep_eta, goodlep_phi, goodlep_E)")

# Save data for statistical analysis tutorial (rf618_mixture_models.py) 
df.Snapshot("tree", ROOT.gROOT.GetTutorialDir().Data() + "/analysis/dataframe/df106_HiggsToFourLeptons.root", ["m4l", "sample_category", "weight"])

# Book histograms for the four different samples: data, higgs, zz and other (this is specific to this particular analysis)
histos = []
for sample_category in ["data", "higgs", "zz", "other"]:
    histos.append(
        df.Filter(f'sample_category == "{sample_category}"').Histo1D(
            ROOT.RDF.TH1DModel(f"{sample_category}", "m4l", 24, 80, 170),
            "m4l",
            "weight",
        )
    )

# Evaluate the systematic uncertainty

# The systematic uncertainty in this analysis is the MC scale factor uncertainty that depends on lepton
# kinematics such as pT or pseudorapidity.
# Muons uncertainties are negligible, as stated in https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/MUON-2018-03/.
# Electrons uncertainties are evaluated based on the plots available in https://doi.org/10.48550/arXiv.1908.00005.
# The uncertainties are linearly interpolated, using the `TGraph::Eval()` method, to cover a range of pT values covered by the analysis.

# Create a VaryHelper to interpolate the available data.
ROOT.gInterpreter.Declare(
    """
using namespace ROOT::VecOps;

class VaryHelper
{
    const std::vector<double> x{5.50e3, 5.52e3, 12.54e3, 17.43e3, 22.40e3, 27.48e3, 30e3, 10000e3};
    const std::vector<double> y{0.06628, 0.06395, 0.06396, 0.03372, 0.02441, 0.01403, 0, 0};
    TGraph graph;

public:
    VaryHelper() : graph(x.size(), x.data(), y.data()) {}
    RVec<double> operator()(const double &w, const RVecF &pt, const RVec<unsigned int> &type)
    {
        const auto v = Mean(Map(pt[type == 11], [this](auto p)
        {return this->graph.Eval(p); })
        );
        return RVec<double>{(1 + v) * w, (1 - v) * w};
    }
};

VaryHelper variationsFactory;
"""
)

# Use the Vary method to add the systematic variations to the total MC scale factor ("weight") of the analysis.
df_variations_mc = (
    df.Filter("isMC == true")
    .Vary("weight", "variationsFactory(weight, goodlep_pt, goodlep_type)", ["up", "down"])
    .Histo1D(ROOT.RDF.TH1DModel("Invariant Mass", "m4l", 24, 80, 170), "m4l", "weight")
)
histos_mc = ROOT.RDF.Experimental.VariationsFor(df_variations_mc)

# We reached the end of the analysis part. We now evaluate the total MC uncertainty based on the variations.
# No computation graph was triggered yet, we trigger the computation graph for all histograms at once now,
# by calling 'histos_mc["nominal"].GetXaxis()'.
# Note, in this case the uncertainties are symmetric.
for i in range(0, histos_mc["nominal"].GetXaxis().GetNbins()):
    (
        histos_mc["nominal"].SetBinError(
            i, (histos_mc["weight:up"].GetBinContent(i) - histos_mc["nominal"].GetBinContent(i))
        )
    )

# Make the plot of the data, individual MC contributions and the total MC scale factor systematic variations.

# Set styles
ROOT.gROOT.SetStyle("ATLAS")

# Create canvas with pad
c1 = ROOT.TCanvas("c", "", 600, 600)
pad = ROOT.TPad("upper_pad", "", 0, 0, 1, 1)
pad.SetTickx(False)
pad.SetTicky(False)
pad.Draw()
pad.cd()

# Draw stack with MC contributions
stack = ROOT.THStack()

# Retrieve values of the data and MC histograms in order to plot them.
# Draw cloned histograms to preserve graphics when original objects goes out of scope
# Note: GetValue() action operation is performed after all lazy actions of the RDF were defined first.
h_data = histos[0].GetValue().Clone()
h_higgs = histos[1].GetValue().Clone()
h_zz = histos[2].GetValue().Clone()
h_other = histos[3].GetValue().Clone()

for h, color in zip([h_other, h_zz, h_higgs], [ROOT.kViolet - 9, ROOT.kAzure - 9, ROOT.kRed + 2]):
    h.SetLineWidth(1)
    h.SetLineColor(1)
    h.SetFillColor(color)
    stack.Add(h)

stack.Draw("HIST")
stack.GetXaxis().SetLabelSize(0.04)
stack.GetXaxis().SetTitleSize(0.045)
stack.GetXaxis().SetTitleOffset(1.3)
stack.GetXaxis().SetTitle("m_{4l}^{H#rightarrow ZZ} [GeV]")
stack.GetYaxis().SetLabelSize(0.04)
stack.GetYaxis().SetTitleSize(0.045)
stack.GetYaxis().SetTitle("Events")
stack.SetMaximum(35)
stack.GetYaxis().ChangeLabel(1, -1, 0)

# Draw MC scale factor and variations
histos_mc["nominal"].SetFillColor(ROOT.kBlack)
histos_mc["nominal"].SetFillStyle(3254)
h_nominal = histos_mc["nominal"].DrawClone("E2 same")
histos_mc["weight:up"].SetLineColor(ROOT.kGreen + 2)
h_weight_up = histos_mc["weight:up"].DrawClone("HIST SAME")
histos_mc["weight:down"].SetLineColor(ROOT.kBlue + 2)
h_weight_down = histos_mc["weight:down"].DrawClone("HIST SAME")

# Draw data histogram
h_data.SetMarkerStyle(20)
h_data.SetMarkerSize(1.2)
h_data.SetLineWidth(2)
h_data.SetLineColor(ROOT.kBlack)
h_data.Draw("E SAME")  # Draw raw data with errorbars

# Add legend
legend = ROOT.TLegend(0.57, 0.65, 0.94, 0.94)
legend.SetTextFont(42)
legend.SetFillStyle(0)
legend.SetBorderSize(0)
legend.SetTextSize(0.025)
legend.SetTextAlign(32)
legend.AddEntry(h_data, "Data", "lep")
legend.AddEntry(h_higgs, "Higgs MC", "f")
legend.AddEntry(h_zz, "ZZ MC", "f")
legend.AddEntry(h_other, "Other MC", "f")
legend.AddEntry(h_weight_down, "Total MC Variations Down", "l")
legend.AddEntry(h_weight_up, "Total MC Variations Up", "l")
legend.AddEntry(h_nominal, "Total MC Uncertainty", "f")
legend.Draw()

text = ROOT.TLatex()
text.SetTextFont(72)
text.SetTextSize(0.04)
text.DrawLatexNDC(0.19, 0.85, "ATLAS")
text.SetTextFont(42)
text.DrawLatexNDC(0.19 + 0.15, 0.85, "Open Data")
text.SetTextSize(0.035)
text.DrawLatexNDC(0.21, 0.80, "#sqrt{s} = 13 TeV, 10 fb^{-1}")

c1.Update()

# Save the plot
c1.SaveAs("df106_HiggsToFourLeptons_python.png")
print("Saved figure to df106_HiggsToFourLeptons_python.png")
