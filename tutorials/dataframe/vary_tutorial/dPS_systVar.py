## \file
## \ingroup tutorial_dataframe
## \notebook -draw
## Perform systematic variations on event data and compare them in a histogram.
##
## This tutorial shows how to perform systematic variations on the lepton scale factors,
## and see how these would affect events resulting in a four-lepton decay, in particular including
## those contributions that result from a Higgs decay.
## Lepton scale factors are applied to leptons in MC simulations to correct for the differences in the
## reconstruction, identification and trigger mechanisms used for MC samples and real data.
## Using the Vary method, we can look at the how the uncertainties of these scale factors scale with the 
## invariant mass of the respective decay.
## The event selection is based on the tutorial (https://root.cern.ch/doc/master/df106__HiggsToFourLeptons_8py.html),
## where the different contributions to this four-lepton decay are analyzed in greater detail.
##
## \macro_code
## \macro_image
##
## \date August 2022
## \author Julia Mathe (CERN)

import ROOT

# Enable Multithreading
ROOT.EnableImplicitMT();    

# Load the input data from real data and MC simulations.
# df_data = ROOT::RDataFrame("mini;", ["https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/Data/data_A.4lep.root",
#                                            "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/Data/data_B.4lep.root",
#                                            "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/Data/data_C.4lep.root",
#                                            "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/Data/data_D.4lep.root"])
#
# df_mc = ROOT::RDataFrame("mini;" ["https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/MC/mc_345060.ggH125_ZZ4lep.4lep.root", 
#                                        "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/MC/mc_344235.VBFH125_ZZ4lep.4lep.root",
#                                        "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/MC/mc_363490.llll.4lep.root",
#                                        "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/MC/mc_361106.Zee.4lep.root",
#                                        "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/MC/mc_361107.Zmumu.4lep.root"])

# On desktop using downloaded files:
df_data = ROOT.RDataFrame("mini;", ["data_A.4lep.root",
                                    "data_B.4lep.root",
                                    "data_C.4lep.root",
                                    "data_D.4lep.root"])

# Use DefinePerSample to encode information for each MC dataset
df_mc = ROOT.RDataFrame("mini;", ["mc_345060.ggH125_ZZ4lep.4lep.root", 
                                  "mc_344235.VBFH125_ZZ4lep.4lep.root",
                                  "mc_363490.llll.4lep.root",
                                  "mc_361106.Zee.4lep.root",
                                  "mc_361107.Zmumu.4lep.root"])

# The cross sections and sums of weights necessary to compute the weight factors for MC processes are 
# different for each MC dataset. For simplicity, it is best to hardcode these values (that can be found 
# in json-files correspoding to the datasets) and the luminosity, too.
# This can easily be done using the DefinePerSample method. 

lumi = 10064.0

ROOT.gInterpreter.Declare(
"""
float xsecs(unsigned int slot, const ROOT::RDF::RSampleInfo &id){
                if (id.Contains("mc_345060.ggH125_ZZ4lep.4lep")){return 0.0060239f;}
                if (id.Contains("mc_344235.VBFH125_ZZ4lep.4lep.root")){return 0.0004633012f;}
                if (id.Contains("mc_363490.llll.4lep.root")){return 1.2578f;}
                if (id.Contains("mc_361106.Zee.4lep.root")){return 1950.5295f;}
                else {return  1950.6321f;}
}
""")
df_mc_xsecs = df_mc.DefinePerSample("xsecs", "xsecs(rdfslot_, rdfsampleinfo_)")

ROOT.gInterpreter.Declare(
"""
float sumws(unsigned int slot, const ROOT::RDF::RSampleInfo &id){
                if (id.Contains("mc_345060.ggH125_ZZ4lep.4lep")){return 27881776.6536f;}
                if (id.Contains("mc_344235.VBFH125_ZZ4lep.4lep.root")){return 3680490.83243f;}
                if (id.Contains("mc_363490.llll.4lep.root")){return 7538705.8077f;}
                if (id.Contains("mc_361106.Zee.4lep.root")){return 150277594200.0f;}
                else {return 147334691090.0f;}
}
""")
df_mc_sumws = df_mc_xsecs.DefinePerSample("sumws", "sumws(rdfslot_, rdfsampleinfo_)")

# We must further apply a MC correction for the ZZ decay due to missing gg->ZZ processes.
ROOT.gInterpreter.Declare(
"""
float scale(unsigned int slot, const ROOT::RDF::RSampleInfo &id){
                if (id.Contains("mc_363490.llll.4lep.root")){return 1.3f;}
                else {return 1.0f;}
}
""")

df_mc_scale = df_mc_sumws.DefinePerSample("scale", "scale(rdfslot_, rdfsampleinfo_)")

# Now we can select the events that pass the electron or muon trigger, and that also have four good leptons 
# that conserve carge and lepton numbers. In addition, we apply extra cuts for the lepton flavour and 
# define new columns with the kinematics of good leptons. Lastly, we apply a high momentum cut.

# This function selects only events with "good" leptons that meet specific preselection criteria that can 
# be found at (http://opendata.atlas.cern/release/2020/documentation/datasets/objects.html).
ROOT.gInterpreter.Declare("""
using cRVecF = const ROOT::RVecF &;
bool GoodElectronsAndMuons(const ROOT::RVecI & type, cRVecF pt, cRVecF eta, cRVecF phi, cRVecF e, cRVecF trackd0pv, cRVecF tracksigd0pv, cRVecF z0)
{
    for (size_t i = 0; i < type.size(); i++) {
        ROOT::Math::PtEtaPhiEVector p(pt[i] / 1000.0, eta[i], phi[i], e[i] / 1000.0);
        if (type[i] == 11) {
            if (pt[i] < 7000 || abs(eta[i]) > 2.47 || abs(trackd0pv[i] / tracksigd0pv[i]) > 5 || abs(z0[i] * sin(p.Theta())) > 0.5) return false;
        } else {
            if (abs(trackd0pv[i] / tracksigd0pv[i]) > 5 || abs(z0[i] * sin(p.Theta())) > 0.5) return false;
        }
    }
    return true;
}
""")

df_4l_data = df_data.Filter("trigE || trigM")\
                    .Define("good_lep", "abs(lep_eta) < 2.5 && lep_pt > 5000 && lep_ptcone30 / lep_pt < 0.3 && lep_etcone20 / lep_pt < 0.3")\
                    .Filter("Sum(good_lep) == 4")\
                    .Filter("Sum(lep_charge[good_lep]) == 0")\
                    .Define("goodlep_sumtypes", "Sum(lep_type[good_lep])")\
                    .Filter("goodlep_sumtypes == 44 || goodlep_sumtypes == 52 || goodlep_sumtypes == 48")\
                    .Filter("GoodElectronsAndMuons(lep_type[good_lep], lep_pt[good_lep], lep_eta[good_lep], lep_phi[good_lep], lep_E[good_lep], lep_trackd0pvunbiased[good_lep], lep_tracksigd0pvunbiased[good_lep], lep_z0[good_lep])")\
                    .Define("goodlep_pt", "lep_pt[good_lep]")\
                    .Define("goodlep_eta", "lep_eta[good_lep]")\
                    .Define("goodlep_phi", "lep_phi[good_lep]")\
                    .Define("goodlep_E", "lep_E[good_lep]")\
                    .Filter("goodlep_pt[0] > 25000 && goodlep_pt[1] > 15000 && goodlep_pt[2] > 10000");\

df_4l_mc = df_mc_scale.Filter("trigE || trigM")\
                      .Define("good_lep", "abs(lep_eta) < 2.5 && lep_pt > 5000 && lep_ptcone30 / lep_pt < 0.3 && lep_etcone20 / lep_pt < 0.3")\
                      .Filter("Sum(good_lep) == 4")\
                      .Filter("Sum(lep_charge[good_lep]) == 0")\
                      .Define("goodlep_sumtypes", "Sum(lep_type[good_lep])")\
                      .Filter("goodlep_sumtypes == 44 || goodlep_sumtypes == 52 || goodlep_sumtypes == 48")\
                      .Filter("GoodElectronsAndMuons(lep_type[good_lep], lep_pt[good_lep], lep_eta[good_lep], lep_phi[good_lep], lep_E[good_lep], lep_trackd0pvunbiased[good_lep], lep_tracksigd0pvunbiased[good_lep], lep_z0[good_lep])")\
                      .Define("goodlep_type", "lep_type[good_lep]")\
                      .Define("goodlep_pt", "lep_pt[good_lep]")\
                      .Define("goodlep_eta", "lep_eta[good_lep]")\
                      .Define("goodlep_phi", "lep_phi[good_lep]")\
                      .Define("goodlep_E", "lep_E[good_lep]")\
                      .Filter("goodlep_pt[0] > 25000 && goodlep_pt[1] > 15000 && goodlep_pt[2] > 10000");\

# In addition, we must define the "weight" column to be varied for the MC datasets.
df_with_weight = df_4l_mc.Define("weight", "scaleFactor_ELE * scaleFactor_MUON * scaleFactor_LepTRIGGER * scaleFactor_PILEUP * scale * mcWeight * xsecs / sumws * {}".format(lumi))

# Before using the Vary method, we must define functions to get the variations of the lepton scale factors.
# The variations of these scale factors depend on the kinematics of the leptons involved, and are thus
# different, depending on the lepton's transverse momentum (p_{T}) and pseudorapidity (#eta).
# In particular, we are interested in the systematic uncertainty. 
# It is define as the quadratic sum of the systematic and statistical uncertainties of the lepton Scale factors.
# The uncertainties for electrons were taken from the analysis in https://doi.org/10.48550/arXiv.1908.00005. 
# For muons, the uncertainties are negligible (cv. https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/MUON-2018-03/).

# The uncertainties of the lepton scale factors can be read from datapoints of publicly available plots and interpolate linearly using the ROOT::Math::Interpolator.
# Now we are ready to perform systematic variations on the MC datasets using the Vary method. 
# The input consists of the column to be varied, a function to compute the variations to be performed, here expressed through a C++ VaryHelper,
# and the new output columns that contain the varied values of the given column. 
# The VaryHelper is necessary to perform the linear interpolation of the datapoints, so that the uncertainties can be expressed as functions of the lepton type and momenta.
# The average of the electron uncertainties is used for correction. 
ROOT.gInterpreter.Declare("""
#include <Math/Interpolator.h>
using namespace ROOT::VecOps;
using namespace ROOT::Math;

class VaryHelper 
{
    std::vector<double> x = {5.50 * 10e2, 5.52 * 10e2, 12.54 * 10e2, 17.43 * 10e2, 22.40 * 10e2, 27.48 * 10e2, 30 * 10e2, 10000 * 10e2};
    std::vector<double> y = {0.06628, 0.06395, 0.06396, 0.03372, 0.02441, 0.01403, 0, 0};
    Interpolator inter;
public:
    VaryHelper() : inter(x, y, Interpolation::kLINEAR) {}

    RVec<double> operator()(const double &w, RVec<float> &pt, RVec<unsigned int> &type)
    {
        const auto v = Mean(Map(pt[type == 11], [this](auto p) {return this->inter.Eval(p); }));                                                                                              
        return RVec<double>{(1 + v) * w, (1 - v) * w};
    }
};

VaryHelper myHelper;
""")
df_with_variations_mc = df_with_weight.Vary("weight", "myHelper(weight, goodlep_pt, goodlep_type)", ["up", "down"])

ROOT.gInterpreter.Declare("""
float ComputeInvariantMass(cRVecF pt, cRVecF eta, cRVecF phi, cRVecF e)
{
    ROOT::Math::PtEtaPhiEVector p1(pt[0], eta[0], phi[0], e[0]);
    ROOT::Math::PtEtaPhiEVector p2(pt[1], eta[1], phi[1], e[1]);
    ROOT::Math::PtEtaPhiEVector p3(pt[2], eta[2], phi[2], e[2]);
    ROOT::Math::PtEtaPhiEVector p4(pt[3], eta[3], phi[3], e[3]);
    return (p1 + p2 + p3 + p4).M() / 1000;
}
""")

# Since we want to see how the histogram of the invariant mass changes, we compute it.
df_mass_var_mc = df_with_variations_mc.Define("m4l", "ComputeInvariantMass(goodlep_pt, goodlep_eta, goodlep_phi, goodlep_E)")\
                                      .Histo1D(ROOT.RDF.TH1DModel("Invariant Mass", "m4l", 24, 80, 170), "m4l", "weight")


# Now, we are ready to plot the variations for this histogram using the VariationsFor method.
c_mc = ROOT.TCanvas("c_mc", " ", 600, 600)
histos_mc = ROOT.RDF.Experimental.VariationsFor(df_mass_var_mc)
(histos_mc)["weight:up"].SetLabelSize(0.04)
(histos_mc)["weight:up"].SetTitleSize(0.04)
(histos_mc)["weight:up"].SetMaximum(26)
(histos_mc)["weight:up"].SetTitle("")
(histos_mc)["weight:up"].GetXaxis().SetTitle("m_{4l}^{H#rightarrow ZZ} [GeV]")
(histos_mc)["weight:up"].GetYaxis().SetTitle("Events")
(histos_mc)["weight:up"].SetStats(0)
(histos_mc)["weight:up"].GetYaxis().ChangeLabel(1, -1, 0)
(histos_mc)["weight:up"].Draw("HIST")
(histos_mc)["weight:down"].SetLineColor(3)
(histos_mc)["weight:down"].SetStats(0)
(histos_mc)["weight:down"].Draw("HIST sames")
(histos_mc)["nominal"].SetLineColor(2)
(histos_mc)["nominal"].SetStats(0)
(histos_mc)["nominal"].Draw("HIST sames")

# We can also print the variations we have defined.
for k in histos_mc.GetKeys():
    print(k)

# To compare the variations against the real data, we need to plot that too.
df_h_mass_data = df_4l_data.Define("m4l", "ComputeInvariantMass(goodlep_pt, goodlep_eta, goodlep_phi, goodlep_E)")\
                           .Histo1D(ROOT.RDF.TH1DModel("Invariant Mass", "m4l", 24, 80, 170), "m4l")
h_mass_data = df_h_mass_data.GetPtr()
h_mass_data.SetLineColor(1)
h_mass_data.SetStats(0)
h_mass_data.Draw("E sames")

# Add a legend to the plot.
legend = ROOT.TLegend(0.50, 0.48, 0.9, 0.7)
legend.SetTextFont(42)
legend.SetFillStyle(0)
legend.SetBorderSize(0)
legend.SetTextSize(0.04)
legend.SetTextAlign(32)
legend.AddEntry(((histos_mc)["weight:down"]), "variation: down", "l")
legend.AddEntry(((histos_mc)["weight:up"]), "variation: up", "l")
legend.AddEntry(((histos_mc)["nominal"]), "nominal", "l")
legend.AddEntry(h_mass_data, "Data", "lep")
legend.Draw("Same")

# Add some extra labels referring to the respective Run of the ATLAS experiment.
atlas_label = ROOT.TLatex()
atlas_label.SetTextFont(72)
atlas_label.SetTextSize(0.045)
atlas_label.DrawLatexNDC(0.21, 0.81, "ATLAS")
data_label = ROOT.TLatex()
data_label.SetTextFont(42)
atlas_label.SetTextSize(0.045)
data_label.DrawLatexNDC(0.21 + 0.16, 0.81, "Open Data")
header = ROOT.TLatex()
data_label.SetTextFont(42)
header.SetTextSize(0.04)
header.DrawLatexNDC(0.21, 0.75, "#sqrt{s} = 13 TeV, 10 fb^{-1}")

# Save the plot.
c_mc.SaveAs("SF_varied_4L_Decay.png")
# Note that the lepton scale factor uncertainties show signficiant effects at lower masses and decrease with higher energies.
