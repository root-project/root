## \file
## \ingroup tutorial_dataframe
## \notebook -draw
## Perform systematic variations on event data and compare them in a histogram.
##
## This tutorial shows how to perform systematic variations on the muon efficiency scale factors,
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

# We must further apply a MC correction for the ZZ decay due to missing gg->ZZ process
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
                      .Define("goodlep_pt", "lep_pt[good_lep]")\
                      .Define("goodlep_eta", "lep_eta[good_lep]")\
                      .Define("goodlep_phi", "lep_phi[good_lep]")\
                      .Define("goodlep_E", "lep_E[good_lep]")\
                      .Filter("goodlep_pt[0] > 25000 && goodlep_pt[1] > 15000 && goodlep_pt[2] > 10000");\

# In addition, we must define the "weight" column to be varied for the MC datasets.
df_with_weight = df_4l_mc.Define("weight", "scaleFactor_ELE * scaleFactor_MUON * scaleFactor_LepTRIGGER * scaleFactor_PILEUP * mcWeight * xsecs / sumws * {}".format(lumi))

# Before using the Vary method, we must define functions to get the variations of the lepton scale factors.
# The variations of these scale factors depend on the kinematics of the leptons involved, and are thus
# different, depending on the lepton's transverse momentum (p_{T}) and pseudorapidity (#eta).
# In particular, we are interested in the systematic uncertainty. 
# It is define as the quadratic sum of the systematic and statistical uncertainties of the lepton Scale factors.
# Since the positive and the negative variations may be different, we define two respective functions.
ROOT.gInterpreter.Declare("""
Float_t getVarUp(const Float_t LepEta, const Float_t LepPhi)
{
    // We get the systematic and statistical uncertainties on the scale factors.
    static TFile f("Reco_HighPt_Z.root");
    static TH2F *h1 = (TH2F*)f.Get("SF_SYS_1UP_2018");
    static TH2F *h2 = (TH2F*)f.Get("SF_STAT_1UP_2018");

    // The Phi and Eta values can be extracted from the 2D histogram.
    float_t v1  = h1->GetBinContent(h1->GetXaxis()->FindBin(LepPhi),h1->GetYaxis()->FindBin(LepEta));
    float_t v2  = h2->GetBinContent(h2->GetXaxis()->FindBin(LepPhi),h2->GetYaxis()->FindBin(LepEta));
    float_t v = TMath::Sqrt(v1*v1 + v2*v2);
    return v;
}
""")

ROOT.gInterpreter.Declare("""
Float_t getVarDown(const Float_t LepEta, const Float_t LepPhi)
{

    static TFile f("Reco_HighPt_Z.root");
    static TH2F *h3 = (TH2F*)f.Get("SF_SYS_1DN_2018");
    static TH2F *h4 = (TH2F*)f.Get("SF_STAT_1DN_2018");
    float_t v3  = h3->GetBinContent(h3->GetXaxis()->FindBin(LepPhi),h3->GetYaxis()->FindBin(LepEta));
    float_t v4  = h4->GetBinContent(h4->GetXaxis()->FindBin(LepPhi),h4->GetYaxis()->FindBin(LepEta));
    float_t v = TMath::Sqrt(v3*v3 + v4*v4);
    return v;
}
""")

# Now we are ready to perform systematic variations on the MC datasets using the Vary method.
# For this, we need to define a function that implements our variations.
# The input consists of the column to be varied, here the muon scale factor, a lamdbda function 
# to compute the variations to be performed, here a scaling by the scale factor variations, 
# and the new output columns that contain the varied values of the given column. 
ROOT.gInterpreter.Declare("""
using namespace ROOT::VecOps;
RVec<double> var(const double &x, RVec<float_t> &eta, RVec<float_t> &phi)
{
    const long unsigned int N = eta.size();
    float_t u = 0; float_t d = 0;
    for (std::size_t i = 0; i < N; ++i) {
    u += getVarUp(eta[i], phi[i]);
    d += getVarDown(eta[i], phi[i]);
    };
    float_t up = 1 + u;
    float_t down = 1 - d;
    return RVec<double>({up * x, down * x});
    return RVec<double>({up * x, down * x});
}
""")

df_with_variations_mc = df_with_weight.Vary("weight", "var(weight, goodlep_eta, goodlep_phi)", {"up", "down"})

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

# Since we want to see how the histogram of the invariant mass changes, we must compute that next.
df_mass_var_mc = df_with_variations_mc.Define("m4l", "ComputeInvariantMass(goodlep_pt, goodlep_eta, goodlep_phi, goodlep_E)")\
                                      .Histo1D(ROOT.RDF.TH1DModel("Invariant Mass", "m4l", 24, 80, 170), "m4l", "weight")

# Now, we are ready to plot the variations for this histogram using the VariationsFor method.
c_mc = ROOT.TCanvas("c_mc", " ", 600, 600) # Create a new canvas to plot the variations.
histos_mc = ROOT.RDF.Experimental.VariationsFor(df_mass_var_mc)
(histos_mc)["weight:up"].SetLabelSize(0.04)
(histos_mc)["weight:up"].SetTitleSize(0.04)
(histos_mc)["weight:up"].SetMaximum(26)
(histos_mc)["weight:up"].SetTitle("Systematic variations of the muon scale factor")
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

# To compare the variations against the real data, we also need to plot that.
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

# It can be seen that the systematic varations are very small and decrease at higher energies.
c_mc.SaveAs("SF_varied_4L_Decay.png")
