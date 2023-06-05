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
##
## The analysis is translated to a RDataFrame workflow processing about 300 MB of simulated events and data.
##
## \macro_image
## \macro_code
## \macro_output
##
## \date March 2020
## \author Stefan Wunsch (KIT, CERN)
## \date May 2023
## \author Marta Czurylo (CERN)

import ROOT
import os

# Create the RDataFrame from the spec json file. The df106_HiggsToFourLeptons_spec.json is provided in the same folder as this tutorial

my_spec = os.path.join(ROOT.gROOT.GetTutorialsDir(), "dataframe", "df106_HiggsToFourLeptons_spec.json")

df = ROOT.RDF.Experimental.FromSpec(my_spec) # Creates a single dataframe for all the samples

# Access metadata information that is stored in the JSON config file of the RDataFrame
# The metadata contained in the JSON file is accessible within a `DefinePerSample` call, through the `RDFSampleInfo` class

df = df.DefinePerSample("xsecs", 'rdfsampleinfo_.GetD("xsecs")')
df = df.DefinePerSample("lumi", 'rdfsampleinfo_.GetD("lumi")') 
df = df.DefinePerSample("sumws", 'rdfsampleinfo_.GetD("sumws")')
df = df.DefinePerSample("sample_category", 'rdfsampleinfo_.GetS("sample_category")')

# Select events for the analysis

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

# Select electron or muon trigger
df = df.Filter("trigE || trigM")

# Select events with exactly four good leptons conserving charge and lepton numbers
# Note that all collections are RVecs and good_lep is the mask for the good leptons.
# The lepton types are PDG numbers and set to 11 or 13 for an electron or muon
# irrespective of the charge.

df = df.Define("good_lep", "abs(lep_eta) < 2.5 && lep_pt > 5000 && lep_ptcone30 / lep_pt < 0.3 && lep_etcone20 / lep_pt < 0.3")\
       .Filter("Sum(good_lep) == 4")\
       .Filter("Sum(lep_charge[good_lep]) == 0")\
       .Define("goodlep_sumtypes", "Sum(lep_type[good_lep])")\
       .Filter("goodlep_sumtypes == 44 || goodlep_sumtypes == 52 || goodlep_sumtypes == 48")

# Apply additional cuts depending on lepton flavour
df = df.Filter("GoodElectronsAndMuons(lep_type[good_lep], lep_pt[good_lep], lep_eta[good_lep], lep_phi[good_lep], lep_E[good_lep], lep_trackd0pvunbiased[good_lep], lep_tracksigd0pvunbiased[good_lep], lep_z0[good_lep])")

# Create new columns with the kinematics of good leptons
df = df.Define("goodlep_pt", "lep_pt[good_lep]")\
       .Define("goodlep_eta", "lep_eta[good_lep]")\
       .Define("goodlep_phi", "lep_phi[good_lep]")\
       .Define("goodlep_E", "lep_E[good_lep]")

# Select leptons with high transverse momentum
df = df.Filter("goodlep_pt[0] > 25000 && goodlep_pt[1] > 15000 && goodlep_pt[2] > 10000")

# Reweighting of the samples is different for "data" and "MC". This is the function to add reweighting for MC samples
ROOT.gInterpreter.Declare("""
double weights(float scaleFactor_1, float scaleFactor_2, float scaleFactor_3, float scaleFactor_4, float mcWeight_, double xsecs_, double sumws_, double lumi_)
{
    double weight_ = scaleFactor_1 * scaleFactor_2 * scaleFactor_3 * scaleFactor_4 * mcWeight_ * xsecs_/sumws_ * lumi_;
    return (weight_);
}
""")

# Use DefinePerSample to define which samples need reweighting
df = df.DefinePerSample("reweighting", 'rdfsampleinfo_.Contains("mc")')
df = df.Define("weight", 'double x; if (reweighting) {return weights(scaleFactor_ELE, scaleFactor_MUON, scaleFactor_LepTRIGGER, scaleFactor_PILEUP, mcWeight, xsecs, sumws, lumi);} else {return 1.;}')
                        
# Compute invariant mass of the four lepton system and make a histogram
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
df = df.Define("m4l", "ComputeInvariantMass(goodlep_pt, goodlep_eta, goodlep_phi, goodlep_E)")

# Book histograms for the four different samples: data, higgs, zz and other (this is specific to this particular analysis)
histos = [] 
for sample_category in ["data", "higgs", "zz", "other"]:
    histos.append(df.Filter(f"sample_category == \"{sample_category}\"").Histo1D(ROOT.RDF.TH1DModel(f"{sample_category}", "m4l", 24, 80, 170), "m4l", "weight"))


h_data = histos[0].GetValue()
h_higgs = histos[1].GetValue()
h_zz = histos[2].GetValue()
h_other = histos[3].GetValue()

# Apply MC correction for ZZ due to missing gg->ZZ process
h_zz.Scale(1.3) 

# Create the plot

# Set styles
ROOT.gROOT.SetStyle("ATLAS")

# Create canvas with pad
c = ROOT.TCanvas("c", "", 600, 600)
pad = ROOT.TPad("upper_pad", "", 0, 0, 1, 1)
pad.SetTickx(False)
pad.SetTicky(False)
pad.Draw()
pad.cd()

# Draw stack with MC contributions

stack = ROOT.THStack()

for h, color in zip([h_other, h_zz, h_higgs], [(155, 152, 204), (100, 192, 232), (191, 34, 41)]):
    h.SetLineWidth(1)
    h.SetLineColor(1)
    h.SetFillColor(ROOT.TColor.GetColor(*color))
    stack.Add(h)

stack.Draw("HIST")
stack.GetXaxis().SetLabelSize(0.04)
stack.GetXaxis().SetTitleSize(0.045)
stack.GetXaxis().SetTitleOffset(1.3)
stack.GetXaxis().SetTitle("m_{4l}^{H#rightarrow ZZ} [GeV]")
stack.GetYaxis().SetTitle("Events")
stack.GetYaxis().SetLabelSize(0.04)
stack.GetYaxis().SetTitleSize(0.045)
stack.SetMaximum(33)
stack.GetYaxis().ChangeLabel(1, -1, 0)

h_data.SetMarkerStyle(20)
h_data.SetMarkerSize(1.2)
h_data.SetLineWidth(2)
h_data.SetLineColor(ROOT.kBlack)
h_data.Draw("E SAME") #draw data with errorbars 

# Add legend
legend = ROOT.TLegend(0.60, 0.65, 0.92, 0.92)
legend.SetTextFont(42)
legend.SetFillStyle(0)
legend.SetBorderSize(0)
legend.SetTextSize(0.04)
legend.SetTextAlign(32)
legend.AddEntry(h_data, "Data" ,"lep")
legend.AddEntry(h_higgs, "Higgs", "f")
legend.AddEntry(h_zz, "ZZ", "f")
legend.AddEntry(h_other, "Other", "f")
legend.Draw("SAME")

# Add ATLAS label
text = ROOT.TLatex()
text.SetNDC()
text.SetTextFont(72)
text.SetTextSize(0.045)
text.DrawLatex(0.21, 0.86, "ATLAS")
text.SetTextFont(42)
text.DrawLatex(0.21 + 0.16, 0.86, "Open Data")
text.SetTextSize(0.04)
text.DrawLatex(0.21, 0.80, "#sqrt{s} = 13 TeV, 10 fb^{-1}")

# Save the plot
c.SaveAs("df106_HiggsToFourLeptons.png")
print("Saved figure to df106_HiggsToFourLeptons.png")
