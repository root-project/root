/// \file
/// \ingroup tutorial_dataframe
/// \notebook -draw
/// The Higgs to four lepton analysis from the ATLAS Open Data release of 2020, with RDataFrame.
///
/// This tutorial is the Higgs to four lepton analysis from the ATLAS Open Data release in 2020
/// (http://opendata.atlas.cern/release/2020/documentation/). The data was taken with the ATLAS detector
/// during 2016 at a center-of-mass energy of 13 TeV. The decay of the Standard Model Higgs boson
/// to two Z bosons and subsequently to four leptons is called the "golden channel". The selection leads
/// to a narrow invariant mass peak on top a relatively smooth and small background, revealing
/// the Higgs at 125 GeV.
/// The analysis is translated to an RDataFrame workflow processing about 300 MB of simulated events and data.
///
/// Lepton selection efficiency corrections ("scale factors") are applied to simulated samples to correct for the
/// differences in the trigger, reconstruction, and identification efficiencies in simulation compared to real data.
/// Systematic uncertainties for those scale factors are evaluated and the Vary function of RDataFrame is used to
/// propagate the variations to the final four leptons mass distribution.
///
/// See the [corresponding spec json file](https://github.com/root-project/root/blob/master/tutorials/analysis/dataframe/df106_HiggsToFourLeptons_spec.json).
///
/// \macro_code
/// \macro_image
///
/// \date March 2020, August 2022, August 2023
/// \authors Stefan Wunsch (KIT, CERN), Julia Mathe (CERN), Marta Czurylo (CERN)

#include "TInterpreter.h"
#include <Math/Vector4D.h>
#include <ROOT/RDFHelpers.hxx>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <TCanvas.h>
#include <TGraph.h>
#include <TH1D.h>
#include <THStack.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TProfile.h>
#include <TStyle.h>

using namespace ROOT::VecOps;
using PtEtaPhiEVectorF = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<float>>;
using ROOT::RVecF;
using ROOT::RDF::RSampleInfo;
using namespace ROOT::RDF::Experimental;

// Define functions needed in the analysis
// Select events for the analysis
bool GoodElectronsAndMuons(const ROOT::RVecI &type, const RVecF &pt, const RVecF &eta, const RVecF &phi, const RVecF &e,
                           const RVecF &trackd0pv, const RVecF &tracksigd0pv, const RVecF &z0)
{
   for (size_t i = 0; i < type.size(); i++) {
      PtEtaPhiEVectorF p(0.001 * pt[i], eta[i], phi[i], 0.001 * e[i]);
      if (type[i] == 11) {
         if (pt[i] < 7000 || abs(eta[i]) > 2.47 || abs(trackd0pv[i] / tracksigd0pv[i]) > 5 ||
             abs(z0[i] * sin(p.Theta())) > 0.5)
            return false;
      } else {
         if (abs(trackd0pv[i] / tracksigd0pv[i]) > 5 || abs(z0[i] * sin(p.Theta())) > 0.5)
            return false;
      }
   }
   return true;
}

// Compute the invariant mass of a four-lepton-system.
float ComputeInvariantMass(const RVecF &pt, const RVecF &eta, const RVecF &phi, const RVecF &e)
{
   PtEtaPhiEVectorF p1(pt[0], eta[0], phi[0], e[0]);
   PtEtaPhiEVectorF p2(pt[1], eta[1], phi[1], e[1]);
   PtEtaPhiEVectorF p3(pt[2], eta[2], phi[2], e[2]);
   PtEtaPhiEVectorF p4(pt[3], eta[3], phi[3], e[3]);
   return 0.001 * (p1 + p2 + p3 + p4).M();
}

void df106_HiggsToFourLeptons()
{

   // Enable Multithreading
   ROOT::EnableImplicitMT();

   // Create the RDataFrame from the spec json file. The df106_HiggsToFourLeptons_spec.json is provided in the same
   // folder as this tutorial
   std::string dataset_spec = gROOT->GetTutorialsDir() + std::string("/dataframe/df106_HiggsToFourLeptons_spec.json");
   ROOT::RDataFrame df = ROOT::RDF::Experimental::FromSpec(dataset_spec);

   // Add the ProgressBar feature
   ROOT::RDF::Experimental::AddProgressBar(df);

#ifndef __CLING__
   // If this tutorial is compiled, rather than run as a ROOT macro, the interpreter needs to be fed the signatures
   // of all the functions we want to JIT in our analysis, as well as any type used in those signatures.
   // clang-format off
   gInterpreter->Declare(
      "using ROOT::RVecF;"
      "bool GoodElectronsAndMuons(const ROOT::RVecI &type, const RVecF &pt, const RVecF &eta, const RVecF &phi, const RVecF &e,"
                           "const RVecF &trackd0pv, const RVecF &tracksigd0pv, const RVecF &z0);"
      "float ComputeInvariantMass(const RVecF &pt, const RVecF &eta, const RVecF &phi, const RVecF &e);"
   );
   // clang-format on
#endif

   // Perform the analysis
   // Access metadata information that is stored in the JSON config file of the RDataFrame
   // The metadata contained in the JSON file is accessible within a `DefinePerSample` call, through the `RSampleInfo`
   // class
   auto df_analysis =
      df.DefinePerSample("xsecs", [](unsigned int slot, const RSampleInfo &id) { return id.GetD("xsecs"); })
         .DefinePerSample("lumi", [](unsigned int slot, const RSampleInfo &id) { return id.GetD("lumi"); })
         .DefinePerSample("sumws", [](unsigned int slot, const RSampleInfo &id) { return id.GetD("sumws"); })
         .DefinePerSample("sample_category",
                          [](unsigned int slot, const RSampleInfo &id) { return id.GetS("sample_category"); })
         // Apply an MC correction for the ZZ decay due to missing gg->ZZ process
         .DefinePerSample("scale",
                          [](unsigned int slot, const ROOT::RDF::RSampleInfo &id) {
                             return id.Contains("mc_363490.llll.4lep.root") ? 1.3f : 1.0f;
                          })
         // Select electron or muon trigger
         .Filter("trigE || trigM")
         // Select events with exactly four good leptons conserving charge and lepton numbers
         // Note that all collections are RVecs and good_lep is the mask for the good leptons.
         // The lepton types are PDG numbers and set to 11 or 13 for an electron or muon
         // irrespective of the charge.
         .Define("good_lep",
                 "abs(lep_eta) < 2.5 && lep_pt > 5000 && lep_ptcone30 / lep_pt < 0.3 && lep_etcone20 / lep_pt < 0.3")
         .Filter("Sum(good_lep) == 4")
         .Filter("Sum(lep_charge[good_lep]) == 0")
         .Define("goodlep_sumtypes", "Sum(lep_type[good_lep])")
         .Filter("goodlep_sumtypes == 44 || goodlep_sumtypes == 52 || goodlep_sumtypes == 48")
         // Apply additional cuts depending on lepton flavour
         .Filter(
            "GoodElectronsAndMuons(lep_type[good_lep], lep_pt[good_lep], lep_eta[good_lep], lep_phi[good_lep], "
            "lep_E[good_lep], lep_trackd0pvunbiased[good_lep], lep_tracksigd0pvunbiased[good_lep], lep_z0[good_lep])")
         // Create new columns with the kinematics of good leptons
         .Define("goodlep_pt", "lep_pt[good_lep]")
         .Define("goodlep_eta", "lep_eta[good_lep]")
         .Define("goodlep_phi", "lep_phi[good_lep]")
         .Define("goodlep_E", "lep_E[good_lep]")
         .Define("goodlep_type", "lep_type[good_lep]")
         // Select leptons with high transverse momentum
         .Filter("goodlep_pt[0] > 25000 && goodlep_pt[1] > 15000 && goodlep_pt[2] > 10000")
         // Compute invariant mass
         .Define("m4l", "ComputeInvariantMass(goodlep_pt, goodlep_eta, goodlep_phi, goodlep_E)")
         // Reweighting of the samples is different for "data" and "MC"
         .DefinePerSample("reweighting", [](unsigned int slot, const RSampleInfo &id) { return id.Contains("mc"); });

   // Define the weight column (scale factor) for the MC samples
   auto df_mc = df_analysis.Filter("reweighting == true")
                   .Define("weight", ("scaleFactor_ELE * scaleFactor_MUON * scaleFactor_LepTRIGGER * "
                                      "scaleFactor_PILEUP * mcWeight * scale * xsecs / sumws * lumi"));

   // Book histograms for individual MC samples
   auto df_higgs = df_mc.Filter(R"(sample_category == "higgs")")
                      .Histo1D<float>(ROOT::RDF::TH1DModel("higgs", "m4l", 24, 80, 170), "m4l", "weight");
   auto df_zz = df_mc.Filter("sample_category == \"zz\"")
                   .Histo1D<float>(ROOT::RDF::TH1DModel("zz", "m4l", 24, 80, 170), "m4l", "weight");
   auto df_other = df_mc.Filter("sample_category == \"other\"")
                      .Histo1D<float>(ROOT::RDF::TH1DModel("other", "m4l", 24, 80, 170), "m4l", "weight");

   // Book the invariant mass histogram for the data
   auto df_h_mass_data = df_analysis.Filter("reweighting == false")
                            .Filter("sample_category == \"data\"")
                            .Define("weight_", []() { return 1; })
                            .Histo1D<float>(ROOT::RDF::TH1DModel("data", "m4l", 24, 80, 170), "m4l", "weight_");

   // Evaluate the systematic uncertainty

   // The systematic uncertainty in this analysis is the MC scale factor uncertainty that depends on lepton
   // kinematics such as pT or pseudorapidity.
   // Muons uncertainties are negligible, as stated in https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/MUON
   // Electrons uncertainties are evaluated based on the plots available in https://doi.org/10.48550/arXiv.1908.0
   // The uncertainties are linearly interpolated, using the `TGraph::Eval()` method, to cover a range of pT values
   // covered by the analysis.
   const std::vector<double> x{5.50e3, 5.52e3, 12.54e3, 17.43e3, 22.40e3, 27.48e3, 30e3, 10000e3};
   const std::vector<double> y{0.06628, 0.06395, 0.06396, 0.03372, 0.02441, 0.01403, 0, 0};
   TGraph graph(x.size(), x.data(), y.data());

   //  Use the Vary method to add the systematic variations to the total MC scale factor ("weight") of the analysis
   //  The input consists of the input column to be varied and the lambda function to compute the systematic variations.
   //  The new output columns contain the varied values of the input column.
   auto df_with_variations_mc =
      df_mc
         .Vary("weight",
               [&graph](double x, const RVecF &pt, const RVec<unsigned int> &type) {
                  const auto v = Mean(Map(pt[type == 11], [&graph](auto p) { return graph.Eval(p); }));
                  return RVec<double>{(1 + v) * x, (1 - v) * x};
               },
               {"weight", "goodlep_pt", "goodlep_type"}, {"up", "down"})
         .Histo1D<float>(ROOT::RDF::TH1DModel("Invariant Mass", "m4l", 24, 80, 170), "m4l", "weight");

   // Create the total MC scale factor histograms: "nominal", "weight:up" and "weight:down".
   auto histos_mc = VariationsFor(df_with_variations_mc);

   // Evaluate the total MC uncertainty based on the variations. Note, in this case the uncertainties are symmetric.
   for (unsigned int i = 0; i < histos_mc["nominal"].GetXaxis()->GetNbins(); i++) {
      histos_mc["nominal"].SetBinError(
         i, (histos_mc["weight:up"].GetBinContent(i) - histos_mc["nominal"].GetBinContent(i)));
   }

   // Make the plot of the data, individual MC contributions and the total MC scale factor systematic variations.
   gROOT->SetStyle("ATLAS");

   // Create canvas with pad
   auto c = new TCanvas("c", " ", 600, 600);
   auto pad = new TPad("upper_pad", "", 0, 0, 1, 1);
   pad->SetTickx(0);
   pad->SetTicky(0);
   pad->Draw();
   pad->cd();

   // Draw stack with MC contributions
   // Draw cloned histograms to preserve graphics when original objects goes out of scope
   df_other->SetFillColor(kViolet - 9);
   df_zz->SetFillColor(kAzure - 9);
   df_higgs->SetFillColor(kRed + 2);

   auto stack = new THStack("stack", "");
   auto h_other = static_cast<TH1 *>(df_other->Clone());
   stack->Add(h_other);
   auto h_zz = static_cast<TH1 *>(df_zz->Clone());
   stack->Add(h_zz);
   auto h_higgs = static_cast<TH1 *>(df_higgs->Clone());
   stack->Add(h_higgs);
   stack->Draw("HIST");

   // stack histogram can be accessed only after drawing
   stack->GetHistogram()->SetTitle("");
   stack->GetHistogram()->GetXaxis()->SetLabelSize(0.035);
   stack->GetHistogram()->GetXaxis()->SetTitleSize(0.045);
   stack->GetHistogram()->GetXaxis()->SetTitleOffset(1.3);
   stack->GetHistogram()->GetXaxis()->SetTitle("m_{4l}^{H#rightarrow ZZ} [GeV]");
   stack->GetHistogram()->GetYaxis()->SetLabelSize(0.035);
   stack->GetHistogram()->GetYaxis()->SetTitleSize(0.045);
   stack->GetHistogram()->GetYaxis()->SetTitle("Events");
   stack->SetMaximum(35);
   stack->GetHistogram()->GetYaxis()->ChangeLabel(1, -1, 0);

   // Draw MC scale factor and variations
   histos_mc["nominal"].SetFillColor(kBlack);
   histos_mc["nominal"].SetFillStyle(3254);
   auto h_nominal = histos_mc["nominal"].DrawClone("E2 same");
   histos_mc["weight:up"].SetLineColor(kGreen + 2);
   auto h_weight_up = histos_mc["weight:up"].DrawClone("HIST same");
   histos_mc["weight:down"].SetLineColor(kBlue + 2);
   auto h_weight_down = histos_mc["weight:down"].DrawClone("HIST same");

   // Draw data histogram
   df_h_mass_data->SetMarkerStyle(20);
   df_h_mass_data->SetMarkerSize(1.);
   df_h_mass_data->SetLineWidth(2);
   df_h_mass_data->SetLineColor(kBlack);
   df_h_mass_data->SetStats(false);
   auto h_mass_data = df_h_mass_data->DrawClone("E sames");

   // Add legend
   auto legend = new TLegend(0.57, 0.65, 0.94, 0.94);
   legend->SetTextFont(42);
   legend->SetFillStyle(0);
   legend->SetBorderSize(0);
   legend->SetTextSize(0.025);
   legend->SetTextAlign(32);
   legend->AddEntry(h_mass_data, "Data", "lep");
   legend->AddEntry(h_higgs, "Higgs MC", "f");
   legend->AddEntry(h_zz, "ZZ MC", "f");
   legend->AddEntry(h_other, "Other MC", "f");
   legend->AddEntry(h_weight_down, "Total MC Variations Down", "l");
   legend->AddEntry(h_weight_up, "Total MC Variations Up", "l");
   legend->AddEntry(h_nominal, "Total MC Uncertainty", "f");
   legend->Draw();

   // Add ATLAS label
   TLatex atlas_label;
   atlas_label.SetTextFont(70);
   atlas_label.SetTextSize(0.04);
   atlas_label.DrawLatexNDC(0.19, 0.85, "ATLAS");
   TLatex data_label;
   data_label.SetTextFont(42);
   data_label.DrawLatexNDC(0.19 + 0.13, 0.85, "Open Data");
   TLatex header;
   data_label.SetTextFont(42);
   header.SetTextSize(0.035);
   header.DrawLatexNDC(0.21, 0.8, "#sqrt{s} = 13 TeV, 10 fb^{-1}");

   // Save the plot.
   c->SaveAs("df106_HiggsToFourLeptons_cpp.png");
   std::cout << "Saved figure to df106_HiggsToFourLeptons_cpp.png" << std::endl;
}

int main()
{
   df106_HiggsToFourLeptons();
}
