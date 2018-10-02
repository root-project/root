/// \file
/// \ingroup tutorial_dataframe
/// \notebook -draw
/// This tutorial illustrates how NanoAOD files can be processed with ROOT
/// dataframes. The NanoAOD-like input file is filled with events from
/// CMS OpenData containing muon candidates from 2011 data
/// ([DOI: 10.7483/OPENDATA.CMS.RZ34.QR6N](http://opendata.cern.ch/record/17)).
/// The script matches muon pairs
/// and produces an histogram of the dimuon mass spectrum showing resonances
/// up to the Z mass.
///
/// \macro_image
/// \macro_code
///
/// \date August 2018
/// \author Stefan Wunsch (KIT, CERN)

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "TCanvas.h"
#include "TH1D.h"
#include "TLatex.h"
#include "TLorentzVector.h"
#include "TStyle.h"

using namespace ROOT::VecOps;

void df102_NanoAODDimuonAnalysis()
{
   // Enable multi-threading
   ROOT::EnableImplicitMT();

   // Create dataframe from NanoAOD file
   ROOT::RDataFrame df("Events", "http://root.cern.ch/files/NanoAOD_DoubleMuon_CMS2011OpenData.root");

   // Select events with more than two muons
   auto df_filtered = df.Filter("nMuon>=2", "More than two muons");

   // Find muon pair with highest pt and opposite charge
   auto find_pair = [](const RVec<float> &pt, const RVec<int> &charge) {
      // Get indices that sort the muon pts in descending order
      const auto idx = Reverse(Argsort(pt));

      // Find muon with second-highest pt and opposite charge
      const auto i1 = idx[0];
      for (size_t i = 1; i < idx.size(); i++) {
         const auto i2 = idx[i];
         if (charge[i1] != charge[i2]) {
            return RVec<size_t>({i1, i2});
         }
      }

      // Return empty selection if no candidate matches
      return RVec<size_t>({});
   };
   auto df_pair = df_filtered.Define("Muon_pair", find_pair, {"Muon_pt", "Muon_charge"})
                             .Filter("Muon_pair.size() == 2", "Found valid pair");

   // Compute invariant mass of the di-muon system
   auto compute_mass = [](RVec<float> &pt, RVec<float> &eta, RVec<float> &phi,
                          RVec<float> &mass, RVec<size_t> &idx) {
      // Compose four-vectors of both muons
      TLorentzVector p1;
      const auto i1 = idx[0];
      p1.SetPtEtaPhiM(pt[i1], eta[i1], phi[i1], mass[i1]);

      TLorentzVector p2;
      const auto i2 = idx[1];
      p2.SetPtEtaPhiM(pt[i2], eta[i2], phi[i2], mass[i2]);

      // Add four-vectors to build di-muon system and return the invariant mass
      return (p1 + p2).M();
   };
   auto df_mass = df_pair.Define("Dimuon_mass", compute_mass,
                                 {"Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_pair"});

   // Produce histogram of di-muon mass spectrum
   auto h = df_mass.Histo1D({"Dimuon_mass", "Dimuon_mass", 20000, 0.25, 300}, "Dimuon_mass")
                   .GetValue();

   // Make plot
   gStyle->SetOptStat(0);
   gStyle->SetTextFont(42);
   auto c = new TCanvas("c", "c", 800, 600);
   c->SetLogx();
   c->SetLogy();

   h.SetTitle("");
   h.GetXaxis()->SetTitle("Invariant di-muon mass (GeV)");
   h.GetXaxis()->SetTitleSize(0.04);
   h.GetYaxis()->SetTitle("N_{Events}");
   h.GetYaxis()->SetTitleSize(0.04);
   h.DrawClone();

   TLatex label;
   label.SetNDC(true);
   label.DrawLatex(0.175, 0.740, "#eta");
   label.DrawLatex(0.205, 0.785, "#rho,#omega");
   label.DrawLatex(0.270, 0.750, "#phi");
   label.DrawLatex(0.400, 0.800, "J/#psi");
   label.DrawLatex(0.415, 0.680, "#psi'");
   label.DrawLatex(0.485, 0.760, "Y(1,2,3S)");
   label.DrawLatex(0.755, 0.620, "Z");
   label.DrawLatex(0.170, 0.350, "#bf{CMS Open Data}");
   label.DrawLatex(0.170, 0.275, "#bf{#sqrt{s} = 7 TeV}");
   label.DrawLatex(0.170, 0.200, "#bf{L_{int} = 2.4 fb^{-1}}");
   label.SetTextSize(0.032);
   label.DrawLatex(0.10, 0.920, "Run2011A Double Muon Dataset (DOI: 10.7483/OPENDATA.CMS.RZ34.QR6N)");

   c->SaveAs("nanoaod_dimuon_spectrum.pdf");
}

int main()
{
   df102_NanoAODDimuonAnalysis();
}
