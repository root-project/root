/// \file
/// \ingroup tutorial_histv7
/// \notebook -js
/// Show how NanoAOD files can be processed with RDataFrame and RHist.
///
/// This tutorial is based on df102_NanoAODDimuonAnalysis.C.
/// It illustrates how NanoAOD files can be processed with ROOT
/// dataframes. The NanoAOD-like input files are filled with 66 mio. events
/// from CMS OpenData containing muon candidates part of 2012 dataset
/// ([DOI: 10.7483/OPENDATA.CMS.YLIC.86ZZ](http://opendata.cern.ch/record/6004)
/// and [DOI: 10.7483/OPENDATA.CMS.M5AD.Y3V3](http://opendata.cern.ch/record/6030)).
/// The macro matches muon pairs and produces an histogram of the dimuon mass
/// spectrum showing resonances up to the Z mass.
/// Note that the bump at 30 GeV is not a resonance but a trigger effect.
///
/// More details about the dataset can be found on
/// [the CERN Open Data portal](http://opendata.web.cern.ch/record/12341).
///
/// \macro_image
/// \macro_code
/// \macro_output
///
/// \date August 2018, February 2026
/// \author Stefan Wunsch (KIT, CERN), the ROOT Team

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RHist.hxx>
#include <ROOT/RVec.hxx>
#include <ROOT/Hist/ConvertToTH1.hxx>
#include <TCanvas.h>
#include <TH1.h>
#include <TLatex.h>
#include <TStyle.h>

using namespace ROOT::VecOps;

void hist102_NanoAODDimuonAnalysis()
{
   // Enable multi-threading
   ROOT::EnableImplicitMT();

   // Create dataframe from NanoAOD files
   ROOT::RDataFrame df("Events", "root://eospublic.cern.ch//eos/opendata/cms/derived-data/AOD2NanoAODOutreachTool/"
                                 "Run2012BC_DoubleMuParked_Muons.root");

   // For simplicity, select only events with exactly two muons and require opposite charge
   auto df_2mu = df.Filter("nMuon == 2", "Events with exactly two muons");
   auto df_os = df_2mu.Filter("Muon_charge[0] != Muon_charge[1]", "Muons with opposite charge");

   // Compute invariant mass of the dimuon system
   auto df_mass = df_os.Define("Dimuon_mass", InvariantMass<float>, {"Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass"});

   // Make histogram of dimuon mass spectrum
   auto hist = df_mass.Hist(30000, {0.25, 300}, "Dimuon_mass");

   // Request cut-flow report
   auto report = df.Report();

   // Convert histogram to TH1, set title and axis labels
   auto h = ROOT::Experimental::Hist::ConvertToTH1D(*hist);
   h->SetName("Dimuon_mass");
   h->SetTitle("Dimuon mass;m_{#mu#mu} (GeV);N_{Events}");

   // Produce plot
   gStyle->SetOptStat(0);
   gStyle->SetTextFont(42);
   auto c = new TCanvas("c", "", 800, 700);
   c->SetLogx();
   c->SetLogy();

   h->GetXaxis()->SetTitleSize(0.04);
   h->GetYaxis()->SetTitleSize(0.04);
   h->DrawClone();

   TLatex label;
   label.SetNDC(true);
   label.DrawLatex(0.175, 0.740, "#eta");
   label.DrawLatex(0.205, 0.775, "#rho,#omega");
   label.DrawLatex(0.270, 0.740, "#phi");
   label.DrawLatex(0.400, 0.800, "J/#psi");
   label.DrawLatex(0.415, 0.670, "#psi'");
   label.DrawLatex(0.485, 0.700, "Y(1,2,3S)");
   label.DrawLatex(0.755, 0.680, "Z");
   label.SetTextSize(0.040);
   label.DrawLatex(0.100, 0.920, "#bf{CMS Open Data}");
   label.SetTextSize(0.030);
   label.DrawLatex(0.630, 0.920, "#sqrt{s} = 8 TeV, L_{int} = 11.6 fb^{-1}");

   c->SaveAs("hist102_dimuon_spectrum.pdf");

   // Print cut-flow report
   report->Print();
}

int main()
{
   hist102_NanoAODDimuonAnalysis();
}
