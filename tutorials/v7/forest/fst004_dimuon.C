/// \file
/// \ingroup tutorial_forest
/// \notebook
/// Convert CMS open data from a TTree to RForest.
/// This tutorial illustrates data conversion and data processing with RForest and RDataFrame.  In contrast to the
/// LHCb open data tutorial, the data model in this tutorial is not tabular but entries have variable lengths vectors
/// Based on RDataFrame's df102_NanoAODDimuonAnalysis.C
///
/// \macro_image
/// \macro_code
///
/// \date April 2019
/// \author The ROOT Team

// NOTE: The RForest classes are experimental at this point.
// Functionality, interface, and data format is still subject to changes.
// Do not use for real data!

R__LOAD_LIBRARY(libROOTForest)

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RForest.hxx>
#include <ROOT/RForestDS.hxx>
#include <ROOT/RVec.hxx>

#include <TCanvas.h>
#include <TH1D.h>
#include <TLatex.h>
#include <TStyle.h>
#include <TSystem.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <utility>

// Import classes from experimental namespace for the time being
using RInputForest = ROOT::Experimental::RInputForest;
using ROutputForest = ROOT::Experimental::ROutputForest;
using RForestDS = ROOT::Experimental::RForestDS;

constexpr char const* gTreeFile = "http://root.cern.ch/files/NanoAOD_DoubleMuon_CMS2011OpenData.root";
constexpr char const* gForestFile = "naod_dimuon_forest.root";


using ColNames_t = std::vector<std::string>;

// This is a custom action for RDataFrame. It does not support parallelism!
// This action writes data from an RDataFrame entry into a forest. It is templated on the
// types of the columns to be written and can be used as a generic file format converter.
template <typename... ColumnTypes_t>
class RForestHelper : public ROOT::Detail::RDF::RActionImpl<RForestHelper<ColumnTypes_t...>> {
public:
   using Result_t = ROutputForest;
private:
   using ColumnValues_t = std::tuple<std::shared_ptr<ColumnTypes_t>...>;

   std::string fForestName;
   std::string fRootFile;
   ColNames_t fColNames;
   ColumnValues_t fColumnValues;
   static constexpr const auto fNColumns = std::tuple_size<ColumnValues_t>::value;
   std::shared_ptr<ROutputForest> fForest;

   template<std::size_t... S>
   void InitializeImpl(std::index_sequence<S...>) {
      auto eventModel = ROOT::Experimental::RForestModel::Create();
      std::initializer_list<int> expander{
         (std::get<S>(fColumnValues) = eventModel->MakeField<ColumnTypes_t>(fColNames[S]), 0)...};
      fForest = std::move(ROutputForest::Create(std::move(eventModel), fForestName, fRootFile));
   }

   template<std::size_t... S>
   void ExecImpl(std::index_sequence<S...>, ColumnTypes_t... values) {
      std::initializer_list<int> expander{(*std::get<S>(fColumnValues) = values, 0)...};
   }

public:
   RForestHelper(std::string_view forestName, std::string_view rootFile, const ColNames_t& colNames)
      : fForestName(forestName), fRootFile(rootFile), fColNames(colNames)
   {
      InitializeImpl(std::make_index_sequence<fNColumns>());
   }

   RForestHelper(RForestHelper&&) = default;
   RForestHelper(const RForestHelper&) = delete;
   std::shared_ptr<ROutputForest> GetResultPtr() const { return fForest; }

   void Initialize() {}

   void InitTask(TTreeReader *, unsigned int) {}

   /// This is a method executed at every entry
   void Exec(unsigned int slot, ColumnTypes_t... values)
   {
      ExecImpl(std::make_index_sequence<fNColumns>(), values...);
      fForest->Fill();
   }

   void Finalize()
   {
      fForest->CommitCluster();
   }

   std::string GetActionName() { return "RForest Writer"; }
};


/// Return the invariant mass of multiple particles given the collections of the
/// quantities transverse momentum (pt), rapidity (eta), azimuth (phi) and mass.
///
/// The function computes the invariant mass of multiple particles with the
/// four-vectors (pt, eta, phi, mass).
///
/// In contrast to ROOT's built-in version, this version works on std::vector instead of RVec.
template <typename T>
T InvariantMassStdVector(
   const std::vector<T>& pt, const std::vector<T>& eta, const std::vector<T>& phi, const std::vector<T>& mass)
{
   assert(pt.size() == eta.size() == phi.size() == mass.size() == 2);
   // Conversion from (mass, pt, eta, phi) to (e, x, y, z) coordinate system
   const auto x0 = pt[0] * cos(phi[0]);   const auto x1 = pt[1] * cos(phi[1]);
   const auto y0 = pt[0] * sin(phi[0]);   const auto y1 = pt[1] * sin(phi[1]);
   const auto z0 = pt[0] * sinh(eta[0]);  const auto z1 = pt[1] * sinh(eta[1]);
   const auto e0 = sqrt(x0 * x0 + y0 * y0 + z0 * z0 + mass[0] * mass[0]);
   const auto e1 = sqrt(x1 * x1 + y1 * y1 + z1 * z1 + mass[1] * mass[1]);

   // Addition of particle four-vectors
   const auto xs = x0 + x1;
   const auto ys = y0 + y1;
   const auto zs = z0 + z1;
   const auto es = e0 + e1;

   // Return invariant mass with (+, -, -, -) metric
   return sqrt(es * es - xs * xs - ys * ys - zs * zs);
}

// We use an RDataFrame custom snapshotter to convert between TTree and RForest.
// The snapshotter is templated; we construct the conversion C++ code as a string and hand it over to Cling
void Convert() {
   // Use df to list the branch types and names of the input tree
   ROOT::RDataFrame df("Events", gTreeFile);

   // Construct the types for the template instantiation and the column names from the dataframe
   std::string typeList = "<";
   std::string columnList = "{";
   auto columnNames = df.GetColumnNames();
   for (auto name : columnNames) {
      auto typeName = df.GetColumnType(name);
      // Skip ULong64_t for the time being, RForest support will be added at a later point
      if (typeName == "ULong64_t") continue;
      columnList += "\"" + name + "\",";
      typeList += typeName + ",";
   }
   *columnList.rbegin() = '}';
   *typeList.rbegin() = '>';

   std::string code = "{";
   // Convert the first 4 million events
   code += "auto df = std::make_unique<ROOT::RDataFrame>(\"Events\", \"" + std::string(gTreeFile)
         + "\")->Range(0, 4000000);";
   code += "ColNames_t colNames = " + columnList + ";";
   code += "RForestHelper" + typeList + " helper{\"Events\", \"" + std::string(gForestFile) + "\", colNames};";
   code += "*df.Book" + typeList + "(std::move(helper), colNames);";
   code += "}";

   gInterpreter->ProcessLine(code.c_str());
}


void fst004_dimuon() {
   if (gSystem->AccessPathName(gForestFile))
      Convert();

   // Create an input forest unique pointer
   auto forest = RInputForest::Create("Events", gForestFile);
   std::cout << forest->Print();

   // Create a data frame from the input forest
   auto df = std::make_unique<ROOT::RDataFrame>(std::make_unique<RForestDS>(forest.get()));

   // As of this point, the tutorial is identical to df102_NanoAODDimuonAnalysis expect the use of
   // InvariantMassStdVector instead of InvariantMass

   // For simplicity, select only events with exactly two muons and require opposite charge
   auto df_2mu = df->Filter("nMuon == 2", "Events with exactly two muons");
   auto df_os = df_2mu.Filter("Muon_charge[0] != Muon_charge[1]", "Muons with opposite charge");

   // Compute invariant mass of the dimuon system
   auto df_mass = df_os.Define("Dimuon_mass", InvariantMassStdVector<float>, {"Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass"});
   auto df_size = df_os.Define("eta_size", "Muon_mass.size()");

   // Make histogram of dimuon mass spectrum
   auto h = df_mass.Histo1D({"Dimuon_mass", "Dimuon_mass", 30000, 0.25, 300}, "Dimuon_mass");

   // Request cut-flow report
   auto report = df_mass.Report();

   // Produce plot
   gStyle->SetOptStat(0); gStyle->SetTextFont(42);
   auto c = new TCanvas("c", "", 800, 700);
   c->SetLogx(); c->SetLogy();

   h->SetTitle("");
   h->GetXaxis()->SetTitle("m_{#mu#mu} (GeV)"); h->GetXaxis()->SetTitleSize(0.04);
   h->GetYaxis()->SetTitle("N_{Events}"); h->GetYaxis()->SetTitleSize(0.04);
   h->DrawCopy();

   TLatex label; label.SetNDC(true);
   label.DrawLatex(0.175, 0.740, "#eta");
   label.DrawLatex(0.205, 0.775, "#rho,#omega");
   label.DrawLatex(0.270, 0.740, "#phi");
   label.DrawLatex(0.400, 0.800, "J/#psi");
   label.DrawLatex(0.415, 0.670, "#psi'");
   label.DrawLatex(0.485, 0.700, "Y(1,2,3S)");
   label.DrawLatex(0.755, 0.680, "Z");
   label.SetTextSize(0.040); label.DrawLatex(0.100, 0.920, "#bf{CMS Open Data}");
   label.SetTextSize(0.030); label.DrawLatex(0.630, 0.920, "#sqrt{s} = 8 TeV, L_{int} = 11.6 fb^{-1}");

   // Print cut-flow report
   report->Print();
}
