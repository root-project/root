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

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleDS.hxx>
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
using RNTupleDS = ROOT::Experimental::RNTupleDS;

constexpr char const* kTreeFileName = "http://root.cern.ch/files/NanoAOD_DoubleMuon_CMS2011OpenData.root";
constexpr char const* kForestFileName = "ntpl004_dimuon.root";


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
   int fCounter;

   template<std::size_t... S>
   void InitializeImpl(std::index_sequence<S...>) {
      auto eventModel = ROOT::Experimental::RNTupleModel::Create();
      // Create the fields and the shared pointers to the connected values
      std::initializer_list<int> expander{
         (std::get<S>(fColumnValues) = eventModel->MakeField<ColumnTypes_t>(fColNames[S]), 0)...};
      fForest = std::move(ROutputForest::Recreate(std::move(eventModel), fForestName, fRootFile));
   }

   template<std::size_t... S>
   void ExecImpl(std::index_sequence<S...>, ColumnTypes_t... values) {
      // For every entry, set the destination of the forest's default entry's shared pointers to the given values,
      // which are provided by RDataFrame
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

   void Initialize()
   {
      fCounter = 0;
   }

   void InitTask(TTreeReader *, unsigned int) {}

   /// This is a method executed at every entry
   void Exec(unsigned int slot, ColumnTypes_t... values)
   {
      // Populate the forest's fields data locations with the provided values, then write to disk
      ExecImpl(std::make_index_sequence<fNColumns>(), values...);
      fForest->Fill();
      if (++fCounter % 100000 == 0)
         std::cout << "Wrote " << fCounter << " entries" << std::endl;
   }

   void Finalize()
   {
      fForest->CommitCluster();
   }

   std::string GetActionName() { return "RForest Writer"; }
};


/// A wrapper for ROOT's InvariantMass function that takes std::vector instead of RVecs
template <typename T>
T InvariantMassStdVector(
   const std::vector<T>& pt, const std::vector<T>& eta, const std::vector<T>& phi, const std::vector<T>& mass)
{
   assert(pt.size() == eta.size() == phi.size() == mass.size() == 2);

   ROOT::VecOps::RVec<float> rvPt(pt);
   ROOT::VecOps::RVec<float> rvEta(eta);
   ROOT::VecOps::RVec<float> rvPhi(phi);
   ROOT::VecOps::RVec<float> rvMass(mass);

   return InvariantMass(rvPt, rvEta, rvPhi, rvMass);
}

// We use an RDataFrame custom snapshotter to convert between TTree and RForest.
// The snapshotter is templated; we construct the conversion C++ code as a string and hand it over to Cling
void Convert() {
   // Use df to list the branch types and names of the input tree
   ROOT::RDataFrame df("Events", kTreeFileName);

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
   code += "auto df = std::make_unique<ROOT::RDataFrame>(\"Events\", \"" + std::string(kTreeFileName)
         + "\")->Range(0, 4000000);";
   code += "ColNames_t colNames = " + columnList + ";";
   code += "RForestHelper" + typeList + " helper{\"Events\", \"" + std::string(kForestFileName) + "\", colNames};";
   code += "*df.Book" + typeList + "(std::move(helper), colNames);";
   code += "}";

   gInterpreter->ProcessLine(code.c_str());
}


void ntpl004_dimuon() {
   // Support for multi-threading comes at a later point, for the time being do not enable
   // ROOT::EnableImplicitMT();

   if (gSystem->AccessPathName(kForestFileName))
      Convert();

   auto df = ROOT::Experimental::MakeNTupleDataFrame("Events", kForestFileName);

   // As of this point, the tutorial is identical to df102_NanoAODDimuonAnalysis except the use of
   // InvariantMassStdVector instead of InvariantMass

   // For simplicity, select only events with exactly two muons and require opposite charge
   auto df_2mu = df.Filter("nMuon == 2", "Events with exactly two muons");
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
