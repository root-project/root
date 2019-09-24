/// \file
/// \ingroup tutorial_ntuple
/// \notebook
/// Convert LHCb run 1 open data from a TTree to RNTuple.
/// This tutorial illustrates data conversion for a simple, tabular data model.
/// For reading, the tutorial shows the use of an ntuple View, which selectively accesses specific fields.
/// If a view is used for reading, there is no need to define the data model as an RNTupleModel first.
/// The advantage of a view is that it directly accesses RNTuple's data buffers without making an additional
/// memory copy.
///
/// \macro_image
/// \macro_code
///
/// \date April 2019
/// \author The ROOT Team

// NOTE: The RNTuple classes are experimental at this point.
// Functionality, interface, and data format is still subject to changes.
// Do not use for real data!

#include <ROOT/RField.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>

#include <TBranch.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TH1F.h>
#include <TLeaf.h>
#include <TSystem.h>
#include <TTree.h>

#include <cassert>
#include <chrono>
#include <memory>
#include <vector>

// Import classes from experimental namespace for the time being
using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RFieldBase = ROOT::Experimental::Detail::RFieldBase;
using RNTupleReader = ROOT::Experimental::RNTupleReader;
using RNTupleWriter = ROOT::Experimental::RNTupleWriter;

constexpr char const* kTreeFileName = "http://root.cern.ch/files/LHCb/lhcb_B2HHH_MagnetUp.root";
constexpr char const* kNTupleFileName = "ntpl003_lhcbOpenData.root";


void Convert() {
   std::unique_ptr<TFile> f(TFile::Open(kTreeFileName));
   assert(f && ! f->IsZombie());

   // Get a unique pointer to an empty RNTuple model
   auto model = RNTupleModel::Create();

   // We create RNTuple fields based on the types found in the TTree
   // This simple approach only works for trees with simple branches and only one leaf per branch
   auto tree = f->Get<TTree>("DecayTree");
   for (auto b : TRangeDynCast<TBranch>(*tree->GetListOfBranches())) {
      // The dynamic cast to TBranch should never fail for GetListOfBranches()
      assert(b);

      // We assume every branch has a single leaf
      TLeaf *l = static_cast<TLeaf*>(b->GetListOfLeaves()->First());

      // Create an ntuple field with the same name and type than the tree branch
      auto field = RFieldBase::Create(l->GetName(), l->GetTypeName());
      std::cout << "Convert leaf " << l->GetName() << " [" << l->GetTypeName() << "]"
                << " --> " << "field " << field->GetName() << " [" << field->GetType() << "]" << std::endl;

      // Hand over ownership of the field to the ntuple model.  This will also create a memory location attached
      // to the model's default entry, that will be used to place the data supposed to be written
      model->AddField(std::unique_ptr<RFieldBase>(field));

      // We connect the model's default entry's memory location for the new field to the branch, so that we can
      // fill the ntuple with the data read from the TTree
      void *fieldDataPtr = model->GetDefaultEntry()->GetValue(l->GetName()).GetRawPtr();
      tree->SetBranchAddress(b->GetName(), fieldDataPtr);
   }

   // The new ntuple takes ownership of the model
   auto ntuple = RNTupleWriter::Recreate(std::move(model), "DecayTree", kNTupleFileName);

   auto nEntries = tree->GetEntries();
   for (decltype(nEntries) i = 0; i < nEntries; ++i) {
      tree->GetEntry(i);
      ntuple->Fill();

      if (i && i % 100000 == 0)
         std::cout << "Wrote " << i << " entries" << std::endl;
   }
}


void ntpl003_lhcbOpenData()
{
   if (gSystem->AccessPathName(kNTupleFileName))
      Convert();

   // Create histogram of the flight distance

   // We open the ntuple without specifiying an explicit model first, but instead use a view on the field we are
   // interested in
   auto ntuple = RNTupleReader::Open("DecayTree", kNTupleFileName);

   // The view wraps a read-only double value and accesses directly the ntuple's data buffers
   auto viewFlightDistance = ntuple->GetView<double>("B_FlightDistance");

   auto c = new TCanvas("c", "B Flight Distance", 200, 10, 700, 500);
   TH1F h("h", "B Flight Distance", 200, 0, 140);
   h.SetFillColor(48);

   auto t1 = std::chrono::high_resolution_clock::now();
   for (auto i : ntuple->GetViewRange()) {
      // Note that we do not load an entry in this loop, i.e. we avoid the memory copy of loading the data into
      // the memory location given by the entry
      h.Fill(viewFlightDistance(i));
   }
   auto t2 = std::chrono::high_resolution_clock::now();
   h.DrawCopy();
   
   
   
   // Create a vector containing the same filename 20 times.
   std::vector<std::string> vec(20);
   for (auto &v : vec) {
      v = kNTupleFileName;
   }
   
   // Creates a ntuple where the same data is concatenated 20 times.
   auto ntuple2 = RNTupleReader::Open("DecayTree", vec);

   auto viewFlightDistance2 = ntuple2->GetView<double>("B_FlightDistance");
   auto c2 = new TCanvas("c2", "B Flight Distance Chain", 200, 10, 700, 500);
   TH1F h2("h", "B Flight Distance Chain", 200, 0, 140);
   h2.SetFillColor(48);


   auto t3 = std::chrono::high_resolution_clock::now();
   // The ViewRange is 20 times longer than for the previous RNTupleReader
   for (auto i : ntuple2->GetViewRange()) {
      h2.Fill(viewFlightDistance2(i));
   }
   auto t4 = std::chrono::high_resolution_clock::now();
   
   // Draws the same graph with 20 times more entries
   h2.DrawCopy();
   
   auto durationSingleFile = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
   auto durationChainOfFiles = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
   
   std::cout << "It takes " << durationChainOfFiles / durationSingleFile << " times longer when the same file is concatenated 20 times in a chain.\n";
}
