/// \file
/// \ingroup tutorial_forest
/// \notebook
/// Convert LHCb run 1 open data from a TTree to RForest.
/// This tutorial illustrates data conversion for a simple, tabular data model.
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

#include <ROOT/RField.hxx>
#include <ROOT/RForest.hxx>
#include <ROOT/RForestModel.hxx>

#include <TBranch.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TH1F.h>
#include <TLeaf.h>
#include <TSystem.h>
#include <TTree.h>

#include <cassert>
#include <memory>
#include <vector>

// Import classes from experimental namespace for the time being
using RForestModel = ROOT::Experimental::RForestModel;
using RFieldBase = ROOT::Experimental::Detail::RFieldBase;
using RInputForest = ROOT::Experimental::RInputForest;
using ROutputForest = ROOT::Experimental::ROutputForest;

constexpr char const* kTreeFile = "http://root.cern.ch/files/LHCb/lhcb_B2HHH_MagnetUp.root";
constexpr char const* kForestFile = "fst003_lhcbOpenData.root";


void Convert() {
   auto f = TFile::Open(kTreeFile);
   assert(f != nullptr);

   // Get a unique pointer to an empty RForest model
   auto model = RForestModel::Create();

   // We create RForest fields based on the types found in the TTree
   // This simple approach only works for trees with simple branches and only one leaf per branch
   auto tree = f->Get<TTree>("DecayTree");
   auto branchIter = tree->GetListOfBranches()->MakeIterator();
   TBranch *b;
   std::vector<TBranch*> branches;
   // --> use begin, end
   while ((b = static_cast<TBranch*>(branchIter->Next())) != nullptr) {
      // We assume every branch has a single leaf
      TLeaf *l = static_cast<TLeaf*>(b->GetListOfLeaves()->First());

      // Create a forest field with the same name and type than the tree branch
      auto field = RFieldBase::Create(l->GetName(), l->GetTypeName());
      std::cout << "Convert leaf " << l->GetName() << " [" << l->GetTypeName() << "]"
                << " --> " << "field " << field->GetName() << " [" << field->GetType() << "]" << std::endl;

      // Hand over ownership of the field to the forest model.  This will also create a memory location attached
      // to the model's default entry, that will be used to place the data supposed to be written
      model->AddField(std::unique_ptr<RFieldBase>(field));

      // We connect the model's default entry's memory location for the new field to the branch, so that we can
      // fill the forest with the data read from the TTree
      void *fieldDataPtr = model->GetDefaultEntry()->GetValue(l->GetName()).GetRawPtr();
      TBranch *branchRead = nullptr;
      tree->SetBranchAddress(b->GetName(), fieldDataPtr);
      branches.push_back(branchRead);
   }

   // The new forest takes ownership of the model
   auto forest = ROutputForest::Create(std::move(model), "DecayTree", kForestFile);

   auto nEntries = tree->GetEntries();
   for (decltype(nEntries) i = 0; i < nEntries; ++i) {
      tree->GetEntry(i);
      forest->Fill();

      if (i && i % 100000 == 0)
         std::cout << "Wrote " << i << " entries" << std::endl;
   }
}


void fst003_lhcbOpenData()
{
   if (gSystem->AccessPathName(kForestFile))
      Convert();

   // Create histogram of the flight distance
   auto model = RForestModel::Create();
   // We only add the one necessary field to the model, which results in a model that is compatible with the
   // data on disk
   auto fldDist = model->MakeField<double>("B_FlightDistance");

   // We use the reduced model to open the forest file
   auto forest = RInputForest::Open(std::move(model), "DecayTree", kForestFile);

   TCanvas *c = new TCanvas("c", "B Flight Distance", 200, 10, 700, 500);
   TH1F *h = new TH1F("h", "B Flight Distance", 200, 0, 140);
   h->SetFillColor(48);

   for (unsigned int i = 0; i < forest->GetNEntries(); ++i) {
      // Populate the created field for every entry...
      forest->LoadEntry(i);
      // ...and use it to fill the histogram
      h->Fill(*fldDist);
   }

   h->DrawCopy();
}
