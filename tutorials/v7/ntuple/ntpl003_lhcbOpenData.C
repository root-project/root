/// \file
/// \ingroup tutorial_ntuple
/// \notebook
/// Convert LHCb run 1 open data from a TTree to RForest.
/// This tutorial illustrates data conversion for a simple, tabular data model.
/// For reading, the tutorial shows the use of a Forest View, which selectively accesses specific fields.
/// If a view is used for reading, there is no need to define the data model as an RNTupleModel first.
/// The advantage of a view is that it directly accesses RForest's data buffers without making an additional
/// memory copy.
///
/// \macro_image
/// \macro_code
///
/// \date April 2019
/// \author The ROOT Team

// NOTE: The RForest classes are experimental at this point.
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
   assert(f.is_valid() && ! f->IsZombie());

   // Get a unique pointer to an empty RForest model
   auto model = RNTupleModel::Create();

   // We create RForest fields based on the types found in the TTree
   // This simple approach only works for trees with simple branches and only one leaf per branch
   auto tree = f->Get<TTree>("DecayTree");
   std::vector<TBranch*> branches;
   for (TObject *obj : *(tree->GetListOfBranches())) {
      auto b = static_cast<TBranch*>(obj);
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
      TBranch *branchRead = nullptr;
      tree->SetBranchAddress(b->GetName(), fieldDataPtr);
      branches.push_back(branchRead);
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

   TCanvas *c = new TCanvas("c", "B Flight Distance", 200, 10, 700, 500);
   TH1F *h = new TH1F("h", "B Flight Distance", 200, 0, 140);
   h->SetFillColor(48);

   for (auto i : ntuple->GetViewRange()) {
      // Note that we do not load an entry in this loop, i.e. we avoid the memory copy of loading the data into
      // the memory location given by the entry
      h->Fill(viewFlightDistance(i));
   }

   h->DrawCopy();
}
