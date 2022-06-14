/// \file
/// \ingroup tutorial_ntuple
/// \notebook
/// Work with befriended RNTuples.  Adapted from tree3.C
///
/// \macro_image
/// \macro_code
///
/// \date April 2020
/// \author The ROOT Team

// NOTE: The RNTuple classes are experimental at this point.
// Functionality, interface, and data format is still subject to changes.
// Do not use for real data!

// Until C++ runtime modules are universally used, we explicitly load the ntuple library.  Otherwise
// triggering autoloading from the use of templated types would require an exhaustive enumeration
// of "all" template instances in the LinkDef file.
R__LOAD_LIBRARY(ROOTNTuple)

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>

#include <TCanvas.h>
#include <TH1F.h>
#include <TMath.h>
#include <TRandom.h>

#include <vector>

constexpr char const* kNTupleMainFileName = "ntpl006_data.root";
constexpr char const* kNTupleFriendFileName = "ntpl006_reco.root";

using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RNTupleReader = ROOT::Experimental::RNTupleReader;
using RNTupleWriter = ROOT::Experimental::RNTupleWriter;

void Generate()
{
   constexpr int kMaxTrack = 500;

   auto modelData = RNTupleModel::Create();
   auto fldPx = modelData->MakeField<std::vector<float>>("px");
   auto fldPy = modelData->MakeField<std::vector<float>>("py");
   auto fldPz = modelData->MakeField<std::vector<float>>("pz");

   auto modelFriend = RNTupleModel::Create();
   auto fldPt = modelFriend->MakeField<std::vector<float>>("pt");

   auto ntupleData = RNTupleWriter::Recreate(std::move(modelData), "data", kNTupleMainFileName);
   auto ntupleReco = RNTupleWriter::Recreate(std::move(modelFriend), "reco", kNTupleFriendFileName);

   for (int i=0; i < 1000; i++) {
      int ntracks = gRandom->Rndm() * (kMaxTrack - 1);
      for (int n = 0; n < ntracks; n++) {
         fldPx->emplace_back(gRandom->Gaus( 0, 1));
         fldPy->emplace_back(gRandom->Gaus( 0, 2));
         fldPz->emplace_back(gRandom->Gaus(10, 5));
         fldPt->emplace_back(TMath::Sqrt(fldPx->at(n) * fldPx->at(n) + fldPy->at(n) * fldPy->at(n)));
      }
      ntupleData->Fill();
      ntupleReco->Fill();

      fldPx->clear();
      fldPy->clear();
      fldPz->clear();
      fldPt->clear();
   }
}


void ntpl006_friends()
{
   Generate();

   std::vector<RNTupleReader::ROpenSpec> friends{
      {"data", kNTupleMainFileName},
      {"reco", kNTupleFriendFileName} };
   auto ntuple = RNTupleReader::OpenFriends(friends);

   auto c = new TCanvas("c", "", 200, 10, 700, 500);
   TH1F h("h", "pz {pt > 3.}", 100, -15, 35);

   auto viewPz = ntuple->GetView<float>("data.pz._0");
   auto viewPt = ntuple->GetView<float>("reco.pt._0");
   for (auto i : viewPt.GetFieldRange()) {
      if (viewPt(i) > 3.)
         h.Fill(viewPz(i));
   }

   h.SetFillColor(48);
   h.DrawCopy();
}
