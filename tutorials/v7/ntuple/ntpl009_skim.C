/// \file
/// \ingroup tutorial_ntuple
/// \notebook
/// Example of converting data stored in a TTree into an RNTuple
///
/// \macro_image
/// \macro_code
///
/// \date December 2022
/// \author The ROOT Team

// NOTE: The RNTuple classes are experimental at this point.
// Functionality, interface, and data format is still subject to changes.
// Do not use for real data!

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>

#include <TRandom.h>

#include <cstdint>

// Import classes from experimental namespace for the time being.
using RNTuple = ROOT::Experimental::RNTuple;
using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RNTupleReader = ROOT::Experimental::RNTupleReader;
using RNTupleWriter = ROOT::Experimental::RNTupleWriter;

// Input and output.
constexpr char const *kNTupleInputName = "ntpl";
constexpr char const *kNTupleInputFileName = "ntpl009_input.root";
constexpr char const *kNTupleOutputName = "ntpl_skim";
constexpr char const *kNTupleOutputFileName = "ntpl009_skim.root";
constexpr int kNEvents = 25000;

static void Write()
{
   auto model = RNTupleModel::Create();

   auto fldVpx = model->MakeField<std::vector<float>>("vpx");
   auto fldVpy = model->MakeField<std::vector<float>>("vpy");
   auto fldVpz = model->MakeField<std::vector<float>>("vpz");
   auto fldRand = model->MakeField<float>("rand");

   auto writer = RNTupleWriter::Recreate(std::move(model), kNTupleInputName, kNTupleInputFileName);

   gRandom->SetSeed();
   for (int i = 0; i < kNEvents; i++) {
      int npx = static_cast<int>(gRandom->Rndm(1) * 15);

      fldVpx->clear();
      fldVpy->clear();
      fldVpz->clear();

      for (int j = 0; j < npx; ++j) {
         float px, py, pz;
         gRandom->Rannor(px, py);
         pz = px * px + py * py;

         fldVpx->emplace_back(px);
         fldVpy->emplace_back(py);
         fldVpz->emplace_back(pz);
      }
      *fldRand = gRandom->Rndm(1);

      writer->Fill();
   }
}

void ntpl009_skim()
{
   Write();

   auto reader = RNTupleReader::Open(kNTupleInputName, kNTupleInputFileName);
   auto inputModel = reader->GetModel().lock();
   auto inputEntry = inputModel->GetDefaultEntry().lock();

   auto skimModel = inputModel->Clone();
   skimModel->Unfreeze();
   skimModel->MakeField<std::uint16_t>("skip", 0);

   auto writer = RNTupleWriter::Recreate(std::move(skimModel), kNTupleOutputName, kNTupleOutputFileName);
   auto skimEntry = writer->CreateEntry(inputEntry.get()).lock();

   auto pRand = inputEntry->GetRaw<float>("rand");
   auto pSkip = skimEntry->GetRaw<std::uint16_t>("skip");
   for (unsigned int i = 0; i < reader->GetNEntries(); ++i) {

      reader->LoadEntry(i);
      if (*pRand < 0.7) {
         (*pSkip)++;
         continue;
      }
      writer->Fill(*skimEntry);
      *pSkip = 0;
   }
}
