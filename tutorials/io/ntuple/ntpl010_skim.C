/// \file
/// \ingroup tutorial_ntuple
/// \notebook
/// Example creating a derived RNTuple
///
/// \macro_image
/// \macro_code
///
/// \date February 2024
/// \author The ROOT Team

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include <TCanvas.h>
#include <TH1F.h>
#include <TRandom.h>

#include <cstdint>

// Input and output.
constexpr char const *kNTupleInputName = "ntpl";
constexpr char const *kNTupleInputFileName = "ntpl010_input.root";
constexpr char const *kNTupleOutputName = "ntpl_skim";
constexpr char const *kNTupleOutputFileName = "ntpl010_skim.root";
constexpr int kNEvents = 25000;

static void Write()
{
   auto model = ROOT::RNTupleModel::Create();

   auto fldVpx = model->MakeField<std::vector<float>>("vpx");
   auto fldVpy = model->MakeField<std::vector<float>>("vpy");
   auto fldVpz = model->MakeField<std::vector<float>>("vpz");
   auto fldN = model->MakeField<float>("n");

   auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), kNTupleInputName, kNTupleInputFileName);

   gRandom->SetSeed();
   for (int i = 0; i < kNEvents; i++) {
      *fldN = static_cast<int>(gRandom->Rndm(1) * 15);

      fldVpx->clear();
      fldVpy->clear();
      fldVpz->clear();

      for (int j = 0; j < *fldN; ++j) {
         float px, py, pz;
         gRandom->Rannor(px, py);
         pz = px * px + py * py;

         fldVpx->emplace_back(px);
         fldVpy->emplace_back(py);
         fldVpz->emplace_back(pz);
      }

      writer->Fill();
   }
}

void ntpl010_skim()
{
   Write();

   auto reader = ROOT::RNTupleReader::Open(kNTupleInputName, kNTupleInputFileName);

   auto skimModel = ROOT::RNTupleModel::Create();
   // Loop through the top-level fields of the input RNTuple
   for (const auto &value : reader->GetModel().GetDefaultEntry()) {
      // Drop "n" field from skimmed dataset
      if (value.GetField().GetFieldName() == "n")
         continue;

      // Add a renamed clone of the other fields to the skim model
      const std::string newName = "skim_" + value.GetField().GetFieldName();
      skimModel->AddField(value.GetField().Clone(newName));
      // Connect input and output field
      skimModel->GetDefaultEntry().BindValue<void>(newName, value.GetPtr<void>());
   }

   // Add an additional field to the skimmed dataset
   auto ptrSkip = skimModel->MakeField<std::uint16_t>("skip");

   auto writer = ROOT::RNTupleWriter::Recreate(std::move(skimModel), kNTupleOutputName, kNTupleOutputFileName);

   auto hSkip = new TH1F("h", "distribution of skipped entries", 10, 0, 10);
   auto ptrN = reader->GetModel().GetDefaultEntry().GetPtr<float>("n");
   for (auto numEntry : *reader) {
      reader->LoadEntry(numEntry);
      if (*ptrN <= 7) {
         (*ptrSkip)++;
         continue;
      }
      writer->Fill();
      hSkip->Fill(*ptrSkip);
      *ptrSkip = 0;
   }

   TCanvas *c1 = new TCanvas("", "Skimming Example", 200, 10, 700, 500);
   hSkip->DrawCopy();
}
