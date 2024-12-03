/// \file
/// \ingroup tutorial_ntuple
/// \notebook
/// Demonstrate the RNTupleProcessor for vertical compositions (chains) of RNTuples
///
/// \macro_image
/// \macro_code
///
/// \date April 2024
/// \author The ROOT Team

// NOTE: The RNTuple classes are experimental at this point.
// Functionality and interface are still subject to changes.

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RNTupleProcessor.hxx>

#include <TCanvas.h>
#include <TH1F.h>
#include <TRandom.h>

// Import classes from the `Experimental` namespace for the time being.
using ROOT::Experimental::RNTupleModel;
using ROOT::Experimental::RNTupleOpenSpec;
using ROOT::Experimental::RNTupleProcessor;
using ROOT::Experimental::RNTupleWriter;

// Number of events to generate for each ntuple.
constexpr int kNEvents = 10000;

void Write(std::string_view ntupleName, std::string_view ntupleFileName)
{
   auto model = RNTupleModel::Create();

   auto fldVpx = model->MakeField<std::vector<float>>("vpx");
   auto fldVpy = model->MakeField<std::vector<float>>("vpy");
   auto fldVpz = model->MakeField<std::vector<float>>("vpz");
   auto fldN = model->MakeField<std::uint64_t>("vn");

   auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, ntupleFileName);

   for (int i = 0; i < kNEvents; ++i) {
      fldVpx->clear();
      fldVpy->clear();
      fldVpz->clear();

      *fldN = gRandom->Integer(15);
      for (int j = 0; j < *fldN; ++j) {
         float px, py, pz;
         gRandom->Rannor(px, py);
         pz = px * px + py * py;

         fldVpx->emplace_back(px);
         fldVpy->emplace_back(py);
         fldVpz->emplace_back(pz);
      }

      ntuple->Fill();
   }
}

void Read(const std::vector<RNTupleOpenSpec> &ntuples)
{
   auto c = new TCanvas("c", "RNTupleProcessor Example", 200, 10, 700, 500);
   TH1F hPx("h", "This is the px distribution", 100, -4, 4);
   hPx.SetFillColor(48);

   auto model = RNTupleModel::Create();
   auto ptrPx = model->MakeField<std::vector<float>>("vpx");

   // By passing a model to the processor, we can use the pointers to field values created upon model creation during
   // processing. When no model is provided, a default model is created based on the first ntuple specified.
   // Access to the entry values in this case can be achieved through RNTupleProcessor::GetEntry() or through its
   // iterator.
   auto processor = RNTupleProcessor::CreateChain(ntuples, std::move(model));

   for (const auto &entry : *processor) {
      // The RNTupleProcessor provides some additional bookkeeping information. The local entry number is reset each
      // a new ntuple in the chain is opened for processing.
      if (processor->GetLocalEntryNumber() == 0) {
         std::cout << "Processing " << ntuples.at(processor->GetCurrentNTupleNumber()).fNTupleName << " ("
                   << processor->GetNEntriesProcessed() << " total entries processed so far)" << std::endl;
      }

      // We can use the pointer to the field obtained while creating our model to read the field's data for the current
      // entry.
      for (auto x : *ptrPx) {
         hPx.Fill(x);
      }
   }

   hPx.DrawCopy();
}

void ntpl012_processor()
{
   // The ntuples to generate and subsequently process. The model of the first ntuple will be used to construct the
   // entry used by the processor.
   std::vector<RNTupleOpenSpec> ntuples = {
      {"ntuple1", "ntuple1.root"}, {"ntuple2", "ntuple2.root"}, {"ntuple3", "ntuple3.root"}};

   for (const auto &ntuple : ntuples) {
      Write(ntuple.fNTupleName, ntuple.fStorage);
   }

   Read(ntuples);
}
