/// \file
/// \ingroup tutorial_ntuple
/// \notebook
/// Demonstrate the RNTupleComposer and RNTupleProcessor for vertical compositions (chains) of RNTuples
///
/// \macro_image
/// \macro_code
///
/// \date April 2024
/// \author The ROOT Team

// NOTE: The RNTupleComposer, RNTupleProcessor and related classes are experimental at this point.
// Functionality and interface are still subject to changes.

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RNTupleProcessor.hxx>

#include <TCanvas.h>
#include <TH1F.h>
#include <TRandom.h>

// Import classes from the `Experimental` namespace for the time being.
using ROOT::Experimental::RNTupleComposer;
using ROOT::Experimental::RNTupleOpenSpec;
using ROOT::Experimental::RNTupleProcessor;

// Number of events to generate for each ntuple.
constexpr int kNEvents = 10000;

void Write(std::string_view ntupleName, std::string_view fileName)
{
   auto model = ROOT::RNTupleModel::Create();

   auto fldVpx = model->MakeField<std::vector<float>>("vpx");
   auto fldVpy = model->MakeField<std::vector<float>>("vpy");
   auto fldVpz = model->MakeField<std::vector<float>>("vpz");
   auto fldN = model->MakeField<std::uint64_t>("vn");

   auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), ntupleName, fileName);

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

      writer->Fill();
   }
}

void Read(const std::vector<RNTupleOpenSpec> &ntuples)
{
   auto c = new TCanvas("c", "RNTupleComposer chain Example", 200, 10, 700, 500);
   TH1F hPx("h", "This is the px distribution", 100, -4, 4);
   hPx.SetFillColor(48);

   // The chain-based composition can be created by passing a list of RNTupleOpenSpecs, describing the name and location
   // of each RNTuple in the chain.
   auto composer = RNTupleComposer::CreateChain(ntuples);
   int prevProcessorNumber{-1};

   // Access to the composition's fields is done by first requesting them through RNTupleComposer::RequestField(). The
   // returned value can be used to read the current entry's value for that particular field.
   auto px = composer->RequestField<std::vector<float>>("vpx");

   // Create the processor responsible for reading entries from the composer created above.
   RNTupleProcessor processor(*composer);

   // The iterator value is the index of the current entry being processed.
   for (auto idx : processor) {
      // The RNTupleProcessor provides some additional bookkeeping information, such as the current processor number.
      if (static_cast<int>(composer->GetCurrentProcessorNumber()) > prevProcessorNumber) {
         prevProcessorNumber = composer->GetCurrentProcessorNumber();
         std::cout << "Processing `ntuple" << prevProcessorNumber + 1 << "` (" << idx + 1
                   << " total entries processed so far)" << std::endl;
      }

      // We use the value returned from requesting the field to read its data for the current entry.
      for (auto x : *px) {
         hPx.Fill(x);
      }
   }

   std::cout << "Processed a total of " << processor.GetNEntriesProcessed() << " entries" << std::endl;

   hPx.DrawCopy();
}

void ntpl012_processor_chain()
{
   Write("ntuple1", "ntuple1.root");
   Write("ntuple2", "ntuple2.root");
   Write("ntuple3", "ntuple3.root");

   // The ntuples to generate, compose and subsequently process. The schema of the first ntuple will be used to
   // construct the entry used by the composer.
   std::vector<RNTupleOpenSpec> ntuples = {
      {"ntuple1", "ntuple1.root"}, {"ntuple2", "ntuple2.root"}, {"ntuple3", "ntuple3.root"}};

   Read(ntuples);
}
