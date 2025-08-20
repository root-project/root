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

// NOTE: The RNTupleProcessor and related classes are experimental at this point.
// Functionality and interface are still subject to changes.

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RNTupleProcessor.hxx>

#include <TCanvas.h>
#include <TH1F.h>
#include <TRandom.h>

// Import classes from the `Experimental` namespace for the time being.
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
   auto c = new TCanvas("c", "RNTupleProcessor Example", 200, 10, 700, 500);
   TH1F hPx("h", "This is the px distribution", 100, -4, 4);
   hPx.SetFillColor(48);

   // The chain-based processor can be created by passing a list of RNTupleOpenSpecs, describing the name and location
   // of each ntuple in the chain.
   auto processor = RNTupleProcessor::CreateChain(ntuples);

   // Access to the processor's fields is done through views, similar to the RNTupleReader.
   auto pxView = processor->GetView<std::vector<float>>("vpx");

   for (auto entryIdx : *processor) {
      // We can use the previously created view to read the field's data for the current entry.
      for (auto x : pxView(entryIdx)) {
         hPx.Fill(x);
      }
   }

   std::cout << "Processed a total of " << processor->GetCurrentEntryIndex() + 1 << " entries" << std::endl;

   hPx.DrawCopy();
}

void ntpl012_processor_chain()
{
   Write("ntuple1", "ntuple1.root");
   Write("ntuple2", "ntuple2.root");
   Write("ntuple3", "ntuple3.root");

   // The ntuples to generate and subsequently process. The model of the first ntuple will be used to construct the
   // entry used by the processor.
   std::vector<RNTupleOpenSpec> ntuples = {
      {"ntuple1", "ntuple1.root"}, {"ntuple2", "ntuple2.root"}, {"ntuple3", "ntuple3.root"}};

   Read(ntuples);
}
