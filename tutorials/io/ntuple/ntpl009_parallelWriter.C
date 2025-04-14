/// \file
/// \ingroup tutorial_ntuple
/// \notebook
/// Example of multi-threaded writes using RNTupleParallelWriter.  Adapted from the ntpl007_mtFill tutorial.
///
/// \macro_image
/// \macro_code
///
/// \date Feburary 2024
/// \author The ROOT Team

// NOTE: The RNTupleParallelWriter class is experimental at this point.
// Functionality and interface are still subject to changes.

#include <ROOT/RNTupleFillContext.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleParallelWriter.hxx>
#include <ROOT/RNTupleReader.hxx>

#include <TCanvas.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TRandom.h>
#include <TRandom3.h>
#include <TStyle.h>
#include <TSystem.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <utility>

// Import classes from experimental namespace for the time being
using ROOT::Experimental::RNTupleParallelWriter;

// Where to store the ntuple of this example
constexpr char const *kNTupleFileName = "ntpl009_parallelWriter.root";

// Number of parallel threads to fill the ntuple
constexpr int kNWriterThreads = 4;

// Number of events to generate is kNEventsPerThread * kNWriterThreads
constexpr int kNEventsPerThread = 25000;

// Thread function to generate and write events
void FillData(RNTupleParallelWriter *writer)
{
   static std::atomic<std::uint32_t> gThreadId;
   const auto threadId = ++gThreadId;

   auto prng = std::make_unique<TRandom3>();
   prng->SetSeed();

   auto fillContext = writer->CreateFillContext();
   auto entry = fillContext->CreateEntry();

   auto id = entry->GetPtr<std::uint32_t>("id");
   *id = threadId;
   auto vpx = entry->GetPtr<std::vector<float>>("vpx");
   auto vpy = entry->GetPtr<std::vector<float>>("vpy");
   auto vpz = entry->GetPtr<std::vector<float>>("vpz");

   for (int i = 0; i < kNEventsPerThread; i++) {
      vpx->clear();
      vpy->clear();
      vpz->clear();

      int npx = static_cast<int>(prng->Rndm(1) * 15);
      // Set the field data for the current event
      for (int j = 0; j < npx; ++j) {
         float px, py, pz;
         prng->Rannor(px, py);
         pz = px * px + py * py;

         vpx->emplace_back(px);
         vpy->emplace_back(py);
         vpz->emplace_back(pz);
      }

      fillContext->Fill(*entry);
   }
}

// Generate kNEvents with multiple threads in kNTupleFileName
void Write()
{
   // Create the data model
   auto model = ROOT::RNTupleModel::CreateBare();
   model->MakeField<std::uint32_t>("id");
   model->MakeField<std::vector<float>>("vpx");
   model->MakeField<std::vector<float>>("vpy");
   model->MakeField<std::vector<float>>("vpz");

   // Create RNTupleWriteOptions to make the writing commit multiple clusters (so that "Entry Id vs Thread Id" shows the
   // interleaved clusters).
   ROOT::RNTupleWriteOptions options;
   options.SetApproxZippedClusterSize(1024 * 1024);

   // We hand-over the data model to a newly created ntuple of name "NTuple", stored in kNTupleFileName
   auto writer = RNTupleParallelWriter::Recreate(std::move(model), "NTuple", kNTupleFileName, options);

   std::vector<std::thread> threads;
   for (int i = 0; i < kNWriterThreads; ++i)
      threads.emplace_back(FillData, writer.get());
   for (int i = 0; i < kNWriterThreads; ++i)
      threads[i].join();

   // The writer unique pointer goes out of scope here.  On destruction, the writer flushes unwritten data to disk
   // and closes the attached ROOT file.
}

// For all of the events, histogram only one of the written vectors
void Read()
{
   auto reader = ROOT::RNTupleReader::Open("NTuple", kNTupleFileName);
   auto viewVpx = reader->GetView<float>("vpx._0");

   gStyle->SetOptStat(0);

   TCanvas *c1 = new TCanvas("c2", "Multi-Threaded Filling Example", 200, 10, 1500, 500);
   c1->Divide(2, 1);

   c1->cd(1);
   TH1F h("h", "This is the px distribution", 100, -4, 4);
   h.SetFillColor(48);
   // Iterate through all values of vpx in all events
   for (auto i : viewVpx.GetFieldRange())
      h.Fill(viewVpx(i));
   // Prevent the histogram from disappearing
   h.DrawCopy();

   c1->cd(2);
   auto nEvents = reader->GetNEntries();
   auto viewId = reader->GetView<std::uint32_t>("id");
   TH2F hFillSequence("", "Entry Id vs Thread Id;Entry Sequence Number;Filling Thread", 100, 0, nEvents, 100, 0,
                      kNWriterThreads + 1);
   for (auto i : reader->GetEntryRange())
      hFillSequence.Fill(i, viewId(i));
   hFillSequence.DrawCopy();
}

void ntpl009_parallelWriter()
{
   Write();
   Read();
}
