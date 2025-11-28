/// \file
/// \ingroup tutorial_ntuple
/// \notebook
/// Example of efficient multi-threaded reading when multiple threads share a single reader.
///
/// \macro_image
/// \macro_code
///
/// \date October 2025
/// \author The ROOT Team

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include <TCanvas.h>
#include <TGraph.h>
#include <TRandom3.h>
#include <TStyle.h>

#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// Where to store the ntuple of this example
constexpr char const *kNTupleFileName = "ntpl017_shared_reader.root";

struct Point {
   float fX;
   float fY;
};

void Write()
{
   auto model = ROOT::RNTupleModel::Create();
   auto ptrPoint = model->MakeField<Point>("point");

   auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", kNTupleFileName);

   for (int i = 0; i < 10000; ++i) {
      if (i % 1000 == 0)
         writer->CommitCluster();

      auto prng = std::make_unique<TRandom3>();
      prng->SetSeed();

      ptrPoint->fX = prng->Rndm(1);
      ptrPoint->fY = prng->Rndm(1);
      writer->Fill();
   }
}

template <bool InformedT>
void ProcessEntries(ROOT::RNTupleReader *reader, const std::chrono::microseconds &usPerEvent,
                    std::vector<int> *countLoadedClusters)
{
   static std::mutex gLock;

   static std::atomic<int> gThreadId;
   const auto threadId = ++gThreadId;

   static std::atomic<int> gNEntriesDone;

   const auto N = reader->GetNEntries();

   auto token = reader->CreateActiveEntryToken();
   for (int i = threadId; i < N; i += 2) {
      {
         std::lock_guard<std::mutex> guard(gLock);
         if constexpr (InformedT)
            token.SetEntryNumber(i);
         reader->LoadEntry(i);
      }

      std::this_thread::sleep_for(usPerEvent);

      countLoadedClusters->at(++gNEntriesDone) =
         reader->GetMetrics().GetCounter("RNTupleReader.RPageSourceFile.nClusterLoaded")->GetValueAsInt();
   }
}

void ReadNaive()
{
   auto reader = ROOT::RNTupleReader::Open("ntpl", kNTupleFileName);
   reader->EnableMetrics();

   const auto N = reader->GetNEntries();
   std::vector<int> countLoadedClusters(N);

   std::array<std::thread, 2> threads;
   threads[0] = std::thread(ProcessEntries<false>, reader.get(), 100us, &countLoadedClusters);
   threads[1] = std::thread(ProcessEntries<false>, reader.get(), 200us, &countLoadedClusters);
   for (auto &t : threads) {
      t.join();
   }

   gStyle->SetOptStat(0);

   TCanvas *canvas = new TCanvas("", "Shared Reader Example", 200, 10, 1500, 500);
   //canvas->Divide(2, 1);

   //
   // canvas->cd(1);

   auto graph = new TGraph();
   for (unsigned int i = 0; i < N; ++i) {
      graph->SetPoint(i, i, countLoadedClusters[i]);
   }
   graph->Draw("ALP");
}

void ReadInformed()
{
   auto reader = ROOT::RNTupleReader::Open("ntpl", kNTupleFileName);
   reader->EnableMetrics();

   const auto N = reader->GetNEntries();
   std::vector<int> countLoadedClusters(N);

   std::array<std::thread, 2> threads;
   threads[0] = std::thread(ProcessEntries<true>, reader.get(), 100us, &countLoadedClusters);
   threads[1] = std::thread(ProcessEntries<true>, reader.get(), 200us, &countLoadedClusters);
   for (auto &t : threads) {
      t.join();
   }

   gStyle->SetOptStat(0);

   TCanvas *canvas = new TCanvas("", "Shared Reader Example", 200, 10, 1500, 500);
   //canvas->Divide(2, 1);

   //
   // canvas->cd(1);

   auto graph = new TGraph();
   for (unsigned int i = 0; i < N; ++i) {
      graph->SetPoint(i, i, countLoadedClusters[i]);
   }
   graph->Draw("ALP");
}

void ntpl017_shared_reader()
{
   Write();
   ReadNaive();
   ReadInformed();
}
