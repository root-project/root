/// \file
/// \ingroup tutorial_ntuple
/// \notebook
/// Example of efficient multi-threaded reading when multiple threads share a single reader.
/// In this example, two threads share the work as follows: the first thread processes all the even entries numbers
/// and the second thread all the odd entry numbers. The second thread works twice as slow as the first one.
///
/// As a result, the threads need the same clusters and pages but at different points in time.
/// With the naive way of using the reader, this read pattern will result in cache thrashing.
///
/// Using the "active entry token" API, on the other hand, the reader will be informed about which entries
/// need to be kept in the caches and cache thrashing is prevented.
///
/// \macro_image
/// \macro_code
///
/// \date October 2025
/// \author The ROOT Team

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include <TAxis.h>
#include <TCanvas.h>
#include <TGraph.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TRandom3.h>
#include <TROOT.h>
#include <TStyle.h>

#include <array>
#include <atomic>
#include <chrono>
#include <exception>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// Where to store the ntuple of this example
constexpr char const *kNTupleFileName = "ntpl017_shared_reader.root";

/// The sample class that is stored in the RNTuple
struct Point2D {
   float fX;
   float fY;
};

/// Whether we read with setting active entry tokens (informed) or not (naive)
enum class EReadMode {
   kNaive,
   kInformed
};

/// Nicify output of EReadMode values
std::ostream &operator<<(std::ostream &os, const EReadMode &e)
{
   switch (e) {
   case EReadMode::kNaive: os << "naive"; break;
   case EReadMode::kInformed: os << "informed"; break;
   default: os << "???";
   }
   return os;
}

// To be reset between ProcessEntries calls to Read()
static std::atomic<int> gNEntriesDone;
static std::atomic<int> gThreadId;

/// The read thread's main function
void ProcessEntries(ROOT::RNTupleReader &reader, EReadMode readMode, const std::chrono::microseconds &usPerEvent,
                    std::vector<int> &sumLoadedClusters, std::vector<int> &sumUnsealedPages,
                    std::vector<int> &nClusters, std::vector<int> &nPages)
{
   static std::mutex gLock;

   const int threadId = gThreadId++;

   auto token = reader.CreateActiveEntryToken();
   for (unsigned int i = threadId; i < reader.GetNEntries(); i += 2) {
      {
         std::lock_guard<std::mutex> guard(gLock);

         // The only difference between naive and informed reading: in informed reading, we indicate which
         // entry we are going to use before loading it.
         switch (readMode) {
         case EReadMode::kInformed: token.SetEntryNumber(i); break;
         case EReadMode::kNaive: break;
         default: std::terminate(); // never here
         }

         reader.LoadEntry(i);
      }

      std::this_thread::sleep_for(usPerEvent);

      const int entriesDone = gNEntriesDone++;
      sumLoadedClusters.at(entriesDone) =
         reader.GetMetrics().GetCounter("RNTupleReader.RPageSourceFile.nClusterLoaded")->GetValueAsInt();
      sumUnsealedPages[entriesDone] =
         reader.GetMetrics().GetCounter("RNTupleReader.RPageSourceFile.nPageUnsealed")->GetValueAsInt();
      nClusters[entriesDone] =
         reader.GetMetrics().GetCounter("RNTupleReader.RPageSourceFile.RClusterPool.nCluster")->GetValueAsInt();
      nPages[entriesDone] =
         reader.GetMetrics().GetCounter("RNTupleReader.RPageSourceFile.RPagePool.nPage")->GetValueAsInt();
   }
}

void Read(EReadMode readMode, std::vector<int> &sumLoadedClusters, std::vector<int> &sumUnsealedPages,
          std::vector<int> &nClusters, std::vector<int> &nPages)
{
   auto reader = ROOT::RNTupleReader::Open("ntpl", kNTupleFileName);
   reader->EnableMetrics();

   gNEntriesDone = 0;
   gThreadId = 0;

   const auto N = reader->GetNEntries();
   sumLoadedClusters.resize(N);
   sumUnsealedPages.resize(N);
   nClusters.resize(N);
   nPages.resize(N);

   std::array<std::thread, 2> threads;
   threads[0] = std::thread(ProcessEntries, std::ref(*reader), readMode, 100us, std::ref(sumLoadedClusters),
                            std::ref(sumUnsealedPages), std::ref(nClusters), std::ref(nPages));
   threads[1] = std::thread(ProcessEntries, std::ref(*reader), readMode, 200us, std::ref(sumLoadedClusters),
                            std::ref(sumUnsealedPages), std::ref(nClusters), std::ref(nPages));
   for (auto &t : threads) {
      t.join();
   }

   std::cout << "Reading in mode '" << readMode << "':" << std::endl;
   std::cout << "===========================" << std::endl;
   reader->GetMetrics().Print(std::cout);
   std::cout << "===========================" << std::endl << std::endl;
}

void Write()
{
   auto model = ROOT::RNTupleModel::Create();
   auto ptrPoint = model->MakeField<Point2D>("point");

   auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", kNTupleFileName);

   for (int i = 0; i < 1000; ++i) {
      if (i % 100 == 0)
         writer->CommitCluster();

      auto prng = std::make_unique<TRandom3>();
      prng->SetSeed();

      ptrPoint->fX = prng->Rndm(1);
      ptrPoint->fY = prng->Rndm(1);
      writer->Fill();
   }
}

TGraph *GetGraph(const std::vector<int> &counts)
{
   auto graph = new TGraph();
   for (unsigned int i = 0; i < counts.size(); ++i) {
      graph->SetPoint(i, i, counts[i]);
   }
   graph->GetXaxis()->SetTitle("Number of processed entries");
   return graph;
}

void ntpl017_shared_reader()
{
   ROOT::EnableImplicitMT();

   Write();
   ROOT::RNTupleReader::Open("ntpl", kNTupleFileName)->PrintInfo(ROOT::ENTupleInfo::kStorageDetails);

   std::vector<int> sumLoadedClusters;
   std::vector<int> sumUnsealedPages;
   std::vector<int> nClusters;
   std::vector<int> nPages;
   TLatex latex;

   gStyle->SetOptStat(0);
   gStyle->SetLineWidth(2);
   gStyle->SetMarkerStyle(8);
   TCanvas *canvas = new TCanvas("", "Shared Reader Example", 200, 10, 1500, 1000);

   canvas->Divide(2, 2);

   Read(EReadMode::kNaive, sumLoadedClusters, sumUnsealedPages, nClusters, nPages);

   canvas->cd(1);
   auto graph1 = GetGraph(sumUnsealedPages);
   graph1->SetLineColor(kRed);
   graph1->Draw("AL");
   auto graph2 = GetGraph(sumLoadedClusters);
   graph2->SetLineColor(kBlue);
   graph2->Draw("SAME L");

   auto legend = new TLegend(0.125, 0.725, 0.625, 0.875);
   legend->AddEntry(graph1, "Number of decompressed pages", "l");
   legend->AddEntry(graph2, "Number of loaded clusters", "l");
   legend->Draw();

   latex.SetTextAlign(22);
   latex.DrawLatexNDC(0.5, 0.95, "Naive Reading");

   canvas->cd(3);

   auto graph3 = GetGraph(nPages);
   graph3->SetMarkerColor(kRed);
   graph3->GetYaxis()->SetNdivisions(8);
   graph3->GetYaxis()->SetRangeUser(-0.5, 14);
   graph3->Draw("AP");

   auto graph4 = GetGraph(nClusters);
   graph4->SetMarkerColor(kBlue);
   graph4->Draw("SAME P");

   legend = new TLegend(0.35, 0.725, 0.85, 0.875);
   legend->AddEntry(graph3, "Number of currently cached pages", "p");
   legend->AddEntry(graph4, "Number of currently cached clusters", "p");
   legend->Draw();

   // ===============================

   Read(EReadMode::kInformed, sumLoadedClusters, sumUnsealedPages, nClusters, nPages);

   canvas->cd(2);
   auto graph5 = GetGraph(sumUnsealedPages);
   graph5->SetLineColor(kRed);
   graph5->Draw("AL");

   auto graph6 = GetGraph(sumLoadedClusters);
   graph6->SetLineColor(kBlue);
   graph6->Draw("SAME L");

   latex.SetTextAlign(22);
   latex.DrawLatexNDC(0.5, 0.95, "Informed Reading");

   canvas->cd(4);

   auto graph7 = GetGraph(nPages);
   graph7->SetMarkerColor(kRed);
   graph7->GetYaxis()->SetNdivisions(8);
   graph7->GetYaxis()->SetRangeUser(-0.5, 14);
   graph7->Draw("AP");

   auto graph8 = GetGraph(nClusters);
   graph8->SetMarkerColor(kBlue);
   graph8->Draw("SAME P");
}
