#include "ROOT/TBufferMerger.hxx"

#include "TROOT.h" // For EnableThreadSafety

#include "TTree.h"

#include "benchmark/benchmark.h"

#include <random>
#include <string>
#include <sstream>
#include <sys/stat.h>

using namespace ROOT::Experimental;

static void BM_TBufferFile_CreateEmpty(benchmark::State &state)
{
   const char *filename = "empty.root";
   while (state.KeepRunning()) {
      TBufferMerger m(std::unique_ptr<TMemFile>(new TMemFile(filename, "RECREATE")));
   }
}
BENCHMARK(BM_TBufferFile_CreateEmpty);

TBufferMerger *Merger = nullptr;
static void BM_TBufferFile_GetFile(benchmark::State &state)
{
   ROOT::EnableThreadSafety();
   using namespace ROOT::Experimental;
   if (state.thread_index == 0) {
      // Setup code here.
      // FIXME: We should have a way to pass an externally constructed file or stream to
      // TFile*, this would allow us to create in-memory files and avoid killing disks
      // when we benchmark IO.
      Merger = new TBufferMerger(std::unique_ptr<TMemFile>(new TMemFile("virtual_file.root", "RECREATE")));
   }
   while (state.KeepRunning()) {
      // Run the test as normal.
      auto myFile = Merger->GetFile();
   }
   if (state.thread_index == 0) {
      // Teardown code here.
      delete Merger;
   }
}
BENCHMARK(BM_TBufferFile_GetFile)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_TBufferFile_GetFile)->Unit(benchmark::kMicrosecond)->UseRealTime()->ThreadPerCpu();
BENCHMARK(BM_TBufferFile_GetFile)->Unit(benchmark::kMicrosecond)->UseRealTime()->ThreadRange(1, 256);

/// Creates a TMemFile, fills a TTree with random numbers. The data is written if it exceeds 32MB.
inline void FillTreeWithRandomData(TBufferMerger &merger, size_t nEntriesPerWorker = 24 * 1024, int flush = 32 )
{
   thread_local std::default_random_engine g;
   std::uniform_real_distribution<double> dist(0.0, 1.0);

   auto f = merger.GetFile();
   auto t = new TTree("random", "random");
   t->ResetBit(kMustCleanup);
   t->SetAutoFlush(-(flush) * 1024 * 1024); // Flush at exceeding 32MB

   double rng;

   t->Branch("random", &rng);

   long entries = 0;
   for (size_t i = 0; i < nEntriesPerWorker; ++i) {
      rng = dist(g);
      t->Fill();
      ++entries;
      auto atflush = t->GetAutoFlush();
      if (entries == atflush) {
         entries = 0;
         f->Write();
      }
   }
   f->Write();
}

static void BM_TBufferFile_FillTreeWithRandomData(benchmark::State &state)
{
   ROOT::EnableThreadSafety();
   using namespace ROOT::Experimental;
   if (state.thread_index == 0) {
      // Setup code here.
      Merger = new TBufferMerger(std::unique_ptr<TMemFile>(new TMemFile("virtual_file.root", "RECREATE")));
   }
   int flush = state.range(0);
   while (state.KeepRunning())
      FillTreeWithRandomData(*Merger, flush);
   auto size = Merger->GetFile()->GetSize();
   std::stringstream ss;
   ss << size;
   state.SetLabel(ss.str());
   if (state.thread_index == 0) {
      // Teardown code here.
      delete Merger;
   }
}
BENCHMARK(BM_TBufferFile_FillTreeWithRandomData)->Unit(benchmark::kMicrosecond)->Arg(32);
BENCHMARK(BM_TBufferFile_FillTreeWithRandomData)->Unit(benchmark::kMicrosecond)->Arg(32)->UseRealTime()->ThreadPerCpu();
BENCHMARK(BM_TBufferFile_FillTreeWithRandomData)->Unit(benchmark::kMicrosecond)->Range(1, 32)->UseRealTime()->ThreadRange(1, 256);

// Define our main.
BENCHMARK_MAIN();
