/// \file
/// \ingroup tutorial_multicore
///
/// Fill an Ntuple in distinct workers, and write the output to a file.
/// This tutorial illustrates the basics of how it's possible with ROOT
/// to write simultaneously to a single output file using TBufferMerger.
///
/// \macro_code
///
/// \author Guilherme Amadio
/// \date May 2017

#include <ROOT/TBufferMerger.hxx>

using ROOT::Experimental::TBufferMerger;
using ROOT::Experimental::TBufferMergerFile;

#include <random>
#include <thread>

// A simple function to fill the ntuple with random values
void fill(TNtuple &ntuple, size_t n)
{
   std::random_device rd;
   std::default_random_engine rng(rd());
   std::normal_distribution<double> dist(0.0, 1.0);

   auto gaussian_random = [&]() { return dist(rng); };

   for (auto i : ROOT::TSeqI(n)) ntuple.Fill(gaussian_random());
}

void mt103_fillNtuples()
{
   // Avoid unnecessary output
   gROOT->SetBatch();

   // Make ROOT thread-safe
   ROOT::EnableThreadSafety();

   // Total number of events
   const size_t nEvents = 65535;

   // Match number of threads to what the hardware can do
   const size_t nWorkers = std::thread::hardware_concurrency();

   // Split work in equal parts
   const size_t nEventsPerWorker = nEvents / nWorkers;

   // Create the TBufferMerger
   TBufferMerger merger("mp103_fillNtuple.root");

   // Define what each worker will do
   auto work_function = [&]() {
      auto f = merger.GetFile();
      TNtuple ntrand("ntrand", "Random Numbers", "r");
      fill(ntrand, nEventsPerWorker);
      ntrand.Write();
      f->Write();
   };

   // Create worker threads
   std::vector<std::thread> workers;

   for (auto i : ROOT::TSeqI(nWorkers)) workers.emplace_back(work_function);

   // Make sure workers are done
   for (auto &&worker : workers) worker.join();
}
