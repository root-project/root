/// \file
/// \ingroup tutorial_ntuple
/// \notebook
/// Example of staged cluster committing in multi-threaded writing using RNTupleParallelWriter.
///
/// \macro_code
///
/// \date September 2024
/// \author The ROOT Team

// NOTE: The RNTupleParallelWriter is experimental at this point.
// Functionality and interface are still subject to changes.

#include <ROOT/RNTupleFillContext.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleParallelWriter.hxx>

#include <TRandom3.h>

#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

// Import classes from experimental namespace for the time being
using ROOT::Experimental::RNTupleParallelWriter;

// Where to store the ntuple of this example
constexpr char const *kNTupleFileName = "ntpl013_staged.root";

// Number of parallel threads to fill the ntuple
constexpr int kNWriterThreads = 4;

// Number of events to generate is kNEventsPerThread * kNWriterThreads
constexpr int kNEventsPerThread = 25000;

// Number of events per block
constexpr int kNEventsPerBlock = 10000;

// Thread function to generate and write events
void FillData(int id, RNTupleParallelWriter *writer)
{
   // static variables that are shared between threads; this is done for simplicity in this tutorial, use proper data
   // structures in real code!
   static std::mutex g_Mutex;
   static ROOT::NTupleSize_t g_WrittenEntries = 0;

   using generator = std::mt19937;
   generator gen;

   // Create a fill context and turn on staged cluster committing.
   auto fillContext = writer->CreateFillContext();
   fillContext->EnableStagedClusterCommitting();
   auto entry = fillContext->CreateEntry();

   auto eventId = entry->GetPtr<std::uint32_t>("eventId");
   auto eventIdStart = id * kNEventsPerThread;
   auto rndm = entry->GetPtr<float>("rndm");

   for (int i = 0; i < kNEventsPerThread; i++) {
      // Prepare the entry with an id and a random number.
      *eventId = eventIdStart + i;
      auto d = static_cast<double>(gen()) / generator::max();
      *rndm = static_cast<float>(d);

      // Fill might auto-flush a cluster, which will be staged.
      fillContext->Fill(*entry);
   }

   // It is important to first FlushCluster() so that a cluster with the remaining entries is staged.
   fillContext->FlushCluster();
   {
      std::lock_guard g(g_Mutex);
      fillContext->CommitStagedClusters();
      std::cout << "Thread #" << id << " wrote events #" << eventIdStart << " - #"
                << (eventIdStart + kNEventsPerThread - 1) << " as entries #" << g_WrittenEntries << " - #"
                << (g_WrittenEntries + kNEventsPerThread - 1) << std::endl;
      g_WrittenEntries += kNEventsPerThread;
   }
}

// Generate kNEvents with multiple threads in kNTupleFileName
void Write()
{
   std::cout << " === Writing with staged cluster committing ===" << std::endl;

   // Create the data model
   auto model = ROOT::RNTupleModel::CreateBare();
   model->MakeField<std::uint32_t>("eventId");
   model->MakeField<float>("rndm");

   // Create RNTupleWriteOptions to make the writing commit multiple clusters.
   // This is for demonstration purposes only to have multiple clusters per
   // thread that are implicitly flushed, and should not be copied into real
   // code!
   ROOT::RNTupleWriteOptions options;
   options.SetApproxZippedClusterSize(32'000);

   // We hand over the data model to a newly created ntuple of name "NTuple", stored in kNTupleFileName
   auto writer = RNTupleParallelWriter::Recreate(std::move(model), "NTuple", kNTupleFileName, options);

   std::vector<std::thread> threads;
   for (int i = 0; i < kNWriterThreads; ++i)
      threads.emplace_back(FillData, i, writer.get());
   for (int i = 0; i < kNWriterThreads; ++i)
      threads[i].join();

   // The writer unique pointer goes out of scope here.  On destruction, the writer flushes unwritten data to disk
   // and closes the attached ROOT file.
}

void FillDataInBlocks(int id, RNTupleParallelWriter *writer)
{
   // static variables that are shared between threads; this is done for simplicity in this tutorial, use proper data
   // structures in real code!
   static std::mutex g_Mutex;
   static ROOT::NTupleSize_t g_WrittenEntries = 0;

   using generator = std::mt19937;
   generator gen;

   // Create a fill context and turn on staged cluster committing.
   auto fillContext = writer->CreateFillContext();
   fillContext->EnableStagedClusterCommitting();
   auto entry = fillContext->CreateEntry();

   auto eventId = entry->GetPtr<std::uint32_t>("eventId");
   auto eventIdStart = id * kNEventsPerThread;
   int startOfBlock = 0;
   auto rndm = entry->GetPtr<float>("rndm");

   for (int i = 0; i < kNEventsPerThread; i++) {
      // Prepare the entry with an id and a random number.
      *eventId = eventIdStart + i;
      auto d = static_cast<double>(gen()) / generator::max();
      *rndm = static_cast<float>(d);

      // Fill might auto-flush a cluster, which will be staged.
      fillContext->Fill(*entry);

      if ((i + 1) % kNEventsPerBlock == 0) {
         // Decide to flush this cluster and logically append all staged clusters to the ntuple.
         fillContext->FlushCluster();
         {
            std::lock_guard g(g_Mutex);
            fillContext->CommitStagedClusters();
            auto firstEvent = eventIdStart + startOfBlock;
            auto lastEvent = eventIdStart + i;
            std::cout << "Thread #" << id << " wrote events #" << firstEvent << " - #" << lastEvent << " as entries #"
                      << g_WrittenEntries << " - #" << (g_WrittenEntries + kNEventsPerBlock - 1) << std::endl;
            g_WrittenEntries += kNEventsPerBlock;
            startOfBlock += kNEventsPerBlock;
         }
      }
   }

   // Flush the remaining data and commit staged clusters.
   fillContext->FlushCluster();
   {
      std::lock_guard g(g_Mutex);
      fillContext->CommitStagedClusters();
      auto firstEvent = eventIdStart + startOfBlock;
      auto lastEvent = eventIdStart + kNEventsPerThread - 1;
      auto numEvents = kNEventsPerThread - startOfBlock;
      std::cout << "Thread #" << id << " wrote events #" << firstEvent << " - #" << lastEvent << " as entries #"
                << g_WrittenEntries << " - #" << (g_WrittenEntries + numEvents - 1) << std::endl;
      g_WrittenEntries += numEvents;
   }
}

// Generate kNEvents with multiple threads in kNTupleFileName, and sequence them in blocks of kNEventsPerBlock entries
void WriteInBlocks()
{
   std::cout << "\n === ... with sequencing in blocks of " << kNEventsPerBlock << " events ===" << std::endl;

   // Create the data model
   auto model = ROOT::RNTupleModel::CreateBare();
   model->MakeField<std::uint32_t>("eventId");
   model->MakeField<float>("rndm");

   // Create RNTupleWriteOptions to make the writing commit multiple clusters.
   // This is for demonstration purposes only to have multiple clusters per
   // thread and also per block that are implicitly flushed, and can be mixed
   // with explicit calls to FlushCluster(). This should not be copied into real
   // code!
   ROOT::RNTupleWriteOptions options;
   options.SetApproxZippedClusterSize(32'000);

   // We hand over the data model to a newly created ntuple of name "NTuple", stored in kNTupleFileName
   auto writer = RNTupleParallelWriter::Recreate(std::move(model), "NTuple", kNTupleFileName, options);

   std::vector<std::thread> threads;
   for (int i = 0; i < kNWriterThreads; ++i)
      threads.emplace_back(FillDataInBlocks, i, writer.get());
   for (int i = 0; i < kNWriterThreads; ++i)
      threads[i].join();

   // The writer unique pointer goes out of scope here.  On destruction, the writer flushes unwritten data to disk
   // and closes the attached ROOT file.
}

void ntpl013_staged()
{
   Write();
   WriteInBlocks();
}
