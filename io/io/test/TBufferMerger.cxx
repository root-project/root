#include "RConfigure.h"
#include "ROOT/TBufferMerger.hxx"

#include "ROOT/TTaskGroup.hxx"

#include "TFile.h"
#include "TROOT.h"
#include "TTree.h"

#include <atomic>
#include <cstdio>
#include <future>
#include <memory>
#include <thread>
#include <sys/stat.h>

#include "gtest/gtest.h"

using namespace ROOT::Experimental;

static void Fill(TTree *tree, int init, int count)
{
   int n = 0;

   tree->Branch("n", &n, "n/I");

   for (int i = 0; i < count; ++i) {
      n = init + i;
      tree->Fill();
   }
}

static bool FileExists(const char *name)
{
   struct stat buffer;
   return stat(name, &buffer) == 0;
}

TEST(TBufferMerger, CreateAndDestroy)
{
   TBufferMerger merger("tbuffermerger_create.root");
}

TEST(TBufferMerger, CreateAndDestroyWithAttachedFiles)
{
   ROOT::EnableThreadSafety();

   {
      TBufferMerger merger("tbuffermerger_create.root");

      auto f1 = merger.GetFile();
      auto f2 = merger.GetFile();
      auto f3 = merger.GetFile();
   }

   EXPECT_TRUE(FileExists("tbuffermerger_create.root"));
}

TEST(TBufferMerger, SequentialTreeFill)
{
   int nevents = 1024;

   ROOT::EnableThreadSafety();

   {
      TBufferMerger merger("tbuffermerger_sequential.root");

      auto myfile = merger.GetFile();
      auto mytree = new TTree("mytree", "mytree");

      // The resetting of the kCleanup bit below is necessary to avoid leaving
      // the management of this object to ROOT, which leads to a race condition
      // that may cause a crash once all threads are finished and the final
      // merge is happening
      mytree->ResetBit(kMustCleanup);

      Fill(mytree, 0, nevents);
      myfile->Write();
   }

   EXPECT_TRUE(FileExists("tbuffermerger_sequential.root"));
}

TEST(TBufferMerger, ParallelTreeFill)
{
   int nthreads = 4;
   int nevents = 256;

   ROOT::EnableThreadSafety();

   {
      TBufferMerger merger("tbuffermerger_parallel.root");
      std::vector<std::thread> threads;
      for (int i = 0; i < nthreads; ++i) {
         threads.emplace_back([=, &merger]() {
            auto myfile = merger.GetFile();
            auto mytree = new TTree("mytree", "mytree");

            // The resetting of the kCleanup bit below is necessary to avoid leaving
            // the management of this object to ROOT, which leads to a race condition
            // that may cause a crash once all threads are finished and the final
            // merge is happening
            mytree->ResetBit(kMustCleanup);

            Fill(mytree, i * nevents, nevents);
            myfile->Write();
         });
      }

      for (auto &&t : threads)
         t.join();
   }

   EXPECT_TRUE(FileExists("tbuffermerger_parallel.root"));
}

TEST(TBufferMerger, CheckTreeFillResults)
{
   int sum_s, sum_p;

   ASSERT_TRUE(FileExists("tbuffermerger_sequential.root"));

   { // sum of all branch values in sequential mode
      TFile f("tbuffermerger_sequential.root");
      auto t = (TTree *)f.Get("mytree");

      int n, sum = 0;
      int nentries = (int)t->GetEntries();

      t->SetBranchAddress("n", &n);

      for (int i = 0; i < nentries; ++i) {
         t->GetEntry(i);
         sum += n;
      }

      sum_s = sum;
   }

   ASSERT_TRUE(FileExists("tbuffermerger_parallel.root"));

   { // sum of all branch values in parallel mode
      TFile f("tbuffermerger_parallel.root");
      auto t = (TTree *)f.Get("mytree");

      int n, sum = 0;
      int nentries = (int)t->GetEntries();

      t->SetBranchAddress("n", &n);

      for (int i = 0; i < nentries; ++i) {
         t->GetEntry(i);
         sum += n;
      }

      sum_p = sum;
   }

   // Note: 0 + 1 + ... + 1024 = 523776

   EXPECT_EQ(523776, sum_s);
   EXPECT_EQ(523776, sum_p);
}

TEST(TBufferMerger, RegisterCallbackThreads)
{
   const char *testfile = "tbuffermerger_threads.root";
   int events = 1000;
   int events_per_task = 50;
   int tasks = events / events_per_task;
   std::atomic<int> launched(0);
   std::atomic<int> processed(0);

   {
      ROOT::EnableThreadSafety();
      TBufferMerger merger(testfile);

      /* define a task: create and push some events */
      auto task = [&]() {
         auto myfile = merger.GetFile();
         auto mytree = new TTree("mytree", "mytree");

         mytree->ResetBit(kMustCleanup);

         int n = 1;
         mytree->Branch("n", &n, "n/I");

         for (int i = 0; i < events_per_task; ++i)
            mytree->Fill();

         myfile->Write();

         processed += events_per_task;
      };

      std::vector<std::thread> workers;

      /* callback: launches new tasks when called */
      merger.RegisterCallback([&]() {
         int i = 0;
         while (launched < tasks && ++i <= 2) {
            workers.emplace_back(task);
            ++launched;
         }
      });

      launched = 1;
      workers.emplace_back(task);

      /* wait until all tasks have been launched */
      while (launched != tasks)
         ;

      /* wait until all tasks have finished running */
      for (auto &w : workers)
         w.join();

      ASSERT_EQ(processed, events);
   }

   ASSERT_TRUE(FileExists(testfile));

   /* open file and check data integrity */
   {
      TFile f(testfile);
      auto t = (TTree *)f.Get("mytree");

      int n, sum = 0;
      int nentries = (int)t->GetEntries();

      EXPECT_EQ(nentries, events);

      t->SetBranchAddress("n", &n);

      for (int i = 0; i < nentries; ++i) {
         t->GetEntry(i);
         sum += n;
      }

      EXPECT_EQ(sum, events);
   }

   remove(testfile);
}

TEST(TBufferMerger, RegisterCallbackTasks)
{
   using namespace ROOT::Experimental;

   const char *testfile = "tbuffermerger_tasks.root";

   std::mutex m;
   int events = 1000;
   int events_per_task = 50;
   int tasks = events / events_per_task;
   volatile int launched = 0;
   volatile int processed = 0;

   {
#ifdef R__USE_IMT
      ROOT::EnableImplicitMT();
#endif
      TBufferMerger merger(testfile);

      /* define a task: create and push some events */
      auto task = [&]() {
         auto myfile = merger.GetFile();
         auto mytree = new TTree("mytree", "mytree");

         mytree->ResetBit(kMustCleanup);

         int n = 1;
         mytree->Branch("n", &n, "n/I");

         for (int i = 0; i < events_per_task; ++i)
            mytree->Fill();

         myfile->Write();

         std::lock_guard<std::mutex> lk(m);
         processed += events_per_task;
      };

      TTaskGroup tg;

      /* callback: launches new tasks when called */
      merger.RegisterCallback([&]() {
         int i = 0;
         while (launched < tasks && ++i <= 2) {
            tg.Run(task);
            ++launched;
         }
      });

      launched = 1;
      tg.Run(task);

      /* wait until all tasks have been launched */
      while (launched != tasks)
         ;

      /* wait until all tasks have finished running */
      tg.Wait();

      ASSERT_EQ(processed, events);
   }

   ASSERT_TRUE(FileExists(testfile));

   /* open file and check data integrity */
   {
      TFile f(testfile);
      auto t = (TTree *)f.Get("mytree");

      int n, sum = 0;
      int nentries = (int)t->GetEntries();

      EXPECT_EQ(nentries, events);

      t->SetBranchAddress("n", &n);

      for (int i = 0; i < nentries; ++i) {
         t->GetEntry(i);
         sum += n;
      }

      EXPECT_EQ(sum, events);
   }

   remove(testfile);
}
