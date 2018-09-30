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

   tree->ResetBranchAddresses();
}

static bool FileExists(const char *name)
{
   struct stat buffer;
   return stat(name, &buffer) == 0;
}

static void RemoveFile(const char *name)
{
   if (remove(name) != 0) {
      perror("failed to remove file");
      exit(1);
   }
}

TEST(TBufferMerger, CreateAndDestroy)
{
   TBufferMerger merger("tbuffermerger_create.root");

   RemoveFile("tbuffermerger_create.root");
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

   RemoveFile("tbuffermerger_create.root");
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

TEST(TBufferMerger, AutoSave)
{
   int nevents = 16384;
   int nthreads = 8;
   int events_per_thread = nevents / nthreads;

   ROOT::EnableThreadSafety();

   {
      TBufferMerger merger("tbuffermerger_autosave.root");

      merger.SetAutoSave(16 * 1024 * 1024); // Auto save every 16MB

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

            Fill(mytree, i * events_per_thread, events_per_thread);
            myfile->Write();
         });
      }

      for (auto &&t : threads)
         t.join();
   }

   EXPECT_TRUE(FileExists("tbuffermerger_autosave.root"));

   { // sum of all branch values in sequential mode
      TFile f("tbuffermerger_autosave.root");
      auto t = (TTree *)f.Get("mytree");

      int nentries = (int)t->GetEntries();

      EXPECT_EQ(nevents, nentries);
   }

   RemoveFile("tbuffermerger_autosave.root");
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
      ASSERT_TRUE(t != nullptr);

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

   RemoveFile("tbuffermerger_sequential.root");
   RemoveFile("tbuffermerger_parallel.root");
}
