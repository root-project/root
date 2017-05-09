#include "ROOT/TBufferMerger.hxx"

#include "TFile.h"
#include "TROOT.h"
#include "TTree.h"

#include <memory>
#include <thread>

#include "gtest/gtest.h"

using namespace ROOT::Experimental;

static void Fill(std::shared_ptr<TTree> tree, int init, int count)
{
   int n = 0;

   tree->Branch("n", &n, "n/I");

   for (int i = 0; i < count; ++i) {
      n = init + i;
      tree->Fill();
   }
}

TEST(TBufferMerger, CreateAndDestroy)
{
   TBufferMerger merger("tbuffermerger_create.root");
}

TEST(TBufferMerger, CreateAndDestroyWithAttachedFiles)
{
   TBufferMerger merger("tbuffermerger_create.root");

   auto f1 = merger.GetFile();
   auto f2 = merger.GetFile();
   auto f3 = merger.GetFile();
}

TEST(TBufferMerger, SequentialTreeFill)
{
   int nevents = 1024;

   ROOT::EnableThreadSafety();

   {
      TBufferMerger merger("tbuffermerger_sequential.root");

      auto myfile = merger.GetFile();
      auto mytree = std::make_shared<TTree>("mytree", "mytree");

      Fill(mytree, 0, nevents);

      mytree->Write();
      myfile->Write();
   }
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
            auto mytree = std::make_shared<TTree>("mytree", "mytree");

            Fill(mytree, i * nevents, nevents);

            mytree->Write();
            myfile->Write();
         });
      }

      for (auto &&t : threads) t.join();
   }
}

TEST(TBufferMerger, CheckTreeFillResults)
{
   int sum_s, sum_p;

   { // sum of all branch values in sequential mode
      auto f = std::unique_ptr<TFile>(TFile::Open("tbuffermerger_sequential.root"));
      auto t = std::unique_ptr<TTree>((TTree *)f->Get("mytree"));

      int n, sum = 0;
      int nentries = (int)t->GetEntries();

      t->SetBranchAddress("n", &n);

      for (int i = 0; i < nentries; ++i) {
         t->GetEntry(i);
         sum += n;
      }

      sum_s = sum;
   }

   { // sum of all branch values in parallel mode
      auto f = std::unique_ptr<TFile>(TFile::Open("tbuffermerger_parallel.root"));
      auto t = std::unique_ptr<TTree>((TTree *)f->Get("mytree"));

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
