#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TRandom.h"

#include "gtest/gtest.h"

class TTreeClusterTest : public ::testing::Test {
protected:
   virtual void SetUp()
   {
      auto random = new TRandom(836);
      auto file = new TFile("TTreeClusterTest.root", "RECREATE");
      auto tree = new TTree("tree", "A test tree");
      auto tree2 = new TTree("tree2", "A test tree with one-basket-per-cluster");
      tree2->SetBit(TTree::kOnlyFlushAtCluster);
      tree->SetAutoFlush(500);
      tree2->SetAutoFlush(500);
      Double_t data = 0;
      auto branch = tree->Branch("branch", &data);
      auto branch2 = tree2->Branch("branch", &data);

      for (Int_t ev = 0; ev < 1000; ev++) {
         data = random->Gaus(100, 7);
         tree->Fill();
         tree2->Fill();
         if (ev == 100) {
            // 8 bytes per double; 500 doubles per event cluster; we need at
            // least 4000 bytes to hold the data of one cluster.  Setting the
            // basket size well under this to force multiple baskets per event
            // cluster in "regular mode".
            branch->SetBasketSize(100);
            branch2->SetBasketSize(100);
         }
      }
      file->Write();
      delete random;
      delete tree;
      delete file;
   }
};

TEST_F(TTreeClusterTest, countBaskets)
{
   auto file = new TFile("TTreeClusterTest.root");
   auto tree = static_cast<TTree *>(file->Get("tree"));
   auto tree2 = static_cast<TTree *>(file->Get("tree2"));
   auto branch = tree->GetBranch("branch");
   auto branch2 = tree2->GetBranch("branch");

   ASSERT_TRUE(branch->GetWriteBasket() == 102);
   ASSERT_TRUE(branch2->GetWriteBasket() == 2);

   delete file;
}
