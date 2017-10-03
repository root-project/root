#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TRandom.h"

#include "gtest/gtest.h"

class TBranchTest : public ::testing::Test {
protected:
   virtual void SetUp()
   {
      random = new TRandom(837);
      f = new TFile("TBranchTestTree.root", "RECREATE");
      myTree = new TTree("myTree", "A test tree");
      myTree->SetAutoSave(10);
      Float_t data = 0;
      myTree->Branch("branch0", &data);

      for (Int_t ev = 0; ev < 100; ev++) {
         data = random->Gaus(100, 7);
         myTree->Fill();
         if (ev % 10 == 7) {
            myTree->FlushBaskets();
         }
      }
      f->Write();
      delete myTree;
      delete f;
   }

   virtual void TearDown()
   {
      myTree->DropBaskets();
      delete branch;
      delete myTree;
      delete random;
      delete f;
   }

   TRandom *random;
   TTree *myTree;
   TFile *f;
   TBranch *branch;
};

TEST_F(TBranchTest, nonePreviousTest)
{
   f = new TFile("TBranchTestTree.root");
   myTree = (TTree *)f->Get("myTree");
   branch = myTree->GetBranch("branch0");

   myTree->SetClusterPrefetch(false);
   myTree->SetMaxVirtualSize(0);

   // Checks for normal behavior when change is
   // not being used
   branch->GetEntry(0);
   ASSERT_TRUE(branch->GetListOfBaskets()->At(0));
   ASSERT_FALSE(branch->GetListOfBaskets()->At(1));

   branch->GetEntry(10);
   ASSERT_FALSE(branch->GetListOfBaskets()->At(0));
   ASSERT_FALSE(branch->GetListOfBaskets()->At(1));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(2));
   ASSERT_FALSE(branch->GetListOfBaskets()->At(3));
}

TEST_F(TBranchTest, onePreviousTest)
{
   f = new TFile("TBranchTestTree.root");
   myTree = (TTree *)f->Get("myTree");
   branch = myTree->GetBranch("branch0");

   // Checks to make sure only first basket is loaded
   branch->GetEntry(0);
   ASSERT_TRUE(branch->GetListOfBaskets()->At(0));
   ASSERT_FALSE(branch->GetListOfBaskets()->At(1));

   myTree->SetClusterPrefetch(true);

   // Checks to make sure the whole cluster is loaded
   branch->GetEntry(10);
   ASSERT_FALSE(branch->GetListOfBaskets()->At(0));
   ASSERT_FALSE(branch->GetListOfBaskets()->At(1));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(2));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(3));

   // Checks to make sure clusters are being removed
   // from memory, with none retained
   branch->GetEntry(20);
   ASSERT_FALSE(branch->GetListOfBaskets()->At(2));
   ASSERT_FALSE(branch->GetListOfBaskets()->At(3));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(4));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(5));

   myTree->SetMaxVirtualSize(-1);

   // Checks to make sure previous is retained in memory
   branch->GetEntry(30);
   ASSERT_TRUE(branch->GetListOfBaskets()->At(4));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(5));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(6));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(7));

   // Checks to make sure clusters are being removed
   // from memory
   branch->GetEntry(40);
   ASSERT_FALSE(branch->GetListOfBaskets()->At(4));
   ASSERT_FALSE(branch->GetListOfBaskets()->At(5));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(6));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(7));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(8));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(9));
}

TEST_F(TBranchTest, twoPreviousTest)
{
   f = new TFile("TBranchTestTree.root");
   myTree = (TTree *)f->Get("myTree");
   branch = myTree->GetBranch("branch0");

   myTree->SetMaxVirtualSize(-2);

   // Checks to make sure the whole cluster is loaded
   branch->GetEntry(0);
   ASSERT_TRUE(branch->GetListOfBaskets()->At(0));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(1));

   // Checks to make sure previous is retained in memory
   branch->GetEntry(10);
   ASSERT_TRUE(branch->GetListOfBaskets()->At(0));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(1));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(2));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(3));

   // Checks to make sure previous is retained in memory
   branch->GetEntry(20);
   ASSERT_TRUE(branch->GetListOfBaskets()->At(0));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(1));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(2));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(3));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(4));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(5));

   // Checks to make sure clusters are being removed
   // from memory
   branch->GetEntry(30);
   ASSERT_FALSE(branch->GetListOfBaskets()->At(0));
   ASSERT_FALSE(branch->GetListOfBaskets()->At(1));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(2));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(3));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(4));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(5));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(6));
   ASSERT_TRUE(branch->GetListOfBaskets()->At(7));
}
