#include <TBasket.h>
#include <TBranch.h>
#include <TFile.h>
#include <TRandom.h>
#include <TTree.h>

#include <gtest/gtest.h>

class TTreeTest : public ::testing::Test {
protected:
   void SetUp() override
   {
      auto random = new TRandom(836);
      auto file = new TFile("TTreeCompression.root", "RECREATE");
      auto tree = new TTree("tree", "A test tree");
      Double_t data = 0;
      tree->Branch("branch", &data);
      for (Int_t ev = 0; ev < 1000; ev++) {
         data = random->Gaus(100, 7);
         tree->Fill();
      }
      file->Write();
      delete random;
      delete tree;
      delete file;
   }
};

TEST_F(TTreeTest, testDefaultCompression)
{
   auto file = new TFile("TTreeCompression.root");
   auto tree = (TTree *)file->Get("tree");
   auto branch = (TBranch *)tree->GetBranch("branch");

   auto compress = file->GetCompressionSettings();
   ASSERT_EQ(compress, 101);

   compress = branch->GetCompressionSettings();
   ASSERT_EQ(compress, 101);

   delete file;
}
