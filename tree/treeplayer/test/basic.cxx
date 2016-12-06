#include "TLeaf.h"
#include "TROOT.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"

#include "gtest/gtest.h"

TTree* MakeTree() {
   double x[3]{};
   struct {
      unsigned int ny;
      int* y = nullptr;
   } yData;

   TTree* tree = new TTree("T", "test tree");
   tree->Branch("one", &x, "x[3]/D");
   tree->Branch("two", &yData, "ny/i:y[ny]/I");

   x[1] = 42.;
   yData.ny = 42;
   yData.y = new int[42]{};
   yData.y[0] = 17;
   tree->Fill();

   x[2] = 43.;
   yData.ny = 5;
   yData.y[4] = 7;
   tree->Fill();
   delete [] yData.y;

   tree->ResetBranchAddresses();

   return tree;
}

TEST(TTreeReaderBasic, Interfaces) {
   TTree* tree = MakeTree();

   TTreeReader tr(tree);
   TTreeReaderArray<double> x(tr, "one.x");
   TTreeReaderArray<int> y(tr, "two.y");
   TTreeReaderValue<unsigned int> ny(tr, "two.ny");

   // Before reading data:
   EXPECT_NE(tr.begin(), tr.end());
   //MISSING: EXPECT_EQ(2, tr.end() - tr.begin());
   EXPECT_EQ(-1, tr.GetCurrentEntry());
   EXPECT_EQ(2, tr.GetEntries(false));
   EXPECT_EQ(TTreeReader::kEntryNotLoaded, tr.GetEntryStatus());
   EXPECT_EQ(tree, tr.GetTree());
   EXPECT_FALSE(tr.IsChain());

   EXPECT_EQ(nullptr, ny.GetAddress());
   EXPECT_FALSE(ny.IsValid());
   EXPECT_STREQ("two.ny", ny.GetBranchName());
   EXPECT_EQ(ROOT::Internal::TTreeReaderValueBase::kReadNothingYet, ny.GetReadStatus());
   EXPECT_EQ(ROOT::Internal::TTreeReaderValueBase::kSetupNotSetup, ny.GetSetupStatus());
   EXPECT_EQ(ROOT::Internal::TTreeReaderValueBase::kReadNothingYet, ny.ProxyRead());

   // Skip to second entry:
   EXPECT_TRUE(tr.Next());
   EXPECT_TRUE(tr.Next());

   EXPECT_NE(tr.begin(), tr.end());
   //MISSING: EXPECT_EQ(2, tr.end() - tr.begin());
   EXPECT_EQ(1, tr.GetCurrentEntry());
   EXPECT_EQ(2, tr.GetEntries(false));


   EXPECT_EQ(5u, *ny);
   EXPECT_NE(nullptr, ny.GetAddress());
   EXPECT_TRUE(ny.IsValid());
   EXPECT_EQ(ny.GetAddress(), ny.Get());
   EXPECT_STREQ("two", ny.GetBranchName());
   EXPECT_STREQ("ny", ny.GetLeaf()->GetName());
   EXPECT_EQ(ROOT::Internal::TTreeReaderValueBase::kReadSuccess, ny.GetReadStatus());
   EXPECT_EQ(ROOT::Internal::TTreeReaderValueBase::kSetupMatch, ny.GetSetupStatus());
   EXPECT_EQ(ROOT::Internal::TTreeReaderValueBase::kReadSuccess, ny.ProxyRead());

   EXPECT_EQ(3, x.GetSize());
   EXPECT_EQ(5, y.GetSize());
   EXPECT_DOUBLE_EQ(43., x[2]);
   //FAILS: EXPECT_EQ(7, y[4]);


   EXPECT_FALSE(tr.Next());
   //FAILS: EXPECT_FALSE(ny.IsValid());
}


TEST(TTreeReaderBasic, ErrorProbing) {
   TTreeReader tr("doesNotExist", gROOT);
   EXPECT_EQ(TTreeReader::kEntryNoTree, tr.GetEntryStatus());
   EXPECT_EQ(nullptr, tr.GetTree());

   tr.SetTree(nullptr);
   EXPECT_EQ(TTreeReader::kEntryNoTree, tr.GetEntryStatus());
   EXPECT_EQ(nullptr, tr.GetTree());
}


TEST(TTreeReaderBasic, Range) {
   TTree* tree = MakeTree();
   TTreeReader tr(tree);

   EXPECT_EQ(TTreeReader::kEntryValid, tr.SetEntriesRange(1, 1));
   EXPECT_EQ(0, tr.GetCurrentEntry());
   EXPECT_TRUE(tr.Next());
   EXPECT_EQ(1, tr.GetCurrentEntry());
   EXPECT_EQ(TTreeReader::kEntryValid, tr.GetEntryStatus());

   // Read beyond end:
   EXPECT_FALSE(tr.Next());
   EXPECT_EQ(TTreeReader::kEntryNotFound, tr.GetEntryStatus());
   EXPECT_EQ(-1, tr.GetCurrentEntry());
}
