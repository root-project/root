#include "TLeaf.h"
#include "TROOT.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"

#include "gtest/gtest.h"

TTree* MakeTree() {
   double x[3]{};
   unsigned int ny;
   int* y = nullptr;

   TTree* tree = new TTree("T", "test tree");
   tree->Branch("one", &x, "x/D[3]");
   tree->Branch("two", &ny, "ny/i");
   tree->Branch("three", &y, "y/I[ny]");

   x[1] = 42.;
   ny = 42;
   y = new int[42]{};
   y[0] = 17;
   tree->Fill();
   delete [] y;

   x[2] = 43.;
   ny = 5;
   y = new int[5]{};
   y[4] = 7;
   tree->Fill();
   delete [] y;

   tree->ResetBranchAddresses();

   return tree;
}

TEST(TTreeReaderBasic, Interfaces) {
   TTree* tree = MakeTree();

   TTreeReader tr(tree);
   TTreeReaderArray<double> x(tr, "one.x");
   TTreeReaderArray<int> y(tr, "three.y");
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


   EXPECT_EQ(5, *ny);
   EXPECT_NE(nullptr, ny.GetAddress());
   EXPECT_EQ(ny.GetAddress(), ny.Get());
   EXPECT_STREQ("two", ny.GetBranchName());
   EXPECT_STREQ("ny", ny.GetLeaf()->GetName());
   EXPECT_EQ(ROOT::Internal::TTreeReaderValueBase::kReadSuccess, ny.GetReadStatus());
   EXPECT_EQ(ROOT::Internal::TTreeReaderValueBase::kSetupMatch, ny.GetSetupStatus());
   EXPECT_EQ(ROOT::Internal::TTreeReaderValueBase::kReadSuccess, ny.ProxyRead());

   //FAILS: EXPECT_EQ(3, x.GetSize());
   //FAILS: EXPECT_EQ(5, y.GetSize());
   //FAILS: EXPECT_DOUBLE_EQ(42., x[2]);
   //FAILS: EXPECT_EQ(7, y[4]);


   EXPECT_FALSE(tr.Next());
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
   EXPECT_EQ(1, tr.GetCurrentEntry());
}
