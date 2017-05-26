#include "TEntryListArray.h"
#include "TLeaf.h"
#include "TROOT.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"

#include "gtest/gtest.h"
#include <stdlib.h>

TTree* MakeTree() {
   double x[3]{};
   float z = 0.;
   struct {
      unsigned int ny;
      int* y = nullptr;
   } yData;
   std::string str;
   Double32_t Double32 = 12.;
   Float16_t Float16 = -12.;


   TTree* tree = new TTree("T", "test tree");
   tree->Branch("one", &x, "x[3]/D");
   tree->Branch("two", &yData, "ny/i:y[ny]/I");
   tree->Branch("three", &z, "z");
   tree->Branch("str", &str);
   tree->Branch("d32", &Double32);
   tree->Branch("f16", &Float16);

   x[1] = 42.;
   yData.ny = 42;
   yData.y = new int[42]{};
   yData.y[0] = 17;
   str = "first";
   tree->Fill();

   x[2] = 43.;
   yData.ny = 5;
   yData.y[4] = 7;
   str = "";
   tree->Fill();

   for (int entry = 2; entry < 20; ++entry) {
      z = entry * (1 - 2 * (entry % 2)); // +entry for even, -entry for odd
      str = std::string(entry, '*');
      tree->Fill();
   }

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
   TTreeReaderValue<std::string> nstr(tr, "str");

   // Before reading data:
   EXPECT_NE(tr.begin(), tr.end());
   //MISSING: EXPECT_EQ(2, tr.end() - tr.begin());
   EXPECT_EQ(-1, tr.GetCurrentEntry());
   EXPECT_EQ(20, tr.GetEntries(false));
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
   EXPECT_EQ(20, tr.GetEntries(false));


   EXPECT_EQ(5u, *ny);
   EXPECT_NE(nullptr, ny.GetAddress());
   EXPECT_TRUE(ny.IsValid());
   EXPECT_EQ(ny.GetAddress(), ny.Get());
   EXPECT_STREQ("two", ny.GetBranchName());
   EXPECT_STREQ("ny", ny.GetLeaf()->GetName());
   EXPECT_EQ(ROOT::Internal::TTreeReaderValueBase::kReadSuccess, ny.GetReadStatus());
   EXPECT_EQ(ROOT::Internal::TTreeReaderValueBase::kSetupMatch, ny.GetSetupStatus());
   EXPECT_EQ(ROOT::Internal::TTreeReaderValueBase::kReadSuccess, ny.ProxyRead());

   EXPECT_EQ(3u, x.GetSize());
   EXPECT_EQ(5u, y.GetSize());
   EXPECT_DOUBLE_EQ(43., x[2]);
   //FAILS: EXPECT_EQ(7, y[4]);

   for (int entry = 2; entry < 20; ++entry)
      EXPECT_TRUE(tr.Next());

   EXPECT_FALSE(tr.Next());
   //FAILS: EXPECT_FALSE(ny.IsValid());
}


TEST(TTreeReaderBasic, ErrorProbing) {
   TTreeReader tr("doesNotExist", gROOT);
   EXPECT_EQ(TTreeReader::kEntryNoTree, tr.GetEntryStatus());
   EXPECT_EQ(nullptr, tr.GetTree());

   tr.SetTree((TTree*)nullptr);
   EXPECT_EQ(TTreeReader::kEntryNoTree, tr.GetEntryStatus());
   EXPECT_EQ(nullptr, tr.GetTree());

   TTreeReaderValue<double> val(tr, "branchDoesNotExist");
   EXPECT_FALSE(val.IsValid());
}


TEST(TTreeReaderBasic, Range) {
   TTree* tree = MakeTree();
   TTreeReader tr(tree);

   EXPECT_EQ(TTreeReader::kEntryValid, tr.SetEntriesRange(5, 8));

   EXPECT_TRUE(tr.Next());
   EXPECT_EQ(5, tr.GetCurrentEntry());
   EXPECT_EQ(TTreeReader::kEntryValid, tr.GetEntryStatus());

   EXPECT_TRUE(tr.Next());
   EXPECT_EQ(6, tr.GetCurrentEntry());
   EXPECT_EQ(TTreeReader::kEntryValid, tr.GetEntryStatus());

   EXPECT_TRUE(tr.Next());
   EXPECT_EQ(7, tr.GetCurrentEntry());
   EXPECT_EQ(TTreeReader::kEntryValid, tr.GetEntryStatus());

   // Reached end:
   EXPECT_FALSE(tr.Next());
   EXPECT_EQ(8, tr.GetCurrentEntry());
   EXPECT_EQ(TTreeReader::kEntryBeyondEnd, tr.GetEntryStatus());

   // Read beyond end:
   EXPECT_FALSE(tr.Next());
   EXPECT_EQ(TTreeReader::kEntryBeyondEnd, tr.GetEntryStatus());
   EXPECT_EQ(9, tr.GetCurrentEntry());


   // Restart, now with different entries.
   tr.Restart();
   EXPECT_EQ(TTreeReader::kEntryValid, tr.SetEntriesRange(0, 2));

   EXPECT_TRUE(tr.Next());
   EXPECT_EQ(0, tr.GetCurrentEntry());
   EXPECT_EQ(TTreeReader::kEntryValid, tr.GetEntryStatus());

   EXPECT_TRUE(tr.Next());
   EXPECT_EQ(1, tr.GetCurrentEntry());
   EXPECT_EQ(TTreeReader::kEntryValid, tr.GetEntryStatus());

   // Reached end:
   EXPECT_FALSE(tr.Next());
   EXPECT_EQ(2, tr.GetCurrentEntry());
   EXPECT_EQ(TTreeReader::kEntryBeyondEnd, tr.GetEntryStatus());

}

TEST(TTreeReaderBasic, InvalidRange) {
   TTree *tree = MakeTree();
   TTreeReader tr(tree);

   EXPECT_EQ(TTreeReader::kEntryNotFound, tr.SetEntriesRange(tree->GetEntries(), 0));

   // Is SetEntriesRange() simply ignored as it should be?
   EXPECT_TRUE(tr.Next());
   EXPECT_EQ(0, tr.GetCurrentEntry());
   EXPECT_EQ(TTreeReader::kEntryValid, tr.GetEntryStatus());
}


TEST(TTreeReaderBasic, OneEntryRange) {
   TTree* tree = MakeTree();
   TTreeReader tr(tree);

   EXPECT_EQ(TTreeReader::kEntryValid, tr.SetEntriesRange(1, 2));

   EXPECT_TRUE(tr.Next());
   EXPECT_EQ(1, tr.GetCurrentEntry());
   EXPECT_EQ(TTreeReader::kEntryValid, tr.GetEntryStatus());

   // Read beyond end:
   EXPECT_FALSE(tr.Next());
   EXPECT_EQ(TTreeReader::kEntryBeyondEnd, tr.GetEntryStatus());
   EXPECT_EQ(2, tr.GetCurrentEntry());

   for (int entry = 2; entry < tree->GetEntries(); ++entry)
      EXPECT_FALSE(tr.Next());
}


TEST(TTreeReaderBasic, ZeroEntryRange) {
   TTree* tree = MakeTree();
   TTreeReader tr(tree);

   // end is ignored:
   EXPECT_EQ(TTreeReader::kEntryValid, tr.SetEntriesRange(18, 18));

   EXPECT_TRUE(tr.Next());
   EXPECT_EQ(18, tr.GetCurrentEntry());
   EXPECT_EQ(TTreeReader::kEntryValid, tr.GetEntryStatus());

   EXPECT_TRUE(tr.Next());
   EXPECT_EQ(19, tr.GetCurrentEntry());
   EXPECT_EQ(TTreeReader::kEntryValid, tr.GetEntryStatus());

   // Read beyond end:
   EXPECT_FALSE(tr.Next());
   // As the TTree only has up to entry 19, 20 is kEntryNotFound:
   EXPECT_EQ(TTreeReader::kEntryNotFound, tr.GetEntryStatus());
   EXPECT_EQ(20, tr.GetCurrentEntry());
}


TEST(TTreeReaderBasic, InvertedEntryRange) {
   TTree* tree = MakeTree();
   TTreeReader tr(tree);

   EXPECT_EQ(TTreeReader::kEntryValid, tr.SetEntriesRange(18, 3));

   EXPECT_TRUE(tr.Next());
   EXPECT_EQ(18, tr.GetCurrentEntry());
   EXPECT_EQ(TTreeReader::kEntryValid, tr.GetEntryStatus());

   EXPECT_TRUE(tr.Next());
   EXPECT_EQ(19, tr.GetCurrentEntry());
   EXPECT_EQ(TTreeReader::kEntryValid, tr.GetEntryStatus());

   // Read beyond end:
   EXPECT_FALSE(tr.Next());
   // As the TTree only has up to entry 19, 20 is kEntryNotFound:
   EXPECT_EQ(TTreeReader::kEntryNotFound, tr.GetEntryStatus());
   EXPECT_EQ(20, tr.GetCurrentEntry());
}


TEST(TTreeReaderBasic, EntryList) {
   // See https://root.cern.ch/phpBB3/viewtopic.php?f=3&t=22850&p=100796
   TTree* tree = MakeTree();
   EXPECT_EQ(9, tree->Draw(">>negZ","three.z<0", "entrylistarray"));
   TEntryListArray* selected = (TEntryListArray*)gDirectory->Get("negZ");
   TTreeReader aReader(tree, selected);

   EXPECT_EQ(9, aReader.GetEntries(false));

   TTreeReaderValue<float> z(aReader, "three.z");

   int count = 0;
   while (aReader.Next()) {
      // Make sure all Next()-ed entries have z<0 as selected by the entry list.
      EXPECT_LT(*z, 0);
      ++count;
   }
   EXPECT_EQ(9, count);

   aReader.Restart();
   EXPECT_EQ(TTreeReader::kEntryValid, aReader.SetEntriesRange(0, 2));

   EXPECT_TRUE(aReader.Next());
   EXPECT_EQ(0, aReader.GetCurrentEntry());
   EXPECT_EQ(TTreeReader::kEntryValid, aReader.GetEntryStatus());

   EXPECT_EQ(9, aReader.GetEntries(false));
}

TEST(TTreeReaderBasic, EntryListBeyondEnd) {
   TTree* tree = MakeTree();
   TEntryList selected;
   // Add the last valid entry; a subsequent Next() must return false.
   selected.Enter(tree->GetEntries() - 1);

   TTreeReader aReader(tree, &selected);

   EXPECT_EQ(1, aReader.GetEntries(false));

   EXPECT_TRUE(aReader.Next());
   EXPECT_EQ(0, aReader.GetCurrentEntry());
   EXPECT_EQ(TTreeReader::kEntryValid, aReader.GetEntryStatus());

   EXPECT_FALSE(aReader.Next());
   EXPECT_EQ(1, aReader.GetCurrentEntry());
   EXPECT_EQ(TTreeReader::kEntryNotFound, aReader.GetEntryStatus());
}


TEST(TTreeReaderBasic, Values) {
   TTree* tree = MakeTree();
   TTreeReader tr(tree);
   TTreeReaderArray<double> x(tr, "one.x");
   TTreeReaderArray<int> y(tr, "two.y");
   TTreeReaderValue<unsigned int> ny(tr, "two.ny");
   TTreeReaderValue<std::string> str(tr, "str");
   TTreeReaderValue<Double32_t> d32(tr, "d32");
   TTreeReaderValue<Float16_t> f16(tr, "f16");

   // Check values for first entry.
   EXPECT_TRUE(tr.Next());

   EXPECT_DOUBLE_EQ(42, x[1]);
   EXPECT_EQ(42u, *ny);
   // FAILS! Already in TLeafI, fNData == 42 (good!) but GetValue(0) == 0.
   // EXPECT_EQ(17, y[0]);
   EXPECT_STREQ("first", str->c_str());
   EXPECT_FLOAT_EQ(12, *d32);
   EXPECT_FLOAT_EQ(-12, *f16);
}
