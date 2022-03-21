#include "TEntryList.h"
#include "TFile.h"
#include "TTree.h"

#include "gtest/gtest.h"

TEST(TEntryList, SimpleEnter) {
   TEntryList e;
   e.Enter(0);
   e.Enter(2);
   e.Enter(4);

   const auto n = e.GetN();
   EXPECT_EQ(n, 3);
   for (int i = 0; i < n; ++i)
      EXPECT_EQ(e.GetEntry(i), i * 2);
}

TEST(TEntryList, EnterWithTree) {
   TTree t("t", "t");
   for (int i = 0; i < 5; ++i)
      t.Fill();

   TEntryList e;
   e.Enter(0, &t);
   e.Enter(2, &t);
   e.Enter(4, &t);

   EXPECT_EQ(e.GetLists(), nullptr);
   const auto n = e.GetN();
   EXPECT_EQ(n, 3);
   for (int i = 0; i < n; ++i)
      EXPECT_EQ(e.GetEntry(i), i * 2);
}

TEST(TEntryList, EnterWithMultipleTrees) {
   TTree t1("t1", "t1");
   for (int i = 0; i < 5; ++i)
      t1.Fill();
   TTree t2("t2", "t2");
   for (int i = 0; i < 11; ++i)
      t2.Fill();

   TEntryList e;
   e.Enter(0, &t1);
   e.Enter(2, &t1);
   e.Enter(4, &t1);

   e.Enter(6, &t2);
   e.Enter(8, &t2);
   e.Enter(10, &t2);

   EXPECT_EQ(e.GetLists()->GetEntries(), 2);
   const auto n = e.GetN();
   EXPECT_EQ(n, 6);
   for (int i = 0; i < n; ++i)
      EXPECT_EQ(e.GetEntry(i), i * 2);
}

TEST(TEntryList, EnterByTreeName) {
   TEntryList e;
   e.Enter(0, "t1", "");
   e.Enter(2, "t1", "");
   e.Enter(4, "t1", "");

   e.Enter(6, "t2", "");
   e.Enter(8, "t2", "");
   e.Enter(10, "t2", "");

   EXPECT_EQ(e.GetLists()->GetEntries(), 2);
   const auto n = e.GetN();
   EXPECT_EQ(n, 6);
   for (int i = 0; i < n; ++i)
      EXPECT_EQ(e.GetEntry(i), i * 2);
}
