#include <string.h>

#include "TEntryList.h"
#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"

#include "gtest/gtest.h"

void FillTree(const char *filename, const char *treeName, int nevents) {
   TFile f{filename, "RECREATE"};
   TTree t{treeName, treeName};

   int b;
   t.Branch("b1", &b);

   for (int i = 0; i < nevents; ++i) {
      b = i;
      t.Fill();
   }

   t.Write();
   f.Close();
}

TEST(TEntryList, addSubList) {

   const auto start1{0};
   const auto end1{20};
   auto treename1{"entrylist_addsublist_tree10entries"};
   auto filename1{"entrylist_addsublist_tree10entries.root"};
   const auto nentries1{10};

   const auto start2{0};
   const auto end2{10};
   auto treename2{"entrylist_addsublist_tree20entries"};
   auto filename2{"entrylist_addsublist_tree20entries.root"};
   const auto nentries2{20};

   FillTree(filename1, treename1, nentries1);
   FillTree(filename2, treename2, nentries2);

   TEntryList elist;
   TEntryList sublist1{"", "", treename1, filename1};
   TEntryList sublist2{"", "", treename2, filename2};

   for(auto entry = start1; entry < end1; entry++){
      sublist1.Enter(entry);
   }

   for(auto entry = start2; entry < end2; entry++){
      sublist2.Enter(entry);
   }

   elist.AddSubList(&sublist1);
   elist.AddSubList(&sublist2);

   EXPECT_EQ(elist.GetN(), 30);
   EXPECT_TRUE(elist.GetLists() != nullptr);
   EXPECT_EQ(elist.GetLists()->GetEntries(), 2);
   EXPECT_EQ(strcmp(static_cast<TEntryList *>(elist.GetLists()->At(0))->GetTreeName(), treename1), 0);
   EXPECT_EQ(strcmp(static_cast<TEntryList *>(elist.GetLists()->At(0))->GetFileName(), filename1), 0);
   EXPECT_EQ(strcmp(static_cast<TEntryList *>(elist.GetLists()->At(1))->GetTreeName(), treename2), 0);
   EXPECT_EQ(strcmp(static_cast<TEntryList *>(elist.GetLists()->At(1))->GetFileName(), filename2), 0);

   gSystem->Unlink(filename1);
   gSystem->Unlink(filename2);
}
