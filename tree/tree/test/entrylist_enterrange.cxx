#include <memory>
#include <string.h>

#include "TEntryList.h"
#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"

#include "gtest/gtest.h"

void FillTree(const char *filename, const char *treeName, int nevents)
{
   TFile f{filename, "RECREATE"};
   TTree t{treeName, treeName};

   int b;
   t.Branch("b1", &b);

   for (int i = 0; i < nevents; ++i) {
      b = i * 10;
      t.Fill();
   }

   t.Write();
   f.Close();
}

TEST(TEntryList, enterRange)
{
   const auto start1{10};
   const auto end1{20};
   auto treename1{"entrylist_enterrange_tree20entries"};
   auto filename1{"entrylist_enterrange_tree20entries.root"};
   const auto nentries1{20};

   FillTree(filename1, treename1, nentries1);

   TFile f{filename1};
   std::unique_ptr<TTree> t{f.Get<TTree>(treename1)};
   int b1;
   t->SetBranchAddress("b1", &b1);

   TEntryList elist{"", "", treename1, filename1};
   elist.EnterRange(start1, end1);

   t->SetEntryList(&elist);

   EXPECT_EQ(elist.GetN(), 10);
   EXPECT_EQ(t->GetEntries(), 20);
   // The branch "b1" of the test tree holds integers in the range [0, 20)
   // multiplied by 10. We traverse the tree with the entry numbers in the
   // TEntryList and check that they have the expected value.
   for (auto entry = 0; entry < elist.GetN(); entry++) {
      t->GetEntry(elist.GetEntry(entry));
      // We added the range of entries with values [10, 20) multiplied by 10 to
      // the TEntryList
      auto correct_value = (entry + 10) * 10;
      EXPECT_EQ(b1, correct_value);
   }

   gSystem->Unlink(filename1);
}
