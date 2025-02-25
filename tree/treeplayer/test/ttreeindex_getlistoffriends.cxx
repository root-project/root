#include "TBranch.h"
#include "TChain.h"
#include "TCollection.h" // TRangeDynCast
#include "TFile.h"
#include "TFriendElement.h"
#include "TObjArray.h"
#include "TTree.h"

#include "gtest/gtest.h"

#include <string>
#include <string_view>
#include <vector>

void write_data(std::string_view treename, std::string_view filename)
{
   TFile f{filename.data(), "update"};
   TTree t{treename.data(), treename.data()};
   int runNumber{};
   int eventNumber{};
   float val{};
   t.Branch("runNumber", &runNumber, "runNumber/I");
   t.Branch("eventNumber", &eventNumber, "eventNumber/I");
   t.Branch("val", &val, "val/F");
   if (treename == "main") {
      for (auto rn = 0; rn < 3; rn++) {
         runNumber = rn;
         for (auto en = 0; en < 5; en++) {
            eventNumber = en;
            val = en * rn;
            t.Fill();
         }
      }
   } else {
      for (auto rn = 0; rn < 3; rn++) {
         runNumber = rn;
         for (auto en = 4; en >= 0; en--) {
            eventNumber = en;
            val = en * rn;
            t.Fill();
         }
      }
   }

   f.Write();
}

struct TTreeIndexGH_17820 : public ::testing::Test {
   constexpr static auto fFileName{"ttreeindex_getlistoffriends_gh_17820.root"};
   constexpr static auto fMainName{"main"};
   constexpr static auto fFriendName{"friend"};

   static void SetUpTestCase()
   {
      write_data(fMainName, fFileName);
      write_data(fFriendName, fFileName);
   }

   static void TearDownTestCase() { std::remove(fFileName); }
};

void expect_branch_names(const TObjArray *branches, const std::vector<std::string> &branchNames)
{
   auto nBranchNames = branchNames.size();
   decltype(nBranchNames) nBranches{};
   for (const auto *br : TRangeDynCast<TBranch>(branches)) {
      EXPECT_STREQ(br->GetName(), branchNames[nBranches].c_str());
      nBranches++;
   }
   EXPECT_EQ(nBranches, nBranchNames);
}

// Regression test for https://github.com/root-project/root/issues/17820
TEST_F(TTreeIndexGH_17820, RunTest)
{
   TChain mainChain{fMainName};
   mainChain.AddFile(fFileName);

   TChain friendChain{fFriendName};
   friendChain.AddFile(fFileName);
   friendChain.BuildIndex("runNumber", "eventNumber");

   mainChain.AddFriend(&friendChain);

   // Calling GetEntries used to mess with the fTree data member of the main
   // chain, not connecting it to the friend chain and thus losing the list
   // of friends. This in turn corrupted the list of branches.
   mainChain.GetEntries();

   const auto *listOfBranches = mainChain.GetListOfBranches();
   ASSERT_TRUE(listOfBranches);

   const std::vector<std::string> expectedNames{"runNumber", "eventNumber", "val"};

   expect_branch_names(listOfBranches, expectedNames);

   const auto *curTree = mainChain.GetTree();
   ASSERT_TRUE(curTree);

   const auto *listOfFriends = mainChain.GetTree()->GetListOfFriends();
   ASSERT_TRUE(listOfFriends);
   EXPECT_EQ(listOfFriends->GetEntries(), 1);

   auto *friendTree = dynamic_cast<TTree *>(dynamic_cast<TFriendElement *>(listOfFriends->At(0))->GetTree());
   ASSERT_TRUE(friendTree);

   EXPECT_STREQ(friendTree->GetName(), fFriendName);
   const auto *friendFile = friendTree->GetCurrentFile();
   ASSERT_TRUE(friendFile);
   EXPECT_STREQ(friendFile->GetName(), fFileName);

   const auto *friendBranches = friendTree->GetListOfBranches();
   expect_branch_names(friendBranches, expectedNames);
}
