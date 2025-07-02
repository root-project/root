#include <TChain.h>
#include <TFile.h>
#include <TSystem.h>
#include <TTree.h>
#include <TEntryList.h>
#include <TDirectory.h>

#include "gtest/gtest.h"

class TTreeCache;

// ROOT-10672
TEST(TChain, GetReadCacheBug)
{
   const auto treename = "tree";
   const auto filename = "tchain_getreadcachebug.root";
   {
      TFile f(filename, "recreate");
      ASSERT_FALSE(f.IsZombie());
      TTree t(treename, treename);
      t.Fill();
      t.Write();
      f.Close();
   }

   TChain chain(treename);
   chain.Add(filename);
   chain.GetEntry(0);
   TFile *currentfile = chain.GetCurrentFile();
   TTreeCache *treecache = chain.GetReadCache(currentfile, true);
   EXPECT_NE(treecache, nullptr);

   gSystem->Unlink(filename);
}

// https://its.cern.ch/jira/browse/ROOT-7855
TEST(TChain, CloneTreeZeroEntries)
{
   auto filename = "file7855.root";
   {
      TFile f(filename, "RECREATE");
      TTree t("tree", "tree");
      int n;
      t.Branch("n", &n, "n/I");
      t.Write();
      f.Close();
   }
   auto c = new TChain("tree");
   c->Add(filename);
   auto cc = c->CloneTree(-1, "fast OptimizeBaskets");
   EXPECT_NE(cc, nullptr);
   EXPECT_NE(cc->FindBranch("n"), nullptr);
   auto tc = c->GetTree()->CloneTree(-1, "fast OptimizeBaskets");
   EXPECT_NE(tc, nullptr);
   EXPECT_NE(tc->FindBranch("n"), nullptr);
   gSystem->Unlink(filename);
}

// ROOT-7097, ROOT-8505
TEST(TChain, GetMinMaxEntryList)
{
   std::unique_ptr<TFile> file1(TFile::Open("t1_7067.root", "RECREATE"));
   TTree t1("t", "");
   int value;
   t1.Branch("value", &value);
   value = 0;
   t1.Fill();
   value = 1;
   t1.Fill();
   value = 2;
   t1.Fill();
   value = 3;
   t1.Fill();
   file1->Write();
   file1->Close();

   std::unique_ptr<TFile> file2(TFile::Open("t2_7067.root", "RECREATE"));
   TTree t2("t", "");
   // int value;
   t2.Branch("value", &value);
   value = 10;
   t2.Fill();
   value = 11;
   t2.Fill();
   value = 12;
   t2.Fill();
   value = 13;
   t2.Fill();
   file2->Write();
   file2->Close();

   TChain ch("t");
   ch.AddFile("t1_7067.root");
   ch.AddFile("t2_7067.root");
   EXPECT_FLOAT_EQ(ch.GetMinimum("value"), 0.);
   EXPECT_FLOAT_EQ(ch.GetMaximum("value"), 13.);
   ch.Draw(">>myList", "value<11 && value >1", "entrylist");
   TEntryList *myList = static_cast<TEntryList *>(gDirectory->Get("myList"));
   ch.SetEntryList(myList);
   EXPECT_FLOAT_EQ(ch.GetMinimum("value"), 2.);
   EXPECT_FLOAT_EQ(ch.GetMaximum("value"), 10.);
   gSystem->Unlink("t1_7067.root");
   gSystem->Unlink("t2_7067.root");
}

// https://its.cern.ch/jira/browse/ROOT-8112
TEST(TChain, UncommonFileExtension)
{
   const auto dirname = "hsimple8112.root";
   const auto filename = "hsimple8112.root/hsimple8112.root.2";
   gSystem->mkdir(dirname);
   const auto treename = "tree";
   {
      TFile f(filename, "recreate");
      ASSERT_FALSE(f.IsZombie());
      TTree t(treename, treename);
      t.Fill();
      t.Fill();
      t.Write();
      f.Close();
   }
   {
      TChain chain(treename);
      chain.AddFile(filename);
      EXPECT_EQ(chain.GetEntries(), 2);
   }
   gSystem->Unlink(filename);
   gSystem->Unlink(dirname);
}

// Originally reproducer of https://github.com/root-project/root/issues/7567
// but see also https://github.com/root-project/root/issues/19220 for an update
// The test parameters are
// - int: number of files (1, 2), only relevant for TChain.
// - bool: call `GetEntries` at the beginning or not.
// - bool: call `SetBranchStatus("*", false)` or not.
// - bool: call `SetBranchStatus("random", true)` or not.
class SetBranchStatusInteraction : public ::testing::TestWithParam<std::tuple<int, bool, bool, bool>> {};

TEST_P(SetBranchStatusInteraction, TestTChain)
{
   const auto [nFiles, callGetEntries, deactivateAllBranches, activateSingleBranch] = GetParam();

   const auto treename = "ntuple";
   const auto filename = "$ROOTSYS/tutorials/hsimple.root";

   TChain chain(treename);
   for (auto i = 0; i < nFiles; ++i) {
      chain.Add(filename);
   }
   auto nEntries = callGetEntries ? chain.GetEntries() : chain.GetEntriesFast();
   if (deactivateAllBranches)
      chain.SetBranchStatus("*", 0);
   // read a single branch
   float random = 0.333333;
   TBranch *b_random{nullptr};
   // attention! SetBranchAddress!=SetBranchStatus. When all the branches are deactivated because of a previous
   // `SetBranchStatus("*", false)` call, the only way to actually read the "random" branch is to activate it first!
   if (deactivateAllBranches && activateSingleBranch)
      chain.SetBranchStatus("random", true);
   chain.SetBranchAddress("random", &random, &b_random);
   for (decltype(nEntries) i = 0; i < chain.GetEntriesFast(); ++i) {
      auto bytes = chain.GetEntry(i);
      if (deactivateAllBranches) {
         if (activateSingleBranch)
            ASSERT_GT(bytes, 0);
         else
            ASSERT_EQ(bytes, 0);
      } else {
         // by default the branches will be activated
         ASSERT_GT(bytes, 0);
      }
   }
}

TEST_P(SetBranchStatusInteraction, TestTTree)
{
   const auto [_, callGetEntries, deactivateAllBranches, activateSingleBranch] = GetParam();

   const auto treename = "ntuple";
   const auto filename = "$ROOTSYS/tutorials/hsimple.root";

   auto tfile = std::make_unique<TFile>(filename);
   auto *ttree = tfile->Get<TTree>(treename);

   auto nEntries = callGetEntries ? ttree->GetEntries() : ttree->GetEntriesFast();
   if (deactivateAllBranches)
      ttree->SetBranchStatus("*", 0);
   // read a single branch
   float random = 0.333333;
   TBranch *b_random{nullptr};
   // attention! SetBranchAddress!=SetBranchStatus. When all the branches are deactivated because of a previous
   // `SetBranchStatus("*", false)` call, the only way to actually read the "random" branch is to activate it first!
   if (deactivateAllBranches && activateSingleBranch)
      ttree->SetBranchStatus("random", true);
   ttree->SetBranchAddress("random", &random, &b_random);
   for (decltype(nEntries) i = 0; i < nEntries; ++i) {
      auto bytes = ttree->GetEntry(i);
      if (deactivateAllBranches) {
         if (activateSingleBranch)
            ASSERT_GT(bytes, 0);
         else
            ASSERT_EQ(bytes, 0);
      } else {
         // by default the branches will be activated
         ASSERT_GT(bytes, 0);
      }
   }
}

INSTANTIATE_TEST_SUITE_P(
   RunTests, SetBranchStatusInteraction,
   ::testing::Combine(
      // number of files, only relevant for TChain
      ::testing::Values(1, 2),
      // call `GetEntries` at the beginning or not
      ::testing::Values(true, false),
      // call `SetBranchStatus("*", false)` or not
      ::testing::Values(true, false),
      // call `SetBranchStatus("random", true)` or not
      ::testing::Values(true, false)),
   // Extra parenthesis around lambda to avoid preprocessor errors, see
   // https://stackoverflow.com/questions/79438894/lambda-with-structured-bindings-inside-macro-call
   ([](const testing::TestParamInfo<SetBranchStatusInteraction::ParamType> &paramInfo) {
      auto &&[nFiles, callGetEntries, deactivateAllBranches, activateSingleBranch] = paramInfo.param;
      // googletest only accepts ASCII alphanumeric characters for labels
      std::string label{"f"};
      label += std::to_string(nFiles);
      label += "e";
      label += std::to_string(callGetEntries);
      label += "d";
      label += std::to_string(deactivateAllBranches);
      label += "a";
      label += std::to_string(activateSingleBranch);
      return label;
   }));
