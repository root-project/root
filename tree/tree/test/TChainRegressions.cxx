#include <TChain.h>
#include <TFile.h>
#include <TSystem.h>
#include <TTree.h>
 
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