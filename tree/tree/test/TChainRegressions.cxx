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
   TEntryList* myList = static_cast<TEntryList *>(gDirectory->Get("myList"));
   ch.SetEntryList(myList);
   EXPECT_FLOAT_EQ(ch.GetMinimum("value"), 2.);
   EXPECT_FLOAT_EQ(ch.GetMaximum("value"), 10.);
}