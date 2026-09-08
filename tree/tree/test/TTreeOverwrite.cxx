#include "gtest/gtest.h"

#include <ROOT/TestSupport.hxx>

#include <TFile.h>
#include <TTree.h>

TEST(TTree, Overwrite)
{
   ROOT::TestSupport::FileRaii fileGuard("test_ttree_overwrite.root");

   auto f = TFile::Open(fileGuard.GetPath().c_str(), "RECREATE");
   auto t = new TTree("t", "");
   f->Write();
   f->Close();
   delete f;

   f = TFile::Open(fileGuard.GetPath().c_str(), "UPDATE");
   EXPECT_FALSE(f->TestBit(TFile::kRecovered));
   t = f->Get<TTree>("t");
   t->Delete("all");
   f->Write();
   f->Close();
   delete f;

   f = TFile::Open(fileGuard.GetPath().c_str(), "UPDATE");
   // Empty files are always "recovered" because they don't contain keys; however, TFile::Recover() doesn't
   // set the kRecovered bit because there are no keys to recover.
   EXPECT_FALSE(f->TestBit(TFile::kRecovered));
   t = new TTree("t", "");
   f->Write();
   f->Close();
   delete f;

   f = TFile::Open(fileGuard.GetPath().c_str(), "UPDATE");
   EXPECT_FALSE(f->TestBit(TFile::kRecovered));
   EXPECT_NE(nullptr, f->Get<TTree>("t"));
   f->Close();
   delete f;
}
