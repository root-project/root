#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TBranch.h"

#include "gtest/gtest.h"

TEST(TTreeTruncatedDatatypes, float16double32leaves)
{
   const auto ofileName = "float16double32leaves.root";
   TFile f(ofileName, "RECREATE");
   auto t = new TTree("t", "t");

   Float16_t  f1 = 0., f2 = 0.;
   Double32_t d1 = 0., d2 = 0.;
   t->Branch("f1", &f1, "f1/f");
   t->Branch("f2", &f2, "f2/f[0,1000000,20]");
   t->Branch("d1", &d1, "d1/d");
   t->Branch("d2", &d2, "d2/d[0,1000000,20]");
   t->Fill();

   f1 = f2 = 1.;
   d1 = d2 = 1.;
   t->Fill();

   f1 = f2 = 999999.;
   d1 = d2 = 999999.;
   t->Fill();

   t->Write();
   delete t;

   t = (TTree*) f.Get("t");
   t->SetBranchAddress("f1", &f1);
   t->SetBranchAddress("f2", &f2);
   t->SetBranchAddress("d1", &d1);
   t->SetBranchAddress("d2", &d2);

   t->GetEntry(0);
   EXPECT_FLOAT_EQ (0., f1);
   EXPECT_FLOAT_EQ (0., f2);
   EXPECT_DOUBLE_EQ(0., d1);
   EXPECT_DOUBLE_EQ(0., d2);

   t->GetEntry(1);
   EXPECT_FLOAT_EQ (1.,               f1);
   EXPECT_FLOAT_EQ (0.95367432,       f2); // Here the precision kicks in
   EXPECT_DOUBLE_EQ(1.,               d1);
   EXPECT_DOUBLE_EQ(0.95367431640625, d2); // Here the precision kicks in

   t->GetEntry(2);
   EXPECT_FLOAT_EQ (999936.,            f1);  // Here the precision kicks in
   EXPECT_FLOAT_EQ (999999.,            f2);
   EXPECT_DOUBLE_EQ(999999.,            d1);
   EXPECT_DOUBLE_EQ(999999.04632568359, d2);

   delete t;
   f.Close();
   gSystem->Unlink(ofileName);
}

// https://its.cern.ch/jira/browse/ROOT-10149
TEST(TTreeTruncatedDatatypes, LeafCounter)
{
   const auto ofileName = "leafcounter10149.root";
   Long64_t nEntries = 10; // must be < 63
   {
      TFile f(ofileName, "RECREATE");
      TTree *t = new TTree("t", "t");
      int n;
      Double32_t arr[64];
      t->Branch("n", &n);
      t->Branch("arr", arr, "arr[n]/d[0,100,32]");
      t->Branch("arr_def", arr, "arr_def[n]/d");
      t->Branch("arr_fix", arr, "arr_fix[64]/d[0,100,32]");
      t->Branch("arr_fix_def", arr, "arr_fix_def[64]/d");
      t->Branch("single", arr, "single/d[0,100,32]");
      t->Branch("single_def", arr, "single_def/d");
      for (auto name : {"arr", "arr_def", "arr_fix", "arr_fix_def", "single", "single_def"}) {
         auto b = t->GetBranch(name);
         EXPECT_NE(b, nullptr);
         auto l = b->GetLeaf(name);
         EXPECT_NE(l, nullptr);
         int len;
         auto b_c = l->GetLeafCounter(len);
         if (strcmp(name, "arr") == 0 || strcmp(name, "arr_def") == 0) {
            EXPECT_NE(b_c, nullptr);
            EXPECT_STREQ(b_c->GetName(), "n");
         } else if (strcmp(name, "arr_fix") == 0 || strcmp(name, "arr_fix_def") == 0) {
            EXPECT_EQ(b_c, nullptr);
            EXPECT_EQ(len, 64);
         } else if (strcmp(name, "single") == 0 || strcmp(name, "single_def") == 0) {
            EXPECT_EQ(b_c, nullptr);
            EXPECT_EQ(len, 1);
         }
      }
      // Fill tree
      for (Long64_t i = 0; i < nEntries; ++i) {
         n = i;
         for (int j = 0; j < 64; ++j) {
            arr[j] = i + 0.01 * j;
         }
         t->Fill();
      }
      t->Write();
      f.Close();
   }
   {
      TFile f(ofileName, "READ");
      TTree *t = f.Get<TTree>("t");
      for (auto name : {"arr", "arr_def", "arr_fix", "arr_fix_def", "single", "single_def"}) {
         auto b = t->GetBranch(name);
         EXPECT_NE(b, nullptr);
         auto l = b->GetLeaf(name);
         EXPECT_NE(l, nullptr);
         int len;
         auto b_c = l->GetLeafCounter(len);
         if (strcmp(name, "arr") == 0 || strcmp(name, "arr_def") == 0) {
            EXPECT_NE(b_c, nullptr);
            EXPECT_STREQ(b_c->GetName(), "n");
         } else if (strcmp(name, "arr_fix") == 0 || strcmp(name, "arr_fix_def") == 0) {
            EXPECT_EQ(b_c, nullptr);
            EXPECT_EQ(len, 64);
         } else if (strcmp(name, "single") == 0 || strcmp(name, "single_def") == 0) {
            EXPECT_EQ(b_c, nullptr);
            EXPECT_EQ(len, 1);
         }
         int n;
         Double32_t arr[64];
         t->SetBranchAddress("n", &n);
         t->SetBranchAddress(name, arr);
         const auto tol = 0.001;
         for (Long64_t i = 0; i < nEntries; ++i) {
            t->GetEntry(i);
            EXPECT_EQ(n, i);
            if (strcmp(name, "arr") == 0 || strcmp(name, "arr_def") == 0) {
               for (int j = 0; j < n; ++j) {
                  EXPECT_NEAR(i + 0.01 * j, arr[j], tol);
               }
            } else if (strcmp(name, "arr_fix") == 0 || strcmp(name, "arr_fix_def") == 0) {
               for (int j = 0; j < 64; ++j) {
                  EXPECT_NEAR(i + 0.01 * j, arr[j], tol);
               }
            } else if (strcmp(name, "single") == 0 || strcmp(name, "single_def") == 0) {
               EXPECT_NEAR(i, arr[0], tol);
            }
         }
         t->ResetBranchAddresses();
      }
      f.Close();
   }
   gSystem->Unlink(ofileName);
}
