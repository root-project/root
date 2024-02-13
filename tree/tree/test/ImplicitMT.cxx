#include "TChain.h"
#include "TFile.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TTree.h"

#include "gtest/gtest.h"

#ifdef R__USE_IMT

// ROOT-9668
TEST(TTreeImplicitMT, flushBaskets)
{
   ROOT::EnableImplicitMT();
   const auto ofileName = "flushBasketsMT.root";
   TFile f(ofileName, "RECREATE");
   TTree t("t", "t");
   double b1 = 0.;
   double b2 = 0.;
   double b3 = 0.;
   t.Branch("branch1", &b1);
   t.Branch("branch2", &b2);
   t.Fill();
   t.Write(); // This invokes FlushBaskets!
   t.Branch("branch3", &b3);
   t.Fill();
   t.Write(); // This invokes FlushBaskets!
   f.Close();
   gSystem->Unlink(ofileName);
}

//# 8720
TEST(TChainImplicitMT, propagateToTTree)
{
   ROOT::DisableImplicitMT();

   const auto fname0 = "propagateToTTree0.root";
   const auto fname1 = "propagateToTTree1.root";
   for (const auto fname : {fname0, fname1}) {
      TFile f(fname, "RECREATE");
      TTree t("e", "e");
      t.Fill();
      t.Write();
      EXPECT_FALSE(t.GetImplicitMT());
   }

   TChain c("e");
   c.SetImplicitMT(true);
   c.Add(fname0);
   c.Add(fname1);
   c.LoadTree(0);
   EXPECT_TRUE(c.GetTree()->GetImplicitMT());
   c.LoadTree(1);
   EXPECT_TRUE(c.GetTree()->GetImplicitMT());

   gSystem->Unlink(fname0);
   gSystem->Unlink(fname1);
}

#endif // R__USE_IMT
