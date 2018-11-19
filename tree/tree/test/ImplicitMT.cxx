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

#endif // R__USE_IMT
