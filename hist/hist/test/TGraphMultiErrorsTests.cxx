#include "TFile.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TGraphMultiErrors.h"

#include "gtest/gtest.h"

TEST(TGraphMultiErrorsTests, tgraphmultierrorstest)
{
   const auto ofileName = "tgraphmultierrorstest.root";
   TFile f(ofileName, "RECREATE");

   double ax[]      = {0, 1, 2, 3, 4};
   double ay[]      = {0, 2, 4, 1, 3};
   double aexl[]    = {0.3, 0.3, 0.3, 0.3, 0.3};
   double aexh[]    = {0.3, 0.3, 0.3, 0.3, 0.3};
   auto   aeylstat = new double[5]  {1, 0.5, 1, 0.5, 1};
   auto   aeyhstat = new double[5]  {0.5, 1, 0.5, 1, 2.};
   auto   aeylsys  = new double[5]  {0.5, 0.4, 0.8, 0.3, 1.2};
   auto   aeyhsys  = new double[5]  {0.6, 0.7, 0.6, 0.4, 0.8};
   auto   aeyl     = new double*[2] {aeylstat, aeylsys};
   auto   aeyh     = new double*[2] {aeyhstat, aeyhsys};

   auto gme = new TGraphMultiErrors(5, 2, ax, ay, aexl, aexh, aeyl, aeyh, TGraphMultiErrors::kOnlyFirst);
   gme->SetMarkerStyle(20);
   gme->SetLineColor(kRed);
   gme->GetAttLine(0)->SetLineColor(kRed);
   gme->GetAttLine(1)->SetLineColor(kBlue);
   gme->GetAttFill(1)->SetFillStyle(0);

   gme->Write("gme");
   delete gme;

   gme = (TGraphMultiErrors*) f.Get("gme");

   EXPECT_DOUBLE_EQ(gme->GetErrorY(0), 0.79056941504209488);
   EXPECT_DOUBLE_EQ(gme->GetErrorY(4), 1.5811388300841898);

   gme->SetSumErrorsMode(TGraphMultiErrors::kSquareSum);

   EXPECT_DOUBLE_EQ(gme->GetErrorY(1), 0.97467943448089633);
   EXPECT_DOUBLE_EQ(gme->GetErrorY(3), 0.86602540378443871);

   gme->SetSumErrorsMode(TGraphMultiErrors::kAbsSum);

   EXPECT_DOUBLE_EQ(gme->GetErrorY(2), 1.4916433890176297);

   gme->DeleteYError(1);

   EXPECT_EQ(gme->GetNYErrors(), 1);

   gme->DeleteYError(0); // Must not work!

   EXPECT_EQ(gme->GetNYErrors(), 1);

   gme->AddYError(5, aeylsys, aeyhsys);

   EXPECT_EQ(gme->GetNYErrors(), 2);
   EXPECT_DOUBLE_EQ(gme->GetErrorY(0), 1.3152946437965904);

   gme->Set(4);
   gme->Set(5);
   gme->SetPoint(4, ax[4], ay[4]);
   gme->SetPointEX(4, aexl[4], aexh[4]);
   gme->SetPointEY(4, 0, aeyl[0][4], aeyh[0][4]);
   gme->SetPointEY(4, 1, aeyl[1][4], aeyh[1][4]);

   EXPECT_DOUBLE_EQ(gme->GetErrorY(4), 2.5179356624028344);

   delete gme;
   f.Close();
   gSystem->Unlink(ofileName);
}
