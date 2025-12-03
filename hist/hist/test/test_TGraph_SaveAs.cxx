#include "gtest/gtest.h"

#include "TGraph.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"

/// Tests for TGraph::SaveAs
/// Testing all three supported formats and conversion into different errors

const int NP = 5;

const double PREC = 1e-5; // in text file only 5-6 digits are stored

Double_t arrx[NP] = { 1, 2, 3, 4, 5 };
Double_t arry[NP] = { 5, 4, 3, 2, 1 };
Double_t arrex[NP] = { 0.2, 0.3, 0.4, 0.5, 0.6 };
Double_t arrexl[NP] = { 0.25, 0.35, 0.45, 0.55, 0.65 };
Double_t arrexh[NP] = { 0.15, 0.25, 0.35, 0.45, 0.55 };
Double_t arrey[NP] = { 0.6, 0.5, 0.4, 0.3, 0.2 };
Double_t arreyl[NP] = { 0.67, 0.57, 0.47, 0.37, 0.27 };
Double_t arreyh[NP] = { 0.53, 0.43, 0.33, 0.23, 0.13 };

TEST(TGraphsa, SaveGraphAsCSV)
{
   TGraph gr(NP, arrx, arry);

   gr.SaveAs("graph.csv");

   auto gr1 = new TGraph("graph.csv", "");
   EXPECT_EQ(gr1->GetN(), NP);

   for (int n = 0; n < NP; ++n) {
      EXPECT_EQ(arrx[n], gr1->GetPointX(n));
      EXPECT_EQ(arry[n], gr1->GetPointY(n));
   }
}


TEST(TGraphsa, SaveGraphAsTSV)
{
   TGraph gr(NP, arrx, arry);

   gr.SaveAs("graph.tsv");

   auto gr1 = new TGraph("graph.tsv", "");
   EXPECT_EQ(gr1->GetN(), NP);

   for (int n = 0; n < NP; ++n) {
      EXPECT_EQ(arrx[n], gr1->GetPointX(n));
      EXPECT_EQ(arry[n], gr1->GetPointY(n));
   }
}

TEST(TGraphsa, SaveGraphAsTXT)
{
   TGraph gr(NP, arrx, arry);

   gr.SaveAs("graph.txt");

   auto gr1 = new TGraph("graph.txt", "");
   EXPECT_EQ(gr1->GetN(), NP);

   for (int n = 0; n < NP; ++n) {
      EXPECT_EQ(arrx[n], gr1->GetPointX(n));
      EXPECT_EQ(arry[n], gr1->GetPointY(n));
   }
}

TEST(TGraphsa, SaveGraphErrorsAsCSV)
{
   TGraphErrors gr(NP, arrx, arry, arrex, arrey);

   gr.SaveAs("grapherrors.csv", "asroot"); // << asroot important for order of values

   auto gr1 = new TGraphErrors("grapherrors.csv", "");
   EXPECT_EQ(gr1->GetN(), NP);

   for (int n = 0; n < NP; ++n) {
      EXPECT_EQ(arrx[n], gr1->GetPointX(n));
      EXPECT_EQ(arry[n], gr1->GetPointY(n));
      EXPECT_NEAR(arrex[n], gr1->GetErrorX(n), PREC);
      EXPECT_NEAR(arrey[n], gr1->GetErrorY(n), PREC);
   }
}


TEST(TGraphsa, SaveGraphErrorsAsTSV)
{
   TGraphErrors gr(NP, arrx, arry, arrex, arrey);

   gr.SaveAs("grapherrors.tsv", "asroot"); // << asroot important for order of values

   auto gr1 = new TGraphErrors("grapherrors.tsv", "");
   EXPECT_EQ(gr1->GetN(), NP);

   for (int n = 0; n < NP; ++n) {
      EXPECT_EQ(arrx[n], gr1->GetPointX(n));
      EXPECT_EQ(arry[n], gr1->GetPointY(n));
      EXPECT_NEAR(arrex[n], gr1->GetErrorX(n), PREC);
      EXPECT_NEAR(arrey[n], gr1->GetErrorY(n), PREC);
   }
}

TEST(TGraphsa, SaveGraphErrorsAsTXT)
{
   TGraphErrors gr(NP, arrx, arry, arrex, arrey);

   gr.SaveAs("grapherrors.txt", "asroot"); // << asroot important for order of values

   auto gr1 = new TGraphErrors("grapherrors.txt", "");
   EXPECT_EQ(gr1->GetN(), NP);

   for (int n = 0; n < NP; ++n) {
      EXPECT_EQ(arrx[n], gr1->GetPointX(n));
      EXPECT_EQ(arry[n], gr1->GetPointY(n));
      EXPECT_NEAR(arrex[n], gr1->GetErrorX(n), PREC);
      EXPECT_NEAR(arrey[n], gr1->GetErrorY(n), PREC);
   }
}


TEST(TGraphsa, SaveGraphAsymmErrorsAsCSV)
{
   TGraphAsymmErrors gr(NP, arrx, arry, arrexl, arrexh, arreyl, arreyh);

   gr.SaveAs("graphasymmrrors.csv", "asroot"); // << asroot important for order of values

   auto gr1 = new TGraphAsymmErrors("graphasymmrrors.csv", "");
   EXPECT_EQ(gr1->GetN(), NP);

   for (int n = 0; n < NP; ++n) {
      EXPECT_EQ(arrx[n], gr1->GetPointX(n));
      EXPECT_EQ(arry[n], gr1->GetPointY(n));
      EXPECT_NEAR(arrexl[n], gr1->GetErrorXlow(n), PREC);
      EXPECT_NEAR(arrexh[n], gr1->GetErrorXhigh(n), PREC);
      EXPECT_NEAR(arreyl[n], gr1->GetErrorYlow(n), PREC);
      EXPECT_NEAR(arreyh[n], gr1->GetErrorYhigh(n), PREC);
   }
}


TEST(TGraphsa, SaveGraphAsymmErrorsAsTSV)
{
   TGraphAsymmErrors gr(NP, arrx, arry, arrexl, arrexh, arreyl, arreyh);

   gr.SaveAs("graphasymmrrors.tsv", "asroot"); // << asroot important for order of values

   auto gr1 = new TGraphAsymmErrors("graphasymmrrors.tsv", "");
   EXPECT_EQ(gr1->GetN(), NP);

   for (int n = 0; n < NP; ++n) {
      EXPECT_EQ(arrx[n], gr1->GetPointX(n));
      EXPECT_EQ(arry[n], gr1->GetPointY(n));
      EXPECT_NEAR(arrexl[n], gr1->GetErrorXlow(n), PREC);
      EXPECT_NEAR(arrexh[n], gr1->GetErrorXhigh(n), PREC);
      EXPECT_NEAR(arreyl[n], gr1->GetErrorYlow(n), PREC);
      EXPECT_NEAR(arreyh[n], gr1->GetErrorYhigh(n), PREC);
   }
}

TEST(TGraphsa, SaveGraphAsymmErrorsAsTXT)
{
   TGraphAsymmErrors gr(NP, arrx, arry, arrexl, arrexh, arreyl, arreyh);

   gr.SaveAs("graphasymmrrors.txt", "asroot"); // << asroot important for order of values

   auto gr1 = new TGraphAsymmErrors("graphasymmrrors.txt", "");
   EXPECT_EQ(gr1->GetN(), NP);

   for (int n = 0; n < NP; ++n) {
      EXPECT_EQ(arrx[n], gr1->GetPointX(n));
      EXPECT_EQ(arry[n], gr1->GetPointY(n));
      EXPECT_NEAR(arrexl[n], gr1->GetErrorXlow(n), PREC);
      EXPECT_NEAR(arrexh[n], gr1->GetErrorXhigh(n), PREC);
      EXPECT_NEAR(arreyl[n], gr1->GetErrorYlow(n), PREC);
      EXPECT_NEAR(arreyh[n], gr1->GetErrorYhigh(n), PREC);
   }
}

TEST(TGraphsa, SaveGraphAsymmErrorsAsOnlyErrorsCSV)
{
   TGraphAsymmErrors gr(NP, arrx, arry, arrexl, arrexh, arreyl, arreyh);

   gr.SaveAs("graphasymmrrors_reduce.csv", "asroot errors"); // << asroot important for order of values

   auto gr1 = new TGraphErrors("graphasymmrrors_reduce.csv", "");
   EXPECT_EQ(gr1->GetN(), NP);

   for (int n = 0; n < NP; ++n) {
      EXPECT_EQ(arrx[n], gr1->GetPointX(n));
      EXPECT_EQ(arry[n], gr1->GetPointY(n));
      EXPECT_NEAR(gr.GetErrorX(n), gr1->GetErrorX(n), PREC);
      EXPECT_NEAR(gr.GetErrorY(n), gr1->GetErrorY(n), PREC);
   }
}

