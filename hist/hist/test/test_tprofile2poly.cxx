#include "TClass.h"
#include "TList.h"
#include "TRandom3.h"
#include "TProfile2D.h"
#include "TProfile2Poly.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <iostream>

using namespace std;

Float_t delta=0.00000000001;


void FillForTest(TProfile2D* tp2d, TProfile2Poly* tpp, TRandom& ran) {
   Double_t value, weight;
   Double_t px, py;

   for (Int_t i = 0; i < 1000000; i++) {
      px = ran.Gaus(5,2);
      py = ran.Gaus(4,2);
      value = ran.Gaus(20, 5);
      //value = ran.Uniform(0, 20);
      weight = ran.Gaus(17,20);
      tp2d->Fill(px, py, value, weight);
      tpp->Fill(px, py, value, weight);
   }
}

void globalStatsCompare(TProfile2D* tp2d, TProfile2Poly* tpp) {
   for(Int_t c=1; c<=3; ++c) {
      ASSERT_NEAR(tp2d->GetMean(c), tpp->GetMean(c), delta);
      ASSERT_NEAR(tp2d->GetMeanError(c), tpp->GetMeanError(c), delta);
      ASSERT_NEAR(tp2d->GetStdDev(c), tpp->GetStdDev(c), delta);
   }
}

void binContentCompare(TProfile2D* tp2d, TProfile2Poly* tpp) {
   Double_t cont1, cont2;
   for(Double_t y=0.5; y<10; y+=2.0) {
      for(Double_t x=0.5; x<10; x+=2.0) {
         cont1 = tp2d->GetBinContent(tp2d->FindBin(x,y));
         cont2 = tpp->GetBinContent(tpp->FindBin(x,y));
         ASSERT_NEAR(cont1, cont2, delta);
      }
   }
   // test overflow
   cont1 = tp2d->GetBinContent(tp2d->FindBin(11,11));
   cont2 = tpp->GetBinContent(tpp->FindBin(11,11));
   ASSERT_NEAR(cont1, cont2, delta);
}

void binEntriesCompare(TProfile2D* tp2d, TProfile2Poly* tpp) {
   Double_t cont1, cont2;
   for(Double_t y=0.5; y<10; y+=2.0) {
      for(Double_t x=0.5; x<10; x+=2.0) {
         cont1 = tp2d->GetBinEffectiveEntries(tp2d->FindBin(x,y));
         cont2 = tpp->GetBinEffectiveEntries(tpp->FindBin(x,y));
         ASSERT_NEAR(cont1, cont2, delta);
      }

   }
   // test overflow
   cont1 = tp2d->GetBinEffectiveEntries(tp2d->FindBin(11,11));
   cont2 = tpp->GetBinEffectiveEntries(tpp->FindBin(11,11));
   ASSERT_NEAR(cont1, cont2, delta);

}

void binErrorCompare(TProfile2D* tp2d, TProfile2Poly* tpp) {
   Double_t cont1, cont2;
   for(Double_t y=0.5; y<10; y+=2.0) {
      for(Double_t x=0.5; x<10; x+=2.0) {
         cont1 = tp2d->GetBinError(tp2d->FindBin(x,y));
         cont2 = tpp->GetBinError(tpp->FindBin(x,y));
         // std::cout << x << "  " << y << "  " <<  tpp->FindBin(x,y) << "  " <<  tpp->GetBinContent(tpp->FindBin(x,y))
         //           << "   " << tpp->GetBinEffectiveEntries(tpp->FindBin(x,y)) << std::endl;
         ASSERT_NEAR(cont1, cont2, delta);
      }
   }
   // test overflow
   cont1 = tp2d->GetBinError(tp2d->FindBin(11,11));
   cont2 = tpp->GetBinError(tpp->FindBin(11,11));
   ASSERT_NEAR(cont1, cont2, delta);
}

void SetupPlots(TProfile2Poly* TP2P_2, TProfile2Poly* TP2P, TProfile2D* TP2D_2, TProfile2D* TP2D, TString opt = "")
{
     
   TP2D->SetErrorOption(opt);
   TP2D_2->SetErrorOption(opt);
   if (opt == "S") {
      TP2P->SetErrorOption(kERRORSPREAD);
      TP2P_2->SetErrorOption(kERRORSPREAD);
   } else {
      TP2P->SetErrorOption(kERRORMEAN);
      TP2P_2->SetErrorOption(kERRORMEAN);
   }
       
   double minx = -10; double maxx = 10;
   double miny = -10; double maxy = 10;
   double binsz = 2;

   for (double i = minx; i < maxx; i += binsz) {
      for (double j = miny; j < maxy; j += binsz) {
         TP2P->AddBin(i, j, i + binsz, j + binsz);
         TP2P_2->AddBin(i, j, i + binsz, j + binsz);
      }
   }

}

void test_globalStats() {

   TH1::StatOverflows(true);
   
   auto TP2D = new TProfile2D("1", "1", 10, -10, 10, 10, -10, 10, 0, 0);
   auto TP2D_2 = new TProfile2D("2", "2", 10, -10, 10, 10, -10, 10, 0, 0);

   auto TP2P = new TProfile2Poly();
   auto TP2P_2 = new TProfile2Poly();

   SetupPlots(TP2P_2, TP2P, TP2D_2, TP2D);

   TRandom3 ran(1);

   // ----- first  comparison
   FillForTest(TP2D, TP2P, ran);
   globalStatsCompare(TP2D, TP2P);

   // ----- second  comparison
   FillForTest(TP2D_2, TP2P_2, ran);
   globalStatsCompare(TP2D_2, TP2P_2);

   // ----- Merging first and second one and then comparing
   TList DMerge;
   TList PMerge;

   DMerge.Add(TP2D_2);
   PMerge.Add(TP2P_2);

   TP2D->Merge(&DMerge);
   TP2P->Merge(&PMerge);

   globalStatsCompare(TP2D, TP2P);

   DMerge.Clear();
   PMerge.Clear();

   delete TP2D;
   delete TP2D_2;
   delete TP2P;
   delete TP2P_2;
}

void test_binEntryStats() {
   auto TP2D = new TProfile2D("1", "1", 10, -10, 10, 10, -10, 10, 0, 0);
   auto TP2D_2 = new TProfile2D("2", "2", 10, -10, 10, 10, -10, 10, 0, 0);

   auto TP2P = new TProfile2Poly();
   auto TP2P_2 = new TProfile2Poly();

   SetupPlots(TP2P_2, TP2P, TP2D_2, TP2D);

   TRandom3 ran(2);

   // ----- first  comparison
   FillForTest(TP2D, TP2P, ran);
   binContentCompare(TP2D, TP2P);
   binEntriesCompare(TP2D, TP2P);

   // ----- second  comparison
   FillForTest(TP2D_2, TP2P_2, ran);
   binContentCompare(TP2D_2, TP2P_2);
   binEntriesCompare(TP2D_2, TP2P_2);

   // ----- Merging first and second one and then comparing
   TList DMerge;
   TList PMerge;

   DMerge.Add(TP2D_2);
   PMerge.Add(TP2P_2);

   TP2D->Merge(&DMerge);
   TP2P->Merge(&PMerge);

   binContentCompare(TP2D, TP2P);
   binEntriesCompare(TP2D, TP2P);

   DMerge.Clear();
   PMerge.Clear();

   delete TP2D;
   delete TP2D_2;
   delete TP2P;
   delete TP2P_2;
}


void test_binErrorSpreadStats() {
   auto TP2D = new TProfile2D("1", "1", 10, -10, 10, 10, -10, 10, 0, 0);
   auto TP2D_2 = new TProfile2D("2", "2", 10, -10, 10, 10, -10, 10, 0, 0);

   auto TP2P = new TProfile2Poly();
   auto TP2P_2 = new TProfile2Poly();

   SetupPlots(TP2P_2, TP2P, TP2D_2, TP2D, "S");

   TRandom3 ran(3);

   // ----- first  comparison
   FillForTest(TP2D, TP2P, ran);
   binErrorCompare(TP2D, TP2P);

   // ----- second  comparison
   FillForTest(TP2D_2, TP2P_2, ran);
   binErrorCompare(TP2D_2, TP2P_2);

   // ----- Merging first and second one and then comparing
   TList DMerge;
   TList PMerge;

   DMerge.Add(TP2D_2);
   PMerge.Add(TP2P_2);

   TP2D->Merge(&DMerge);
   TP2P->Merge(&PMerge);

   binErrorCompare(TP2D, TP2P);

   DMerge.Clear();
   PMerge.Clear();

   delete TP2D;
   delete TP2D_2;
   delete TP2P;
   delete TP2P_2;
}

void test_binErrorMeanStats() {
   auto TP2D = new TProfile2D("1", "1", 10, -10, 10, 10, -10, 10, 0, 0);
   auto TP2D_2 = new TProfile2D("2", "2", 10, -10, 10, 10, -10, 10, 0, 0);

   auto TP2P = new TProfile2Poly();
   auto TP2P_2 = new TProfile2Poly();

   SetupPlots(TP2P_2, TP2P, TP2D_2, TP2D,"");

   TRandom3 ran(3);

   // ----- first  comparison
   FillForTest(TP2D, TP2P, ran);
   binErrorCompare(TP2D, TP2P);

   // ----- second  comparison
   FillForTest(TP2D_2, TP2P_2, ran);
   binErrorCompare(TP2D_2, TP2P_2);

   // ----- Merging first and second one and then comparing
   TList DMerge;
   TList PMerge;

   DMerge.Add(TP2D_2);
   PMerge.Add(TP2P_2);

   TP2D->Merge(&DMerge);
   TP2P->Merge(&PMerge);

   binErrorCompare(TP2D, TP2P);

   DMerge.Clear();
   PMerge.Clear();

   delete TP2D;
   delete TP2D_2;
   delete TP2P;
   delete TP2P_2;
}


// ------------ TEST CALLS ------------

TEST(TProfile2Poly, GlobalCompare) {
   test_globalStats();
}

TEST(TProfile2Poly, BinContentCompare) {
   test_binEntryStats();
}

TEST(TProfile2Poly, BinErrorSpreadCompare) {
   test_binErrorSpreadStats();
}
TEST(TProfile2Poly, BinErrorMeanCompare) {
   test_binErrorMeanStats();
}
