#include "TClass.h"
#include "TList.h"
#include "TRandom.h"
#include "TProfile2D.h"
#include "TProfile2Poly.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <iostream>

using namespace std;

Float_t delta=0.00000000001;


void FillForTest(TProfile2D* tp2d, TProfile2Poly* tpp, TRandom& ran) {
    Double_t value, weight;
    Float_t px, py;

    for (Int_t i = 0; i < 1000000; i++) {
        gRandom->Rannor(px, py);
        value = ran.Gaus(20, 5);
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
    for(Double_t y=0.5; y<10; y+=1.0) {
        for(Double_t x=0.5; x<10; x+=1.0) {
            cont1 = tp2d->GetBinContent(tp2d->FindBin(x,y));
            cont2 = tpp->GetBinContent(tpp->FindBin(x,y));
            ASSERT_NEAR(cont1, cont2, delta);
        }

    }
}

void binErrorCompare(TProfile2D* tp2d, TProfile2Poly* tpp) {
    Double_t cont1, cont2;
    for(Double_t y=0.5; y<10; y+=1.0) {
        for(Double_t x=0.5; x<10; x+=1.0) {
            cont1 = tp2d->GetBinError(tp2d->FindBin(x,y));
            cont2 = tpp->GetBinError(tpp->FindBin(x,y));
            ASSERT_NEAR(cont1, cont2, delta);
        }
    }
}

void SetupPlots(TProfile2Poly* TP2P_2, TProfile2Poly* TP2P, TProfile2D* TP2D_2, TProfile2D* TP2D)
{
    Option_t* opt = new Option_t('s');
    TP2D->SetErrorOption(opt);
    TP2D_2->SetErrorOption(opt);

    float minx = -10; float maxx = 10;
    float miny = -10; float maxy = 10;
    float binsz = 2;

    for (float i = minx; i < maxx; i += binsz) {
        for (float j = miny; j < maxy; j += binsz) {
            TP2P->AddBin(i, j, i + binsz, j + binsz);
            TP2P_2->AddBin(i, j, i + binsz, j + binsz);
        }
    }
}

void test_globalStats() {
    auto TP2D = new TProfile2D("1", "1", 10, -10, 10, 10, -10, 10, 0, 0);
    auto TP2D_2 = new TProfile2D("2", "2", 10, -10, 10, 10, -10, 10, 0, 0);

    auto TP2P = new TProfile2Poly();
    auto TP2P_2 = new TProfile2Poly();

    SetupPlots(TP2P_2, TP2P, TP2D_2, TP2D);

    TRandom ran;

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

    TRandom ran;

    // ----- first  comparison
    FillForTest(TP2D, TP2P, ran);
    binContentCompare(TP2D, TP2P);

    // ----- second  comparison
     FillForTest(TP2D_2, TP2P_2, ran);
     binContentCompare(TP2D_2, TP2P_2);

    // ----- Merging first and second one and then comparing
     TList DMerge;
     TList PMerge;

     DMerge.Add(TP2D_2);
     PMerge.Add(TP2P_2);

     TP2D->Merge(&DMerge);
     TP2P->Merge(&PMerge);

     binContentCompare(TP2D, TP2P);

     delete TP2D;
     delete TP2D_2;
     delete TP2P;
     delete TP2P_2;
}


void test_binErrorStats() {
    auto TP2D = new TProfile2D("1", "1", 10, -10, 10, 10, -10, 10, 0, 0);
    auto TP2D_2 = new TProfile2D("2", "2", 10, -10, 10, 10, -10, 10, 0, 0);

    auto TP2P = new TProfile2Poly();
    auto TP2P_2 = new TProfile2Poly();

    SetupPlots(TP2P_2, TP2P, TP2D_2, TP2D);

    TRandom ran;

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

     delete TP2D;
     delete TP2D_2;
     delete TP2P;
     delete TP2P_2;
}


// ------------ TEST CALLS ------------

TEST(TProfile2Poly, GlobalCompare) {
    test_globalStats();
}

TEST(TProfile2Poly, BinEntriesCompare) {
    test_binEntryStats();
}

TEST(TProfile2Poly, BinErrorCompare) {
    test_binErrorStats();
}
